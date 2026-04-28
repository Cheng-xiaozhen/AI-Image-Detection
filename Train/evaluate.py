import os
import json
import argparse
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import Trainer, TrainingArguments, default_data_collator, set_seed

try:
    import torch.distributed as torch_dist

    _TORCH_DIST_AVAILABLE = True
except Exception:
    torch_dist = None
    _TORCH_DIST_AVAILABLE = False

try:
    from sklearn.metrics import average_precision_score, roc_auc_score

    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

from models.registry import create_model, list_models
from train.utils.dataset_builder import ImageSample, build_image_index, ForgeryImageDataset


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def build_compute_metrics(num_classes: int):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        labels = labels.reshape(-1)

        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.reshape(-1)

        if num_classes == 1 or logits.ndim == 1:
            probs = _sigmoid(logits)
            preds = (probs >= 0.5).astype(int)
            metrics = {"accuracy": float((preds == labels).mean())}
            if _SKLEARN_AVAILABLE:
                try:
                    metrics["auc"] = float(roc_auc_score(labels, probs))
                except ValueError:
                    metrics["auc"] = float("nan")
                try:
                    metrics["ap"] = float(average_precision_score(labels, probs))
                except ValueError:
                    metrics["ap"] = float("nan")
            return metrics

        probs = _softmax(logits)
        preds = probs.argmax(axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    return compute_metrics


def resolve_image_handler(
    model,
) -> Tuple[Optional[Any], Optional[Any]]:
    handler = None
    if hasattr(model, "get_image_processor_or_transform"):
        handler = model.get_image_processor_or_transform()
    if handler is None:
        return None, None
    if isinstance(handler, tuple) and len(handler) == 2:
        train_or_processor, eval_transform = handler
        if hasattr(train_or_processor, "model_input_names"):
            return train_or_processor, None
        return None, eval_transform
    if hasattr(handler, "model_input_names"):
        return handler, None
    return None, handler


def cleanup_distributed() -> None:
    if not _TORCH_DIST_AVAILABLE:
        return
    if torch_dist is not None and torch_dist.is_available() and torch_dist.is_initialized():
        torch_dist.destroy_process_group()


def build_dataset_from_samples(
    root_dir: Path,
    processor,
    transform,
    samples: Sequence[ImageSample],
) -> Optional[ForgeryImageDataset]:
    if not samples:
        return None

    return ForgeryImageDataset(
        root_dir=root_dir,
        processor=processor,
        transform=transform,
        return_source_model=False,
        samples=samples,
    )


def evaluate_additional_datasets(
    trainer: Trainer,
    eval_root: Path,
    processor,
    transform,
    all_metrics: dict,
) -> None:
    
    # 1. 如果 eval_root 是单一模型目录，直接评估
    direct_real = (eval_root / "0_real").is_dir()
    direct_fake = (eval_root / "1_fake").is_dir()
    if direct_real or direct_fake:
        eval_samples = build_image_index(eval_root)
        whole_dataset = build_dataset_from_samples(eval_root, processor, transform, eval_samples)
        if whole_dataset is not None:
            whole_metrics = trainer.evaluate(
                eval_dataset=whole_dataset,
                metric_key_prefix="external_eval",
            )
            if trainer.is_world_process_zero():
                print(f"Metrics for external_eval: {whole_metrics}")
            all_metrics["external_eval"] = whole_metrics

        for subset_name, subset_label in (("0_real", 0), ("1_fake", 1)):
            subset_samples = [sample for sample in eval_samples if sample.label == subset_label]
            subset_dataset = build_dataset_from_samples(
                eval_root,
                processor,
                transform,
                subset_samples,
            )
            if subset_dataset is None:
                continue

            subset_metrics = trainer.evaluate(
                eval_dataset=subset_dataset,
                metric_key_prefix=subset_name,
            )
            if trainer.is_world_process_zero():
                print(f"Metrics for {subset_name}: {subset_metrics}")
            all_metrics[subset_name] = subset_metrics
        return

    # 2. eval_root 包含多个模型子目录，分别评估每个子目录
    for model_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        model_samples = build_image_index(model_dir, source_model_override=model_dir.name)
        model_dataset = build_dataset_from_samples(model_dir, processor, transform, model_samples)
        if model_dataset is None:
            continue

        model_metrics = trainer.evaluate(
            eval_dataset=model_dataset,
            metric_key_prefix=model_dir.name,
        )
        if trainer.is_world_process_zero():
            print(f"Metrics for {model_dir.name}: {model_metrics}")
        all_metrics[model_dir.name] = model_metrics


def load_checkpoint_weights(model, checkpoint_dir: str) -> None:
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_path}"
        )

    safetensors_path = checkpoint_path / "model.safetensors"
    bin_path = checkpoint_path / "pytorch_model.bin"

    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError(
                "safetensors is required to load model.safetensors"
            ) from exc
        state_dict = load_file(str(safetensors_path))
        print(f"Loaded checkpoint weights from {safetensors_path}")
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
        print(f"Loaded checkpoint weights from {bin_path}")
    else:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_path}. Expected model.safetensors or pytorch_model.bin."
        )

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"Missing keys when loading checkpoint: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Unexpected keys when loading checkpoint: {load_result.unexpected_keys}")



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate forgery detection models.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--eval_data_dir", type=str, required=True, help="Evaluation dataset directory.")
    parser.add_argument(
        "--model_name",
        default="unifd",
        help=f"Model name. Available: {', '.join(list_models())}",
    )
    parser.add_argument("--num_classes", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)

    
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=50)
    
    parser.add_argument("--dataloader_num_workers", type=int, default=8)

    parser.add_argument("--fp16", action="store_true")

    

    return parser.parse_args()


def main():
    try:
        args = parse_args()
        set_seed(args.seed)

        model = create_model(
            args.model_name,
        )
        load_checkpoint_weights(model, args.checkpoint)
        processor, transform = resolve_image_handler(model)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            do_train=False,
            do_eval=True,
            save_strategy="no",
            report_to="none",
            fp16=args.fp16,
            ddp_find_unused_parameters=False,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=True,
            seed=args.seed,
        )

        os.makedirs(args.output_dir, exist_ok=True)

        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=build_compute_metrics(args.num_classes),
            data_collator=default_data_collator,
        )

        all_metrics = {}
        evaluate_additional_datasets(
            trainer=trainer,
            eval_root=Path(args.eval_data_dir),
            processor=processor,
            transform=transform,
            all_metrics=all_metrics,
        )

        if trainer.is_world_process_zero():
            report_path = Path(args.output_dir) / "eval_report.jsonc"
            with open(report_path, "a", encoding="utf-8") as f: # 先以追加模式写入 
                json.dump(all_metrics, f, ensure_ascii=False, indent=2)
                f.write("\n")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
