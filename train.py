import os
import argparse
import shlex
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple
import json
import numpy as np
from transformers import Trainer, TrainingArguments, TrainerCallback, default_data_collator, set_seed

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
from utils.dataset_builder import ImageSample, build_image_index, split_samples, ForgeryImageDataset,build_datasets_index


class StopOnAccuracyThresholdCallback(TrainerCallback):
    def __init__(self, accuracy_threshold: float = 0.99):
        self.accuracy_threshold = accuracy_threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return control

        accuracy = metrics.get("eval_accuracy")
        if accuracy is not None and accuracy >= self.accuracy_threshold:
            control.should_training_stop = True
            print(
                f"Reached eval_accuracy={accuracy:.4f} >= {self.accuracy_threshold:.2f}. "
                "Stopping training early."
            )

        return control


class EarlyStopOnPlateauCallback(TrainerCallback):
    def __init__(self, patience: int = 3, metric_name: str = "eval_accuracy"):
        self.patience = patience
        self.metric_name = metric_name
        self.best_metric: Optional[float] = None
        self.num_bad_epochs = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return control

        current_metric = metrics.get(self.metric_name)
        if current_metric is None:
            return control

        if self.best_metric is None or current_metric > self.best_metric:
            self.best_metric = current_metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            print(
                f"{self.metric_name} did not improve for {self.num_bad_epochs} epoch(s). "
                f"Best={self.best_metric:.4f}, Current={current_metric:.4f}"
            )

        if self.num_bad_epochs >= self.patience:
            control.should_training_stop = True
            print(
                f"Early stopping triggered: {self.metric_name} has not improved for "
                f"{self.patience} consecutive epochs."
            )

        return control


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


# def resolve_image_handler(model) -> Tuple[object, object]:
#     handler = None
#     if hasattr(model, "get_image_processor_or_transform"):
#         handler = model.get_image_processor_or_transform()
#     if handler is None:
#         return None, None
#     if hasattr(handler, "model_input_names"):
#         return handler, None
#     return None, handler

def resolve_image_train_eval_handler(
    model,
) -> Tuple[Optional[Callable], Optional[Callable]]:
    if hasattr(model, "get_image_processor_or_transform"):
        train_transform, eval_transform = model.get_image_processor_or_transform()
        return train_transform, eval_transform
    return None, None


def cleanup_distributed() -> None:
    if not _TORCH_DIST_AVAILABLE:
        return
    if torch_dist is not None and torch_dist.is_available() and torch_dist.is_initialized():
        torch_dist.destroy_process_group()


def build_dataset_from_samples(
    root_dir: Path,
    eval_transform,
    samples: Sequence[ImageSample],
) -> Optional[ForgeryImageDataset]:
    if not samples:
        return None

    return ForgeryImageDataset(
        root_dir=root_dir,
        processor=None,
        transform=eval_transform,
        return_source_model=False,
        samples=samples,
    )


def evaluate_additional_datasets(trainer, eval_root: Path, eval_transform, all_metrics: dict) -> None:
    # 1. 如果eval_root下直接有0real和1fake两个子目录，则整体评测一次，再分别评测这两个子目录
    direct_real = (eval_root / "0real").is_dir()
    direct_fake = (eval_root / "1fake").is_dir()
    if direct_real or direct_fake:
        eval_samples = build_image_index(eval_root)
        whole_dataset = build_dataset_from_samples(eval_root, eval_transform, eval_samples)
        if whole_dataset is not None:
            whole_metrics = trainer.evaluate(
                eval_dataset=whole_dataset,
                metric_key_prefix="external_eval",
            )
            if trainer.is_world_process_zero():
                print(f"Metrics for external_eval: {whole_metrics}")
            all_metrics["external_eval"] = whole_metrics

        for subset_name, subset_label in (("0real", 0), ("1fake", 1)):
            subset_samples = [sample for sample in eval_samples if sample.label == subset_label]
            subset_dataset = build_dataset_from_samples(eval_root, eval_transform, subset_samples)
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
    
    # 2. eval_root下没有直接的0real/1fake子目录，eval_root下每个子目录都是一个模型来源，对每个子目录进行评测
    for model_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        model_samples = build_image_index(model_dir)
        model_dataset = build_dataset_from_samples(model_dir, eval_transform, model_samples)
        if model_dataset is None:
            continue

        model_metrics = trainer.evaluate(
            eval_dataset=model_dataset,
            metric_key_prefix=model_dir.name,
        )
        if trainer.is_world_process_zero():
            print(f"Metrics for {model_dir.name}: {model_metrics}")
        all_metrics[model_dir.name] = model_metrics


def find_latest_checkpoint(output_dir: str) -> str:
    checkpoints = []
    output_path = Path(output_dir)
    if not output_path.exists():
        return ""
    for entry in output_path.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("checkpoint-"):
            continue
        step_text = name.split("checkpoint-")[-1]
        if step_text.isdigit():
            checkpoints.append((int(step_text), str(entry)))
    if not checkpoints:
        return ""
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def parse_eval_roots(eval_data_dir: Optional[str]) -> Sequence[Path]:
    if not eval_data_dir:
        return []

    dir_candidates = shlex.split(eval_data_dir)
    eval_roots = []
    for dir_path in dir_candidates:
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            print(f"Skipping invalid eval_data_dir: {path}")
            continue
        eval_roots.append(path)
    return eval_roots


def parse_args():
    parser = argparse.ArgumentParser(description="Train forgery detection models.")
    parser.add_argument("--data_dir", required=True, help="Dataset root directory.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default=None,
        help=(
            "Evaluation dataset directory (optional). "
            "Supports one or multiple directories separated by spaces."
        ),
    )
    parser.add_argument(
        "--model_name",
        default="unifd",
        help=f"Model name. Available: {', '.join(list_models())}",
    )
    parser.add_argument("--num_classes", type=int, default=1)

    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify", action="store_true", default=True)
    parser.add_argument("--no_stratify", dest="stratify", action="store_false")

    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--metric_for_best_model", default="accuracy")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    #parser.add_argument("--no_resume", dest="resume", action="store_false")
    

    return parser.parse_args()


def main():
    try:
        args = parse_args()
        set_seed(args.seed)

        model = create_model(
            args.model_name,
        )

        train_transform, eval_transform = resolve_image_train_eval_handler(model)

        samples = build_datasets_index(args.data_dir) # 尝试一下使用多个数据集的混合样本进行训练
        train_samples, val_samples = split_samples(
            samples=samples,
            val_ratio=args.val_ratio,
            seed=args.seed,
            stratify=args.stratify,
        )

        train_dataset = ForgeryImageDataset(
            root_dir=args.data_dir,
            processor=None,
            transform=train_transform,
            return_source_model=False,
            samples=train_samples,
        )
        val_dataset = ForgeryImageDataset(
            root_dir=args.data_dir,
            processor=None,
            transform=eval_transform,
            return_source_model=False,
            samples=val_samples,
        )

        metric_for_best = args.metric_for_best_model or (
            "auc" if args.num_classes == 1 else "accuracy"
        )

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best,
            save_total_limit=args.save_total_limit,
            greater_is_better=True,
            logging_steps=args.logging_steps,
            report_to=["tensorboard"],
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
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=build_compute_metrics(args.num_classes),
            data_collator=default_data_collator,
            callbacks=[
                StopOnAccuracyThresholdCallback(accuracy_threshold=0.99),
                EarlyStopOnPlateauCallback(patience=2, metric_name="eval_accuracy"),
            ],
        )

        resume_checkpoint = ""
        if args.resume:
            resume_checkpoint = find_latest_checkpoint(args.output_dir)
            if resume_checkpoint:
                if trainer.is_world_process_zero():
                    print(f"Resuming from checkpoint: {resume_checkpoint}")
            else:
                if trainer.is_world_process_zero():
                    print("No checkpoint found. Starting training from scratch.")

        trainer.train(resume_from_checkpoint=resume_checkpoint or None)
        #trainer.save_model(os.path.join(args.output_dir, "final_model")) 节省空间暂时不保存

        all_metrics = {}
        val_metrics = trainer.evaluate()
        if trainer.is_world_process_zero():
            print(f"Final validation metrics: {val_metrics}")
        all_metrics["validation"] = val_metrics

        eval_roots = parse_eval_roots(args.eval_data_dir)
        if eval_roots:
            if len(eval_roots) == 1:
                evaluate_additional_datasets(
                    trainer=trainer,
                    eval_root=eval_roots[0],
                    eval_transform=eval_transform,
                    all_metrics=all_metrics,
                )
            else:
                multi_eval_metrics = {}
                for eval_root in eval_roots:
                    per_root_metrics = {}
                    evaluate_additional_datasets(
                        trainer=trainer,
                        eval_root=eval_root,
                        eval_transform=eval_transform,
                        all_metrics=per_root_metrics,
                    )
                    multi_eval_metrics[str(eval_root)] = per_root_metrics
                all_metrics["multi_eval"] = multi_eval_metrics

        if trainer.is_world_process_zero():
            # 把指标结果写入 JSON 文件，以追加模式写入
            with open(Path(args.output_dir) / "eval_report.json", "a") as f:
                json.dump(all_metrics, f, indent=2)
                f.write("\n")
    finally:
        cleanup_distributed()

    


if __name__ == "__main__":
    main()