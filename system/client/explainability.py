from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


@dataclass
class ExplainResult:
    file_path: str
    score: float
    label: int
    preprocessing_latency_ms: float
    inference_latency_ms: float
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_checkpoint_weights(model: torch.nn.Module, checkpoint_dir: str) -> None:
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    safetensors_path = checkpoint_path / "model.safetensors"
    bin_path = checkpoint_path / "pytorch_model.bin"

    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("safetensors is required to load model.safetensors") from exc
        state_dict = load_file(str(safetensors_path))
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_path}. Expected model.safetensors or pytorch_model.bin."
        )

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"Missing keys when loading checkpoint: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Unexpected keys when loading checkpoint: {load_result.unexpected_keys}")


class GradCAMExplainer:
    def __init__(
        self,
        model_name: str = "convnext2_tiny",
        checkpoint_dir: str = "logs/convnext2_tiny/final_model",
        threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        from models import create_model

        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.threshold = float(threshold)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = create_model(model_name)
        _load_checkpoint_weights(self.model, checkpoint_dir)
        self.model.to(self.device)
        self.model.eval()

        self.processor: Optional[Any] = None
        self.transform: Optional[Any] = None
        handler_fn = getattr(self.model, "get_image_processor_or_transform", None)
        if callable(handler_fn):
            handler: Any = handler_fn()
            if isinstance(handler, tuple) and len(handler) == 2:
                maybe_processor, maybe_transform = handler
                if hasattr(maybe_processor, "model_input_names"):
                    self.processor = maybe_processor
                else:
                    self.transform = maybe_processor
                if self.transform is None:
                    self.transform = maybe_transform
            elif hasattr(handler, "model_input_names"):
                self.processor = handler
            else:
                self.transform = handler

        self.target_layer = self._resolve_target_layer()

    def _resolve_target_layer(self) -> torch.nn.Module:
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            raise RuntimeError("The model does not expose a backbone for Grad-CAM")

        stages = getattr(backbone, "stages", None)
        if stages is not None and len(stages) > 0:
            return stages[-1]

        blocks = getattr(backbone, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            return blocks[-1]

        if hasattr(backbone, "layer4"):
            return backbone.layer4

        raise RuntimeError("Unable to resolve a spatial feature layer for Grad-CAM")

    def _load_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        if self.transform is not None:
            tensor = self.transform(image)
            if isinstance(tensor, torch.Tensor):
                return tensor

        if self.processor is not None:
            processed = self.processor(images=image, return_tensors="pt")
            return processed["pixel_values"][0]

        image = image.resize((384, 384))
        array = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std

    @staticmethod
    def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:
        cam = cam - cam.min()
        max_value = cam.max()
        if float(max_value) <= 1e-8:
            return torch.zeros_like(cam)
        return cam / max_value

    @staticmethod
    def _colorize_heatmap(cam: torch.Tensor, image_size: Sequence[int], label: int) -> Image.Image:
        cam_np = cam.detach().cpu().numpy()
        cam_np = np.clip(cam_np, 0.0, 1.0)
        heatmap = Image.fromarray((cam_np * 255).astype(np.uint8), mode="L")
        if hasattr(Image, "Resampling"):
            resample = Image.Resampling.BILINEAR
        else:  # pragma: no cover - older Pillow fallback
            resample = Image.BILINEAR  # type: ignore[attr-defined]
        heatmap = heatmap.resize((int(image_size[0]), int(image_size[1])), resample)
        if label == 1:
            return ImageOps.colorize(heatmap, black="#1f77b4", white="#d62728")
        return ImageOps.colorize(heatmap, black="#eaf3ff", white="#1f77b4")

    def _build_cam(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=activations.shape[-2:], mode="bilinear", align_corners=False)
        return cam[0, 0]

    def explain(self, image_path: str) -> Dict[str, Any]:
        image = self._load_image(image_path)
        original_size = image.size

        preprocess_start = time.perf_counter()
        pixel_values = self._to_tensor(image).unsqueeze(0).to(self.device)
        preprocessing_seconds = time.perf_counter() - preprocess_start

        activations: Dict[str, torch.Tensor] = {}

        def capture_activations(_module, _inputs, output):
            activations["value"] = output[0] if isinstance(output, (tuple, list)) else output

        handle = self.target_layer.register_forward_hook(capture_activations)
        try:
            infer_start = time.perf_counter()
            with torch.enable_grad():
                pixel_values = pixel_values.requires_grad_(True)
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logit = logits.reshape(-1)[0]
                    probability = torch.sigmoid(logit)
                    label = int(float(probability.item()) >= self.threshold)
                    target_score = logit if label == 1 else -logit
                else:
                    probabilities = torch.softmax(logits, dim=-1)
                    label = int(torch.argmax(probabilities, dim=-1).item())
                    probability = probabilities[0, label]
                    target_score = logits[0, label]

                if "value" not in activations:
                    raise RuntimeError("Failed to capture target layer activations for Grad-CAM")

                grads = torch.autograd.grad(target_score, activations["value"], retain_graph=False, create_graph=False)[0]
                cam = self._build_cam(activations["value"], grads)
                cam = self._normalize_cam(cam)
            inference_seconds = time.perf_counter() - infer_start
        finally:
            handle.remove()

        heatmap = self._colorize_heatmap(cam, original_size, label)
        overlay = Image.blend(image, heatmap, alpha=0.45)

        result = ExplainResult(
            file_path=str(Path(image_path).resolve()),
            score=float(probability.item()),
            label=label,
            preprocessing_latency_ms=(preprocessing_seconds * 1000.0),
            inference_latency_ms=(inference_seconds * 1000.0),
            latency_ms=((preprocessing_seconds + inference_seconds) * 1000.0),
        )

        return {
            "prediction": result.to_dict(),
            "original_image": image,
            "heatmap": heatmap,
            "overlay": overlay,
        }
