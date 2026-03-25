import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from models import create_model, list_models


class OnnxExportWrapper(nn.Module):
	def __init__(self, model: nn.Module, export_probs: bool = True):
		super().__init__()
		self.model = model
		self.export_probs = export_probs

	def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
		logits = self.model(pixel_values=pixel_values).logits # 原始模型输出logits
		if self.export_probs: # 如果需要导出概率，则对logits进行后处理
			if logits.shape[-1] == 1:
				return torch.sigmoid(logits)
			return torch.softmax(logits, dim=-1)
		return logits


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export a trained model to ONNX.")
	parser.add_argument(
		"--model_name",
		required=True,
		help=f"Model name. Available: {', '.join(list_models())}",
	)
	parser.add_argument("--num_classes", type=int, default=1)
	parser.add_argument(
		"--checkpoint",
		required=True,
		help="Checkpoint directory that contains model.safetensors or pytorch_model.bin.",
	)
	parser.add_argument(
		"--output",
		required=True,
		help="Output ONNX file path.",
	)
	
	parser.add_argument("--batch_size", type=int, default=1)
	parser.add_argument("--height", type=int, default=None)
	parser.add_argument("--width", type=int, default=None)
	parser.add_argument("--opset", type=int, default=17)
	parser.add_argument(
		"--device",
		default="cuda" if torch.cuda.is_available() else "cpu",
		choices=["cpu", "cuda"],
		help="Device used for export.",
	)
	parser.add_argument(
		"--export_probs",
		action="store_true",
		help="Export post-processed probabilities instead of raw logits.",
	)
	parser.add_argument(
		"--verify",
		action="store_true",
		help="Run ONNX structural validation after export.",
	)
	return parser.parse_args()


def load_checkpoint_weights(model: nn.Module, checkpoint_dir: str) -> None:
	checkpoint_path = Path(checkpoint_dir)
	if not checkpoint_path.exists() or not checkpoint_path.is_dir():
		raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

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


def infer_input_size(model: nn.Module) -> Tuple[int, int]:
	backbone = getattr(model, "backbone", None)
	if backbone is not None:
		for attr_name in ("pretrained_cfg", "default_cfg"):
			cfg = getattr(backbone, attr_name, None)
			if cfg and "input_size" in cfg:
				_, height, width = cfg["input_size"]
				return int(height), int(width)

	if hasattr(model, "get_image_processor_or_transform"):
		try:
			from timm.data.config import resolve_data_config

			config = resolve_data_config({}, model=backbone or model)
			_, height, width = config.get("input_size", (3, 224, 224))
			return int(height), int(width)
		except Exception:
			pass

	return 224, 224


def verify_onnx_file(onnx_path: Path) -> None:
	try:
		import onnx
	except Exception as exc:
		raise RuntimeError(
			"ONNX validation requires the 'onnx' package to be installed."
		) from exc

	model = onnx.load(str(onnx_path))
	onnx.checker.check_model(model)
	print(f"ONNX validation succeeded: {onnx_path}")


def export_to_onnx(args: argparse.Namespace) -> Path:
	if args.device == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA is not available, please use --device cpu.")

	model = create_model(args.model_name, num_classes=args.num_classes)
	load_checkpoint_weights(model, args.checkpoint)
	model.eval()

	height, width = infer_input_size(model)
	if args.height is not None:
		height = args.height
	if args.width is not None:
		width = args.width

	device = torch.device(args.device)
	wrapper = OnnxExportWrapper(model, export_probs=args.export_probs).to(device)
	dummy_input = torch.randn(args.batch_size, 3, height, width, device=device)

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_name = "probabilities" if args.export_probs else "logits"

	with torch.no_grad():
		torch.onnx.export(
			wrapper,
			(dummy_input,),
			str(output_path),
			export_params=True,
			opset_version=args.opset,
			do_constant_folding=True,
			input_names=["pixel_values"],
			output_names=[output_name],
			dynamic_axes={
				"pixel_values": {0: "batch_size"},
				output_name: {0: "batch_size"},
			},
		)

	print(
		f"Exported ONNX model to {output_path} with input shape "
		f"({args.batch_size}, 3, {height}, {width})."
	)
	return output_path


def main() -> None:
	args = parse_args()
	onnx_path = export_to_onnx(args)
	if args.verify:
		verify_onnx_file(onnx_path)


if __name__ == "__main__":
	main()
