import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPORT_DIR = Path(__file__).resolve().parent

for path in (ROOT_DIR, EXPORT_DIR):
	if str(path) not in sys.path:
		sys.path.insert(0, str(path))

from models import create_model, list_models
from export_onnx import OnnxExportWrapper, infer_input_size, load_checkpoint_weights


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Validate whether ONNX outputs are consistent with the original PyTorch model."
	)
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
		"--onnx_path",
		required=True,
		help="Exported ONNX file path.",
	)
	parser.add_argument("--batch_size", type=int, default=1)
	parser.add_argument("--height", type=int, default=None)
	parser.add_argument("--width", type=int, default=None)
	parser.add_argument("--num_samples", type=int, default=10)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--atol", type=float, default=1e-4)
	parser.add_argument("--rtol", type=float, default=1e-3)
	parser.add_argument(
		"--device",
		default="cpu",
		choices=["cpu", "cuda"],
		help="Device used by the PyTorch reference model.",
	)
	parser.add_argument(
		"--provider",
		default=None,
		help="Optional ONNX Runtime provider override, e.g. CPUExecutionProvider or CUDAExecutionProvider.",
	)
	parser.add_argument(
		"--report",
		default=None,
		help="Optional JSON report output path.",
	)
	return parser.parse_args()


def build_session(onnx_path: Path, provider: Optional[str]):
	try:
		ort = importlib.import_module("onnxruntime")
	except Exception as exc:
		raise RuntimeError(
			"ONNX validation requires the 'onnxruntime' package to be installed."
		) from exc

	available_providers = ort.get_available_providers()
	if provider is not None:
		if provider not in available_providers:
			raise RuntimeError(
				f"Provider '{provider}' is not available. Available providers: {available_providers}"
			)
		providers = [provider]
	elif "CUDAExecutionProvider" in available_providers:
		providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
	else:
		providers = ["CPUExecutionProvider"]

	return ort.InferenceSession(str(onnx_path), providers=providers)


def detect_export_probs(output_name: str) -> bool:
	return "prob" in output_name.lower()


def compute_diff_metrics(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
	abs_diff = np.abs(reference - candidate)
	rel_diff = abs_diff / np.maximum(np.abs(reference), 1e-12)
	return {
		"max_abs_diff": float(abs_diff.max(initial=0.0)),
		"mean_abs_diff": float(abs_diff.mean()),
		"max_rel_diff": float(rel_diff.max(initial=0.0)),
		"mean_rel_diff": float(rel_diff.mean()),
	}


def validate_outputs(args: argparse.Namespace) -> Dict[str, Any]:
	if args.device == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA is not available, please use --device cpu.")

	onnx_path = Path(args.onnx_path)
	if not onnx_path.is_file():
		raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	session = build_session(onnx_path, args.provider)
	input_name = session.get_inputs()[0].name # ONNX模型的输入名称
	output_name = session.get_outputs()[0].name # ONNX模型的输出名称
	export_probs = detect_export_probs(output_name) # 根据输出名称判断是否导出的是概率值（如果输出名称中包含"prob"则认为是概率值）

	model = create_model(args.model_name, num_classes=args.num_classes) # 创建PyTorch模型实例
	load_checkpoint_weights(model, args.checkpoint)
	model.eval()

	height, width = infer_input_size(model)
	if args.height is not None:
		height = args.height
	if args.width is not None:
		width = args.width

	device = torch.device(args.device)
	wrapper = OnnxExportWrapper(model, export_probs=export_probs).to(device) # 包装模型以适应ONNX导出，可能会根据是否导出概率值进行不同的处理
	wrapper.eval() # 设置模型为评估模式，dropout等层会导致输出不一致

	sample_reports: List[Dict[str, Any]] = []
	all_passed = True

	for index in range(args.num_samples):
		input_tensor = torch.randn(args.batch_size, 3, height, width, dtype=torch.float32)

		with torch.no_grad():
			reference_output = wrapper(input_tensor.to(device)).detach().cpu().numpy() # 计算PyTorch模型的输出作为参考结果

		onnx_output = session.run([output_name], {input_name: input_tensor.numpy()})[0] # 运行ONNX模型得到输出结果
		metrics = compute_diff_metrics(reference_output, onnx_output) # 计算PyTorch模型输出与ONNX模型输出的差异指标（绝对误差和相对误差）
		passed = bool(
			np.allclose(reference_output, onnx_output, atol=args.atol, rtol=args.rtol) # 检查PyTorch模型输出与ONNX模型输出是否一致（考虑绝对误差和相对误差）
		)
		all_passed = all_passed and passed

		sample_reports.append(
			{
				"sample_index": index,
				"passed": passed,
				**metrics,
			}
		)

	summary = {
		"model_name": args.model_name,
		"checkpoint": args.checkpoint,
		"onnx_path": str(onnx_path),
		"device": args.device,
		"provider": session.get_providers()[0],
		"input_name": input_name,
		"output_name": output_name,
		"export_probs": export_probs,
		"input_shape": [args.batch_size, 3, height, width],
		"num_samples": args.num_samples,
		"atol": args.atol,
		"rtol": args.rtol,
		"passed": all_passed,
		"max_abs_diff": max(item["max_abs_diff"] for item in sample_reports),
		"mean_abs_diff": float(np.mean([item["mean_abs_diff"] for item in sample_reports])),
		"max_rel_diff": max(item["max_rel_diff"] for item in sample_reports),
		"mean_rel_diff": float(np.mean([item["mean_rel_diff"] for item in sample_reports])),
		"samples": sample_reports,
	}

	return summary


def maybe_write_report(report_path: Optional[str], summary: Dict[str, Any]) -> None:
	if report_path is None:
		return

	path = Path(report_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"Validation report saved to {path}")


def main() -> None:
	args = parse_args()
	summary = validate_outputs(args)

	print(json.dumps(summary, ensure_ascii=False, indent=2))
	maybe_write_report(args.report, summary)

	if not summary["passed"]:
		raise SystemExit("ONNX validation failed: outputs differ from the PyTorch model.")

	print("ONNX validation passed: outputs are consistent with the PyTorch model.")


if __name__ == "__main__":
	main()
