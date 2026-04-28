import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from client.inference import TritonInferenceClient, summarize_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton inference client with batch support.")
    parser.add_argument(
        "input_path",
        help="Path to an image file or a directory that contains images.",
    )
    parser.add_argument(
        "--url",
        default="localhost:8000",
        help="Triton HTTP server address.",
    )
    parser.add_argument(
        "--model-name",
        default="convnext2_tiny_fp32",
        help="Triton model name.",
    )
    parser.add_argument(
        "--mode",
        choices=["tensor", "raw"],
        default="tensor",
        help="tensor: client side preprocess; raw: send IMAGE_BYTES to server preprocess model.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per request.")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of in-flight batch requests.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold.")
    parser.add_argument(
        "--backbone-name",
        default="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        help="Backbone name used when mode=tensor.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_client = TritonInferenceClient(
        url=args.url,
        model_name=args.model_name,
        mode=args.mode,
        threshold=args.threshold,
        backbone_name=args.backbone_name,
    )
    image_paths = inference_client.list_image_files(args.input_path)
    predictions = inference_client.predict_paths(
        image_paths,
        batch_size=args.batch_size,
        max_concurrency=args.concurrency,
    )
    payload = {
        "summary": summarize_predictions(predictions),
        "predictions": predictions,
    }
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()