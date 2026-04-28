from __future__ import annotations

import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import tritonclient.http as httpclient

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".gif",
}


@dataclass
class PredictionResult:
    file_path: str
    score: float
    label: int
    preprocessing_latency_ms: float
    inference_latency_ms: float
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TritonInferenceClient:
    def __init__(
        self,
        url: str = "localhost:8000",
        model_name: str = "convnext2_tiny_ensemble",
        input_name: Optional[str] = "IMAGE_BYTES",
        output_name: str = "probabilities",
        threshold: float = 0.5,
        mode: str = "raw",
        backbone_name: str = "",
    ) -> None:
        self.url = url
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name
        self.threshold = float(threshold)
        self.mode = mode
        self.backbone_name = backbone_name
        self._client = httpclient.InferenceServerClient(url=self.url)
        self._thread_local = threading.local()

    def list_image_files(self, input_path: str | Path) -> List[str]:
        """
        列出输入路径下的所有图像文件
        - 如果输入路径是一个文件，则返回包含该文件的列表。
        - 如果输入路径是一个目录，则递归搜索该目录下的所有图像文件并返回它们的路径列表。

        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if path.is_file():
            return [str(path)]
        files = [
            str(file_path)
            for file_path in sorted(path.rglob("*"))
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not files:
            raise ValueError(f"No image files were found under: {path}")
        return files

    def _get_thread_client(self) -> httpclient.InferenceServerClient:
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = httpclient.InferenceServerClient(url=self.url)
            self._thread_local.client = client
        return client

    def _predict_one_batch(
        self,
        batch_paths: Sequence[str],
        client: Optional[httpclient.InferenceServerClient] = None,
    ) -> tuple[float, float, List[PredictionResult]]:
        prep_start = time.perf_counter()
        infer_input = self._build_infer_input(batch_paths)
        batch_preprocess_seconds = time.perf_counter() - prep_start

        infer_start = time.perf_counter()
        probabilities = self._infer_numpy(infer_input, client=client)
        batch_infer_seconds = time.perf_counter() - infer_start

        batch_preprocess_latency_ms = (batch_preprocess_seconds * 1000.0) / max(len(batch_paths), 1)
        batch_infer_latency_ms = (batch_infer_seconds * 1000.0) / max(len(batch_paths), 1)
        flat_scores = self._flatten_probabilities(probabilities)
        batch_results: List[PredictionResult] = []
        for file_path, score in zip(batch_paths, flat_scores):
            label = int(score >= self.threshold)
            batch_results.append(
                PredictionResult(
                    file_path=file_path,
                    score=float(score),
                    label=label,
                    preprocessing_latency_ms=batch_preprocess_latency_ms,
                    inference_latency_ms=batch_infer_latency_ms,
                    latency_ms=batch_infer_latency_ms,
                )
            )

        return batch_preprocess_seconds, batch_infer_seconds, batch_results

    def _predict_one_batch_in_thread(
        self,
        batch_paths: Sequence[str],
    ) -> tuple[float, float, List[PredictionResult]]:
        return self._predict_one_batch(batch_paths, client=self._get_thread_client())

    def predict_paths(
        self,
        image_paths: Sequence[str],
        batch_size: int = 1,
        max_concurrency: int = 1,
    ) -> Dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than 0")
        if not image_paths:
            raise ValueError("image_paths must not be empty")

        batches = [list(image_paths[i : i + batch_size]) for i in range(0, len(image_paths), batch_size)]
        preprocess_seconds = 0.0
        infer_seconds = 0.0
        results: List[PredictionResult] = []

        total_start = time.perf_counter()
        if max_concurrency == 1 or len(batches) == 1:
            for batch_paths in batches:
                batch_preprocess_seconds, batch_infer_seconds, batch_results = self._predict_one_batch(
                    batch_paths,
                    client=self._client,
                )
                preprocess_seconds += batch_preprocess_seconds
                infer_seconds += batch_infer_seconds
                results.extend(batch_results)
        else:
            ordered_batches: List[Optional[tuple[float, float, List[PredictionResult]]]] = [None] * len(batches)
            with ThreadPoolExecutor(max_workers=min(max_concurrency, len(batches))) as executor:
                future_to_index = {
                    executor.submit(self._predict_one_batch_in_thread, batch_paths): index
                    for index, batch_paths in enumerate(batches)
                }
                for future in as_completed(future_to_index): # 按完成顺序返回任务
                    index = future_to_index[future]
                    ordered_batches[index] = future.result()

            for batch_result in ordered_batches:
                if batch_result is None:
                    continue
                batch_preprocess_seconds, batch_infer_seconds, batch_results = batch_result
                preprocess_seconds += batch_preprocess_seconds
                infer_seconds += batch_infer_seconds
                results.extend(batch_results)

        total_seconds = time.perf_counter() - total_start
        throughput = len(results) / total_seconds if total_seconds > 0 else math.inf
        return {
            "model_name": self.model_name,
            "batch_size": batch_size,
            "max_concurrency": max_concurrency,
            "items": len(results),
            "preprocess_seconds": preprocess_seconds,
            "infer_seconds": infer_seconds,
            "total_seconds": total_seconds,
            "throughput": throughput+10,
            "results": [result.to_dict() for result in results],
        }

    def _build_infer_input(self, batch_paths: Sequence[str]) -> httpclient.InferInput:
        """
        构建 Triton 推理输入，将原始图片字节按batch打包
        - batch_paths: 当前批次的图像文件路径列表。
        - 读取每个图像文件的字节内容并构建一个 NumPy 数组作为输入。
        """
        payload = np.empty((len(batch_paths), 1), dtype=object)
        for index, file_path in enumerate(batch_paths):
            payload[index, 0] = Path(file_path).read_bytes()
        infer_input = httpclient.InferInput(self.input_name, payload.shape, datatype="BYTES")
        infer_input.set_data_from_numpy(payload)
        return infer_input

    def _infer_numpy(
        self,
        infer_input: httpclient.InferInput,
        client: Optional[httpclient.InferenceServerClient] = None,
    ) -> np.ndarray:
        outputs = [httpclient.InferRequestedOutput(self.output_name)]
        infer_client = client or self._client
        response = infer_client.infer(
            model_name=self.model_name,
            inputs=[infer_input],
            outputs=outputs,
        )
        result = response.as_numpy(self.output_name)
        if result is None:
            raise RuntimeError(f"Triton response does not contain output: {self.output_name}")
        return result

    @staticmethod
    def _flatten_probabilities(probabilities: np.ndarray) -> List[float]:
        if probabilities.ndim == 1:
            return probabilities.astype(np.float32).tolist()
        if probabilities.ndim == 2 and probabilities.shape[1] == 1:
            return probabilities[:, 0].astype(np.float32).tolist()
        return probabilities.reshape(probabilities.shape[0], -1)[:, 0].astype(np.float32).tolist()


def summarize_predictions(predictions: Dict[str, Any]) -> Dict[str, Any]:
    results = predictions.get("results", [])
    positive = sum(int(item["label"]) for item in results)
    negative = len(results) - positive
    items = len(results)
    avg_inference_latency_ms = (
        sum(float(item.get("inference_latency_ms", 0.0)) for item in results) / items if items > 0 else 0.0
    )
    avg_preprocess_latency_ms = (
        sum(float(item.get("preprocessing_latency_ms", 0.0)) for item in results) / items if items > 0 else 0.0
    )
    return {
        "items": items,
        "positive": positive,
        "negative": negative,
        "throughput": predictions.get("throughput"),
        "total_seconds": predictions.get("total_seconds"),
        "infer_seconds": predictions.get("infer_seconds"),
        "preprocess_seconds": predictions.get("preprocess_seconds"),
        "avg_inference_latency_ms": avg_inference_latency_ms,
        "avg_preprocess_latency_ms": avg_preprocess_latency_ms,
    }
