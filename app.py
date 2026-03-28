from __future__ import annotations

from pathlib import Path
from typing import List

from client.explainability import GradCAMExplainer
from client.inference import TritonInferenceClient, summarize_predictions


def _to_rows(predictions: dict) -> List[List[object]]:
    rows: List[List[object]] = []
    for item in predictions.get("results", []):
        label_value = int(item["label"])
        label_text = "假" if label_value == 1 else "真"
        preprocess_latency_ms = float(item.get("preprocessing_latency_ms", 0.0))
        inference_latency_ms = float(item.get("inference_latency_ms", item.get("latency_ms", 0.0)))
        rows.append(
            [
                Path(item["file_path"]).name,
                label_text,
                label_value,
                round(preprocess_latency_ms, 3),
                round(inference_latency_ms, 3),
            ]
        )
    return rows


def _split_manual_paths(path_text: str) -> List[str]:
    if not path_text:
        return []
    normalized = path_text.replace("\n", ",").replace(";", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _list_image_files_local(input_path: str | Path) -> List[str]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        return [str(path)]
    files = [
        str(file_path)
        for file_path in sorted(path.rglob("*"))
        if file_path.is_file() and file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
    ]
    if not files:
        raise ValueError(f"No image files were found under: {path}")
    return files


def _collect_image_paths(
    uploaded_paths: List[str],
    manual_paths_text: str,
    client: TritonInferenceClient | None,
    single_only: bool,
) -> List[str]:
    """
    合并上传的文件路径和手动输入的路径，去重后返回最终的路径列表。
    - uploaded_paths: 来自文件上传组件的路径列表。
    - manual_paths_text: 来自文本输入的路径字符串，支持逗号、分号或换行分隔。
    - client: 用于调用 list_image_files 的 TritonInferenceClient 实例。
    - single_only: 如果为 True，则只返回一个路径（上传的文件或手动输入的路径）。
    """
    merged: List[str] = []
    for path in uploaded_paths:
        if path:
            merged.append(path)
    for raw_path in _split_manual_paths(manual_paths_text):
        discovered = client.list_image_files(raw_path) if client is not None else _list_image_files_local(raw_path)
        merged.extend(discovered)

    unique_paths: List[str] = []
    seen = set()
    for path in merged:
        normalized = str(Path(path).resolve())
        if normalized not in seen:
            seen.add(normalized)
            unique_paths.append(normalized)

    if single_only and len(unique_paths) > 1:
        return unique_paths[:1]
    # if len(unique_paths) > 100:
    #     import random
    #     return random.sample(unique_paths, 100) # 随机选择100个文件小型测试
    return unique_paths


def _to_batch_stats(summary: dict) -> List[List[object]]:
    return [
        [
            int(summary.get("items", 0)),
            int(summary.get("negative", 0)),
            int(summary.get("positive", 0)),
            round(float(summary.get("avg_inference_latency_ms", 0.0)), 3),
            round(float(summary.get("avg_preprocess_latency_ms", 0.0)), 3),
            round(float(summary.get("throughput", 0.0)), 3),
        ]
    ]


def predict_single(image_path: str, input_path: str, server_url: str, model_name: str, threshold: float):
    client = TritonInferenceClient(
        url=server_url,
        model_name=model_name,
        threshold=threshold,
    )
    selected_paths = _collect_image_paths(
        uploaded_paths=[image_path] if image_path else [],
        manual_paths_text=input_path,
        client=client,
        single_only=True,
    )
    if not selected_paths:
        return {}, []
    predictions = client.predict_paths(selected_paths, batch_size=1)
    summary = summarize_predictions(predictions)
    summary["result"] = predictions["results"][0]
    return summary, _to_rows(predictions)


def predict_batch(
    image_paths: List[str],
    input_paths: str,
    server_url: str,
    model_name: str,
    threshold: float,
    batch_size: int,
    max_concurrency: int,
):
    client = TritonInferenceClient(
        url=server_url,
        model_name=model_name,
        threshold=threshold,
    )
    selected_paths = _collect_image_paths(
        uploaded_paths=image_paths or [],
        manual_paths_text=input_paths,
        client=client,
        single_only=False,
    )
    if not selected_paths:
        return {}, [], []
    predictions = client.predict_paths(
        selected_paths,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )
    summary = summarize_predictions(predictions)
    #return summary, _to_rows(predictions), _to_batch_stats(summary)
    return  _to_rows(predictions), _to_batch_stats(summary)


def predict_explain_single(
    image_path: str,
    input_path: str,
    model_name: str,
    checkpoint_dir: str,
    threshold: float,
):
    selected_paths = _collect_image_paths(
        uploaded_paths=[image_path] if image_path else [],
        manual_paths_text=input_path,
        client=None,
        single_only=True,
    )
    if not selected_paths:
        return {}, [], None, None, None

    explainer = GradCAMExplainer(
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        threshold=threshold,
    )
    explained = explainer.explain(selected_paths[0])
    prediction = explained["prediction"]
    predictions_payload = {"results": [prediction]}
    summary = summarize_predictions(predictions_payload)
    summary["result"] = prediction
    label_value = int(prediction["label"])
    summary["label_text"] = "假" if label_value == 1 else "真"
    summary["model_name"] = model_name
    summary["checkpoint_dir"] = checkpoint_dir
    return summary, _to_rows(predictions_payload), explained["original_image"], explained["heatmap"], explained["overlay"]


def build_app():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("gradio is required to run app.py") from exc

    with gr.Blocks(title="Triton Forgery Detection") as demo:
        gr.Markdown("# Triton 伪造图像检测\n支持单图、批量推理与基础性能统计。")

        with gr.Row():
            server_url = gr.Dropdown(label="Triton 地址", choices=["localhost:8000"], value="localhost:8000")
            model_name = gr.Dropdown(label="模型名", choices=["convnext2_tiny_ensemble", "dinov3_vith16_ensemble"], value="convnext2_tiny_ensemble")
            threshold = gr.Slider(label="阈值", minimum=0.0, maximum=1.0, value=0.5, step=0.01)

        with gr.Tab("单样本推理"):
            single_image = gr.Image(label="输入图片", type="filepath")
            single_path = gr.Textbox(label="或输入图片路径", placeholder="例如: /data/test/1.jpg")
            single_button = gr.Button("开始推理")
            single_summary = gr.JSON(label="汇总", visible=False)
            single_table = gr.Dataframe(
                headers=["文件名", "标签", "真/假(0/1)", "预处理延迟(ms)", "推理延迟(ms)"],
                datatype=["str", "str", "number", "number", "number"],
                label="结果",
            )
            single_button.click(
                fn=predict_single,
                inputs=[single_image, single_path, server_url, model_name, threshold],
                outputs=[single_summary, single_table],
            )

        with gr.Tab("批量推理"):
            multi_images = gr.File(label="批量图片", type="filepath", file_count="multiple")
            batch_paths = gr.Textbox(
                label="或输入文件/目录路径(支持逗号、分号或换行分隔)",
                placeholder="例如: /data/a.jpg, /data/b.png 或 /data/eval_set",
                lines=3,
            )
            batch_size = gr.Slider(label="批大小", minimum=1, maximum=64, value=8, step=1)
            max_concurrency = gr.Slider(label="并发批数", minimum=1, maximum=16, value=4, step=1)
            batch_button = gr.Button("批量推理")
            #batch_summary = gr.JSON(label="汇总")
            batch_table = gr.Dataframe(
                headers=["文件名", "标签", "真/假(0/1)", "预处理延迟(ms)", "推理延迟(ms)"],
                datatype=["str", "str", "number", "number", "number"],
                label="结果",
            )
            batch_stats = gr.Dataframe(
                headers=["总文件数", "真文件数(0)", "假文件数(1)", "平均推理时间(ms)", "平均预处理时间(ms)", "吞吐量(文件/s)"],
                datatype=["number", "number", "number", "number", "number", "number"],
                label="批量统计",
            )
            batch_button.click(
                fn=predict_batch,
                inputs=[multi_images, batch_paths, server_url, model_name, threshold, batch_size, max_concurrency],
                outputs=[batch_table, batch_stats],
            )

        with gr.Tab("解释可视化"):
            explain_image = gr.Image(label="输入图片", type="filepath")
            explain_path = gr.Textbox(label="或输入图片路径", placeholder="例如: /data/test/1.jpg")
            explain_model_name = gr.Dropdown(
                label="本地解释模型",
                choices=["convnext2_tiny", "convnext2"],
                value="convnext2_tiny",
            )
            explain_checkpoint = gr.Textbox(
                label="权重目录",
                value="logs/convnext2_tiny/final_model",
                placeholder="例如: logs/convnext2_tiny/final_model",
            )
            explain_button = gr.Button("生成解释图")
            explain_summary = gr.JSON(label="解释汇总", visible=False)
            explain_table = gr.Dataframe(
                headers=["文件名", "标签", "真/假(0/1)", "预处理延迟(ms)", "推理延迟(ms)"],
                datatype=["str", "str", "number", "number", "number"],
                label="解释结果",
                visible=False,
            )
            with gr.Row():
                explain_original = gr.Image(label="原图", type="pil")
                explain_heatmap = gr.Image(label="热力图", type="pil")
                explain_overlay = gr.Image(label="叠加图", type="pil")
            explain_button.click(
                fn=predict_explain_single,
                inputs=[explain_image, explain_path, explain_model_name, explain_checkpoint, threshold],
                outputs=[explain_summary, explain_table, explain_original, explain_heatmap, explain_overlay],
            )

    return demo


def main() -> None:
    demo = build_app()
    demo.launch()


if __name__ == "__main__":
    main()
