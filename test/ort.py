import onnxruntime as ort
print("ONNX Runtime version:", ort.__version__)
print(ort.get_available_providers())