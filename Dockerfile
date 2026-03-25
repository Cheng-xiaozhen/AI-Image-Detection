FROM nvcr.io/nvidia/tritonserver:23.10-py3
# 安装 PIL (Pillow) 依赖
RUN pip install --no-cache-dir Pillow