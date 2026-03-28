# 启动Docker容器
```
sudo docker run --gpus all --rm -it --net host --shm-size=4g -v /home/chengxiaozhen/Test/SFT-Infra/triton_model_repository:/models my_tritonserver:23.10-py3
```

# 启动TritonServer
```
tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=5 --pinned-memory-pool-byte-size=1073741824
```