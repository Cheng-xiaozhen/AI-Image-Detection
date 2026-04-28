# 启动Docker容器
```
sudo docker run -it --gpus all \
  --name triton_server \
  --net host \
  --shm-size=32g \
  --pids-limit=-1 \
  -v /home/chengxiaozhen/Test/SFT-Infra/system/triton_model_repository:/models \
  my_tritonserver:23.10-py3 \
```

# 启动TritonServer
```
  tritonserver \
  --model-repository=/models \
  --model-control-mode=poll \
  --repository-poll-secs=5 \
  --pinned-memory-pool-byte-size=4294967296
```

# 合起来
```
sudo docker run -it --gpus all \
  --name triton_server \
  --net host \
  --shm-size=32g \
  --pids-limit=-1 \
  -v /home/chengxiaozhen/Test/SFT-Infra/system/triton_model_repository:/models \
  my_tritonserver:23.10-py3 \
  tritonserver \
  --model-repository=/models \
  --model-control-mode=poll \
  --repository-poll-secs=5 \
  --pinned-memory-pool-byte-size=4294967296
```



