# Version 1
当前文件目录结构如下所示:
chengxiaozhen@censoring02:~/Test/SFT-Infra$ tree -L 1
.
├── app.py
├── client
├── Dockerfile
├── evaluate.py
├── evaluate.sh
├── export
├── logs
├── models
├── __pycache__
├── system_design.md
├── test
├── train.py
├── train.sh
├── triton_model_repository
└── utils

8 directories, 7 files

其中
- client: 客户端代码
- design.md: 设计文档
- evaluate.py: 评估代码
- evaluate.sh: 评估脚本
- export: 导出模型代码目录
- logs: 模型训练日志目录
- models: 模型定义代码目录
- __pycache__: python缓存目录
- test: 测试代码目录
- train.py: 训练代码
- train.sh: 训练脚本
- triton_model_repository: triton模型仓库目录
- utils: 工具代码目录

# 需求
* 在批量推理的时候，在批量统计模块，平均推理时间为20ms左右，吞吐量为55文件/s左右，应该如何优化提速？
