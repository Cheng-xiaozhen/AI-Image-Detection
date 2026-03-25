python export_onnx.py \
    --model_name convnext2_tiny \
    --num_classes 1 \
    --checkpoint /home/chengxiaozhen/Test/SFT-Infra/logs/convnext2_tiny/final_model \
    --output /home/chengxiaozhen/Test/SFT-Infra/export/onnx/convnext2_tiny.onnx \
    --batch_size 1 \
    --export_probs \
    --verify