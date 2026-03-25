python validate_onnx.py \
    --model_name convnext2_tiny \
    --num_classes 1 \
    --checkpoint /home/chengxiaozhen/Test/SFT-Infra/logs/convnext2_tiny/final_model \
    --onnx_path /home/chengxiaozhen/Test/SFT-Infra/export/onnx/convnext2_tiny.onnx \
    --batch_size 1 \
    --report /home/chengxiaozhen/Test/SFT-Infra/export/validate_report.json
    
