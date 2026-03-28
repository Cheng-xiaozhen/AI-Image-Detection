python export_onnx.py \
    --model_name dinov3_vith16-vib \
    --num_classes 1 \
    --checkpoint /home/chengxiaozhen/Test/SFT-Infra/logs/dinov3_vith16-vib/final_model \
    --output /home/chengxiaozhen/Test/SFT-Infra/export/ONNX/dinov3_onnx/dinov3_vith16-vib.onnx \
    --batch_size 1 \
    --export_probs \
    --verify