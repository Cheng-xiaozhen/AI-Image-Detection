import torch
import torch.nn as nn
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers.modeling_outputs import ImageClassifierOutput

class CLIP_ViT(nn.Module):
    def __init__(self,backbone_name="vit_small_patch16_384.augreg_in21k_ft_in1k", num_classes=1, fix_backbone=False):
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="token")
        self.backbone.head = nn.Linear(in_features=384, out_features=num_classes,bias=True)
    def forward(self, pixel_values, labels=None):
        logits = self.backbone(pixel_values)
        return ImageClassifierOutput(
            loss=None,
            logits=logits,
        )

    def get_image_processor_or_transform(self):
        config = resolve_data_config({}, model=self.backbone)

        train_transform = create_transform(**config, is_training=True)
        eval_transform = create_transform(**config, is_training=False)

        return train_transform, eval_transform


if __name__ == "__main__":
    model = CLIP_ViT()
    checkpoint_path = "/home/chengxiaozhen/Test/SFT-Infra/logs/CLIP/clip384/final_model/model.safetensors"
    from safetensors.torch import load_file
    state_dict=  load_file(str(checkpoint_path))
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"Missing keys when loading checkpoint: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Unexpected keys when loading checkpoint: {load_result.unexpected_keys}")
    print("Checkpoint loaded successfully.")


