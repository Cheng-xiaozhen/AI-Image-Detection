from .registry import create_model, list_models, register_model

# 导入内置模型以完成注册
from .UniFD.UniFD import UniFD  # noqa: F401
from .CLIP_ViT.CLIP_ViT import CLIP_ViT, CLIP_ViT_Base, CLIP_ViT_Large, CLIP_ViT384
from .ConvNext2.ConvNext2 import ConvNext2
from .ConvNext2.ConvNext2_tiny import ConvNext2Tiny
from .DINOv3.dinov3 import (
    DINOv3Model,
    DINOv3Vit7B,
    DINOv3Vitb16,
    DINOv3Vith16,
    DINOv3Vitl16,
    DINOv3Vits16,
    DINOv3Vith16Linear,
    DINOv3Vith16VIB
)
__all__ = [
	"create_model",
	"list_models",
	"register_model",
	"UniFD",
    "CLIP_ViT",
    "ConvNext2",
	"ConvNext2Tiny",
	"DINOv3Model",
	"DINOv3Vits16",
	"DINOv3Vitb16",
	"DINOv3Vitl16",
	"DINOv3Vith16",
	"DINOv3Vit7B",
    "DINOv3Vith16Linear",
    "DINOv3Vith16VIB",
    "CLIP_ViT_Base",
    "CLIP_ViT_Large",
    "CLIP_ViT384"
]
