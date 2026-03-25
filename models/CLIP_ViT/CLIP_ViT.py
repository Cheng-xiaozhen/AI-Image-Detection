import torch
import torch.nn as nn
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers.modeling_outputs import ImageClassifierOutput

from ..registry import register_model


@register_model("clip_vit")
class CLIP_ViT(nn.Module):
    def __init__(self, backbone_name="vit_base_patch16_clip_224.openai", num_classes=1, fix_backbone=False):
        super(CLIP_ViT, self).__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="token")
        
        hidden_size = getattr(self.backbone, "num_features", None)
        if not isinstance(hidden_size, int):
            raise RuntimeError("timm backbone does not expose integer num_features.")
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes
        
        # 冻结骨干网络 (Linear Probe 的核心)
        if fix_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 定义 Loss 函数 (初始化时确定，避免 forward 重复创建)
        if num_classes == 1:
            self.loss_fct = nn.BCEWithLogitsLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        """
        Args:
            pixel_values: Tensor (Batch, 3, H, W)
            labels: Tensor (Batch) or (Batch, 1)
        """
        features = self.backbone(pixel_values)
        
        # 计算 Logits
        logits = self.fc(features)
        
        loss = None
        if labels is not None:
            if self.num_classes == 1:
                # 二分类 (BCE): 需要将 labels 转为 float 并调整形状为 (B, 1)
                loss = self.loss_fct(logits.view(-1), labels.float().view(-1))
            else:
                # 多分类 (CrossEntropy): labels 应该是 LongTensor (B)
                loss = self.loss_fct(logits, labels.view(-1))

        # 返回 HF 风格的 Output 对象，这样 Trainer 可以自动解析 loss 和 logits
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )
    def extract_feature(self, pixel_values):
        """
        专门用于提取特征的函数，返回 backbone 的特征表示
        Args:
            pixel_values: Tensor (Batch, 3, H, W)
        Returns:
            features: Tensor (Batch, hidden_size)
        """
        with torch.no_grad():
            features = self.backbone(pixel_values)
        return features
    
    def get_image_processor_or_transform(self):
        config = resolve_data_config({}, model=self.backbone)

        train_transform = create_transform(**config, is_training=True)
        eval_transform = create_transform(**config, is_training=False)

        return train_transform, eval_transform

@register_model("clip_vit_384")
class CLIP_ViT384(nn.Module):
    def __init__(self,backbone_name="vit_small_patch16_384.augreg_in21k_ft_in1k", num_classes=1, fix_backbone=False):
        super(CLIP_ViT384, self).__init__()
        self.vit = timm.create_model(backbone_name, pretrained=True)
        self.vit.head = nn.Linear(in_features=384, out_features=num_classes,bias=True)
        self.num_classes = num_classes

        # 定义 Loss 函数 (初始化时确定，避免 forward 重复创建)
        if num_classes == 1:
            self.loss_fct = nn.BCEWithLogitsLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        logits = self.vit(pixel_values)

        loss = None
        if labels is not None:
            if self.num_classes == 1:
                # 二分类 (BCE): 需要将 labels 转为 float 并调整形状为 (B, 1)
                loss = self.loss_fct(logits.view(-1), labels.float().view(-1))
            else:
                # 多分类 (CrossEntropy): labels 应该是 LongTensor (B)
                loss = self.loss_fct(logits, labels.view(-1))

        # 返回 HF 风格的 Output 对象，这样 Trainer 可以自动解析 loss 和 logits
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )
    
    def get_image_processor_or_transform(self):
        config = resolve_data_config({}, model=self.vit)

        train_transform = create_transform(**config, is_training=True)
        eval_transform = create_transform(**config, is_training=False)

        return train_transform, eval_transform

@register_model("clip_vit_224")
class CLIP_ViT224(nn.Module):
    def __init__(self,backbone_name="vit_small_patch16_224.augreg_in21k_ft_in1k", num_classes=1, fix_backbone=False):
        super(CLIP_ViT224, self).__init__()
        self.vit = timm.create_model(backbone_name, pretrained=True)
        self.vit.head = nn.Linear(in_features=384, out_features=num_classes,bias=True)
        self.num_classes = num_classes

        # 定义 Loss 函数 (初始化时确定，避免 forward 重复创建)
        if num_classes == 1:
            self.loss_fct = nn.BCEWithLogitsLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        logits = self.vit(pixel_values)

        loss = None
        if labels is not None:
            if self.num_classes == 1:
                # 二分类 (BCE): 需要将 labels 转为 float 并调整形状为 (B, 1)
                loss = self.loss_fct(logits.view(-1), labels.float().view(-1))
            else:
                # 多分类 (CrossEntropy): labels 应该是 LongTensor (B)
                loss = self.loss_fct(logits, labels.view(-1))

        # 返回 HF 风格的 Output 对象，这样 Trainer 可以自动解析 loss 和 logits
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )
    
    def get_image_processor_or_transform(self):
        config = resolve_data_config({}, model=self.vit)

        train_transform = create_transform(**config, is_training=True)
        eval_transform = create_transform(**config, is_training=False)

        return train_transform, eval_transform




@register_model("clip_vit_base")
class CLIP_ViT_Base(CLIP_ViT):
    def __init__(self, num_classes=1, fix_backbone=False):
        super(CLIP_ViT_Base, self).__init__(
            backbone_name="vit_base_patch16_clip_224.openai",
            num_classes=num_classes,
            fix_backbone=fix_backbone
        )

@register_model("clip_vit_large")
class CLIP_ViT_Large(CLIP_ViT):
    def __init__(self, num_classes=1, fix_backbone=False):
        super(CLIP_ViT_Large, self).__init__(
            backbone_name="vit_large_patch14_clip_224.openai",
            num_classes=num_classes,
            fix_backbone=fix_backbone
        )

