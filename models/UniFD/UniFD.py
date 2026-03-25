import torch
import torch.nn as nn
from transformers import AutoImageProcessor, CLIPVisionModel
from transformers.modeling_outputs import ImageClassifierOutput

from ..registry import register_model


@register_model("unifd")
class UniFD(nn.Module):
    def __init__(self, backbone_name="openai/clip-vit-large-patch14", num_classes=1, fix_backbone=True):
        super(UniFD, self).__init__()
        self.backbone_name = backbone_name
        self.backbone = CLIPVisionModel.from_pretrained(backbone_name)
        
        hidden_size = self.backbone.config.hidden_size
        
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
        outputs = self.backbone(pixel_values=pixel_values)
        
        # pooler_output 是 [CLS] token 经过 projection 后的特征，形状 (Batch, hidden_size)
        features = outputs.pooler_output 
        
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
            outputs = self.backbone(pixel_values=pixel_values)
            features = outputs.pooler_output 
        return features
    
    def get_image_processor_or_transform(self):
        config_processor = getattr(self.backbone.config, "image_processor", None)
        if config_processor is not None:
            return config_processor

        try:
            return AutoImageProcessor.from_pretrained(self.backbone_name)
        except Exception as exc:
            raise RuntimeError(
                "Image processor is not available in config and failed to load via AutoImageProcessor."
            ) from exc