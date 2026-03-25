import torch
import torch.nn as nn
import timm
from transformers.modeling_outputs import ImageClassifierOutput
from ..registry import register_model

@register_model("convnext2_tiny")
class ConvNext2Tiny(nn.Module):
    def __init__(self,backbone_name="timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384",num_classes=1,fix_backbone=False):
        super(ConvNext2Tiny, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        self.backbone = timm.create_model(backbone_name,pretrained=True,num_classes=0,global_pool="avg")

        if fix_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        num_features = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.Linear(num_features,num_features//2),
            nn.BatchNorm1d(num_features//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(num_features//2,num_classes)
        )
    
    def forward(self,pixel_values,labels=None):
        outputs = self.backbone(pixel_values)
        logits = self.head(outputs)

        loss = None
        if labels is not None:
            if self.num_classes == 1:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.float().view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels.view(-1))
        
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )
    
    def get_image_processor_or_transform(self):
        config = timm.data.resolve_data_config({}, model=self.backbone)

        train_transform = timm.data.create_transform(**config,is_training=True)
        eval_transform = timm.data.create_transform(**config,is_training=False)

        return train_transform, eval_transform





