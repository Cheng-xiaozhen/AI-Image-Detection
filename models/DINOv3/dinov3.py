import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data.config import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from transformers.modeling_outputs import ImageClassifierOutput
try:
    from ..registry import register_model
except ImportError:
    models_root = Path(__file__).resolve().parents[1]
    if str(models_root) not in sys.path:
        sys.path.insert(0, str(models_root))
    from registry import register_model

# Alias mapping: your alias in VALID_NAMES -> actual timm model name
# Convention:
#   * *_16 series: maps to DINOv3 ViT patch16 weights (requires newer timm)
_ALIAS_TO_TIMM = {
    'dinov3_vits16': 'vit_small_plus_patch16_dinov3_qkvb',
    'dinov3_vitb16': 'vit_base_patch16_dinov3',
    'dinov3_vitl16': 'vit_large_patch16_dinov3',
    'dinov3_vith16': 'vit_huge_plus_patch16_dinov3_qkvb', 
    'dinov3_vit_7b': 'vit_7b_patch16_dinov3',  
}


def _count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x)->Dict[str, Any]:
        logits = self.fc(x)
        return {
            'logits':logits,
            'aux_loss':None,  
        }

class MLPHead(nn.Module):
    def __init__(self,in_dim:int,hidden_dim:int,out_dim:int=1,dropout:float=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self,x:torch.Tensor)->Dict[str, Any]:
        logits = self.net(x)
        return {
            'logits':logits,
            'aux_loss':None,  
        }
    
class BottleneckMLPHead(nn.Module):
    def __init__(self,in_dim:int,bottleneck_dim:int,out_dim:int=1,dropout:float=0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, out_dim)
        )
    def forward(self,x:torch.Tensor)->Dict[str, Any]:
        logits = self.net(x)
        return {
            'logits':logits,
            'aux_loss':None, 
        }
    
class VIBHeadNoKL(nn.Module):
    """
    仅保留 ViB 的 stochastic bottleneck 结构，但不加 KL 正则。
    用来检验：
    提升是否只是来自更复杂结构 / 随机采样，而不是信息瓶颈约束本身。
    """
    def __init__(self,in_dim:int,hidden_dim:int,latent_dim:int,out_dim:int=1,dropout:float=0.1)->None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.mu_proj = nn.Linear(hidden_dim,latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim,latent_dim)
        self.classifier = nn.Linear(latent_dim, out_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        h = self.encoder(x)
        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h).clamp(min=-30.0, max=20.0)
        z = self.reparameterize(mu, logvar)
        logits = self.classifier(z)
        return {
            "logits": logits,
            "aux_loss": None,
            "z": z,
            "mu": mu,
            "logvar": logvar,
        }

class VIBHead(nn.Module):
    """
    标准 ViB:
    CE/BCE + beta * KL(q(z|x) || N(0, I))
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        out_dim: int = 1,
        dropout: float = 0.1,
        beta: float = 1e-3,
    ):
        super().__init__()
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, out_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL(q(z|x) || N(0, I))
        # shape: [B, latent_dim] -> [B] -> scalar
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl = kl.sum(dim=-1).mean()
        return kl

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        h = self.encoder(x)
        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h).clamp(min=-30.0, max=20.0)
        z = self.reparameterize(mu, logvar)
        logits = self.classifier(z)

        kl = self.kl_divergence(mu, logvar)
        aux_loss = self.beta * kl

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "z": z,
            "mu": mu,
            "logvar": logvar,
        }

def _resolve_local_checkpoint(timm_name: str) -> Optional[str]:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_root.exists():
        return None

    for model_dir in sorted(cache_root.glob(f"models--timm--{timm_name}*")):
        for checkpoint in sorted(model_dir.glob("snapshots/*/model.safetensors"), reverse=True):
            if checkpoint.is_file():
                return str(checkpoint)
    return None

@register_model("dinov3")
class DINOv3Model(nn.Module):
    def __init__(
        self,
        model_name: str = 'dinov3_vit_7b',
        img_size: Optional[int] = None,
        pool_type: str = 'cls',
        layer_strategy: str = 'vib',
        custom_token_indices: Optional[list[int]] = None,
        pos_weight: Optional[float] = None,
        head_dropout: float = 0.1,
        vib_beta: float = 1e-3,
        num_classes: int = 1,
        fix_backbone: bool = True,
        mlp_hidden_dim: Optional[int] = None,
        bottleneck_dim: Optional[int] = None,
        vib_hidden_dim: Optional[int] = None,
        vib_latent_dim: Optional[int] = None,
    ):
        """
        model_name: The value from name.split(':',1)[1] in __init__.py,
                    e.g., 'dinov3_vitl16' or 'dinov3_vitl14'
        img_size: Optional input image size. If specified, timm will automatically interpolate
                  position embeddings to adapt to this size.
        pool_type: Token pooling method
                    - 'all': Average all tokens
                    - 'patch_avg': Average PATCH tokens (default, original timm behavior)
                    - 'cls': Use CLS token
                    - 'reg_avg': Average REG tokens
                    - 'cls+reg': Concatenate CLS and REG average
                    - 'cls+patch': Concatenate CLS and PATCH average (without registers)
                    - 'custom_tokens': Average tokens specified by custom_token_indices
                        custom_token_indices: Custom token index list, only used when pool_type='custom_tokens'
                        Example: [1, 0, 4, 3, 2, 5, 18, 187, 200, 6]
        """
        super().__init__()
        self.model_name = model_name
        self.layer_strategy = layer_strategy
        self.head_dropout = head_dropout
        self.vib_beta = vib_beta
        self.num_classes = num_classes
        self.fix_backbone = fix_backbone
        self.mlp_hidden_dim = mlp_hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.vib_hidden_dim = vib_hidden_dim
        self.vib_latent_dim = vib_latent_dim

        key = model_name.lower()
        timm_name = _ALIAS_TO_TIMM.get(key)
        if timm_name is None:
            raise ValueError(
                f"Unknown DINO alias '{model_name}'. "
                f"Valid: {', '.join(sorted(_ALIAS_TO_TIMM.keys()))}"
            )
        # Create backbone; requires timm version that supports the entry name
        # If img_size is specified, timm will automatically interpolate position embeddings after loading pretrained weights
        default_path = _resolve_local_checkpoint(timm_name)

        if img_size is not None:
            self.backbone = timm.create_model(timm_name, pretrained=True, img_size=img_size)
            #print(f"[DinoV3Model] Created {timm_name} with custom img_size={img_size}")
        else:
            if default_path and os.path.exists(default_path):

                self.backbone = timm.create_model(timm_name, pretrained=False,checkpoint_path=default_path)
                #print(f"[DinoV3Model] Loaded {timm_name} from local checkpoint: {default_path}")
            else:
                #print(f"[Warning] Local path not found. Trying to download pretrained weights...")
                self.backbone = timm.create_model(timm_name, pretrained=True)

        # 一般情况下，DINOv3的backbone的最后是一个直通层Identity()，直接输出特征向量，没有全连接层
        # reset_classifier()可以重置最后一个分类头，但在此处设为0，将Backbone仅用于提取特征
        reset_classifier = getattr(self.backbone, 'reset_classifier', None)
        if callable(reset_classifier):
            reset_classifier(0)

        if self.fix_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        if model_name == 'dinov3_vit_7b':
        # 强制将 Backbone 转为半精度 (FP16)，dinov3-7b 在FP32下占用显存过大
            self.backbone.half()
            #print(f"[DinoV3Model] Backbone 加载完毕，参数已冻结，FP16模式。")

        # Save pool_type configuration
        self.pool_type = pool_type
        #print(f"[DinoV3Model] pool_type={pool_type}")
        self.custom_token_indices = custom_token_indices

        # 自动获取backbone输出向量的维度，才能定义后面的linear层
        feat_dim = getattr(self.backbone, 'num_features', None)
        if feat_dim is None: # 如果获取不到，就用一个假的输入来跑一次前向传播，看看输出的维度是多少
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 256, 256)
                if model_name == 'dinov3_vit_7b':
                    dummy = dummy.half()  # 确保dummy输入与Backbone的dtype一致，避免某些模型在前向传播时因为dtype不匹配而报错
                feats = self._extract_feats(self.backbone, dummy, pool_type=pool_type, custom_token_indices=custom_token_indices)
                feat_dim = feats.shape[-1]
        else:
            # If cls+reg or cls+patch, feature dimension doubles 如果选择拼接特征，linear层的输入维度也要翻倍
            if pool_type in ['cls+reg', 'cls+patch']: 
                feat_dim = feat_dim * 2

        self.feat_dim = feat_dim
        self.mlp_hidden_dim = self.mlp_hidden_dim or max(1, self.feat_dim * 2)
        self.bottleneck_dim = self.bottleneck_dim or max(1, self.feat_dim // 4)
        self.vib_hidden_dim = self.vib_hidden_dim or max(1, self.feat_dim // 4)
        self.vib_latent_dim = self.vib_latent_dim or max(1, self.feat_dim // 8)

        # Setup classifier head  分类头
        self._setup_classifier()

        # 损失函数
        if self.num_classes == 1:
            self.loss_fct = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight) if pos_weight is not None else None
            )
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.head_trainable_params = _count_trainable_parameters(self.head)

        # 打印最终的模型配置信息
        if pool_type == 'custom_tokens':
            print(f"[DinoV3Model] Using pool_type='custom_tokens', indices={custom_token_indices}, feat_dim={feat_dim}")
        else:
            print(f"[DinoV3Model] Using pool_type='{pool_type}', feat_dim={feat_dim}")
        print(
            f"[DinoV3Model] head='{self.layer_strategy}', "
            f"num_classes={self.num_classes}, trainable_head_params={self.head_trainable_params}"
        )

    def _setup_classifier(self):
        """
        设置不同的分类头结构
        支持:
        - linear
        - mlp
        - bottleneck_mlp
        - vib_no_kl
        - vib
        """
        mlp_hidden_dim = self.mlp_hidden_dim or max(1, self.feat_dim * 2)
        bottleneck_dim = self.bottleneck_dim or max(1, self.feat_dim // 4)
        vib_hidden_dim = self.vib_hidden_dim or max(1, self.feat_dim // 4)
        vib_latent_dim = self.vib_latent_dim or max(1, self.feat_dim // 8)

        if self.layer_strategy == 'linear':
            self.head = LinearHead(self.feat_dim, out_dim=self.num_classes)

        elif self.layer_strategy == 'mlp':
            self.head = MLPHead(
                in_dim=self.feat_dim,
                hidden_dim=mlp_hidden_dim,
                out_dim=self.num_classes,
                dropout=self.head_dropout,
            )

        elif self.layer_strategy == 'bottleneck':
            self.head = BottleneckMLPHead(
                in_dim=self.feat_dim,
                bottleneck_dim=bottleneck_dim,
                out_dim=self.num_classes,
                dropout=self.head_dropout,
            )

        elif self.layer_strategy == 'vib_no_kl':
            self.head = VIBHeadNoKL(
                in_dim=self.feat_dim,
                hidden_dim=vib_hidden_dim,
                latent_dim=vib_latent_dim,
                out_dim=self.num_classes,
                dropout=self.head_dropout,
            )

        elif self.layer_strategy == 'vib':
            self.head = VIBHead(
                in_dim=self.feat_dim,
                hidden_dim=vib_hidden_dim,
                latent_dim=vib_latent_dim,
                out_dim=self.num_classes,
                dropout=self.head_dropout,
                beta=self.vib_beta,
            )

        else:
            raise ValueError(
                f"Unknown layer_strategy: {self.layer_strategy}. "
                f"Supported: 'linear', 'mlp', 'bottleneck', 'vib_no_kl', 'vib'"
            )

    def _build_all_tokens_from_dict(self, feats_dict: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        all_tokens = feats_dict.get('x')
        if all_tokens is not None:
            return all_tokens

        all_tokens = feats_dict.get('x_prenorm')
        if all_tokens is not None:
            return all_tokens

        cls_token = feats_dict.get('x_norm_clstoken')
        reg_tokens = feats_dict.get('x_norm_regtokens')
        patch_tokens = feats_dict.get('x_norm_patchtokens')

        pieces = []
        if cls_token is not None:
            pieces.append(cls_token.unsqueeze(1))
        if reg_tokens is not None:
            pieces.append(reg_tokens)
        if patch_tokens is not None:
            pieces.append(patch_tokens)

        if not pieces:
            return None
        return torch.cat(pieces, dim=1)

    def _extract_from_feature_dict(
        self,
        backbone,
        feats_dict: Dict[str, torch.Tensor],
        pool_type: str,
        custom_token_indices: Optional[list[int]] = None,
    ) -> Optional[torch.Tensor]:
        cls_token = feats_dict.get('x_norm_clstoken')
        reg_tokens = feats_dict.get('x_norm_regtokens')
        patch_tokens = feats_dict.get('x_norm_patchtokens')
        all_tokens = self._build_all_tokens_from_dict(feats_dict)

        if pool_type == 'cls' and cls_token is not None:
            return cls_token

        if pool_type == 'patch_avg' and patch_tokens is not None:
            return patch_tokens.mean(dim=1)

        if pool_type == 'reg_avg' and reg_tokens is not None:
            if reg_tokens.shape[1] == 0:
                if cls_token is None:
                    raise ValueError('No REG tokens found and CLS token is unavailable.')
                print('[Warning] No REG tokens found, using CLS instead')
                return cls_token
            return reg_tokens.mean(dim=1)

        if pool_type == 'cls+reg' and cls_token is not None:
            if reg_tokens is not None and reg_tokens.shape[1] > 0:
                reg_feat = reg_tokens.mean(dim=1)
            else:
                reg_feat = cls_token
            return torch.cat([cls_token, reg_feat], dim=-1)

        if pool_type == 'cls+patch' and cls_token is not None and patch_tokens is not None:
            return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)

        if pool_type == 'all' and all_tokens is not None:
            return all_tokens.mean(dim=1)

        if pool_type == 'custom_tokens':
            if custom_token_indices is None or len(custom_token_indices) == 0:
                raise ValueError("custom_token_indices must be provided when pool_type='custom_tokens'")
            if all_tokens is None:
                raise ValueError('custom_tokens pooling requires token sequence output, but none was found.')
            try:
                selected_tokens = all_tokens[:, custom_token_indices, :]
                return selected_tokens.mean(dim=1)
            except IndexError as e:
                raise ValueError(
                    f"Token indices {custom_token_indices} out of range. "
                    f"Total tokens: {all_tokens.shape[1]}. Error: {e}"
                ) from e

        return None

    def _extract_feats(self, backbone, x: torch.Tensor, pool_type: str = 'patch_avg', custom_token_indices: Optional[list[int]] = None) -> torch.Tensor:
        """
        Safely extract pre-logits features from timm model.
        Supports multiple token pooling methods.
        负责从Backbone中提取特征，并根据pool_type进行不同的token池化操作，最终返回特征向量。
        不依赖具体的模型实例，逻辑独立且复用性高

        Args:
            backbone: DINOv3 backbone model
            x: Input images [B, 3, H, W]
            pool_type: Pooling method
                - 'all': Average all tokens
                - 'patch_avg': Average PATCH tokens (default,original timm behavior)
                - 'cls': CLS token
                - 'reg_avg': Average REG tokens
                - 'cls+reg': Concatenate CLS and REG average
                - 'cls+patch': Concatenate CLS and PATCH average (without registers)
                - 'custom_tokens': Average tokens specified by custom_token_indices
            custom_token_indices: Custom token index list, only used when pool_type='custom_tokens'

        Returns:
            feats: [B, D] or [B, 2D] (when pool_type='cls+reg' or 'cls+patch')
        """
        with torch.no_grad():
            feats = backbone.forward_features(x)# timm的前向传播，通常返回shape为 [B, num_tokens, Dim]的张量，对于Dinov3来说，num_tokens通常包含【CLS，Reg1，Reg2，Reg3，Reg4，Patch1，Patch2，...】

        #兼容性处理，某些特定的模型封装可能返回dict格式的特征
        if isinstance(feats, dict):
            extracted = self._extract_from_feature_dict(
                backbone,
                feats,
                pool_type=pool_type,
                custom_token_indices=custom_token_indices,
            )
            if extracted is not None:
                return extracted

            # Fallback: use forward_head for models that only expose pooled outputs in dict form
            return backbone.forward_head(feats, pre_logits=True)

        # Now feats should be [B, num_tokens, D] Tensor
        # Extract features based on pool_type
        if pool_type == 'all':
            return feats.mean(dim=1) # Average all tokens [B, D]
    
        elif pool_type == 'cls':
            # 提取第一个token作为CLS特征
            return feats[:, 0, :]  # [B, D]

        elif pool_type == 'reg_avg':
            # Extract REG tokens (tokens 1-4) and average
            num_prefix = getattr(backbone, 'num_prefix_tokens', 5) # Dinov3中Register token数量默认为4，加上CLS一共5个prefix tokens
            num_reg = num_prefix - 1  # Subtract CLS 剔除 CLS token
            if num_reg > 0:
                reg_tokens = feats[:, 1:1+num_reg, :]  # [B, num_reg, D]
                return reg_tokens.mean(dim=1)  # [B, D] 对REG tokens进行平均池化，得到一个[B, D]的向量
            else:
                # If no REG tokens, fallback to CLS 没有REG token时，就用CLS token作为特征
                print("[Warning] No REG tokens found, using CLS instead")
                return feats[:, 0, :] 

        elif pool_type == 'cls+reg':
            # Concatenate CLS and REG average 拼接CLS和average REG
            cls_feat = feats[:, 0, :]  # [B, D]

            num_prefix = getattr(backbone, 'num_prefix_tokens', 5)
            num_reg = num_prefix - 1
            if num_reg > 0:
                reg_tokens = feats[:, 1:1+num_reg, :]  # [B, num_reg, D]
                reg_feat = reg_tokens.mean(dim=1)  # [B, D]
            else:
                # If no REG tokens, duplicate CLS
                reg_feat = cls_feat

            return torch.cat([cls_feat, reg_feat], dim=-1)  # [B, 2D] 拼接CLS和average REG，得到一个[B, 2D]的向量

        elif pool_type == 'cls+patch':
            # Concatenate CLS and PATCH average (without register tokens) 拼接CLS和average PATCH（不包含REG tokens）
            cls_feat = feats[:, 0, :]  # [B, D] CLS token

            # Get patch tokens starting position (skip CLS and register tokens)
            num_prefix = getattr(backbone, 'num_prefix_tokens', 5)
            patch_tokens = feats[:, num_prefix:, :]  # [B, num_patches, D] 跳过CLS和REG tokens，从第5个token开始就是PATCH tokens
            patch_feat = patch_tokens.mean(dim=1)  # [B, D] 对PATCH tokens进行平均池化，得到一个[B, D]的向量

            return torch.cat([cls_feat, patch_feat], dim=-1)  # [B, 2D] 拼接CLS和average PATCH，得到一个[B, 2D]的向量

        elif pool_type == 'patch_avg':
            # Use original forward_head (PATCH average) 原始的forward_head，会对PATCH tokens进行平均池化，得到一个[B, D]的向量，，dinov3的gobal_pool默认为avg，平均池化Patch Tokens
            return backbone.forward_head(feats, pre_logits=True)  

        elif pool_type == 'custom_tokens':
            # Average custom token indices 根据自定义的token索引列表，提取对应的tokens并进行平均池化
            if custom_token_indices is None or len(custom_token_indices) == 0:
                raise ValueError("custom_token_indices must be provided when pool_type='custom_tokens'")

            # Extract tokens at specified indices
            try:
                selected_tokens = feats[:, custom_token_indices, :]  # [B, num_selected, D] 根据自定义的token索引列表，提取对应的tokens
                return selected_tokens.mean(dim=1)  # [B, D] 对提取的tokens进行平均池化，得到一个[B, D]的向量
            except IndexError as e:
                raise ValueError(f"Token indices {custom_token_indices} out of range. "
                               f"Total tokens: {feats.shape[1]}. Error: {e}")

        else:
            raise ValueError(f"Unknown pool_type: {pool_type}. "
                           f"Valid options: 'patch_avg', 'cls', 'reg_avg', 'cls+reg', 'cls+patch', 'custom_tokens'")

    def forward(self, pixel_values,labels=None):
        if self.model_name == 'dinov3_vit_7b' and pixel_values.dtype != torch.float16:
            pixel_values = pixel_values.half()  # 强制转换为FP16，确保与Backbone一致
        feats = self._extract_feats(self.backbone, pixel_values, pool_type=self.pool_type, custom_token_indices=self.custom_token_indices)   # [B, D] or [B, 2D]
        feats = F.normalize(feats, dim=-1) # L2归一化，DINO系列模型是在超球面上训练的，特征向量的方向包含语义信息，而模长往往包含无关信息（如对比度），归一化可以显著稳定训练并提升分类稳定性
        head_dtype = next(self.head.parameters()).dtype
        feats = feats.to(dtype=head_dtype)  # 将feats转换为与分类头相同的数据类型，避免dtype不匹配的问题
        head_outputs = self.head(feats)
        logits = head_outputs['logits']
        aux_loss = head_outputs.get('aux_loss')

        loss = None
        if labels is not None:
            if self.num_classes == 1:
                loss = self.loss_fct(logits.view(-1), labels.float().view(-1))
            else:
                loss = self.loss_fct(logits, labels.view(-1).long())

            if aux_loss is not None:
                loss = loss + aux_loss.to(loss.dtype)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )

    def get_image_processor_or_transform(self):
        data_config = resolve_model_data_config(self.backbone)
        train_transform = create_transform(**data_config, is_training=True)
        eval_transform = create_transform(**data_config, is_training=False)
        return train_transform, eval_transform


@register_model("dinov3_vits16")
class DINOv3Vits16(DINOv3Model):
    def __init__(self, **kwargs):
        super().__init__(model_name='dinov3_vits16', **kwargs)


@register_model("dinov3_vitb16")
class DINOv3Vitb16(DINOv3Model):
    def __init__(self, **kwargs):
        super().__init__(model_name='dinov3_vitb16', **kwargs)


@register_model("dinov3_vitl16")
class DINOv3Vitl16(DINOv3Model):
    def __init__(self, **kwargs):
        super().__init__(model_name='dinov3_vitl16', **kwargs)


@register_model("dinov3_vith16")
class DINOv3Vith16(DINOv3Model):
    def __init__(self, **kwargs):
        super().__init__(model_name='dinov3_vith16', **kwargs)

@register_model("dinov3_vith16-linear")
class DINOv3Vith16Linear(DINOv3Model):
    def __init__(self, **kwargs):
        super().__init__(model_name='dinov3_vith16', layer_strategy='linear',**kwargs)


@register_model("dinov3_vith16-mlp")
class DINOv3Vith16MLP(DINOv3Model):
    def __init__(self, **kwargs):
        super().__init__(model_name='dinov3_vith16', layer_strategy='mlp',**kwargs)


@register_model("dinov3_vith16-bottleneck")
class DINOv3Vith16Bottleneck(DINOv3Model):
    def __init__(self, **kwargs):
        super().__init__(model_name='dinov3_vith16', layer_strategy='bottleneck',**kwargs)

@register_model("dinov3_vit_7b")
class DINOv3Vit7B(DINOv3Model):
    def __init__(self, **kwargs):
        super().__init__(model_name='dinov3_vit_7b', **kwargs)





if __name__ == "__main__":
    # Simple test
    model = DINOv3Vits16()
    dummy_img = torch.randn(2, 3, 256, 256)
    out = model(dummy_img, labels=torch.tensor([0, 1]))
    print(f"Output loss: {out.loss}")
    print(f"Output logits shape: {out.logits.shape}")
    
    