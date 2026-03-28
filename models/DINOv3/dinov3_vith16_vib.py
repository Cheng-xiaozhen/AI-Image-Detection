import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, cast

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


_TIMM_NAME = "vit_huge_plus_patch16_dinov3_qkvb"


def _resolve_local_checkpoint(timm_name: str) -> Optional[str]:
	cache_root = Path.home() / ".cache" / "huggingface" / "hub"
	if not cache_root.exists():
		return None

	for model_dir in sorted(cache_root.glob(f"models--timm--{timm_name}*")):
		for checkpoint in sorted(model_dir.glob("snapshots/*/model.safetensors"), reverse=True):
			if checkpoint.is_file():
				return str(checkpoint)
	return None


class VIBHead(nn.Module):
	def __init__(
		self,
		in_dim: int,
		hidden_dim: int,
		latent_dim: int,
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
		self.classifier = nn.Linear(latent_dim, 1)

	def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		if self.training:
			std = torch.exp(0.5 * logvar)
			eps = torch.randn_like(std)
			return mu + eps * std
		return mu

	def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
		return kl.sum(dim=-1).mean()

	def forward(self, x: torch.Tensor) -> Dict[str, Any]:
		h = self.encoder(x)
		mu = self.mu_proj(h)
		logvar = self.logvar_proj(h).clamp(min=-30.0, max=20.0)
		z = self.reparameterize(mu, logvar)
		return {"logits": self.classifier(z), "aux_loss": self.beta * self.kl_divergence(mu, logvar)}


@register_model("dinov3_vith16-vib")
class DINOv3Vith16VIB(nn.Module):
	def __init__(
		self,
		img_size: Optional[int] = None,
		fix_backbone: bool = True,
		head_dropout: float = 0.1,
		vib_beta: float = 1e-3,
		vib_hidden_dim: Optional[int] = None,
		vib_latent_dim: Optional[int] = None,
	):
		super().__init__()
		self.model_name = "dinov3_vith16"
		self.num_classes = 1

		default_path = _resolve_local_checkpoint(_TIMM_NAME)
		if img_size is not None:
			self.backbone = timm.create_model(_TIMM_NAME, pretrained=True, img_size=img_size)
		elif default_path and os.path.exists(default_path):
			self.backbone = timm.create_model(_TIMM_NAME, pretrained=False, checkpoint_path=default_path)
		else:
			self.backbone = timm.create_model(_TIMM_NAME, pretrained=True)

		reset_classifier = getattr(self.backbone, "reset_classifier", None)
		if callable(reset_classifier):
			reset_classifier(0)

		if fix_backbone:
			for param in self.backbone.parameters():
				param.requires_grad = False

		feat_dim = int(getattr(self.backbone, "num_features", 1280))
		self.vib_hidden_dim = vib_hidden_dim or max(1, feat_dim // 4)
		self.vib_latent_dim = vib_latent_dim or max(1, feat_dim // 8)
		self.head = VIBHead(
			in_dim=feat_dim,
			hidden_dim=self.vib_hidden_dim,
			latent_dim=self.vib_latent_dim,
			dropout=head_dropout,
			beta=vib_beta,
		)
		self.loss_fct = nn.BCEWithLogitsLoss()

	def _extract_feats(self, x: torch.Tensor) -> torch.Tensor:
		forward_features = getattr(self.backbone, "forward_features")
		feats = forward_features(x)
		return cast(torch.Tensor, feats)[:, 0, :]

	def forward_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
		feats = F.normalize(self._extract_feats(pixel_values), dim=-1)
		return self.head(feats)["logits"]

	def forward(self, pixel_values: torch.Tensor, labels=None) -> ImageClassifierOutput:
		logits = self.forward_logits(pixel_values)
		return ImageClassifierOutput(
			loss=None if labels is None else self.loss_fct(logits.view(-1), labels.float().view(-1)),
			logits=cast(torch.FloatTensor, logits),
		)

	def get_image_processor_or_transform(self):
		data_config = resolve_model_data_config(self.backbone)
		train_transform = create_transform(**data_config, is_training=True)
		eval_transform = create_transform(**data_config, is_training=False)
		return train_transform, eval_transform

if __name__ == "__main__":
    model = DINOv3Vith16VIB()
    train_transform, eval_transform = model.get_image_processor_or_transform()
    print(train_transform)
    print(eval_transform)
