"""
model.py – MetadataAwareEfficientNetV2 for Age Regression

Architecture: Mid-Fusion with Gating Mechanism
==============================================
                     ┌─────────────────────┐
   Image ─────────── │   EfficientNetV2-S  │
                     │   (Early Stages     │
                     │    0-2: ch=64)      │
                     └────────┬────────────┘
                              │ AdaptiveAvgPool → [B, 64]
                              ↓
  Metadata ─────────── MetadataExtractor → [B, 128]
      (sex + view)             │
                              ↓
                     ┌─────────────────────┐
                     │   Gating Mechanism  │ → [B, 256]  (fusion_dim)
                     └────────┬────────────┘
                              │ reshape → [B, 256, 1, 1]  injected back
                              ↓
                     ┌─────────────────────┐
                     │  Remaining Backbone  │ (stages 3+, conv_head, bn2)
                     │  → [B, 1280, 7, 7]  │
                     └────────┬────────────┘
                              │ GlobalAvgPool → [B, 1280]
                              ↓
                     Concat([1280, 256]) → [B, 1536]
                              ↓
                     Regression Head → [B, 1]  (predicted age)
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import timm


# ---------------------------------------------------------------------------
# Metadata Feature Extractor
# ---------------------------------------------------------------------------
class MetadataExtractor(nn.Module):
    """
    Two-layer MLP that projects raw metadata (sex + view_position)
    to a richer embedding.

    Input  shape: [B, metadata_dim]  (default: [B, 2])
    Output shape: [B, out_dim]       (default: [B, 128])
    """

    def __init__(self, in_dim: int = 2, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        hidden = out_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Gating Fusion Module
# ---------------------------------------------------------------------------
class GatingFusion(nn.Module):
    """
    Mid-fusion with a soft gating mechanism.

    Given:
      - early image features : [B, img_early_dim]
      - metadata features    : [B, meta_dim]

    Produces fused representation : [B, fusion_dim]

    The gate is a sigmoid over a linear projection of the metadata,
    which learns to weight image features channel-wise.
    """

    def __init__(
        self,
        img_early_dim: int = 64,
        meta_dim: int = 128,
        fusion_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        combined_dim = img_early_dim + meta_dim

        # Gate: metadata → gate vector in [0,1]^img_early_dim
        self.gate_proj = nn.Sequential(
            nn.Linear(meta_dim, img_early_dim),
            nn.Sigmoid(),
        )

        # Fusion MLP: gated_image ++ metadata → fusion_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        img_early: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        gate = self.gate_proj(meta)                   # [B, img_early_dim]
        gated_img = img_early * gate                  # element-wise gate
        fused = torch.cat([gated_img, meta], dim=1)   # [B, img_early_dim + meta_dim]
        return self.fusion_mlp(fused)                 # [B, fusion_dim]


# ---------------------------------------------------------------------------
# Regression Head
# ---------------------------------------------------------------------------
class RegressionHead(nn.Module):
    """
    Final fully-connected regression head.

    Input  shape: [B, in_dim]
    Output shape: [B, 1]   (raw predicted value, NO activation)
    """

    def __init__(self, in_dim: int = 1536, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
class MetadataAwareEfficientNetV2(nn.Module):
    """
    MetadataAwareEfficientNetV2 – Mid-Fusion with Gating Mechanism
    for Chest X-ray Age Regression.

    Parameters
    ----------
    metadata_dim : int
        Number of metadata features (default 2: sex + view_position).
    pretrained : bool
        Whether to load ImageNet pretrained weights for the backbone.
    in_channels : int
        Number of image channels. 1 for grayscale, 3 for RGB.
        When in_channels=1 the stem conv is adapted automatically.
    img_early_dim : int
        Channels of early image features (after Stages 0-2 of backbone).
        EfficientNetV2-S Stage 2 output = 64.
    meta_embedding_dim : int
        Dimension of the metadata embedding (default 128).
    fusion_dim : int
        Dimension of the mid-fusion output (default 256).
    late_img_dim : int
        Channels after the full backbone conv_head (EfficientNetV2-S = 1280).
    head_dropout : float
        Dropout probability in the regression head.
    """

    # Architecture constants (EfficientNetV2-S)
    BACKBONE_NAME: str = "tf_efficientnetv2_s"
    # Indices of stages that form the "early" part (outputs 64-ch features)
    EARLY_STAGE_INDICES: Tuple[int, ...] = (0, 1, 2)   # stages 0,1,2 → 64 ch

    def __init__(
        self,
        metadata_dim: int = 2,
        pretrained: bool = True,
        in_channels: int = 1,
        img_early_dim: int = 64,
        meta_embedding_dim: int = 128,
        fusion_dim: int = 256,
        late_img_dim: int = 1280,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.img_early_dim = img_early_dim
        self.fusion_dim = fusion_dim
        self.late_img_dim = late_img_dim

        # ------------------------------------------------------------------
        # 1. Load backbone with features_only to access intermediate stages
        # ------------------------------------------------------------------
        backbone_full = timm.create_model(
            self.BACKBONE_NAME,
            pretrained=pretrained,
            in_chans=in_channels,       # handles grayscale automatically
        )

        # ------------------------------------------------------------------
        # 2. Split backbone into early / remaining parts
        # ------------------------------------------------------------------
        # EfficientNetV2-S children:
        #   conv_stem, bn1, blocks, conv_head, bn2, global_pool, classifier
        # backbone_full.blocks is an nn.Sequential of 6 stage groups (0-5)

        self.stem = nn.Sequential(
            backbone_full.conv_stem,
            backbone_full.bn1,
        )

        # Early stages (0, 1, 2) → output 64 channels at stride-8
        self.early_stages = nn.Sequential(
            *[backbone_full.blocks[i] for i in self.EARLY_STAGE_INDICES]
        )

        # Remaining stages (3, 4, 5) + conv_head + bn2
        remaining_block_indices = [
            i for i in range(len(backbone_full.blocks))
            if i not in self.EARLY_STAGE_INDICES
        ]
        self.late_stages = nn.Sequential(
            *[backbone_full.blocks[i] for i in remaining_block_indices],
            backbone_full.conv_head,
            backbone_full.bn2,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)   # → [B, late_img_dim, 1, 1]

        # ------------------------------------------------------------------
        # 3. Early image feature pooling (spatial → vector)
        # ------------------------------------------------------------------
        self.early_pool = nn.AdaptiveAvgPool2d(1)    # [B, 64, H, W] → [B, 64, 1, 1]

        # ------------------------------------------------------------------
        # 4. Metadata extractor
        # ------------------------------------------------------------------
        self.metadata_extractor = MetadataExtractor(
            in_dim=metadata_dim,
            out_dim=meta_embedding_dim,
        )

        # ------------------------------------------------------------------
        # 5. Gating fusion (mid-fusion)
        # ------------------------------------------------------------------
        self.gating_fusion = GatingFusion(
            img_early_dim=img_early_dim,
            meta_dim=meta_embedding_dim,
            fusion_dim=fusion_dim,
        )

        # Projection to inject fused features back into late stages
        # We add the fused vector as a channel-bias after reshaping to [B, fusion_dim, 1, 1]
        # then project down to late_stage_in_channels for channel-wise addition
        # --- actually we pass it as a residual AFTER late stages via concatenation ---

        # ------------------------------------------------------------------
        # 6. Regression head
        # ------------------------------------------------------------------
        final_dim = late_img_dim + fusion_dim   # 1280 + 256 = 1536
        self.regression_head = RegressionHead(
            in_dim=final_dim,
            dropout=head_dropout,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        image    : FloatTensor [B, C, H, W]
        metadata : FloatTensor [B, metadata_dim]

        Returns
        -------
        FloatTensor [B, 1]  — predicted age (raw, no activation)
        """
        # --- Stem ---
        x = self.stem(image)                              # [B, 24, H/2, W/2]

        # --- Early stages → extract early image features ---
        x = self.early_stages(x)                          # [B, 64, H/8, W/8]
        early_feat = self.early_pool(x).flatten(1)        # [B, 64]

        # --- Metadata embedding ---
        meta_feat = self.metadata_extractor(metadata)     # [B, 128]

        # --- Mid-fusion with gating ---
        fused_feat = self.gating_fusion(early_feat, meta_feat)  # [B, 256]

        # --- Remaining backbone (late stages) ---
        x = self.late_stages(x)                           # [B, 1280, 7, 7]
        late_feat = self.global_pool(x).flatten(1)        # [B, 1280]

        # --- Feature concatenation ---
        final_feat = torch.cat([late_feat, fused_feat], dim=1)  # [B, 1536]

        # --- Regression head ---
        out = self.regression_head(final_feat)             # [B, 1]
        return out

    # ------------------------------------------------------------------
    def get_trainable_params(self) -> int:
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze / unfreeze stem + blocks (useful for fine-tuning)."""
        for param in self.stem.parameters():
            param.requires_grad = not freeze
        for param in self.early_stages.parameters():
            param.requires_grad = not freeze
        for param in self.late_stages.parameters():
            param.requires_grad = not freeze
