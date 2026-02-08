"""
Identity Loss Module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from disbeauty.models.encoders.id_encoder.model_irse import Backbone


class IDLoss(nn.Module):
    """
    Identity Loss using Pre-trained ArcFace Backbone (IR-SE50).
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize IDLoss.

        Args:
            pretrained_model_path (str): Path to the pre-trained backbone checkpoint.
        """
        super().__init__()
        # Initialize ArcFace Backbone: IR_SE50
        self.device = config.device
        self.facenet = Backbone(
            input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se"
        )
        self.facenet.load_state_dict(
            torch.load(config.id_encoder_path, map_location=self.device)
        )
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        """
        # If input size is not 256, resize using interpolate
        if self.device == "mps":
            if x.shape[2] != 256:
                x = F.interpolate(
                    x, size=(256, 256), mode="bilinear", align_corners=True
                )
            x = x[:, :, 35:223, 32:220]  # Crop interesting region
            # Replace face_pool (AdaptiveAvgPool) with interpolate to solve MPS compatibility issues
            x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=True)
        else:
            if x.shape[2] != 256:
                x = self.pool(x)
            x = x[:, :, 35:223, 32:220]  # Crop interesting region
            x = self.face_pool(x)

        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate ID Loss (1 - Cosine Similarity).

        Args:
            y_hat (torch.Tensor): Generated images.
            y (torch.Tensor): Target ID images.

        Returns:
            torch.Tensor: Scalar loss.
        """
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)

        # Detach target features to prevent gradient flow back to target encoder (if needed)
        y_feats = y_feats.detach()

        loss = torch.tensor(0.0, device=y.device)
        for i in range(n_samples):
            # Calculate cosine similarity (dot product of normalized vectors, assuming they are normalized?)
            # The Backbone usually returns features. Cosine similarity is typically used.
            # dot product is sufficient if vectors are normalized.
            # Note: ArcFace features are often normalized?
            # If not, strictly, we should normalize.
            # However, preserving original logic:
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target

        return loss / n_samples
