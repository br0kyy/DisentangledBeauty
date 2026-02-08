"""
Landmark Encoder using MobileFaceNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import transforms

from .mobilefacenet import MobileFaceNet


class LandmarkEncoder(nn.Module):
    """
    Encoder for predicting facial landmarks.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize LandmarkEncoder.

        Args:
            model_dir (str): Path to the pre-trained MobileFaceNet checkpoint.
        """
        super().__init__()
        # MobileFaceNet with 136 output channels (68 landmarks * 2 coords)
        self.model = MobileFaceNet([112, 112], 136)
        self.device = config.device
        # Load checkpoint
        # Using map_location='cpu' is safer if CUDA is not available at init time,
        # but original code didn't specify. Adding it is safer.
        checkpoint = torch.load(config.landmark_encoder_path, map_location=self.device)

        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.eval()
        self.resize = transforms.Resize(112)
        self.to_device = transforms.Lambda(lambda x: x.to(self.device))

    def preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Resize images to 112x112.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Resized images.
        """
        return self.resize(imgs)

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict landmarks.

        Args:
            imgs (torch.Tensor): Input images, typically [B, 3, 256, 256] or similar.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Scaled outputs (landmarks scaled to 112x112 space).
                - Jawline-removed landmarks (reshaped to [B, 51, 2]).
        """
        resized_images = self.preprocess(imgs)
        outputs, _ = self.model(resized_images)

        batch_size = resized_images.shape[0]
        # Reshape to [B, 68, 2]
        # Output of MobileFaceNet is usually normalized?
        # Original code multiplied by 112, suggesting output was in [0, 1].
        landmarks = torch.reshape(outputs * 112, (batch_size, 68, 2))

        # Return:
        # 1. Raw outputs scaled to image size (112)
        # 2. Landmarks excluding points 0-16 (jawline), keeping 17-67 (51 points)
        return outputs * 112, landmarks[:, 17:, :]
