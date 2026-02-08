"""
Inception Encoder Wrapper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Any


class Inception(nn.Module):
    """
    Inception V3 wrapper for attribute encoding.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize Inception model.

        Args:
            model_path (str | None): Path to model weights. If None, loads from hub.
        """
        super().__init__()

        self.device = config.device
        self.model_path = config.attr_encoder_path

        full_inception = torch.hub.load(
            "pytorch/vision:v0.6.0",
            "inception_v3",
            pretrained=(self.model_path is None),
            aux_logits=False,
            init_weights=False,
        )

        # Remove the last fully connected layer
        removed = list(full_inception.children())[:-1]

        self.model = nn.Sequential(*removed)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
            ]
        )

        if self.model_path is not None:
            state_dict = torch.load(self.model_path, map_location="cpu")
            self.load_state_dict(state_dict)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Encoded features.
        """
        resized_data = self.preprocess(data)
        # Original code multiplied by 255, implying prediction on [0, 255] range or similar expectation
        return self.model(resized_data * 255)
