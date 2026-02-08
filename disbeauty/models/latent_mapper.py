"""
Latent Mapper model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentMapper(nn.Module):
    """
    MLP that maps concatenated ID and Attribute features to W latent space.
    """

    def __init__(self, config) -> None:
        super().__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(2560, 2048),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(512, 512),
        )

        # Initialize weights
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=slope)
                nn.init.constant_(m.bias, 0)
        if config.mapper_path is not None:
            state_dict = torch.load(config.mapper_path, map_location="cpu")
            self.load_state_dict(state_dict)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_tensor (torch.Tensor): Concatenated features (ID + Attr).

        Returns:
            torch.Tensor: Mapped W latent vector.
        """
        return self.model(input_tensor)
