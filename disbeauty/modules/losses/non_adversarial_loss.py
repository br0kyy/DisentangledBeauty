"""
Non-Adversarial Loss Functions (Reconstruction, L2, Landmark, ID).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim

# Initialize standard loss criteria
# Note: Initializing these at module level works, but typically inside a class or function is safer for device handling.
# However, standard PyTorch loss functions without parameters are stateless, so it's fine.
l1_criterion = nn.L1Loss(reduction="mean")
l2_criterion = nn.MSELoss(reduction="mean")


def rec_loss(
    attr_images: torch.Tensor, generated_images: torch.Tensor, alpha: float
) -> torch.Tensor:
    """
    Calculate Reconstruction Loss (Combined MS-SSIM and L1).

    Args:
        attr_images (torch.Tensor): Target attribute images.
        generated_images (torch.Tensor): Generated images.
        alpha (float): Weight for MS-SSIM term.

    Returns:
        torch.Tensor: Weighted loss.
    """
    ms_ssim_loss = 1 - ms_ssim(
        attr_images, generated_images, data_range=1.0, size_average=True
    )
    l1_loss_value = l1_criterion(attr_images, generated_images)
    return alpha * ms_ssim_loss + (1 - alpha) * l1_loss_value


def id_loss(
    encoded_input_image: torch.Tensor, encoded_generated_image: torch.Tensor
) -> torch.Tensor:
    """
    Calculate simple L1 ID Loss between encoded features.

    Note: This is different from the deep ID loss in id_loss.py.

    Args:
        encoded_input_image (torch.Tensor): Features of input image.
        encoded_generated_image (torch.Tensor): Features of generated image.

    Returns:
        torch.Tensor: L1 loss.
    """
    return l1_criterion(encoded_input_image, encoded_generated_image)


def landmark_loss(
    input_attr_lnd: torch.Tensor, output_lnd: torch.Tensor
) -> torch.Tensor:
    """
    Calculate Landmark Loss (L2).

    Args:
        input_attr_lnd (torch.Tensor): Target landmarks.
        output_lnd (torch.Tensor): Predicted landmarks.

    Returns:
        torch.Tensor: L2 loss.
    """
    loss = l2_criterion(input_attr_lnd, output_lnd)
    return loss


def l2_loss(attr_images: torch.Tensor, generated_images: torch.Tensor) -> torch.Tensor:
    """
    Calculate L2 Loss (MSE) between images.

    Args:
        attr_images (torch.Tensor): Target images.
        generated_images (torch.Tensor): Generated images.

    Returns:
        torch.Tensor: MSE loss.
    """
    loss = l2_criterion(attr_images, generated_images)
    return loss
