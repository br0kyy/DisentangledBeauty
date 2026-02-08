"""
Adversarial loss functions for GAN training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd


def calc_dw_loss(probs: torch.Tensor, label: float) -> torch.Tensor:
    """
    Calculate the Discriminator loss using Binary Cross Entropy.

    Args:
        probs (torch.Tensor): Probabilities predicted by the discriminator.
        label (float): Target label (1.0 for real, 0.0 for fake).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    labels = torch.full((probs.size(0),), label, dtype=torch.float, device=probs.device)
    criterion = nn.BCELoss()
    adversarial_loss = criterion(probs, labels)
    return adversarial_loss


def r1_regularization(
    r1_coefficient: float, probs: torch.Tensor, ws: torch.Tensor
) -> torch.Tensor:
    """
    Compute R1 Regularization term.

    Args:
        r1_coefficient (float): Weighting coefficient for R1 regularization.
        probs (torch.Tensor): Discriminator output probabilities.
        ws (torch.Tensor): Input latent vectors (w).

    Returns:
        torch.Tensor: Scalar R1 regularization loss.
    """
    return (r1_coefficient / 2) * compute_grad2(probs, ws).mean()


def compute_grad2(probs: torch.Tensor, w_input: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared L2 norm of gradients of probabilities with respect to input.

    Args:
        probs (torch.Tensor): Discriminator output.
        w_input (torch.Tensor): Input latent vectors.

    Returns:
        torch.Tensor: Squared gradients.
    """
    batch_size = w_input.size(0)
    grads = autograd.grad(
        outputs=probs,
        inputs=w_input,
        grad_outputs=torch.ones_like(probs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten non-batch dimensions
    grads = grads.view(batch_size, -1)

    # Compute squared norm
    grad_norm = grads.norm(2, dim=1) ** 2
    return grad_norm
