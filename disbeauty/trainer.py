"""
Trainer class for the DisentangledBeauty model.
"""

from __future__ import annotations

import os
from typing import Any

import lpips
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import transforms
from tqdm import tqdm

import wandb
from disbeauty.modules.losses.adversarial_loss import calc_dw_loss, r1_regularization
from disbeauty.modules.losses.non_adversarial_loss import (
    l2_loss,
    landmark_loss,
    rec_loss,
)


class Trainer:
    """
    Trainer class managing the training loop and optimization steps using standard PyTorch models.
    """

    def __init__(
        self,
        config: DictConfig,
        discriminator_optimizer: torch.optim.Optimizer,
        adversarial_mapper_optimizer: torch.optim.Optimizer,
        non_adversarial_mapper_optimizer: torch.optim.Optimizer,
        discriminator: nn.Module,
        generator: nn.Module,
        id_encoder: nn.Module,
        attr_encoder: nn.Module,
        landmark_encoder: nn.Module,
        mapper: nn.Module,
        device: str | torch.device,
    ):
        """
        Initialize the Trainer.

        Args:
            config (DictConfig): Configuration object.
            discriminator_optimizer (torch.optim.Optimizer): Optimizer for the discriminator.
            adversarial_mapper_optimizer (torch.optim.Optimizer): Optimizer for the mapper (adversarial loss).
            non_adversarial_mapper_optimizer (torch.optim.Optimizer): Optimizer for the mapper (reconstruction loss).
            discriminator (nn.Module): Discriminator model.
            generator (nn.Module): Generator model.
            id_encoder (nn.Module): Identity encoder model.
            attr_encoder (nn.Module): Attribute encoder model.
            landmark_encoder (nn.Module): Landmark encoder model.
            mapper (nn.Module): Latent mapper model.
            device (str | torch.device): Computation device.
        """
        self.config = config
        self.device = device

        self.discriminator_optimizer = discriminator_optimizer
        self.adversarial_mapper_optimizer = adversarial_mapper_optimizer
        self.non_adversarial_mapper_optimizer = non_adversarial_mapper_optimizer

        self.discriminator = discriminator
        self.generator = generator
        self.id_encoder = id_encoder
        self.attr_encoder = attr_encoder
        self.landmark_encoder = landmark_encoder
        self.mapper = mapper

        # Initialize LPIPS
        self.lpips_loss = lpips.LPIPS(net="alex")
        self.lpips_loss.eval()
        self.lpips_loss.to(self.device)

        self.step = 0

    def train_discriminator(
        self, real_w: torch.Tensor, generated_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Train the discriminator to distinguish between real and generated W.

        Args:
            real_w (torch.Tensor): Real W latents.
            generated_w (torch.Tensor): Generated W latents (fake).

        Returns:
            tuple: (error_real, prediction_real, error_fake, prediction_fake)
        """
        self.discriminator_optimizer.zero_grad()
        real_w.requires_grad_()

        prediction_real = self.discriminator(real_w).view(-1)
        error_real = calc_dw_loss(prediction_real, 1.0)
        error_real.backward(retain_graph=True)

        r1_error = r1_regularization(self.config.r1_param, prediction_real, real_w)
        r1_error.backward()

        cloned_generated_w = generated_w.clone().detach()
        prediction_fake = self.discriminator(cloned_generated_w).view(-1)
        error_fake = calc_dw_loss(prediction_fake, 0.0)
        error_fake.backward()

        self.discriminator_optimizer.step()

        return error_real, prediction_real, error_fake, prediction_fake

    def train_mapper(
        self, generated_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Train the mapper to fool the discriminator.

        Args:
            generated_w (torch.Tensor): Generated W latents.

        Returns:
            tuple: (discriminative_loss, prediction)
        """
        self.adversarial_mapper_optimizer.zero_grad()
        prediction = self.discriminator(generated_w).view(-1)
        discriminative_loss = calc_dw_loss(prediction, 1.0)
        discriminative_loss.backward()
        self.adversarial_mapper_optimizer.step()

        return discriminative_loss, prediction

    def adversarial_train_step(
        self, real_w: torch.Tensor, fake_data: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Perform an adversarial training step (Discriminator + Mapper).

        Args:
            real_w: Sampled real W latents.
            fake_data: Generated W latents (from mapper).

        Returns:
            tuple: Metrics (error_real, error_fake, etc.)
        """
        error_real, prediction_real, error_fake, prediction_fake = (
            self.train_discriminator(real_w, fake_data)
        )
        g_error, g_pred = self.train_mapper(fake_data)

        # Return detached metrics for logging
        return (
            error_real,
            error_fake,
            torch.mean(prediction_real),
            torch.mean(prediction_fake),
            g_error,
            torch.mean(g_pred),
        )

    def non_adversarial_train_step(
        self,
        id_images: torch.Tensor,
        attr_images: torch.Tensor,
        fake_data: torch.Tensor,
        real_landmarks: torch.Tensor,
        use_rec_extra_term: bool,
    ) -> torch.Tensor:
        """
        Perform non-adversarial training step (Reconstruction, ID, Landmark, etc.).

        Args:
            id_images: Input images for ID.
            attr_images: Input images for Attribute.
            fake_data: Generated W latents (mapped).
            real_landmarks: Ground truth landmarks.
            use_rec_extra_term: Whether to use extra reconstruction terms.

        Returns:
            torch.Tensor: Total loss value.
        """
        self.id_encoder.zero_grad()
        self.landmark_encoder.zero_grad()
        self.generator.zero_grad()
        self.mapper.zero_grad()  # Mapper was missing in original clear loop but should be here?
        # Actually standard practice is to zero the optimizer's params.
        self.non_adversarial_mapper_optimizer.zero_grad()

        total_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)

        # Generate images from the mapped latent W
        generated_images, _ = self.generator(
            [fake_data], input_is_latent=True, return_latents=False
        )

        normalized_generated_images = (generated_images + 1) / 2

        # Ensure correct device
        attr_images = attr_images.to(self.device)
        id_images = id_images.to(self.device)
        real_landmarks = real_landmarks.to(self.device)

        conf = self.config

        # 1. Identity Loss
        if conf.use_id:
            # id_encoder usually extracts features.
            # Assuming self.id_encoder is loss wrapper or returns loss?
            # Original code: self.id_encoder(generated_images, (id_images * 2) - 1)
            # If id_encoder is IDLoss class:
            id_loss_val = conf.lambda_id * self.id_encoder(
                generated_images, (id_images * 2) - 1
            )
            total_loss += id_loss_val
            wandb.log({"id_loss_val": id_loss_val.detach().cpu()}, step=self.step)

        # 2. Landmark Loss
        if conf.use_landmark:
            generated_landmarks, generated_landmarks_nojawline = self.landmark_encoder(
                normalized_generated_images
            )
            landmark_loss_val = (
                landmark_loss(generated_landmarks_nojawline, real_landmarks)
                * conf.lambda_lnd
            )
            total_loss += landmark_loss_val
            wandb.log(
                {"landmark_loss_val": landmark_loss_val.detach().cpu()}, step=self.step
            )

        # 3. Reconstruction Loss (SSIM + L1)
        if use_rec_extra_term and conf.use_reconstruction:
            alpha = 0.84
            rec_loss_val = conf.lambda_rec * rec_loss(
                attr_images, normalized_generated_images, alpha
            )
            total_loss += rec_loss_val
            wandb.log({"rec_loss_val": rec_loss_val.detach().cpu()}, step=self.step)

        # 4. L2 Loss
        if use_rec_extra_term and conf.use_l2:
            l2_loss_val = conf.lambda_l2 * l2_loss(
                attr_images, normalized_generated_images
            )
            total_loss += l2_loss_val
            wandb.log({"l2_loss_val": l2_loss_val.detach().cpu()}, step=self.step)

        # 5. VGG/LPIPS Loss
        if use_rec_extra_term and (not conf.use_adverserial):
            # LPIPS expects input in [-1, 1], so we use `generated_images` directly
            # attr_images is likely [0, 1] so we scale it.
            vgg_loss_val = torch.mean(
                conf.lambda_vgg
                * self.lpips_loss(generated_images, (attr_images * 2) - 1)
            )
            wandb.log({"vgg_loss_val": vgg_loss_val.detach().cpu()}, step=self.step)
            total_loss += vgg_loss_val

        total_loss.backward()
        self.non_adversarial_mapper_optimizer.step()

        return total_loss

    def train(self, dataloader: Any) -> None:
        """
        Main training loop.

        Args:
            dataloader: PyTorch DataLoader yielding batches of data.
        """
        print(f"Starting training on {self.device}...")

        epochs = self.config.epochs

        for epoch in range(epochs):
            pbar = tqdm(dataloader)
            for i, batch in enumerate(pbar):
                self.step += 1

                # Unpack batch: expected (id_img, attr_img, fake_w)
                if len(batch) == 3:
                    id_imgs, attr_imgs, fake_w = batch
                else:
                    continue

                id_imgs = id_imgs.to(self.device)
                attr_imgs = attr_imgs.to(self.device)
                # fake_w/real_w: Latent W to map towards or from?
                # Assuming 'fake_w' in variable name meant 'sample W' or 'real W' from dataset.
                real_w = fake_w.to(self.device).squeeze(1)  # Ensure dim matches

                # Extract landmarks for loss
                with torch.no_grad():
                    # Landmark encoder expects [0, 1]
                    _, real_landmarks = self.landmark_encoder(id_imgs)

                # --- Forward Pass ---
                # 1. Extract Features
                with torch.no_grad():
                    # id_encoder feature extraction
                    # Note: IDLoss wrapper might not expose extract_feats directly if not standard.
                    # Use internal net if needed. assuming models["id_encoder"] usage in main.py.
                    # But here self.id_encoder is the *Loss* module usually?
                    # The original passed 'id_encoder' which was IDLoss.
                    # The IDLoss class likely has 'extract_feats'.

                    # Assuming input range [-1, 1] for ID encoder
                    id_features = self.id_encoder.extract_feats((id_imgs * 2) - 1)
                    id_features = id_features.squeeze().unsqueeze(
                        1
                    )  # [Batch, 1, 512]? or [Batch, 512]

                    # Attribute encoder
                    attr_features = self.attr_encoder(attr_imgs)
                    attr_features = attr_features.squeeze().unsqueeze(1)

                # 2. Map to W
                input_vec = torch.cat((id_features, attr_features), 1)
                # Since dimensions match, simply concat?
                # Assuming id_features is [B, 512], attr_features is [B, 512] -> [B, 1024]?
                # If they were unsqueezed to [B, 1, 512], then concat on dim 1 gives [B, 2, 512]
                # Check LatentMapper input expectation.
                # Assuming simple concat of vectors:
                if id_features.dim() == 3:
                    # Flatten?
                    input_vec = torch.cat(
                        (
                            id_features.view(id_features.size(0), -1),
                            attr_features.view(attr_features.size(0), -1),
                        ),
                        1,
                    )
                else:
                    input_vec = torch.cat((id_features, attr_features), 1)

                mapped_w = self.mapper(input_vec)

                # --- Training Steps ---

                # Adversarial Step
                if self.config.use_adverserial:
                    self.adversarial_train_step(real_w, mapped_w)

                # Non-Adversarial Step
                use_rec_extra = self.step > self.config.rec_warmup  # Example logic
                self.non_adversarial_train_step(
                    id_imgs,
                    attr_imgs,
                    mapped_w,
                    real_landmarks,
                    use_rec_extra_term=True,
                )

                pbar.set_description(f"Epoch {epoch} Step {self.step}")
