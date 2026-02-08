"""
Main entry point for DisentangledBeauty inference.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, cast

import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import utils

from disbeauty.models.discriminator import Discriminator
from disbeauty.models.encoders.inception import Inception
from disbeauty.models.encoders.landmark_encoder.landmark_encoder import LandmarkEncoder
from disbeauty.models.latent_mapper import LatentMapper
from disbeauty.models.stylegan2.model import Generator
from disbeauty.modules.losses.id_loss import IDLoss


def get_parser(**kwargs: Any) -> argparse.ArgumentParser:
    """
    Configure and return the command line argument parser.

    Args:
        **kwargs: Arbitrary keyword arguments passed to ArgumentParser.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(**kwargs)

    # Global settings
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use for computation (e.g., 'cuda:0', 'cpu', 'mps').",
    )

    # Configuration entries
    parser.add_argument(
        "--training_config",
        type=str,
        default="configs/training.yaml",
        help="Path to the training configuration YAML file.",
    )

    # Input/Output paths
    parser.add_argument(
        "--input_image",
        type=str,
        help="Path to the input ID image.",
    )
    parser.add_argument(
        "--target_image",
        type=str,
        help="Path to the target Attribute image.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/result.jpg",
        help="Path to the output file or directory.",
    )
    parser.add_argument(
        "--generator_size",
        type=int,
        default=256,
        help="Resolution of the generated image.",
    )

    # Model Checkpoints
    parser.add_argument(
        "--generator_path",
        type=str,
        default="pretrained_models/550000.pt",
        help="Path to the pre-trained generator checkpoint.",
    )
    parser.add_argument(
        "--id_encoder_path",
        type=str,
        default="pretrained_models/model_ir_se50.pth",
        help="Path to the pre-trained ID encoder (IR-SE50) checkpoint.",
    )
    parser.add_argument(
        "--landmark_encoder_path",
        type=str,
        default="pretrained_models/mobilefacenet_model_best.pth.tar",
        help="Path to the pre-trained landmark encoder checkpoint.",
    )
    parser.add_argument(
        "--aesthetic_model_path",
        type=str,
        default="pretrained_models/ComboNet_SCUTFBP5500.pth",
        help="Path to the pre-trained ComboNet model checkpoint.",
    )

    # Inference Specific Models
    parser.add_argument(
        "--attr_encoder_path",
        type=str,
        default="pretrained_models/inception_v3.pt",
        help="Path to the trained attribute encoder checkpoint.",
    )
    parser.add_argument(
        "--mapper_path",
        type=str,
        default="pretrained_models/latent_mapper.pt",
        help="Path to the trained latent mapper checkpoint.",
    )

    return parser


def build_config(yaml_path: str, args: argparse.Namespace) -> DictConfig:
    """
    Build configuration by merging YAML file and command line arguments.

    Args:
        yaml_path: Path to the YAML configuration file.
        args: Parsed command line arguments.

    Returns:
        DictConfig: Merged configuration object.
    """
    if os.path.exists(yaml_path):
        yaml_cfg = OmegaConf.load(yaml_path)
    else:
        # Fallback to empty config if YAML does not exist
        # (e.g. pure inference without specific yaml)
        yaml_cfg = OmegaConf.create()

    # Convert argparse Namespace to OmegaConf
    # Filter out None values to allow defaults in YAML to persist if not overridden
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    args_cfg = OmegaConf.create(args_dict)

    # Merge: YAML config is base, CLI args override it
    cfg = OmegaConf.merge(yaml_cfg, args_cfg)
    return cast(DictConfig, cfg)


def load_inference_models(cfg: DictConfig) -> dict[str, torch.nn.Module]:
    """
    Load all necessary models for inference.

    Args:
        cfg: Configuration object containing model paths and device settings.

    Returns:
        dict[str, torch.nn.Module]: A dictionary of loaded and initialized models.
    """
    device = cfg.device

    print(f"Loading models to {device}...")

    # 1. ID Encoder (Using IDLoss wrapper as per original implementation)
    # The original code imported IDLoss to use as the encoder wrapper.
    id_encoder = IDLoss(cfg)

    # 2. Attribute Encoder (Inception based)
    # Load state dict first to check structure
    attr_encoder = Inception(cfg)

    # 4. Latent Mapper
    mapper = LatentMapper(cfg)

    # 5. Landmark Encoder
    landmark_encoder = LandmarkEncoder(cfg)

    # 6. Generator (StyleGAN2)
    generator_size = cfg.generator_size
    generator = Generator(generator_size, 512, 8)

    ckpt = torch.load(cfg.generator_path, map_location=device)
    # StyleGAN2 checkpoints usually have a 'g_ema' key for the generator
    if "g_ema" in ckpt:
        generator.load_state_dict(ckpt["g_ema"], strict=False)
    else:
        # Fallback if checkpoint structure is different
        generator.load_state_dict(ckpt, strict=False)

    # Move all models to device and set to eval mode
    models = {
        "id_encoder": id_encoder.to(device).eval(),
        "attr_encoder": attr_encoder.to(device).eval(),
        "mapper": mapper.to(device).eval(),
        "landmark_encoder": landmark_encoder.to(device).eval(),
        "generator": generator.to(device).eval(),
    }

    return models


def main() -> None:
    """
    Main execution function for inference.
    """
    parser = get_parser()
    args = parser.parse_args()

    # Build Configuration
    cfg = build_config(args.training_config, args)
    device = cfg.device

    # Validations
    if not cfg.input_image:
        print("Error: Inference requires --input_image to be specified.")
        return

    # Load Models
    models = load_inference_models(cfg)

    # Preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    # Load ID Image
    input_image_path = Path(cfg.input_image)
    if not input_image_path.exists():
        print(f"Error: Input image not found at {input_image_path}")
        return

    img = Image.open(input_image_path).convert("RGB")
    # unsqueeze(0) for batch dimension: [1, 3, 256, 256]
    id_image_tensor = transform(img).unsqueeze(0).to(device)

    print("Processing...")
    with torch.no_grad():
        # Extract ID features. Input usually scaled to [-1, 1] for some encoders?
        # Original code used `(id_image * 2) - 1`.
        id_features = models["id_encoder"].extract_feats((id_image_tensor * 2) - 1)
        # Ensure shape [1, 512] (assuming 512 dim)
        id_features = torch.squeeze(id_features).unsqueeze(0)

        # Extract current attributes
        current_attr = models["attr_encoder"](id_image_tensor).squeeze().unsqueeze(0)

        # Determine target attributes
        target_attr = current_attr  # Default: reconstruction

        if cfg.target_image:
            target_image_path = Path(cfg.target_image)
            if target_image_path.exists():
                tgt_img = Image.open(target_image_path).convert("RGB")
                tgt_tensor = transform(tgt_img).unsqueeze(0).to(device)
                target_attr = models["attr_encoder"](tgt_tensor).squeeze().unsqueeze(0)
            else:
                print(
                    f"Warning: Target image {cfg.target_image} not found. Using self-reconstruction."
                )

        # Concatenate ID and Attribute features
        # Assuming axis 1 is feature dimension
        input_vec = torch.cat((id_features, target_attr), 1)

        # Map to W space
        w_latent = models["mapper"](input_vec)

        # Generate Image
        # Generator expects list of latents if input_is_latent=True?
        # Checking implementation of StyleGAN2 generator...
        # Usually: generator([latents], ...)
        sample, _ = models["generator"](
            [w_latent], input_is_latent=True, return_latents=False
        )

        # Save Result
        output_path = Path(cfg.output_path)
        # Verify output directory exists
        if output_path.parent.name and not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # If output_path is a directory, append filename
        if output_path.is_dir() or str(output_path).endswith(os.sep):
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / "result.jpg"

        utils.save_image(
            sample,
            str(output_path),
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
