# DisBeauty - Disentangled Beauty Analysis and Manipulation

### Overview

DisBeauty is a deep learning framework designed for disentangled facial beauty analysis and manipulation. It leverages a combination of StyleGAN2, specific attribute encoders, and identity preservation mechanisms to achieve high-quality facial aesthetic enhancement or attribute transfer while maintaining the subject's identity.

### Architecture

The system implements a multi-component pipeline:

1.  **Encoders**:
    -   **Attribute Encoder**: Extracts facial attributes using InceptionV3.
    -   **ID Encoder**: Preserves identity features using IR-SE50.
    -   **Landmark Encoder**: Captures facial geometry and landmarks using MobileFaceNet.
2.  **Latent Mapper**: Maps combined identity and attribute features into the StyleGAN2 latent space.
3.  **Generator**: Synthesis network based on StyleGAN2.
4.  **Aesthetic Evaluation**: Integrated beauty scoring using ComboNet.

### Installation

#### Prerequisites

- Python 3.7+
- CUDA-capable GPU or Apple Silicon (MPS)

#### Install Dependencies

Ensure you have the required packages installed (PyTorch, Torchvision, OmegaConf, Pillow, etc.):

```bash
pip install torch torchvision omegaconf pillow
```

### Model Checkpoints

The system relies on several pretrained models which should be placed in the `pretrained_models/` directory:

-   **Generator**: `550000.pt` - StyleGAN2 generator weights.
-   **ID Encoder**: `model_ir_se50.pth` - IR-SE50 for identity feature extraction.
-   **Landmark Encoder**: MobileFaceNet checkpoint (e.g., `mobilefacenet_model_best.pth.tar`).
-   **Attribute Encoder**: `inception_v3.pt` - InceptionV3 based encoder.
-   **Latent Mapper**: `latent_mapper.pt` - Trained mapper network.
-   **Aesthetic Model**: `ComboNet_SCUTFBP5500.pth` - For aesthetic score evaluation.

### Usage

#### Basic Inference

To run inference using an identity image and an optional target attribute image:

```bash
python main.py --input_image path/to/id_image.jpg --target_image path/to/attr_image.jpg
```

#### Arguments

-   `--input_image`: Path to the input image providing the identity (Required).
-   `--target_image`: Path to the target image providing attributes. If omitted, performs self-reconstruction.
-   `--output_path`: Path to save the generated result (Default: `outputs/result.jpg`).
-   `--device`: Device to use (e.g., `cuda`, `mps`, `cpu`).

### Configuration

The system is configured via YAML files in the `configs/` directory.

-   `configs/training.yaml`: Contains hyperparameters for training and model configuration (loss weights, learning rates, etc.).

### Project Structure

```
DisBeauty/
├── configs/              # Configuration files
│   └── training.yaml
├── disbeauty/            # Main package source code
│   ├── data/             # Data loading and datasets
│   ├── models/           # Model architectures
│   │   ├── encoders/     # ID, Attribute, and Landmark encoders
│   │   ├── stylegan2/    # StyleGAN2 generator components
│   │   ├── discriminator.py
│   │   └── latent_mapper.py
│   └── modules/          # Utility modules and losses
│       ├── comboloss/    # Aesthetic scoring modules
│       └── losses/       # Training loss functions
├── pretrained_models/    # Directory for model checkpoints
├── examples/             # Example images
├── main.py               # Main entry point for inference
└── ffhq.csv              # Dataset metadata
```
