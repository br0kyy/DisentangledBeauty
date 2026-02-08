import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from disbeauty.modules.comboloss.main.inference import FacialBeautyPredictor
from disbeauty.modules.losses.id_loss import IDLoss


class BeautyAndIDScorer:
    def __init__(self, combo_model_path, id_model_path, device=None):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize beauty predictor
        self.beauty_predictor = FacialBeautyPredictor(combo_model_path)
        self.beauty_predictor.model = self.beauty_predictor.model.to(self.device)

        # Initialize ID loss calculator
        self.id_loss = IDLoss(id_model_path)
        self.id_loss.eval()
        self.id_loss = self.id_loss.to(self.device)

        # Define transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_scores(self, original_img, generated_img):
        """Calculate beauty score and ID similarity"""
        # Convert PIL images to tensors
        orig_tensor = self.transform(original_img).unsqueeze(0).to(self.device)
        gen_tensor = self.transform(generated_img).unsqueeze(0).to(self.device)

        # Get beauty score
        with torch.no_grad():
            beauty_score, _ = self.beauty_predictor.model(gen_tensor)
            beauty_score = beauty_score.item()

            # Calculate ID similarity
            id_loss_val = self.id_loss(gen_tensor, orig_tensor)
            id_similarity = 1 - id_loss_val.item()

        return beauty_score, id_similarity

    def save_image_with_scores(
        self, original_img_path, generated_img, output_dir, rate
    ):
        """Save generated image with beauty and ID scores in filename"""
        original_img = Image.open(original_img_path).convert("RGB")
        beauty_score, id_similarity = self.get_scores(original_img, generated_img)

        # Create output filename
        output_filename = f"{rate}_{beauty_score:.3f}_{id_similarity:.3f}.png"
        output_path = os.path.join(output_dir, output_filename)

        # Save image
        generated_img.save(output_path)
        print(f"Saved {output_filename}")
        return output_path, beauty_score, id_similarity


# Usage example:
"""
scorer = BeautyAndIDScorer(
    combo_model_path='path/to/ComboNet_SCUTFBP5500.pth',
    id_model_path='path/to/model_ir_se50.pth'
)

# In your image generation loop:
output_path, beauty_score, id_similarity = scorer.save_image_with_scores(
    original_img_path='path/to/original.jpg',
    generated_img=generated_pil_image,
    output_dir='path/to/output/dir'
)
"""
