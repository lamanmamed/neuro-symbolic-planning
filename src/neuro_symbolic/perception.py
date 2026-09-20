"""Image-to-symbol grounding with the saved CIFAR-100 projection model."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageEncoder(nn.Module):
    """MobileNetV3-small backbone followed by a 128-dimensional projection head."""

    def __init__(self, projection_dim: int = 128) -> None:
        super().__init__()

        # The checkpoint contains the backbone weights, so no network download is needed.
        mobilenet = models.mobilenet_v3_small(weights=None)
        self.backbone = nn.Sequential(*list(mobilenet.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(576, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(images).flatten(1)
        return self.projection(features)


def preprocess_image(image: torch.Tensor) -> torch.Tensor:
    """Convert a raw CHW or BCHW tensor to the format used during training."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected an RGB tensor with shape [3,H,W] or [B,3,H,W].")

    image = image.float()
    if image.max() > 1.0:
        image = image / 255.0

    image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    return (image - mean) / std


def load_projection_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[ImageEncoder, list[str], torch.Tensor, dict]:
    """Load the image encoder and the class vectors stored with its checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_words = list(checkpoint["class_words"])
    text_embeddings = checkpoint["text_embeddings"].to(device)

    model = ImageEncoder(projection_dim=text_embeddings.shape[1]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    metadata = {
        "epoch": int(checkpoint.get("epoch", -1)),
        "val_loss": float(checkpoint.get("val_loss", float("nan"))),
        "val_similarity": float(checkpoint.get("val_similarity", float("nan"))),
    }
    return model, class_words, text_embeddings, metadata


def predict_object(
    image: torch.Tensor,
    checkpoint_path: str | Path,
    top_k: int = 5,
    device: str | torch.device | None = None,
) -> list[tuple[str, float]]:
    """Predict CIFAR-100 labels by cosine similarity in the shared embedding space."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, class_words, text_embeddings, _ = load_projection_model(checkpoint_path, device)
    batch = preprocess_image(image).to(device)

    with torch.no_grad():
        image_embedding = F.normalize(model(batch), p=2, dim=1)
        class_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        similarities = image_embedding @ class_embeddings.T

    values, indices = torch.topk(similarities[0], k=min(top_k, similarities.shape[1]))
    return [(class_words[int(i)], float(v)) for i, v in zip(indices, values)]
