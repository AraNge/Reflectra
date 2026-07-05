import os
from typing import List, Union
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class PretrainedCLIPEncoder(torch.nn.Module):
    """
    Pretrained CLIP wrapper for:
    - image -> CLIP image embedding
    - text -> CLIP text embedding
    - image/text cosine similarity

    This is written as a torch.nn.Module so you can later connect it
    to a projection layer and train image -> CLAP space.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Union[str, None] = None,
        freeze: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

        self.model.to(self.device_name)

        if freeze:
            self.freeze()

        self.model.eval()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        return Image.open(image_path).convert("RGB")

    @torch.no_grad()
    def encode_image(
        self,
        image_paths: Union[str, List[str]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Convert images into CLIP image embeddings.

        Args:
            image_paths: one image path or list of image paths
            normalize: L2-normalize embeddings for cosine similarity

        Returns:
            Tensor shape: [batch_size, embedding_dim]
        """

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images = [self.load_image(path) for path in image_paths]

        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(self.device_name) for k, v in inputs.items()}

        image_features = self.model.get_image_features(**inputs)

        if normalize:
            image_features = F.normalize(image_features, p=2, dim=-1)

        return image_features

    @torch.no_grad()
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Convert text into CLIP text embeddings.
        Useful for testing image-text retrieval on Flickr30k/COCO.
        """

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {k: v.to(self.device_name) for k, v in inputs.items()}

        text_features = self.model.get_text_features(**inputs)

        if normalize:
            text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features

    @torch.no_grad()
    def image_text_similarity(
        self,
        image_paths: Union[str, List[str]],
        texts: Union[str, List[str]],
    ) -> torch.Tensor:
        """
        Compute cosine similarity between images and texts.

        Returns:
            similarity matrix [num_images, num_texts]
        """

        image_embeds = self.encode_image(image_paths, normalize=True)
        text_embeds = self.encode_text(texts, normalize=True)

        return image_embeds @ text_embeds.T

    def forward(self, image_paths: Union[str, List[str]]):
        """
        Forward for future image -> CLAP projection training.

        Returns normalized CLIP image embeddings.
        """

        return self.encode_image(image_paths, normalize=True)


if __name__ == "__main__":
    encoder = PretrainedCLIPEncoder(freeze=True)

    image_path = "data/example_images/example.jpg"

    texts = [
        "a beach party with happy people",
        "a sad rainy city street",
        "a dog running in a park",
        "a dark nightclub with neon lights",
    ]

    scores = encoder.image_text_similarity(image_path, texts)

    print(scores.cpu().numpy())