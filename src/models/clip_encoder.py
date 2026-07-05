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
    def encode_image_tokens(
        self,
        image_paths: Union[str, List[str]],
        include_cls_token: bool = True,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Convert images into CLIP vision token embeddings.

        This is useful for Q-Former, because Q-Former should attend to
        a sequence of visual tokens, not only one global image embedding.

        Args:
            image_paths:
                One image path or list of image paths.

            include_cls_token:
                If True, returns CLS token + patch tokens.
                If False, removes the first CLS token and returns only patch tokens.

            normalize:
                If True, L2-normalize every token embedding.

        Returns:
            Tensor shape:
                [batch_size, num_tokens, vision_hidden_dim]

            Example for openai/clip-vit-base-patch32:
                include_cls_token=True  -> [B, 50, 768]
                include_cls_token=False -> [B, 49, 768]
        """

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images = [self.load_image(path) for path in image_paths]

        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )

        pixel_values = inputs["pixel_values"].to(self.device_name)

        outputs = self.model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )

        image_tokens = outputs.last_hidden_state

        if not include_cls_token:
            image_tokens = image_tokens[:, 1:, :]

        if normalize:
            image_tokens = F.normalize(image_tokens, p=2, dim=-1)

        return image_tokens

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