from pathlib import Path
from typing import Literal, Optional

import torch
from torch import nn
import torch.nn.functional as F

from src.models.clip_encoder import PretrainedCLIPEncoder
from src.models.clap_encoder import PretrainedCLAPEncoder
from src.models.projection_head import MLPProjection, LinearProjection
from src.models.reranker import RerankerType, build_reranker, rerank_topk_similarity


ProjectionType = Literal["mlp", "linear"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


class ReflectraModel(nn.Module):
    """
    Full Reflectra model.

    Components:
    - CLIP image encoder
    - projection head: CLIP space -> CLAP space
    - CLAP text encoder
    - CLAP audio encoder
    - (optional) cross-encoder reranker

    Main goal:
        image -> CLIP image embedding -> projection -> CLAP-compatible embedding

    Then the projected image embedding can be searched against
    Qdrant audio embeddings produced by CLAP audio encoder. If use_reranker
    is enabled, the top candidates from that bi-encoder search are rescored
    by a cross-encoder reranker before being returned — see
    image_audio_similarity below and src/vector_db/rerank_search.py for the
    Qdrant-backed version of the same pipeline.
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        clap_model_name: str = "laion/clap-htsat-unfused",
        projection_type: ProjectionType = "mlp",
        projection_hidden_dim: int = 1024,
        projection_dropout: float = 0.1,
        freeze_clip: bool = True,
        freeze_clap: bool = True,
        normalize: bool = True,
        device: Optional[str] = None,
        projection_checkpoint: str | Path | None = None,
        use_reranker: bool = False,
        reranker_type: RerankerType = "mlp",
        reranker_hidden_dim: int = 512,
        reranker_dropout: float = 0.1,
        reranker_checkpoint: str | Path | None = None,
        reranker_top_k: int = 20,
    ):
        super().__init__()

        self.normalize = normalize

        self.clip = PretrainedCLIPEncoder(
            model_name=clip_model_name,
            freeze=freeze_clip,
            device=device,
        )

        self.clap = PretrainedCLAPEncoder(
            model_name=clap_model_name,
            freeze=freeze_clap,
            device=device,
        )

        self.clip_dim = self._get_clip_embedding_dim()
        self.clap_dim = self._get_clap_embedding_dim()

        if projection_type == "mlp":
            self.image_projection = MLPProjection(
                input_dim=self.clip_dim,
                output_dim=self.clap_dim,
                hidden_dim=projection_hidden_dim,
                dropout=projection_dropout,
                normalize=normalize,
            )
        elif projection_type == "linear":
            self.image_projection = LinearProjection(
                input_dim=self.clip_dim,
                output_dim=self.clap_dim,
                normalize=normalize,
            )
        else:
            raise ValueError(f"Unsupported projection_type: {projection_type}")

        # --- Reranker (optional second stage) ---
        self.use_reranker = use_reranker
        self.reranker_top_k = reranker_top_k
        self.reranker: Optional[nn.Module] = None

        if use_reranker:
            self.reranker = build_reranker(
                reranker_type=reranker_type,
                embed_dim=self.clap_dim,
                hidden_dim=reranker_hidden_dim,
                dropout=reranker_dropout,
            )

        self.to(self.device)

        if projection_checkpoint is not None:
            self.load_projection_checkpoint(projection_checkpoint)

        if use_reranker and reranker_checkpoint is not None:
            self.load_reranker_checkpoint(reranker_checkpoint)
        elif use_reranker and reranker_checkpoint is None:
            print(
                "[WARN] use_reranker=True but no reranker_checkpoint given — "
                "the reranker is randomly initialized and will not improve ranking "
                "until trained (see src.training.train_reranker)."
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    

    def _get_clip_embedding_dim(self) -> int:
        """
        Get CLIP image/text embedding dimension from the loaded CLIP model.
        """

        config = self.clip.model.config

        if hasattr(config, "projection_dim"):
            return int(config.projection_dim)

        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)

        raise ValueError(
            "Could not infer CLIP embedding dimension from model config."
        )

    def _get_clap_embedding_dim(self) -> int:
        """
        Get CLAP audio/text embedding dimension from the loaded CLAP model.
        """

        config = self.clap.model.config

        if hasattr(config, "projection_dim"):
            return int(config.projection_dim)

        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)

        raise ValueError(
            "Could not infer CLAP embedding dimension from model config."
        )

    def encode_image_clip(
        self,
        image_paths: list[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode images with CLIP only.
        Output shape: [batch, clip_dim]
        """

        return self.clip.encode_image(
            image_paths=image_paths,
            normalize=normalize,
        )

    def encode_image(
        self,
        image_paths: list[str],
    ) -> torch.Tensor:
        """
        Encode image into CLAP-compatible embedding space.

        image_paths
            ↓
        CLIP image encoder
            ↓
        projection head
            ↓
        projected image embedding in CLAP space
        """

        clip_image_embeds = self.encode_image_clip(
            image_paths=image_paths,
            normalize=True,
        )

        projected_image_embeds = self.image_projection(clip_image_embeds)

        if self.normalize:
            projected_image_embeds = F.normalize(projected_image_embeds, dim=-1)

        return projected_image_embeds

    def encode_text(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode text into CLAP embedding space.
        """

        return self.clap.encode_text(
            texts=texts,
            normalize=normalize,
        )

    def encode_audio(
        self,
        audio_paths: list[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode audio into CLAP embedding space.
        """

        return self.clap.encode_audio(
            audio_paths=audio_paths,
            normalize=normalize,
        )

    def image_text_similarity(
        self,
        image_paths: list[str],
        texts: list[str],
    ) -> torch.Tensor:
        """
        Similarity between projected image embeddings and CLAP text embeddings.
        """

        image_embeds = self.encode_image(image_paths)
        text_embeds = self.encode_text(texts, normalize=True)

        return image_embeds @ text_embeds.T

    def image_audio_similarity(
        self,
        image_paths: list[str],
        audio_paths: list[str],
        use_reranker: Optional[bool] = None,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Similarity between projected image embeddings and CLAP audio embeddings.

        This is the main two-stage retrieval entrypoint:
            1) bi-encoder similarity (CLIP-projected image vs CLAP audio)
            2) if a reranker is enabled, the top_k candidates per image are
               rescored by the cross-encoder reranker.

        Args:
            use_reranker:
                Overrides self.use_reranker for this call. Pass False to force
                plain bi-encoder scoring even if the model was built with a
                reranker. Defaults to whatever the model was configured with.
            top_k:
                How many top bi-encoder candidates per image get rescored.
                Defaults to self.reranker_top_k. Ignored if reranking is off.
        """

        image_embeds = self.encode_image(image_paths)
        audio_embeds = self.encode_audio(audio_paths, normalize=True)

        similarity = image_embeds @ audio_embeds.T

        should_rerank = self.use_reranker if use_reranker is None else use_reranker

        if should_rerank and self.reranker is not None:
            similarity = rerank_topk_similarity(
                similarity=similarity,
                query_embeds=image_embeds,
                candidate_embeds=audio_embeds,
                reranker=self.reranker,
                top_k=top_k or self.reranker_top_k,
            )

        return similarity

    def text_audio_similarity(
        self,
        texts: list[str],
        audio_paths: list[str],
    ) -> torch.Tensor:
        """
        Native CLAP text-audio similarity.
        """

        text_embeds = self.encode_text(texts, normalize=True)
        audio_embeds = self.encode_audio(audio_paths, normalize=True)

        return text_embeds @ audio_embeds.T

    def forward(
        self,
        image_paths: Optional[list[str]] = None,
        texts: Optional[list[str]] = None,
        audio_paths: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Flexible forward.

        Example outputs:
        - image_embeds
        - text_embeds
        - audio_embeds
        - image_text_logits
        - image_audio_logits (rerank-aware if use_reranker=True)
        - text_audio_logits
        """

        output = {}

        if image_paths is not None:
            output["image_embeds"] = self.encode_image(image_paths)

        if texts is not None:
            output["text_embeds"] = self.encode_text(texts)

        if audio_paths is not None:
            output["audio_embeds"] = self.encode_audio(audio_paths)

        if "image_embeds" in output and "text_embeds" in output:
            output["image_text_logits"] = (
                output["image_embeds"] @ output["text_embeds"].T
            )

        if "image_embeds" in output and "audio_embeds" in output:
            image_audio_logits = output["image_embeds"] @ output["audio_embeds"].T

            if self.use_reranker and self.reranker is not None:
                image_audio_logits = rerank_topk_similarity(
                    similarity=image_audio_logits,
                    query_embeds=output["image_embeds"],
                    candidate_embeds=output["audio_embeds"],
                    reranker=self.reranker,
                    top_k=self.reranker_top_k,
                )

            output["image_audio_logits"] = image_audio_logits

        if "text_embeds" in output and "audio_embeds" in output:
            output["text_audio_logits"] = (
                output["text_embeds"] @ output["audio_embeds"].T
            )

        return output

    def freeze_encoders(self) -> None:
        """
        Freeze CLIP and CLAP encoders.
        Projection remains trainable.
        """

        self.clip.freeze()
        self.clap.freeze()

    def unfreeze_projection(self) -> None:
        for param in self.image_projection.parameters():
            param.requires_grad = True

    def resolve_checkpoint_path(self, checkpoint_path: str | Path) -> Path:
        path = Path(checkpoint_path).expanduser()
        if path.exists():
            return path.resolve()

        candidate = CHECKPOINT_DIR / path
        if candidate.exists():
            return candidate.resolve()

        return path

    def load_projection_checkpoint(
        self,
        checkpoint_path: str | Path,
        strict: bool = True,
    ) -> None:
        path = self.resolve_checkpoint_path(checkpoint_path)
        checkpoint = torch.load(path, map_location=self.device)
        load_result = self.image_projection.load_state_dict(
            checkpoint["projection_state_dict"],
            strict=strict,
        )
        if not strict:
            print(f"Projection checkpoint load result: {load_result}")
        print(f"Loaded projection checkpoint: {path}")

    def load_reranker_checkpoint(
        self,
        checkpoint_path: str | Path,
        strict: bool = True,
    ) -> None:
        if self.reranker is None:
            raise RuntimeError(
                "Model was built with use_reranker=False — there is no reranker "
                "to load weights into."
            )

        path = self.resolve_checkpoint_path(checkpoint_path)
        checkpoint = torch.load(path, map_location=self.device)
        load_result = self.reranker.load_state_dict(
            checkpoint["reranker_state_dict"],
            strict=strict,
        )
        if not strict:
            print(f"Reranker checkpoint load result: {load_result}")
        print(f"Loaded reranker checkpoint: {path}")

    def trainable_parameters(self):
        return [
            name
            for name, param in self.named_parameters()
            if param.requires_grad
        ]
