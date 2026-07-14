from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn
import torch.nn.functional as F


RerankerType = Literal["mlp", "attention"]


class InteractionMLPReranker(nn.Module):
    """
    Cross-encoder-style reranker over bi-encoder embeddings.

    The CLIP/CLAP bi-encoders score every pair with a single dot product
    computed independently per side (fast, but limited: the two embeddings
    never actually "see" each other). This module looks at both embeddings
    jointly and learns nonlinear interaction features on top of them.

    It is only meant to run over a small top-K candidate set returned by the
    bi-encoder + Qdrant stage, not over the full corpus (see
    src/vector_db/rerank_search.py and rerank_topk_similarity below).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # concat(image, audio) + elementwise product + abs diff + cosine scalar
        input_dim = embed_dim * 4 + 1

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        image_embeds: torch.Tensor,
        audio_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        image_embeds: [N, D] (broadcast to one row per candidate)
        audio_embeds: [N, D]

        Returns:
            relevance logits, shape [N]. Higher = more relevant. Not a
            probability; compare candidates within the same query only.
        """

        image_n = F.normalize(image_embeds, dim=-1)
        audio_n = F.normalize(audio_embeds, dim=-1)

        product = image_n * audio_n
        diff = (image_n - audio_n).abs()
        cosine = (image_n * audio_n).sum(dim=-1, keepdim=True)

        features = torch.cat([image_n, audio_n, product, diff, cosine], dim=-1)

        return self.net(features).squeeze(-1)


class AttentionReranker(nn.Module):
    """
    Small cross-attention reranker.

    Treats the projected image embedding and the CLAP audio embedding as a
    2-token sequence and lets a couple of self-attention layers exchange
    information between them before scoring. More expressive than
    InteractionMLPReranker, at the cost of more parameters — use it once you
    have enough LLM-graded pairs (a few thousand image/audio scores) to avoid
    overfitting.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.input_projection = nn.Linear(embed_dim, hidden_dim)

        # learned marker so the model can tell the image token from the audio token
        self.type_embeddings = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        image_embeds: torch.Tensor,
        audio_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        image_embeds: [N, D]
        audio_embeds: [N, D]

        Returns: relevance logits, shape [N]
        """

        image_n = F.normalize(image_embeds, dim=-1)
        audio_n = F.normalize(audio_embeds, dim=-1)

        image_tok = self.input_projection(image_n) + self.type_embeddings[0]
        audio_tok = self.input_projection(audio_n) + self.type_embeddings[1]

        tokens = torch.stack([image_tok, audio_tok], dim=1)  # [N, 2, H]
        encoded = self.encoder(tokens)  # [N, 2, H]

        pooled = encoded.reshape(encoded.size(0), -1)  # [N, 2*H]

        return self.head(pooled).squeeze(-1)


def build_reranker(
    reranker_type: RerankerType = "mlp",
    embed_dim: int = 512,
    hidden_dim: int = 512,
    dropout: float = 0.1,
    num_layers: int = 2,
    num_heads: int = 8,
) -> nn.Module:
    if reranker_type == "mlp":
        return InteractionMLPReranker(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    if reranker_type == "attention":
        return AttentionReranker(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    raise ValueError(f"Unsupported reranker_type: {reranker_type}")


def score_candidates(
    reranker: nn.Module,
    query_embed: torch.Tensor,
    candidate_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    Score a single query embedding against many candidate embeddings.

    query_embed: [D]
    candidate_embeds: [K, D]

    Returns: [K] relevance logits, aligned with candidate_embeds rows.
    """

    if candidate_embeds.numel() == 0:
        return torch.empty(0, device=query_embed.device)

    query_expanded = query_embed.unsqueeze(0).expand(candidate_embeds.size(0), -1)

    return reranker(query_expanded, candidate_embeds)


def _min_max_scale(values: torch.Tensor) -> torch.Tensor:
    value_min = values.min()
    value_max = values.max()

    if (value_max - value_min).abs() < 1e-8:
        return torch.zeros_like(values)

    return (values - value_min) / (value_max - value_min)


def rerank_topk_similarity(
    similarity: torch.Tensor,
    query_embeds: torch.Tensor,
    candidate_embeds: torch.Tensor,
    reranker: nn.Module,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Two-stage rescoring: shared by ReflectraModel.image_audio_similarity and
    the evaluation scripts, so there is exactly one implementation of "rerank
    the bi-encoder's top_k candidates per query".

    Args:
        similarity:
            [Q, C] bi-encoder cosine similarity (query x candidate).
        query_embeds:
            [Q, D] query-side embeddings (e.g. projected image embeddings).
        candidate_embeds:
            [C, D] candidate-side embeddings (e.g. CLAP audio embeddings).
        reranker:
            A trained cross-encoder module (InteractionMLPReranker or
            AttentionReranker) taking (query, candidate) pairs.
        top_k:
            Only the top_k bi-encoder candidates per query are rescored;
            ranking beyond top_k is left untouched. None reranks every
            candidate (fine for small candidate sets, expensive for large
            corpora — always pass a top_k when scoring a full corpus).

    Returns:
        A new similarity tensor, same shape as `similarity`, where the top_k
        block per row has been replaced by (scaled) reranker scores placed
        above every candidate outside top_k.
    """

    reranker.eval()
    reranked = similarity.clone()

    num_candidates = similarity.size(1)
    effective_top_k = num_candidates if top_k is None else min(top_k, num_candidates)

    with torch.no_grad():
        for query_idx in range(similarity.size(0)):
            row = similarity[query_idx]
            _, top_indices = torch.topk(row, effective_top_k)

            rerank_scores = score_candidates(
                reranker=reranker,
                query_embed=query_embeds[query_idx],
                candidate_embeds=candidate_embeds[top_indices],
            )

            scaled = _min_max_scale(rerank_scores)
            floor = reranked[query_idx].min() - 1.0
            reranked[query_idx, top_indices] = floor + 1.0 + scaled

    return reranked
