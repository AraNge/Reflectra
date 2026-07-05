import torch
from torch import nn
import torch.nn.functional as F


class MLPProjection(nn.Module):
    """
    Nonlinear projection.

    Good for:
    - CLIP image embedding -> CLAP embedding space
    - stronger mapping when source and target spaces are different
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()

        self.normalize = normalize

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)

        if self.normalize:
            x = F.normalize(x, dim=-1)

        return x


class LinearProjection(nn.Module):
    """
    Simple linear projection.

    Good for:
    - baseline projection
    - testing whether CLIP and CLAP spaces can be aligned with minimal capacity
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        normalize: bool = True,
    ):
        super().__init__()

        self.normalize = normalize
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)

        if self.normalize:
            x = F.normalize(x, dim=-1)

        return x
    


class QFormerHead(nn.Module):
    """
    Lightweight Q-Former style projection head.

    Purpose:
        CLIP image embeddings -> CLAP embedding space

    Expected input:
        x shape: [batch_size, num_tokens, input_dim]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_query_tokens: int = 8,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()

        self.normalize = normalize
        self.num_query_tokens = num_query_tokens

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, hidden_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.qformer = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )

        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                CLIP token/patch embeddings.
                Shape: [batch_size, num_tokens, input_dim]

        Returns:
            projected:
                CLAP-compatible embedding.
                Shape: [batch_size, num_query_tokens, output_dim]
        """

        if x.ndim != 3:
            raise ValueError(
                f"QFormerHead expects x with shape [B, T, D], got {tuple(x.shape)}"
            )

        batch_size = x.size(0)

        memory = self.input_projection(x)

        queries = self.query_tokens.expand(batch_size, -1, -1)

        query_outputs = self.qformer(
            tgt=queries,
            memory=memory,
        )

        projected_queries = self.output_projection(query_outputs)

        if self.normalize:
            projected_queries = F.normalize(projected_queries, dim=-1)

        return projected_queries