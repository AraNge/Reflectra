from __future__ import annotations

import os
import importlib
from typing import Any

import torch

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

_import_module = importlib.import_module


def _protobuf_safe_import(name: str, package: str | None = None):
    if name == "google._upb._message":
        raise ImportError(name)
    return _import_module(name, package)


importlib.import_module = _protobuf_safe_import
try:
    from qdrant_client import QdrantClient
finally:
    importlib.import_module = _import_module

from src.models.reflectra_model import ReflectraModel
from src.models.reranker import score_candidates
from src.opentelemetry.telemetry import StageTimer
from src.vector_db.qdrant_store import search_vector


def _point_vector_to_tensor(vector: Any, device: torch.device) -> torch.Tensor | None:
    if vector is None:
        return None

    if isinstance(vector, dict):
        if not vector:
            return None
        vector = next(iter(vector.values()))

    return torch.tensor(vector, dtype=torch.float32, device=device)


def search_image_with_rerank(
    client: QdrantClient,
    collection_name: str,
    model: ReflectraModel,
    reranker: torch.nn.Module,
    image_path: str,
    candidate_k: int = 20,
    final_k: int = 10,
    timing_plot_path: str | None = None,
    timing_json_path: str | None = None,
    show_timing_plot: bool = False,
    return_timings: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Two-stage image-to-song search:

        image
          -> CLIP image encoder -> projection -> CLAP-space query vector
          -> Qdrant ANN search over CLAP audio embeddings (candidate_k)
          -> cross-encoder reranker rescoring
          -> final ranked songs (final_k)

    Returns a list of dicts sorted by reranker score, each with:
        - payload: the original Qdrant point payload (audio_path, filename, ...)
        - bi_encoder_score: cosine similarity from Qdrant
        - rerank_score: cross-encoder relevance logit
    """

    timer = StageTimer("image_search_with_rerank")

    with timer.stage("setup"):
        model.eval()
        reranker.eval()

    with timer.stage("encode_query"):
        with torch.no_grad():
            query_embed = model.encode_image([image_path])[0]

    with timer.stage("check_db"):
        candidates = search_vector(
            client=client,
            collection_name=collection_name,
            query_vector=query_embed.cpu().numpy().tolist(),
            limit=candidate_k,
            with_vectors=True,
        )

    if not candidates:
        if timing_json_path is not None:
            timer.save_json(timing_json_path)
        if timing_plot_path is not None:
            timer.save_plot(timing_plot_path, show=show_timing_plot)
        if return_timings:
            return {"results": [], "timings": timer.as_dicts()}
        return []

    with timer.stage("prepare_candidates"):
        stored_vectors = [
            _point_vector_to_tensor(getattr(point, "vector", None), query_embed.device)
            for point in candidates
        ]
        can_use_stored_vectors = all(vector is not None for vector in stored_vectors)

    with timer.stage("encode_candidates"):
        with torch.no_grad():
            if can_use_stored_vectors:
                candidate_embeds = torch.stack(
                    [vector for vector in stored_vectors if vector is not None]
                )
            else:
                candidate_audio_paths = [
                    point.payload["audio_path"]
                    for point in candidates
                    if point.payload is not None and "audio_path" in point.payload
                ]
                if len(candidate_audio_paths) != len(candidates):
                    raise RuntimeError(
                        "Qdrant points did not include vectors or audio_path payloads. "
                        "Reranking needs stored Qdrant vectors or local audio paths."
                    )
                candidate_embeds = model.encode_audio(candidate_audio_paths).to(query_embed.device)

    with timer.stage("reranker"):
        rerank_scores = score_candidates(
            reranker=reranker,
            query_embed=query_embed,
            candidate_embeds=candidate_embeds,
        ).cpu().tolist()

    with timer.stage("format_results"):
        results = [
            {
                "payload": point.payload,
                "bi_encoder_score": float(point.score),
                "rerank_score": float(rerank_score),
            }
            for point, rerank_score in zip(candidates, rerank_scores)
        ]

        results.sort(key=lambda item: item["rerank_score"], reverse=True)
        results = results[:final_k]

    if timing_json_path is not None:
        timer.save_json(timing_json_path)
    if timing_plot_path is not None:
        timer.save_plot(timing_plot_path, show=show_timing_plot)

    if return_timings:
        return {
            "results": results,
            "timings": timer.as_dicts(),
        }

    return results
