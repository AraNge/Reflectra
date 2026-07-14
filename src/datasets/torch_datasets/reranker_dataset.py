from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.utils.benchmark_tables import (
    load_reflectra_score_rows,
    referenced_reflectra_media_ids,
    resolve_reflectra_benchmark_paths,
)
from src.utils.media_tables import load_unpacked_media_index


class RerankerListwiseDataset(Dataset):
    """
    One item = one image query together with every LLM-graded candidate
    audio for it (0..10 relevance, from the Reflectra benchmark).

    Backed by the unpacked image_audio_scores.jsonl / image_table.jsonl /
    audio_table.jsonl produced by
    src.datasets.downloaders.download_reflectra_benchmark — the same files
    used by src.evaluation.evaluate_reflectra.
    """

    def __init__(
        self,
        score_rows: list[dict[str, Any]],
        images_by_id: dict[str, dict[str, Any]],
        audio_by_id: dict[str, dict[str, Any]],
        min_candidates: int = 2,
    ):
        self.records: list[dict[str, Any]] = []

        for row in score_rows:
            image_id = row["image_id"]

            if image_id not in images_by_id:
                continue

            audio_paths = []
            scores = []

            for audio_id, score in zip(row["audio_ids"], row["scores"]):
                if audio_id not in audio_by_id:
                    continue

                audio_paths.append(audio_by_id[audio_id]["audio_path"])
                scores.append(float(score))

            if len(audio_paths) < min_candidates:
                continue

            self.records.append(
                {
                    "image_id": image_id,
                    "image_path": images_by_id[image_id]["image_path"],
                    "audio_paths": audio_paths,
                    "scores": scores,
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


def collate_reranker_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Every query has a different number of candidates, so there is nothing
    # useful to pad/stack here. The training loop iterates over the queries
    # in the batch one at a time (see src/training/train_reranker.py).
    return batch


def load_reranker_dataset(
    benchmark_path: str | Path,
    min_candidates: int = 2,
) -> RerankerListwiseDataset:
    """
    benchmark_path: either the unpacked benchmark directory (containing
    image_audio_scores.jsonl, image_table.jsonl, audio_table.jsonl, images/,
    audio/) or the image_audio_scores.jsonl file directly.
    """

    score_path, benchmark_dir = resolve_reflectra_benchmark_paths(Path(benchmark_path))
    score_rows = load_reflectra_score_rows(score_path)

    required_image_ids, required_audio_ids = referenced_reflectra_media_ids(score_rows)

    images_by_id = load_unpacked_media_index(
        path=benchmark_dir / "image_table.jsonl",
        id_column="image_id",
        path_column="image_path",
        dataset_dir=benchmark_dir,
        required_ids=required_image_ids,
    )
    audio_by_id = load_unpacked_media_index(
        path=benchmark_dir / "audio_table.jsonl",
        id_column="audio_id",
        path_column="audio_path",
        dataset_dir=benchmark_dir,
        required_ids=required_audio_ids,
    )

    return RerankerListwiseDataset(
        score_rows=score_rows,
        images_by_id=images_by_id,
        audio_by_id=audio_by_id,
        min_candidates=min_candidates,
    )
