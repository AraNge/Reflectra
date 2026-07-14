from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import get_nested, load_config
from src.datasets.paths import PROJECT_ROOT
from src.datasets.torch_datasets.reranker_dataset import load_reranker_dataset
from src.metrics.retrieval_metrics import sparse_retrieval_metrics
from src.models.reflectra_model import CHECKPOINT_DIR, ReflectraModel
from src.models.reranker import build_reranker


RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

def pairwise_ranknet_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Weighted pairwise RankNet loss for one query's candidate list.

    pred, target: [K]. Every pair (i, j) with a different target score
    contributes -log(sigmoid(sign(target_i - target_j) * (pred_i - pred_j))),
    weighted by |target_i - target_j| so a (score 10 vs score 0) mismatch
    counts far more than a (score 6 vs score 5) mismatch. Ties contribute 0.
    """

    diff_pred = pred.unsqueeze(0) - pred.unsqueeze(1)      # [K, K], [i, j] = pred_j - pred_i
    diff_target = target.unsqueeze(0) - target.unsqueeze(1)

    sign = torch.sign(diff_target)
    weight = diff_target.abs()
    mask = sign != 0

    if mask.sum() == 0:
        return pred.sum() * 0.0

    logits = sign[mask] * diff_pred[mask]
    weights = weight[mask]

    loss = F.softplus(-logits)

    return (loss * weights).sum() / weights.sum()


# --------------------------------------------------------------------------
# Embedding cache (bi-encoders + projection are frozen, so compute once)
# --------------------------------------------------------------------------

@torch.no_grad()
def precompute_embeddings(
    model: ReflectraModel,
    paths: list[str],
    kind: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    unique_paths = sorted(set(paths))
    embeddings: dict[str, torch.Tensor] = {}

    encode_fn = model.encode_image if kind == "image" else model.encode_audio

    for start in tqdm(range(0, len(unique_paths), batch_size), desc=f"Embedding {kind}"):
        batch_paths = unique_paths[start:start + batch_size]
        batch_embeds = encode_fn(batch_paths).to(device)

        for path, embed in zip(batch_paths, batch_embeds):
            embeddings[path] = embed.detach()

    return embeddings


# --------------------------------------------------------------------------
# Train/test split by exact query count (e.g. 700 train / 300 test)
# --------------------------------------------------------------------------

def split_records_by_count(
    records: list[dict[str, Any]],
    train_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    total_needed = train_size + test_size

    if len(shuffled) < total_needed:
        scale = len(shuffled) / total_needed
        scaled_train_size = max(1, int(round(train_size * scale)))
        scaled_test_size = max(1, len(shuffled) - scaled_train_size)

        print(
            f"[WARN] Only {len(shuffled)} queries available, but "
            f"train_size+test_size={total_needed} requested. "
            f"Scaling down to train={scaled_train_size}, test={scaled_test_size}."
        )
        train_size, test_size = scaled_train_size, scaled_test_size

    train_records = shuffled[:train_size]
    test_records = shuffled[train_size:train_size + test_size]

    return train_records, test_records


# --------------------------------------------------------------------------
# Automatic projection-checkpoint discovery
# --------------------------------------------------------------------------

def find_latest_projection_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(
        (p for p in checkpoint_dir.glob("*.pt") if "projection" in p.name.lower()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No projection checkpoint found in {checkpoint_dir} "
            "(expected a filename containing 'projection', e.g. "
            "'reflectra_projection_flickr30k_1000.pt'). "
            "Pass --projection-checkpoint explicitly."
        )

    return candidates[0]


def resolve_projection_checkpoint(
    value: Optional[str],
    checkpoint_dir: Path,
) -> Path:
    if value is None:
        found = find_latest_projection_checkpoint(checkpoint_dir)
        print(f"[INFO] No --projection-checkpoint given, auto-selected latest: {found}")
        return found

    candidate = Path(value)
    if candidate.exists():
        return candidate

    candidate_in_dir = checkpoint_dir / value
    if candidate_in_dir.exists():
        print(f"[INFO] Resolved projection checkpoint '{value}' -> {candidate_in_dir}")
        return candidate_in_dir

    raise FileNotFoundError(
        f"Projection checkpoint not found: {value} "
        f"(also checked {candidate_in_dir})"
    )


# --------------------------------------------------------------------------
# Dynamic reranker checkpoint naming (no hardcoded filename)
# --------------------------------------------------------------------------

def default_output_name(reranker_type: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"reflectra_reranker_{reranker_type}_{timestamp}.pt"


# --------------------------------------------------------------------------
# Train / eval loop
# --------------------------------------------------------------------------

def run_epoch(
    records: list[dict[str, Any]],
    reranker: torch.nn.Module,
    image_cache: dict[str, torch.Tensor],
    audio_cache: dict[str, torch.Tensor],
    optimizer: Optional[torch.optim.Optimizer],
    batch_queries: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    reranker.train(is_train)

    losses: list[float] = []
    per_query_ndcg: list[float] = []
    per_query_mrr: list[float] = []

    order = list(range(len(records)))
    if is_train:
        random.shuffle(order)

    for batch_start in range(0, len(order), batch_queries):
        batch_indices = order[batch_start:batch_start + batch_queries]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        batch_loss = 0.0
        counted = 0

        for idx in batch_indices:
            record = records[idx]

            image_embed = image_cache[record["image_path"]]
            candidate_embeds = torch.stack(
                [audio_cache[path] for path in record["audio_paths"]]
            )
            target = torch.tensor(
                record["scores"],
                device=image_embed.device,
                dtype=torch.float32,
            )

            query_expanded = image_embed.unsqueeze(0).expand(candidate_embeds.size(0), -1)
            pred = reranker(query_expanded, candidate_embeds)

            loss = pairwise_ranknet_loss(pred, target)

            if is_train:
                loss.backward()

            batch_loss += loss.item()
            counted += 1

            relevance_row = {
                i: score for i, score in enumerate(record["scores"]) if score > 0
            }
            if relevance_row:
                single_metrics = sparse_retrieval_metrics(
                    similarity=pred.detach().cpu().numpy().reshape(1, -1),
                    relevance=[relevance_row],
                    ks=(1, 5, 10),
                    threshold=0.0,
                    exponential_gain=True,
                )
                per_query_ndcg.append(single_metrics["ndcg@10"])
                per_query_mrr.append(single_metrics["mrr"])

        if is_train and counted > 0:
            optimizer.step()

        if counted > 0:
            losses.append(batch_loss / counted)

    metrics = {"loss": sum(losses) / len(losses) if losses else 0.0}

    if per_query_ndcg:
        metrics["ndcg@10"] = sum(per_query_ndcg) / len(per_query_ndcg)
        metrics["mrr"] = sum(per_query_mrr) / len(per_query_mrr)

    return metrics


def save_checkpoint(
    reranker: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "reranker_state_dict": reranker.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        output_path,
    )

    print(f"[INFO] Saved reranker checkpoint to: {output_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(parents=[config_parser])

    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(DEFAULT_BENCHMARK_DIR),
        help=(
            "Unpacked benchmark directory (containing image_audio_scores.jsonl, "
            "image_table.jsonl, audio_table.jsonl, images/, audio/). "
            "Run python -m src.datasets.downloaders.download_reflectra_benchmark first."
        ),
    )

    parser.add_argument("--clip-model-name", type=str, default=get_nested(config, "models", "clip", "openai/clip-vit-base-patch32"))
    parser.add_argument("--clap-model-name", type=str, default=get_nested(config, "models", "clap", "laion/clap-htsat-unfused"))

    parser.add_argument(
        "--projection-checkpoint",
        type=str,
        default=None,
        help=(
            "Path or filename (under checkpoints/) of a trained CLIP->CLAP "
            "projection checkpoint. If omitted, the most recently modified "
            "'*projection*.pt' file in checkpoints/ is used automatically."
        ),
    )
    parser.add_argument("--projection-type", type=str, default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--projection-hidden-dim", type=int, default=1024)

    parser.add_argument("--reranker-type", type=str, default=get_nested(config, "reranker", "type", "mlp"), choices=["mlp", "attention"])
    parser.add_argument("--reranker-hidden-dim", type=int, default=get_nested(config, "reranker", "hidden_dim", 512))
    parser.add_argument("--reranker-dropout", type=float, default=get_nested(config, "reranker", "dropout", 0.1))

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-queries", type=int, default=8, help="Number of image queries per optimizer step.")
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--train-size", type=int, default=700, help="Number of image queries used for training.")
    parser.add_argument("--test-size", type=int, default=300, help="Number of image queries held out for testing.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Reranker checkpoint filename under checkpoints/. If omitted, a "
            "timestamped name is generated automatically, e.g. "
            "reflectra_reranker_mlp_20260714_153000.pt"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[INFO] Device: {device}")

    if args.output_name is None:
        args.output_name = default_output_name(args.reranker_type)
    print(f"[INFO] Reranker checkpoint will be saved as: checkpoints/{args.output_name}")

    dataset = load_reranker_dataset(
        benchmark_path=args.benchmark,
        min_candidates=args.min_candidates,
    )
    print(f"[INFO] Loaded {len(dataset)} image queries with graded audio candidates.")

    if len(dataset) == 0:
        raise RuntimeError(
            "No usable queries found. Check --benchmark and --min-candidates, "
            "and make sure python -m src.datasets.downloaders.download_reflectra_benchmark "
            "has been run."
        )

    train_records, test_records = split_records_by_count(
        records=dataset.records,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"[INFO] Train queries: {len(train_records)} | Test queries: {len(test_records)}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    projection_checkpoint_path = resolve_projection_checkpoint(
        value=args.projection_checkpoint,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    model = ReflectraModel(
        clip_model_name=args.clip_model_name,
        clap_model_name=args.clap_model_name,
        projection_type=args.projection_type,
        projection_hidden_dim=args.projection_hidden_dim,
        freeze_clip=True,
        freeze_clap=True,
        normalize=True,
        device=str(device),
        projection_checkpoint=projection_checkpoint_path,
    )

    model.eval()  # bi-encoders + projection are frozen; only the reranker trains.

    all_image_paths = [record["image_path"] for record in dataset.records]
    all_audio_paths = [path for record in dataset.records for path in record["audio_paths"]]

    print("[INFO] Precomputing frozen bi-encoder embeddings (this happens once, not per epoch)...")
    image_cache = precompute_embeddings(model, all_image_paths, "image", args.encode_batch_size, device)
    audio_cache = precompute_embeddings(model, all_audio_paths, "audio", args.encode_batch_size, device)

    embed_dim = model.clap_dim

    reranker = build_reranker(
        reranker_type=args.reranker_type,
        embed_dim=embed_dim,
        hidden_dim=args.reranker_hidden_dim,
        dropout=args.reranker_dropout,
    ).to(device)

    print(f"[INFO] Reranker type: {args.reranker_type} | embed_dim={embed_dim} | hidden_dim={args.reranker_hidden_dim}")

    optimizer = torch.optim.AdamW(
        reranker.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_path = CHECKPOINT_DIR / args.output_name

    best_test_ndcg = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n[INFO] Epoch {epoch}/{args.epochs}")

        train_metrics = run_epoch(
            records=train_records,
            reranker=reranker,
            image_cache=image_cache,
            audio_cache=audio_cache,
            optimizer=optimizer,
            batch_queries=args.batch_queries,
        )
        epoch_metrics = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}

        if test_records:
            test_metrics = run_epoch(
                records=test_records,
                reranker=reranker,
                image_cache=image_cache,
                audio_cache=audio_cache,
                optimizer=None,
                batch_queries=args.batch_queries,
            )
            epoch_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

            test_ndcg = test_metrics.get("ndcg@10", 0.0)

            if test_ndcg >= best_test_ndcg:
                best_test_ndcg = test_ndcg
                save_checkpoint(
                    reranker=reranker,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=epoch_metrics,
                    args=args,
                    output_path=output_path,
                )
        else:
            save_checkpoint(
                reranker=reranker,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                args=args,
                output_path=output_path,
            )

        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))

    history_path = RESULTS_DIR / f"train_reranker_history_{Path(args.output_name).stem}.json"

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n[INFO] Training history saved to: {history_path}")
    print(f"[INFO] Best reranker checkpoint: {output_path}")
    print(
        "[INFO] Next step — compare bi-encoder vs bi-encoder+reranker with:\n"
        f"  python -m src.evaluation.evaluate_reflectra --checkpoint {projection_checkpoint_path.name} "
        "--output evaluation_results/reflectra_eval_baseline.json\n"
        f"  python -m src.evaluation.evaluate_reflectra --checkpoint {projection_checkpoint_path.name} "
        f"--use_reranker --reranker_checkpoint {output_path.name} "
        "--output evaluation_results/reflectra_eval_reranked.json"
    )


"""
Examples:

python -m src.training.train_reranker \
  --epochs 15 \
  --train-size 700 \
  --test-size 300

python -m src.training.train_reranker \
  --projection-checkpoint reflectra_projection_flickr30k_1000.pt \
  --reranker-type attention \
  --reranker-hidden-dim 256 \
  --train-size 700 \
  --test-size 300 \
  --epochs 20
"""

if __name__ == "__main__":
    main()
