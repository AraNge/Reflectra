import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.torch_datasets.projection_dataset import (
    DEFAULT_IMAGE_METADATA_PATHS,
    load_projection_records,
    create_projection_dataloaders,
)
from src.models.reflectra_model import ReflectraModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"



def symmetric_contrastive_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Contrastive loss for matching image-text pairs.

    image_embeds: [B, D]
    text_embeds:  [B, D]
    """

    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    logits = image_embeds @ text_embeds.T
    logits = logits / temperature

    labels = torch.arange(logits.size(0), device=logits.device)

    image_to_text_loss = F.cross_entropy(logits, labels)
    text_to_image_loss = F.cross_entropy(logits.T, labels)

    return (image_to_text_loss + text_to_image_loss) / 2


@torch.no_grad()
def evaluate_projection(
    model: ReflectraModel,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()

    losses = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        image_paths = batch["image_paths"]
        texts = batch["texts"]

        image_embeds = model.encode_image(image_paths)
        text_embeds = model.encode_text(texts, normalize=True)

        image_embeds = image_embeds.to(device)
        text_embeds = text_embeds.to(device)

        loss = symmetric_contrastive_loss(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            temperature=temperature,
        )

        losses.append(loss.item())

    return {
        "val_loss": sum(losses) / len(losses) if losses else 0.0,
    }


def train_one_epoch(
    model: ReflectraModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    grad_clip: float | None = None,
) -> Dict[str, float]:
    model.train()

    # Encoders stay frozen. Projection is trainable.
    model.clip.eval()
    model.clap.eval()
    model.image_projection.train()

    losses = []

    for batch in tqdm(dataloader, desc="Training"):
        image_paths = batch["image_paths"]
        texts = batch["texts"]

        optimizer.zero_grad(set_to_none=True)

        image_embeds = model.encode_image(image_paths)

        with torch.no_grad():
            text_embeds = model.encode_text(texts, normalize=True)

        image_embeds = image_embeds.to(device)
        text_embeds = text_embeds.to(device)

        loss = symmetric_contrastive_loss(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            temperature=temperature,
        )

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.image_projection.parameters(),
                grad_clip,
            )

        optimizer.step()

        losses.append(loss.item())

    return {
        "train_loss": sum(losses) / len(losses) if losses else 0.0,
    }


def save_checkpoint(
    model: ReflectraModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "projection_state_dict": model.image_projection.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "clip_dim": model.clip_dim,
        "clap_dim": model.clap_dim,
    }

    torch.save(checkpoint, output_path)

    print(f"[INFO] Saved checkpoint to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--clip-model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--clap-model-name", type=str, default="laion/clap-htsat-unfused")

    parser.add_argument(
        "--projection-type",
        type=str,
        default="mlp",
        choices=["linear", "mlp", "qformer"],
    )

    parser.add_argument("--projection-hidden-dim", type=int, default=1024)
    parser.add_argument("--projection-dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default=None)

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=1000)

    parser.add_argument(
        "--dataset-fractions",
        type=str,
        default=None,
        help='Example: "coco_karpathy=0.5,nlphuji/flickr30k=0.8,LiangJian24/EmoSet=1.0"',
    )

    parser.add_argument(
        "--dataset-counts",
        type=str,
        default=None,
        help='Example: "coco_karpathy=50000,nlphuji/flickr30k=10000,LiangJian24/EmoSet=5000"',
    )

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument(
        "--output-name",
        type=str,
        default="reflectra_projection.pt",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"[INFO] Device: {device}")

    train_records, val_records = load_projection_records(
        metadata_paths=DEFAULT_IMAGE_METADATA_PATHS,
        project_root=PROJECT_ROOT,
        train_split=args.train_split,
        val_split=args.val_split,
        dataset_fractions=args.dataset_fractions,
        dataset_counts=args.dataset_counts,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        require_image_exists=True,
    )

    print(f"[INFO] Train records: {len(train_records)}")
    print(f"[INFO] Val records: {len(val_records)}")

    train_loader, val_loader = create_projection_dataloaders(
        train_records=train_records,
        val_records=val_records,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last_train=True,
    )

    model = ReflectraModel(
        clip_model_name=args.clip_model_name,
        clap_model_name=args.clap_model_name,
        projection_type=args.projection_type,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dropout=args.projection_dropout,
        freeze_clip=True,
        freeze_clap=True,
        normalize=True,
        device=str(device),
    )

    model.to(device)
    model.freeze_encoders()
    model.unfreeze_projection()

    print("[INFO] Trainable parameters:")
    for name in model.trainable_parameters():
        print(f"  - {name}")

    optimizer = torch.optim.AdamW(
        model.image_projection.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n[INFO] Epoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=args.temperature,
            grad_clip=args.grad_clip,
        )

        epoch_metrics = {
            "epoch": epoch,
            **train_metrics,
        }

        if val_loader is not None:
            val_metrics = evaluate_projection(
                model=model,
                dataloader=val_loader,
                device=device,
                temperature=args.temperature
            )

            epoch_metrics.update(val_metrics)

            val_loss = val_metrics["val_loss"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                best_path = CHECKPOINT_DIR / args.output_name

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=epoch_metrics,
                    args=args,
                    output_path=best_path,
                )

        else:
            checkpoint_path = CHECKPOINT_DIR / args.output_name

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                args=args,
                output_path=checkpoint_path,
            )

        history.append(epoch_metrics)

        print(json.dumps(epoch_metrics, indent=2))

    history_path = RESULTS_DIR / "train_projection_history.json"

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"[INFO] Training history saved to: {history_path}")


"""
Examples:

python -m src.training.train_projection \
  --projection-type mlp \
  --epochs 5 \
  --batch-size 32 \
  --train-split train

python -m src.training.train_projection \
  --projection-type linear \
  --epochs 5 \
  --batch-size 64 \
  --train-split train

python -m src.training.train_projection \
  --projection-type mlp \
  --dataset-counts "coco_karpathy=50000,LiangJian24/EmoSet=5000" \
  --max-train-samples 55000 \
  --epochs 10 \
  --batch-size 32

python -m src.training.train_projection \
  --projection-type mlp \
  --train-split train \
  --val-split validation \
  --max-val-samples 2000 \
  --epochs 10
"""

if __name__ == "__main__":
    main()
