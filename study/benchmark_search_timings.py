from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.datasets.paths import METADATA_DIR, PROJECT_ROOT
from src.datasets.selection.metadata import load_image_metadata
from src.main import add_search_args, config_arg_parser, load_model, search_image


DEFAULT_METADATA_PATH = METADATA_DIR / "flickr30k_metadata.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots" / "search_timing_benchmark"
STAGE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#138a22",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#17becf",
]


def build_parser() -> argparse.ArgumentParser:
    config_parser, config = config_arg_parser()
    parser = argparse.ArgumentParser(
        description=(
            "Download or reuse Flickr30k images, run Reflectra image search, "
            "and plot p50/p90/p95/p99 stage timings."
        ),
        parents=[config_parser],
    )
    parser.add_argument("--samples", "-n", type=int, required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--seed", type=int, default=11062026)
    parser.add_argument(
        "--percentiles",
        default="50,90,95,99",
        help="Comma-separated percentiles to plot, for example 50,90,95,99.",
    )
    add_search_args(parser, config)
    return parser


def parse_percentiles(value: str) -> list[float]:
    percentiles: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        percentile = float(item)
        if percentile < 0 or percentile > 100:
            raise argparse.ArgumentTypeError("Percentiles must be between 0 and 100.")
        percentiles.append(percentile)

    if not percentiles:
        raise argparse.ArgumentTypeError("At least one percentile is required.")
    return percentiles


def load_flickr_records(metadata_path: Path) -> list[dict[str, Any]]:
    return load_image_metadata(
        metadata_paths=[metadata_path],
        project_root=PROJECT_ROOT,
        require_image_exists=True,
    )


def ensure_flickr_records(
    *,
    samples: int,
    metadata_path: Path,
    skip_download: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if metadata_path.exists():
        records = load_flickr_records(metadata_path)

    if len(records) >= samples:
        return records

    if skip_download:
        raise RuntimeError(
            f"Only found {len(records)} Flickr30k records with local images, "
            f"but --samples requested {samples}."
        )

    from src.datasets.downloaders.download_flickr30k import download_flickr30k_samples

    print(f"Downloading Flickr30k samples up to requested count: {samples}")
    download_flickr30k_samples(samples)
    records = load_flickr_records(metadata_path)

    if len(records) < samples:
        raise RuntimeError(
            f"Only found {len(records)} Flickr30k records after download, "
            f"but --samples requested {samples}."
        )

    return records


def timing_map(timings: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in timings:
        stage = str(row["stage"])
        values[stage] = values.get(stage, 0.0) + float(row["seconds"])
    return values


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def write_raw_timings(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile_label(value: float) -> str:
    if value.is_integer():
        return f"p{int(value)}"
    return f"p{value:g}"


def summarize_timings(
    raw_rows: list[dict[str, Any]],
    percentiles: list[float],
) -> dict[str, Any]:
    stage_order: list[str] = []
    by_stage: dict[str, list[float]] = {}
    totals: list[float] = []

    for row in raw_rows:
        timings = timing_map(row["timings"])
        totals.append(sum(timings.values()))
        for stage in timings:
            if stage not in by_stage:
                by_stage[stage] = [0.0] * (len(totals) - 1)
                stage_order.append(stage)

        for stage in stage_order:
            by_stage.setdefault(stage, []).append(timings.get(stage, 0.0))

    percentile_rows: dict[str, dict[str, float]] = {}
    for percentile in percentiles:
        label = percentile_label(percentile)
        stage_values = {
            stage: float(np.percentile(by_stage[stage], percentile))
            for stage in stage_order
        }
        stage_values["total"] = float(np.percentile(totals, percentile))
        stage_values["stage_sum"] = float(sum(stage_values[stage] for stage in stage_order))
        percentile_rows[label] = stage_values

    return {
        "samples": len(raw_rows),
        "stages": stage_order,
        "percentiles": [percentile_label(p) for p in percentiles],
        "stage_percentiles": percentile_rows,
        "stage_means": {
            stage: float(np.mean(values))
            for stage, values in by_stage.items()
        },
        "total_mean": float(np.mean(totals)) if totals else 0.0,
    }


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stages = summary["stages"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["percentile", *stages, "stage_sum", "total"])
        for label in summary["percentiles"]:
            row = summary["stage_percentiles"][label]
            writer.writerow(
                [
                    label,
                    *[f"{row.get(stage, 0.0):.9f}" for stage in stages],
                    f"{row.get('stage_sum', 0.0):.9f}",
                    f"{row.get('total', 0.0):.9f}",
                ]
            )


def prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/reflectra_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_percentile_overview(summary: dict[str, Any], output_path: Path) -> None:
    plt = prepare_matplotlib()
    stages = summary["stages"]
    labels = summary["percentiles"]
    stage_percentiles = summary["stage_percentiles"]
    x_positions = np.arange(len(labels))
    bottoms = np.zeros(len(labels), dtype=float)
    max_total = max(stage_percentiles[label]["stage_sum"] for label in labels) or 1.0

    fig, ax = plt.subplots(figsize=(13, 7))
    for idx, stage in enumerate(stages):
        heights = np.array([stage_percentiles[label].get(stage, 0.0) for label in labels])
        ax.bar(
            x_positions,
            heights,
            bottom=bottoms,
            color=STAGE_COLORS[idx % len(STAGE_COLORS)],
            edgecolor="white",
            linewidth=0.8,
            label=stage,
        )
        for x_pos, bottom, height in zip(x_positions, bottoms, heights):
            if height >= max_total * 0.045:
                ax.text(
                    x_pos,
                    bottom + height / 2,
                    f"{height:.3f}s",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    fontweight="bold",
                )
        bottoms += heights

    for x_pos, label in zip(x_positions, labels):
        total = stage_percentiles[label]["stage_sum"]
        ax.text(
            x_pos,
            total + max_total * 0.03,
            f"Total: {total:.3f}s",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_title("Stage Time Percentiles: image_search", fontsize=24, pad=18)
    ax.set_ylabel("Seconds", fontsize=15)
    ax.set_xticks(x_positions, labels)
    ax.grid(axis="y", linestyle=(0, (4, 4)), color="#cfcfcf", linewidth=1)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max_total * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_single_percentile(summary: dict[str, Any], label: str, output_path: Path) -> None:
    plt = prepare_matplotlib()
    stages = summary["stages"]
    values = summary["stage_percentiles"][label]
    stage_sum = values["stage_sum"]
    max_total = stage_sum or 1.0

    fig, ax = plt.subplots(figsize=(12, 8))
    bottom = 0.0
    for idx, stage in enumerate(stages):
        height = values.get(stage, 0.0)
        ax.bar(
            [0],
            [height],
            bottom=[bottom],
            width=0.62,
            color=STAGE_COLORS[idx % len(STAGE_COLORS)],
            edgecolor="white",
            linewidth=0.8,
            label=f"{stage} (≈ {height:.3f}s)",
        )
        if height >= max_total * 0.045:
            ax.text(
                0,
                bottom + height / 2,
                f"{height:.3f}s",
                ha="center",
                va="center",
                color="white",
                fontsize=15,
                fontweight="bold",
            )
        else:
            ax.annotate(
                f"{height:.3f}s",
                xy=(0.31, bottom + height),
                xytext=(0.42, bottom + max_total * 0.04),
                arrowprops={"arrowstyle": "-", "color": STAGE_COLORS[idx % len(STAGE_COLORS)]},
                color=STAGE_COLORS[idx % len(STAGE_COLORS)],
                fontsize=13,
                fontweight="bold",
            )
        bottom += height

    ax.text(
        0,
        stage_sum + max_total * 0.025,
        f"Total: {stage_sum:.3f}s",
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_title(f"Stage Time: image_search {label}", fontsize=24, pad=18)
    ax.set_ylabel("Seconds", fontsize=15)
    ax.set_xticks([0], [label])
    ax.set_xlim(-0.55, 0.85)
    ax.set_ylim(0, max_total * 1.2)
    ax.grid(axis="y", linestyle=(0, (4, 4)), color="#cfcfcf", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    percentiles = parse_percentiles(args.percentiles)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    metadata_path = Path(args.metadata_path).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path

    args.timing_json = None
    args.timing_plot = None
    args.show_timing_plot = False
    args.timing_dir = str(output_dir)

    records = ensure_flickr_records(
        samples=args.samples,
        metadata_path=metadata_path,
        skip_download=args.skip_download,
    )
    random.Random(args.seed).shuffle(records)
    selected_records = records[: args.samples]

    print(f"Loaded {len(selected_records)} benchmark images.")
    print("Loading Reflectra model once for the full run...")
    model = load_model(args)

    raw_rows: list[dict[str, Any]] = []
    for index, record in enumerate(tqdm(selected_records, desc="Searching images"), start=1):
        result = search_image(args, record["image_path"], model=model)
        timings = result["timings"]
        raw_rows.append(
            {
                "index": index,
                "image_id": record["image_id"],
                "image_path": record["image_path"],
                "source_dataset": record["source_dataset"],
                "captions": record["captions"],
                "timings": timings,
                "stage_total_seconds": sum(row["seconds"] for row in timings),
                "result_count": len(result.get("results", [])),
                "warnings": result.get("warnings", []),
            }
        )

    summary = summarize_timings(raw_rows, percentiles)
    write_raw_timings(output_dir / "raw_timings.jsonl", raw_rows)
    write_json(output_dir / "summary.json", summary)
    write_summary_csv(output_dir / "summary.csv", summary)
    plot_percentile_overview(summary, output_dir / "stage_percentiles_stacked.png")
    for label in summary["percentiles"]:
        plot_single_percentile(summary, label, output_dir / f"stage_time_image_search_{label}.png")

    print(f"Wrote benchmark outputs to: {output_dir}")


if __name__ == "__main__":
    main()
