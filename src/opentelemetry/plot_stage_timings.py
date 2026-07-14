import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Reflectra stage timings written by src.opentelemetry.telemetry."
    )
    parser.add_argument("timings_json", type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timings_path = Path(args.timings_json)

    with timings_path.open("r", encoding="utf-8") as timings_file:
        payload = json.load(timings_file)

    timings = payload.get("timings", [])
    stages = [str(item["stage"]) for item in timings]
    seconds = [float(item["seconds"]) for item in timings]

    output = (
        Path(args.output)
        if args.output is not None
        else Path("plots") / f"{timings_path.stem}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(stages, seconds, color="#4C78A8")
    plt.ylabel("Seconds")
    plt.title(f"Stage Time: {payload.get('trace_name', timings_path.stem)}")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, seconds):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output, dpi=200)
    print(f"Saved timing plot to: {output}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
