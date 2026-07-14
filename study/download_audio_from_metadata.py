import argparse
import json
from pathlib import Path
from typing import Any

from study.audio_parts import download_audio_from_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a playable 15-second audio clip from returned Reflectra metadata."
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Metadata JSON string, or path to a JSON file containing one result/payload.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/study_downloaded_audio",
        help="Directory to save the downloaded audio file.",
    )
    return parser.parse_args()


def load_metadata(value: str) -> dict[str, Any]:
    path = Path(value).expanduser()
    if path.exists():
        with path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
            return normalize_metadata(metadata)

    return normalize_metadata(json.loads(value))


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if "results" in metadata and isinstance(metadata["results"], list):
        if not metadata["results"]:
            raise ValueError("Search result JSON has an empty results list.")
        first_result = metadata["results"][0]
        if not isinstance(first_result, dict):
            raise ValueError("First search result must be a JSON object.")
        return first_result

    return metadata


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    audio_path = download_audio_from_metadata(
        metadata=metadata,
        output_dir=Path(args.output_dir).expanduser(),
    )
    print(json.dumps({"audio_path": str(audio_path)}, indent=2))


if __name__ == "__main__":
    main()
