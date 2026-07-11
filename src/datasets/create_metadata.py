from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from src.config import get_nested, load_config
from src.datasets.paths import METADATA_DIR, PROJECT_ROOT
from src.utils.audio import audio_payload_for_llm
from src.utils.hashing import make_global_media_id
from src.utils.json import append_jsonl, read_jsonl, write_jsonl
from src.utils.openai_client import create_openai_client


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
}
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
}

IMAGE_PROMPT = """
Return only minified JSON. No markdown. No explanation.
Create 3 concise image-to-music retrieval captions for this image.
Focus on scene, mood, atmosphere, emotion, energy, style, and aesthetics.
Shape: {"captions":["caption 1","caption 2","caption 3"]}
"""

AUDIO_PROMPT = """
Return only minified JSON. No markdown. No explanation.
Create 3 concise image-to-music retrieval captions for this audio.
Focus on mood, atmosphere, emotion, energy, genre/style, instrumentation, texture.
Shape: {"captions":["caption 1","caption 2","caption 3"]}
"""


def iter_media_files(
    paths: list[str],
    extensions: set[str],
) -> list[Path]:
    files: list[Path] = []
    missing_paths: list[Path] = []
    unsupported_files: list[Path] = []

    for value in paths:
        path = Path(value).expanduser()

        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path.resolve())
            continue

        if path.is_file():
            unsupported_files.append(path)
            continue

        if path.is_dir():
            files.extend(
                file.resolve()
                for file in path.rglob("*")
                if file.is_file() and file.suffix.lower() in extensions
            )
            continue

        missing_paths.append(path)

    if missing_paths:
        raise FileNotFoundError(
            "Media input path(s) not found: "
            + ", ".join(str(path) for path in missing_paths)
        )

    if unsupported_files and not files:
        raise ValueError(
            "No supported media files found. Unsupported file input(s): "
            + ", ".join(str(path) for path in unsupported_files)
        )

    return sorted(set(files))


def data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def parse_captions(text: str) -> list[str]:
    text = text.strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            value = {"captions": [line for line in text.splitlines() if line.strip()]}
        else:
            value = json.loads(text[start:end + 1])

    if isinstance(value, dict):
        captions = value.get("captions", [])
    elif isinstance(value, list):
        captions = value
    else:
        captions = []

    cleaned = []
    seen = set()
    for item in captions:
        caption = str(item).strip()
        if caption and caption not in seen:
            seen.add(caption)
            cleaned.append(caption)

    if not cleaned:
        raise ValueError(f"No captions found in model response: {text!r}")

    return cleaned


def response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    choices = getattr(response, "choices", None)
    if choices:
        message = choices[0].message
        content = getattr(message, "content", "")
        return str(content)

    return str(response)


def caption_image(
    client: OpenAI,
    path: Path,
    model: str,
    max_attempts: int,
    max_output_tokens: int,
) -> list[str]:
    content = [
        {"type": "input_text", "text": IMAGE_PROMPT},
        {"type": "input_image", "image_url": data_url(path)},
    ]
    return caption_media(client, content, model, max_attempts, max_output_tokens)


def caption_audio(
    client: OpenAI,
    path: Path,
    model: str,
    max_attempts: int,
    clip_seconds: float | None,
    max_output_tokens: int,
) -> list[str]:
    audio_bytes, audio_format = audio_payload_for_llm(
        path,
        clip_seconds=clip_seconds,
    )
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    content = [
        {"type": "input_text", "text": AUDIO_PROMPT},
        {
            "type": "input_audio",
            "input_audio": {
                "data": encoded,
                "format": audio_format,
            },
        },
    ]
    return caption_media(client, content, model, max_attempts, max_output_tokens)


def caption_media(
    client: OpenAI,
    content: list[dict[str, Any]],
    model: str,
    max_attempts: int,
    max_output_tokens: int,
) -> list[str]:
    for attempt in range(max_attempts):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                max_output_tokens=max_output_tokens,
            )
            return parse_captions(response_text(response))
        except Exception:
            if attempt + 1 >= max_attempts:
                raise
            time.sleep(2**attempt)

    raise RuntimeError("Caption generation failed unexpectedly.")


def existing_media_ids(path: Path, id_field: str) -> set[str]:
    return {
        str(row[id_field])
        for row in read_jsonl(path, missing_ok=True)
        if row.get(id_field)
    }


def make_image_record(
    path: Path,
    captions: list[str],
    source_dataset: str,
    split: str,
) -> dict[str, Any]:
    image_id = make_global_media_id(
        media_type="image",
        source_dataset=source_dataset,
        source_id=path.stem,
        media_path=path,
        project_root=PROJECT_ROOT,
    )
    return {
        "image_id": image_id,
        "image_path": str(path),
        "captions": captions,
        "split": split,
        "source_dataset": source_dataset,
        "source_image_id": path.stem,
    }


def make_audio_record(
    path: Path,
    captions: list[str],
    source_dataset: str,
    split: str,
) -> dict[str, Any]:
    audio_id = make_global_media_id(
        media_type="audio",
        source_dataset=source_dataset,
        source_id=path.stem,
        media_path=path,
        project_root=PROJECT_ROOT,
    )
    return {
        "audio_id": audio_id,
        "audio_path": str(path),
        "captions": captions,
        "split": split,
        "source_dataset": source_dataset,
        "source_audio_id": path.stem,
    }


def generate_metadata(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = create_openai_client(
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        config=args.config_data,
    )

    if args.image_path:
        image_paths = iter_media_files(args.image_path, IMAGE_EXTENSIONS)
        if not image_paths:
            raise ValueError(
                "No supported images found under --image_path. "
                f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}"
            )
        image_output_path = Path(args.image_output)
        if not image_output_path.is_absolute():
            image_output_path = output_dir / image_output_path
        if args.force:
            write_jsonl(image_output_path, [])
        completed = existing_media_ids(image_output_path, "image_id")
        image_records = []

        for path in tqdm(
            image_paths,
            desc="Caption images",
            unit="image",
        ):
            record = make_image_record(
                path=path,
                captions=[],
                source_dataset=args.source_dataset,
                split=args.split,
            )
            if record["image_id"] in completed and not args.force:
                continue

            record["captions"] = caption_image(
                client=client,
                path=path,
                model=args.model,
                max_attempts=args.max_attempts,
                max_output_tokens=args.max_output_tokens,
            )
            image_records.append(record)

            if len(image_records) >= args.flush_every:
                append_jsonl(image_output_path, image_records)
                image_records = []

        append_jsonl(image_output_path, image_records)
        print(f"Wrote image metadata to {image_output_path}")

    if args.audio_path:
        audio_paths = iter_media_files(args.audio_path, AUDIO_EXTENSIONS)
        if not audio_paths:
            raise ValueError(
                "No supported audio found under --audio_path. "
                f"Supported extensions: {', '.join(sorted(AUDIO_EXTENSIONS))}"
            )
        audio_output_path = Path(args.audio_output)
        if not audio_output_path.is_absolute():
            audio_output_path = output_dir / audio_output_path
        if args.force:
            write_jsonl(audio_output_path, [])
        completed = existing_media_ids(audio_output_path, "audio_id")
        audio_records = []

        for path in tqdm(
            audio_paths,
            desc="Caption audio",
            unit="audio",
        ):
            record = make_audio_record(
                path=path,
                captions=[],
                source_dataset=args.source_dataset,
                split=args.split,
            )
            if record["audio_id"] in completed and not args.force:
                continue

            record["captions"] = caption_audio(
                client=client,
                path=path,
                model=args.model,
                max_attempts=args.max_attempts,
                clip_seconds=args.audio_clip_seconds,
                max_output_tokens=args.max_output_tokens,
            )
            audio_records.append(record)

            if len(audio_records) >= args.flush_every:
                append_jsonl(audio_output_path, audio_records)
                audio_records = []

        append_jsonl(audio_output_path, audio_records)
        print(f"Wrote audio metadata to {audio_output_path}")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        default=None,
        help="Path to TOML config. Default: configs/reflectra.toml.",
    )
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Create Reflectra metadata JSONL from local audio/images.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--audio_path",
        "--audio-path",
        nargs="*",
        default=[],
        metavar="DIR",
        help=(
            "Directory root(s) containing songs/audio to scan recursively. "
            "Individual audio files are still accepted."
        ),
    )
    parser.add_argument(
        "--image_path",
        "--image-path",
        nargs="*",
        default=[],
        metavar="DIR",
        help=(
            "Directory root(s) containing images to scan recursively. "
            "Individual image files are still accepted."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(METADATA_DIR),
    )
    parser.add_argument(
        "--audio_output",
        default="custom_audio_metadata.jsonl",
    )
    parser.add_argument(
        "--image_output",
        default="custom_image_metadata.jsonl",
    )
    parser.add_argument(
        "--source_dataset",
        default="custom_llm_captioned",
    )
    parser.add_argument("--split", default="custom")
    parser.add_argument(
        "--model",
        default=(
            get_nested(config, "metadata", "model", None)
            or get_nested(config, "benchmark", "model", "")
        ),
    )
    parser.add_argument(
        "--base_url",
        default=get_nested(config, "llm", "base_url", "") or None,
        help="OpenAI-compatible API base URL. Defaults to [llm].base_url.",
    )
    parser.add_argument(
        "--api_key",
        default=get_nested(config, "llm", "api_key", "") or None,
        help=(
            "API key for the OpenAI-compatible client. Prefer "
            "--api_key_env or [llm].api_key_env for real secrets."
        ),
    )
    parser.add_argument(
        "--api_key_env",
        default=get_nested(config, "llm", "api_key_env", "OPENAI_API_KEY"),
    )
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=int(get_nested(config, "llm", "max_output_tokens", 128)),
        help="Maximum tokens the LLM may generate for each JSON caption reply.",
    )
    parser.add_argument(
        "--audio_clip_seconds",
        type=float,
        default=15.0,
        help=(
            "If an audio file is longer than this, send only a middle clip. "
            "Use 0 to send the full file."
        ),
    )
    parser.add_argument("--flush_every", type=int, default=10)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if not args.audio_path and not args.image_path:
        parser.error("Provide at least one --audio_path or --image_path.")

    if not args.model:
        parser.error("A model is required via --model or configs/reflectra.toml.")

    if args.max_attempts < 1:
        parser.error("--max_attempts must be at least 1.")

    if args.max_output_tokens < 16:
        parser.error("--max_output_tokens must be at least 16.")

    if args.flush_every < 1:
        parser.error("--flush_every must be at least 1.")

    if args.audio_clip_seconds <= 0:
        args.audio_clip_seconds = None

    args.config_data = config
    return args


def main() -> None:
    generate_metadata(parse_args())


if __name__ == "__main__":
    main()
