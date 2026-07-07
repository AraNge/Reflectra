import argparse
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from src.config import get_nested, load_config
from src.datasets.paths import PROJECT_ROOT


DEFAULT_AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index a local music folder into Qdrant with CLAP audio embeddings."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config. Default: configs/reflectra.toml.",
    )

    parser.add_argument(
        "--music-dir",
        type=str,
        default=None,
        help="Directory containing music files to index. Overrides config audio_index.music_dir.",
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Qdrant collection name. Overrides config qdrant.collection_name.",
    )

    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="Qdrant URL. Overrides config qdrant.url.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="CLAP model name. Overrides config models.clap.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for CLAP audio encoding. Overrides config audio_index.batch_size.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )

    parser.add_argument(
        "--vector-size",
        type=int,
        default=None,
        help="Qdrant vector size. Overrides config qdrant.vector_size.",
    )

    parser.add_argument(
        "--extensions",
        type=str,
        default=None,
        help='Comma-separated extensions, e.g. ".wav,.mp3,.flac". Overrides config audio_index.extensions.',
    )

    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def parse_extensions(value: str | None, fallback: List[str]) -> List[str]:
    if value is None:
        extensions = fallback
    else:
        extensions = [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    return [
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    ]


def discover_audio_files(
    music_dir: Path,
    extensions: List[str],
) -> List[Path]:
    if not music_dir.exists():
        raise FileNotFoundError(f"Music directory does not exist: {music_dir}")

    if not music_dir.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {music_dir}")

    extension_set = set(extensions)

    return sorted(
        path.resolve()
        for path in music_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extension_set
    )


def build_payload(audio_path: Path, music_dir: Path) -> Dict[str, Any]:
    relative_path = audio_path.relative_to(music_dir)

    return {
        "audio_id": str(relative_path),
        "audio_path": str(audio_path),
        "relative_path": str(relative_path),
        "filename": audio_path.name,
        "stem": audio_path.stem,
        "extension": audio_path.suffix.lower(),
        "source": "local_music_library",
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    from src.models.clap_encoder import PretrainedCLAPEncoder
    from src.vector_db.qdrant_store import (
        create_collection_if_not_exists,
        get_qdrant_client,
        upsert_vectors,
    )

    music_dir_value = args.music_dir or get_nested(config, "audio_index", "music_dir", "data/music")
    collection_name = args.collection_name or get_nested(config, "qdrant", "collection_name", "reflectra_music_clap")
    qdrant_url = args.qdrant_url or get_nested(config, "qdrant", "url", "http://localhost:6333")
    model_name = args.model_name or get_nested(config, "models", "clap", "laion/clap-htsat-unfused")
    batch_size = args.batch_size or int(get_nested(config, "audio_index", "batch_size", 8))
    vector_size = args.vector_size or int(get_nested(config, "qdrant", "vector_size", 512))
    extensions = parse_extensions(
        value=args.extensions,
        fallback=get_nested(config, "audio_index", "extensions", DEFAULT_AUDIO_EXTENSIONS),
    )

    music_dir = resolve_project_path(music_dir_value)
    audio_files = discover_audio_files(
        music_dir=music_dir,
        extensions=extensions,
    )

    if args.max_files is not None:
        audio_files = audio_files[: args.max_files]

    print(f"[INFO] Music directory: {music_dir}")
    print(f"[INFO] Audio files found: {len(audio_files)}")
    print(f"[INFO] Qdrant URL: {qdrant_url}")
    print(f"[INFO] Qdrant collection: {collection_name}")
    print(f"[INFO] CLAP model: {model_name}")

    if len(audio_files) == 0:
        raise RuntimeError("No audio files found. Check --music-dir and --extensions.")

    client = get_qdrant_client(url=qdrant_url)

    create_collection_if_not_exists(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
    )

    model = PretrainedCLAPEncoder(
        model_name=model_name,
        freeze=True,
    )

    for start in tqdm(range(0, len(audio_files), batch_size), desc="Indexing music"):
        end = start + batch_size
        batch_paths = audio_files[start:end]

        audio_embeddings = model.encode_audio([str(path) for path in batch_paths])
        vectors = audio_embeddings.cpu().numpy().tolist()

        ids = [
            f"local_music:{path.relative_to(music_dir)}"
            for path in batch_paths
        ]

        payloads = [
            build_payload(audio_path=path, music_dir=music_dir)
            for path in batch_paths
        ]

        upsert_vectors(
            client=client,
            collection_name=collection_name,
            ids=ids,
            vectors=vectors,
            payloads=payloads,
            batch_size=batch_size,
        )

    print("[INFO] Done indexing local music CLAP audio embeddings.")


"""
python -m src.vector_db.index_clap_audio_qdrant --music-dir data/music
python -m src.vector_db.index_clap_audio_qdrant --music-dir /path/to/music --collection-name reflectra_music_clap
"""

if __name__ == "__main__":
    main()
