import argparse
import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.datasets.paths import PROJECT_ROOT


DEFAULT_STORAGE_DIR = PROJECT_ROOT / "qdrant_storage"
DEFAULT_STATE_PATH = PROJECT_ROOT / "data" / "study_audio_parts" / "study_fill_state.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "vector_db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or restore a portable Reflectra Qdrant vector DB snapshot."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Archive qdrant_storage and study fill state.")
    create.add_argument("--storage-dir", default=str(DEFAULT_STORAGE_DIR))
    create.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    create.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    create.add_argument("--name", default=None, help="Archive filename. Defaults to timestamped .tar.gz.")

    restore = subparsers.add_parser("restore", help="Restore qdrant_storage and fill state from an archive.")
    restore.add_argument("archive", help="Path to a snapshot .tar.gz created by this script.")
    restore.add_argument("--storage-dir", default=str(DEFAULT_STORAGE_DIR))
    restore.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    restore.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing qdrant_storage/state paths.",
    )

    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def directory_summary(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def create_snapshot(args: argparse.Namespace) -> Path:
    storage_dir = resolve_path(args.storage_dir)
    state_path = resolve_path(args.state_path)
    output_dir = resolve_path(args.output_dir)

    if not storage_dir.exists():
        raise FileNotFoundError(f"Qdrant storage directory not found: {storage_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_name = args.name
    if archive_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_name = f"reflectra_vector_db_{timestamp}.tar.gz"
    if not archive_name.endswith(".tar.gz"):
        archive_name = f"{archive_name}.tar.gz"

    archive_path = output_dir / archive_name
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "Reflectra",
        "format": "reflectra-vector-db-snapshot-v1",
        "paths": {
            "qdrant_storage": "qdrant_storage",
            "study_fill_state": "data/study_audio_parts/study_fill_state.json",
        },
        "qdrant_storage": directory_summary(storage_dir),
        "study_fill_state_included": state_path.exists(),
    }

    with tempfile.TemporaryDirectory(prefix="reflectra_vector_db_") as temp_name:
        temp_dir = Path(temp_name)
        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(manifest_path, arcname="manifest.json")
            archive.add(storage_dir, arcname="qdrant_storage")
            if state_path.exists():
                archive.add(state_path, arcname="data/study_audio_parts/study_fill_state.json")

    return archive_path


def safe_members(archive: tarfile.TarFile, destination: Path) -> list[tarfile.TarInfo]:
    destination = destination.resolve()
    members = archive.getmembers()
    for member in members:
        member_path = (destination / member.name).resolve()
        if destination != member_path and destination not in member_path.parents:
            raise RuntimeError(f"Unsafe archive member path: {member.name}")
    return members


def replace_path(source: Path, destination: Path, force: bool) -> None:
    if destination.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing path without --force: {destination}")
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def restore_snapshot(args: argparse.Namespace) -> None:
    archive_path = resolve_path(args.archive)
    storage_dir = resolve_path(args.storage_dir)
    state_path = resolve_path(args.state_path)

    if not archive_path.exists():
        raise FileNotFoundError(f"Snapshot archive not found: {archive_path}")

    with tempfile.TemporaryDirectory(prefix="reflectra_vector_db_restore_") as temp_name:
        temp_dir = Path(temp_name)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(temp_dir, members=safe_members(archive, temp_dir))

        extracted_storage = temp_dir / "qdrant_storage"
        if not extracted_storage.exists():
            raise RuntimeError("Snapshot does not contain qdrant_storage.")

        replace_path(extracted_storage, storage_dir, args.force)

        extracted_state = temp_dir / "data" / "study_audio_parts" / "study_fill_state.json"
        if extracted_state.exists():
            replace_path(extracted_state, state_path, args.force)


def main() -> None:
    args = parse_args()
    if args.command == "create":
        archive_path = create_snapshot(args)
        print(json.dumps({"snapshot_path": str(archive_path)}, indent=2))
        return

    if args.command == "restore":
        restore_snapshot(args)
        print(json.dumps({"restored": True}, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
