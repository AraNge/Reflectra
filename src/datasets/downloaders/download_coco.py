import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from src.datasets.preprocessing.coco_karpathy import (
    CocoKarpathyPreprocessor,
    write_coco_jsonl,
)
from src.datasets.paths import DATA_DIR, METADATA_DIR, PROJECT_ROOT, ensure_data_dirs


COCO_SPLIT = "val2014"
COCO_IMAGE_URL = "http://images.cocodataset.org/zips/val2014.zip"

KARPATHY_METADATA_URL = "http://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip"
KARPATHY_JSON_NAME = "dataset_coco.json"

DOWNLOAD_DIR = DATA_DIR / "downloads" / "coco"
IMAGE_ROOT = DATA_DIR / "coco_images"
KARPATHY_DIR = DATA_DIR / "coco_karpathy"
KARPATHY_JSON_PATH = METADATA_DIR / KARPATHY_JSON_NAME
CAPTIONS_METADATA_PATH = METADATA_DIR / "coco_captions_metadata.jsonl"

TMP_DIR = Path(tempfile.gettempdir()) / "reflectra_cxc"
CXC_REPO_DIR = TMP_DIR / "Crisscrossed-Captions"
DEFAULT_CXC_OUTPUT_PATH = METADATA_DIR / "coco_karpathy_cxc_sits_val.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download COCO 2014 images and Karpathy COCO caption metadata."
    )

    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Download/prepare metadata only. COCO images are val2014 only.",
    )

    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Download/extract images only.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract existing files.",
    )

    parser.add_argument(
        "--prepare-cxc",
        action="store_true",
        help="Also clone CxC and merge SITS labels into data/metadata.",
    )

    parser.add_argument(
        "--coco-input",
        type=str,
        default=None,
        help=(
            "Optional local path to Karpathy dataset_coco.json. "
            "Used for COCO JSONL export and CxC merging."
        ),
    )

    parser.add_argument(
        "--cxc-output",
        type=str,
        default=str(DEFAULT_CXC_OUTPUT_PATH),
        help="CxC val output path.",
    )

    parser.add_argument(
        "--force-cxc",
        action="store_true",
        help="Delete and re-clone the CxC repo in /tmp.",
    )

    parser.add_argument(
        "--clear-cxc-tmp",
        action="store_true",
        help="Delete temporary CxC files after successful preparation.",
    )

    return parser.parse_args()


def request_for_url(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        },
    )


def download_file(url: str, output_path: Path, force: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        print(f"Already downloaded: {output_path}")
        return

    partial_path = output_path.with_suffix(output_path.suffix + ".part")

    if partial_path.exists():
        partial_path.unlink()

    print(f"\nDownloading: {url}")
    print(f"Saving to: {output_path}")

    request = request_for_url(url)

    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length", 0))

        with open(partial_path, "wb") as f:
            with tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=output_path.name,
            ) as progress:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    f.write(chunk)
                    progress.update(len(chunk))

    partial_path.replace(output_path)
    print(f"Downloaded: {output_path}")


def extract_zip(zip_path: Path, output_dir: Path, marker_name: str, force: bool = False) -> None:
    marker_path = output_dir / marker_name

    if marker_path.exists() and not force:
        print(f"Already extracted: {zip_path}")
        return

    if force and marker_path.exists():
        marker_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting: {zip_path}")
    print(f"Output directory: {output_dir}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.infolist()

        for member in tqdm(members, desc=f"Extracting {zip_path.name}"):
            zip_ref.extract(member, output_dir)

    marker_path.write_text("ok", encoding="utf-8")
    print(f"Extracted: {zip_path}")


def find_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))

    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")

    return matches[0]


def run_command(command: list[str], cwd: Path | None = None) -> None:
    print("\nRunning:")
    print(" ".join(str(part) for part in command))

    subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
    )


def download_coco_images(force: bool = False) -> None:
    zip_path = DOWNLOAD_DIR / f"{COCO_SPLIT}.zip"

    download_file(url=COCO_IMAGE_URL, output_path=zip_path, force=force)
    extract_zip(
        zip_path=zip_path,
        output_dir=IMAGE_ROOT,
        marker_name=f".{COCO_SPLIT}.extracted",
        force=force,
    )


def download_karpathy_metadata(force: bool = False) -> Path:
    zip_path = DOWNLOAD_DIR / "karpathy_coco.zip"

    download_file(
        url=KARPATHY_METADATA_URL,
        output_path=zip_path,
        force=force,
    )

    extract_zip(
        zip_path=zip_path,
        output_dir=KARPATHY_DIR,
        marker_name=".karpathy_metadata.extracted",
        force=force,
    )

    metadata_json = find_file(KARPATHY_DIR, KARPATHY_JSON_NAME)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metadata_json, KARPATHY_JSON_PATH)

    print(f"Karpathy metadata JSON: {KARPATHY_JSON_PATH}")

    return KARPATHY_JSON_PATH


def export_coco_caption_metadata(metadata_json: Path) -> None:
    print(f"\nExporting loader metadata: {CAPTIONS_METADATA_PATH}")

    preprocessor = CocoKarpathyPreprocessor(
        metadata_path=metadata_json,
        image_root=IMAGE_ROOT,
        require_image_exists=False,
    )
    rows = preprocessor.load_rows()
    written = write_coco_jsonl(rows, CAPTIONS_METADATA_PATH)

    print(f"Metadata rows written: {written}")
    print(f"COCO caption metadata: {CAPTIONS_METADATA_PATH}")


def clone_or_update_cxc_repo(force: bool = False) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    if CXC_REPO_DIR.exists() and force:
        print(f"Removing existing CxC repo: {CXC_REPO_DIR}")
        shutil.rmtree(CXC_REPO_DIR)

    if not CXC_REPO_DIR.exists():
        run_command(
            [
                "git",
                "clone",
                "https://github.com/google-research-datasets/Crisscrossed-Captions.git",
                str(CXC_REPO_DIR),
            ]
        )
    else:
        print(f"CxC repo already exists: {CXC_REPO_DIR}")


def find_cxc_setup_file() -> Path:
    candidates = [
        CXC_REPO_DIR / "setup.py",
        CXC_REPO_DIR / "crisscrossed_captions" / "setup.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(CXC_REPO_DIR.rglob("setup.py"))

    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find CxC setup.py under {CXC_REPO_DIR}")


def get_cxc_sits_file(split: str) -> Path:
    path = CXC_REPO_DIR / "data" / f"sits_{split}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Could not find CxC SITS file: {path}")

    return path


def prepare_cxc_sits(
    coco_input: Path,
    output_path: Path,
    force_cxc: bool = False,
    clear_tmp: bool = False,
) -> None:
    clone_or_update_cxc_repo(force=force_cxc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    setup_file = find_cxc_setup_file()
    cxc_input = get_cxc_sits_file("val")

    print("\nPreparing CxC SITS split: val")
    print(f"COCO input: {coco_input}")
    print(f"CxC input: {cxc_input}")
    print(f"Output: {output_path}")

    run_command(
        [
            sys.executable,
            str(setup_file),
            "--coco_input",
            str(coco_input),
            "--cxc_input",
            str(cxc_input),
            "--output",
            str(output_path),
        ],
        cwd=CXC_REPO_DIR,
    )

    validate_cxc_output(output_path)

    if clear_tmp and TMP_DIR.exists():
        print(f"\nClearing temporary CxC directory: {TMP_DIR}")
        shutil.rmtree(TMP_DIR)


def validate_cxc_output(output_json: Path) -> None:
    if not output_json.exists():
        raise FileNotFoundError(f"Expected CxC output not found: {output_json}")

    print(f"Merged CxC output saved to: {output_json}")

    try:
        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            print(f"Top-level keys: {list(data.keys())[:20]}")
        elif isinstance(data, list):
            print(f"Top-level list length: {len(data)}")
    except Exception as e:
        print(f"[WARN] Output exists but could not inspect JSON: {e}")


def main() -> None:
    args = parse_args()

    ensure_data_dirs()
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {DATA_DIR}")
    print(f"COCO image root: {IMAGE_ROOT}")
    print(f"Metadata directory: {METADATA_DIR}")

    metadata_json = None

    if not args.skip_images:
        download_coco_images(force=args.force)

    if not args.skip_metadata:
        if args.coco_input:
            metadata_json = Path(args.coco_input).expanduser().resolve()

            if not metadata_json.exists():
                raise FileNotFoundError(f"--coco-input does not exist: {metadata_json}")

            shutil.copy2(metadata_json, KARPATHY_JSON_PATH)
            print(f"Using local Karpathy metadata: {metadata_json}")
            print(f"Copied metadata JSON to: {KARPATHY_JSON_PATH}")
        else:
            metadata_json = download_karpathy_metadata(force=args.force)

        export_coco_caption_metadata(metadata_json)

    if args.prepare_cxc:
        if metadata_json is None:
            if args.coco_input:
                metadata_json = Path(args.coco_input).expanduser().resolve()
            elif KARPATHY_JSON_PATH.exists():
                metadata_json = KARPATHY_JSON_PATH
            else:
                metadata_json = download_karpathy_metadata(force=args.force)

        output_path = Path(args.cxc_output).expanduser()

        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

        prepare_cxc_sits(
            coco_input=metadata_json.resolve(),
            output_path=output_path.resolve(),
            force_cxc=args.force_cxc,
            clear_tmp=args.clear_cxc_tmp,
        )

    print("\nDone.")


"""
python -m src.datasets.downloaders.download_coco
python -m src.datasets.downloaders.download_coco --skip-metadata
python -m src.datasets.downloaders.download_coco --skip-images --prepare-cxc
"""

if __name__ == "__main__":
    main()
