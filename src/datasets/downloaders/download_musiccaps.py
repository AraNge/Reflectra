import os
import json
import argparse
import subprocess
from datasets import load_dataset


# Find project directory roots
module_dir = os.path.dirname(os.path.abspath(__file__))  # src/modules
project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))  # REFLECTION

# Define centralized parent data directories
hf_cache_dir = os.path.join(project_root, "data", "hf_cache")
audio_output_dir = os.path.join(project_root, "data", "musiccaps_audio")
metadata_output_path = os.path.join(project_root, "data", "musiccaps_metadata.jsonl")

os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(audio_output_dir, exist_ok=True)


def download_clip(ytid, start_s, end_s, output_path):
    video_url = f"https://www.youtube.com/watch?v={ytid}"

    command = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "-x",
        "--audio-format", "wav",
        "-f", "bestaudio",
        "--download-sections", f"*{start_s}-{end_s}",
        "-o", output_path,
        video_url,
    ]

    try:
        subprocess.run(command, check=True)
        return os.path.exists(output_path)
    except subprocess.CalledProcessError:
        return False


def write_metadata_row(meta_file, row, audio_id, file_path, status):
    metadata = {
        "audio_id": audio_id,
        "youtube_url": f"https://www.youtube.com/watch?v={row['ytid']}",
        "start_s": row["start_s"],
        "end_s": row["end_s"],
        "audio_path": file_path,
        "caption": row["caption"],
        "aspect_list": row["aspect_list"],
        "source_dataset": "google/MusicCaps",
        "split": "train",
        "status": status,
    }

    meta_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")


def download_musiccaps_samples(num_samples: int):
    print("Loading MusicCaps metadata...")
    ds = load_dataset("google/MusicCaps", split="train", cache_dir=hf_cache_dir)

    print(f"Dataset columns: {ds.column_names}")
    print(f"Total songs available for download: {len(ds)}")
    print(f"Requested samples: {num_samples}")
    print(f"Audio clips will save to: {audio_output_dir}")
    print(f"Metadata will save to: {metadata_output_path}")

    num_samples = min(num_samples, len(ds))

    print("Starting downloads...")

    downloaded_count = 0
    failed_count = 0
    skipped_count = 0

    with open(metadata_output_path, "w", encoding="utf-8") as meta_file:
        for i, row in enumerate(ds.select(range(num_samples))):
            ytid = row["ytid"]
            start_s = row["start_s"]
            end_s = row["end_s"]
            caption = row["caption"]
            aspects = row["aspect_list"]

            safe_ytid = ytid.replace("/", "_")
            audio_id = f"{safe_ytid}_{start_s}_{end_s}"
            file_name = f"{audio_id}.wav"
            file_path = os.path.join(audio_output_dir, file_name)

            print(f"\n[{i + 1}/{num_samples}] Downloading track {ytid} from {start_s}s to {end_s}s...")

            if os.path.exists(file_path):
                print(f"-> Already exists: {file_path}")
                skipped_count += 1

                write_metadata_row(
                    meta_file=meta_file,
                    row=row,
                    audio_id=audio_id,
                    file_path=file_path,
                    status="already_exists",
                )
                continue

            success = download_clip(ytid, start_s, end_s, file_path)

            if success:
                downloaded_count += 1
                print(f"-> Saved to: {file_path}")
                print(f"-> Mood/Aspects: {aspects}")
                print(f"-> Caption: {caption}")

                write_metadata_row(
                    meta_file=meta_file,
                    row=row,
                    audio_id=audio_id,
                    file_path=file_path,
                    status="downloaded",
                )
            else:
                failed_count += 1
                print(f"-> Failed to download {ytid}. Video may be deleted, private, or region-locked.")

    print("\nDownload summary:")
    print(f"Requested: {num_samples}")
    print(f"Downloaded: {downloaded_count}")
    print(f"Skipped existing: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {audio_output_dir}")
    print(f"Metadata file: {metadata_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download MusicCaps audio clips from YouTube using yt-dlp."
    )

    parser.add_argument(
        "--number",
        "-n",
        type=int,
        default=10,
        help="Number of MusicCaps samples to download. Default: 10.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_musiccaps_samples(num_samples=args.number)