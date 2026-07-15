# Reflectra Study Vector DB

This folder contains the tooling for building a large Qdrant collection of CLAP audio embeddings from public audio datasets. The intended workflow is:

1. Start or resume indexing into local Qdrant.
2. Stop whenever you have enough samples.
3. Snapshot `qdrant_storage` plus the study resume state.
4. Upload the snapshot somewhere durable, such as Google Drive.
5. Restore later to continue indexing or use the DB as-is.

## What Gets Indexed

`study/fill_db.sh` runs [study/fill_clap_audio_qdrant.py](fill_clap_audio_qdrant.py), which streams audio parts from supported datasets, encodes audio with CLAP, and upserts vectors into Qdrant.

Each Qdrant point contains:

- a 512-dimensional CLAP audio vector
- `audio_id`
- `dataset_id`
- `source_dataset`
- `captions`
- `original_id`
- `dataset_key`, for newly indexed points
- `dataset_split`, for newly indexed points
- `dataset_subset`, for newly indexed AudioSet points
- `archive_idx`, for newly indexed MTG-Jamendo points

The GUI uses these fields to re-download and play found songs. Older snapshots without the extra lookup hints still work, but MTG-Jamendo and AudioSet lookup can be slower because the downloader has to search for the matching source item.

Supported study dataset keys:

```text
song_describer
mtg_jamendo_train
mtg_jamendo_validation
audioset_balanced_train
audioset_balanced_test
audioset_unbalanced_train
audioset_unbalanced_test
```

The default study run excludes MusicCaps.

## Index The DB

From the project root:

```bash
bash study/fill_db.sh --target-samples 500000 --part-size 200
```

For a smaller run:

```bash
bash study/fill_db.sh --target-samples 13000 --part-size 200
```

Useful options are passed through to `study.fill_clap_audio_qdrant`:

```bash
bash study/fill_db.sh \
  --target-samples 500000 \
  --part-size 200 \
  --datasets song_describer,mtg_jamendo_train,audioset_balanced_train
```

Important paths:

```text
qdrant_storage/                         Live Qdrant storage mounted into Docker
data/study_audio_parts/study_fill_state.json
data/vector_db/qdrant_storage/          Copy made by fill_db.sh after indexing
```

The fill state records where indexing stopped. Keeping it lets you restore later and continue without starting dataset iteration from the beginning.

## Pause And Continue

You can stop indexing with `Ctrl+C`. The script saves state after each processed part.

To continue later:

```bash
bash study/fill_db.sh --target-samples 500000 --part-size 200
```

The script reads:

```text
data/study_audio_parts/study_fill_state.json
```

and resumes from the current Qdrant collection count.

## Create A Snapshot

Create a portable archive in `data/vector_db/`:

```bash
python -m study.snapshot_vector_db create
```

This writes a timestamped file such as:

```text
data/vector_db/reflectra_vector_db_20260715_081814.tar.gz
```

The archive includes:

- `qdrant_storage/`
- `data/study_audio_parts/study_fill_state.json`, when present
- `manifest.json`

You can choose a custom name:

```bash
python -m study.snapshot_vector_db create --name reflectra_13k.tar.gz
```

After that, upload the `.tar.gz` file to Google Drive or another storage service.

## Restore A Snapshot

Download the snapshot archive back into the project, then restore it:

```bash
python -m study.snapshot_vector_db restore data/vector_db/reflectra_13k.tar.gz --force
```

By default this restores to:

```text
qdrant_storage/
data/study_audio_parts/study_fill_state.json
```

Start Qdrant after restore:

```bash
bash scripts/setup.sh start
```

Now you can either use the DB directly in Reflectra, or continue expanding it:

```bash
bash study/fill_db.sh --target-samples 500000 --part-size 200
```

## Use Without Continuing

If you only want to search with the restored DB:

```bash
bash scripts/setup.sh start
reflectra-gui --checkpoint checkpoints/reflectra_projection_flickr30k_1000.pt
```

The GUI search results include `Play` and `Save` buttons. These use the stored Qdrant metadata to download clips from the source dataset into:

```text
data/study_downloaded_audio/
```

## Search Timing Benchmark

To measure image-search latency over many images and generate p50/p90/p95/p99 stage plots, run:

```bash
python -m study.benchmark_search_timings \
  --samples 1000 \
  --checkpoint checkpoints/reflectra_projection_flickr30k_1000.pt
```

The script reuses the Flickr30k downloader. If fewer local Flickr30k images are available than `--samples`, it downloads enough records into:

```text
data/flickr30k_images/
data/metadata/flickr30k_metadata.jsonl
```

Use already-downloaded images only:

```bash
python -m study.benchmark_search_timings \
  --samples 1000 \
  --skip-download \
  --checkpoint checkpoints/reflectra_projection_flickr30k_1000.pt
```

Useful options:

```bash
python -m study.benchmark_search_timings \
  --samples 1000 \
  --percentiles 50,75,90,95,99 \
  --output-dir plots/search_timing_benchmark \
  --qdrant-url http://localhost:6333 \
  --collection-name reflectra_music_clap \
  --checkpoint checkpoints/reflectra_projection_flickr30k_1000.pt
```

Outputs are written to `plots/search_timing_benchmark/` by default:

```text
raw_timings.jsonl                 One row per image search
summary.json                      Aggregated stage timing data
summary.csv                       Percentile table for spreadsheets
stage_percentiles_stacked.png     p50/p90/p95/p99 stacked bar chart
stage_time_image_search_p50.png   Single-bar p50 plot
stage_time_image_search_p99.png   Single-bar p99 plot
```

Start Qdrant before running the benchmark:

```bash
bash scripts/setup.sh start
```

The benchmark loads Reflectra once, then runs the real `image_search` path for every selected image. The plotted stages match the normal search timing stages, such as `encode_query`, `check_db`, and `format_results`.

## Restore Somewhere Else

To restore into non-default paths:

```bash
python -m study.snapshot_vector_db restore reflectra_13k.tar.gz \
  --storage-dir /path/to/qdrant_storage \
  --state-path /path/to/study_fill_state.json \
  --force
```

If you use a custom Qdrant storage directory, mount that directory into the Qdrant Docker container.

## Check The Current State

Inspect the saved progress:

```bash
cat data/study_audio_parts/study_fill_state.json
```

Check Qdrant point count while Qdrant is running:

```bash
python - <<'PY'
from src.vector_db.qdrant_store import get_qdrant_client

client = get_qdrant_client("http://localhost:6333")
print(client.count(collection_name="reflectra_music_clap", exact=True).count)
PY
```
