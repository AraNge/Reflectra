<div align="center">

<img src="assets/logo.png" alt="Reflectra Logo" width="280"/>

# Reflectra

### Image-to-Music Retrieval with CLIP, CLAP, Qdrant, and Multimodal Embeddings

<p>
  <b>Upload an image. Understand its visual mood. Retrieve matching songs.</b>
</p>

<p>
  <img src="https://img.shields.io/badge/Task-Image%20to%20Music%20Retrieval-8A2BE2" />
  <img src="https://img.shields.io/badge/Model-CLIP%20%2B%20CLAP-00BFFF" />
  <img src="https://img.shields.io/badge/Vector%20DB-Qdrant-FF4FD8" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C" />
</p>

</div>


---

## Overview

Reflectra is a multimodal retrieval system for recommending songs from an input image. The project does not generate music. Instead, it maps images, text descriptions, and audio clips into compatible embedding spaces and retrieves the closest matching audio tracks.

The system flow is:

```text
image
↓
CLIP image encoder
↓
image-to-CLAP projection layer
↓
CLAP embedding space
↓
Qdrant vector search over CLAP audio embeddings
↓
candidate songs
↓
reranker
↓
ranked songs
```


---

## Datasets

Downloaded files should live in the top-level `data/` directory. Python code for loading and downloading datasets should live under `src/datasets/`.

### Audio-text datasets

| Dataset | Main use | Notes |
|---|---|---|
| MusicCaps | Clean text-to-audio evaluation | Stores the human caption plus aspect-list entries as `captions`. Best clean benchmark for CLAP retrieval. |
| Song Describer Dataset | Clean text-to-audio evaluation | Stores available song descriptions as `captions`. |
| MTG-Jamendo | Training / validation / scale evaluation | Builds captions from moods, genres, and instruments, then saves only the clean caption list. |
| AudioSet | Large-scale weak evaluation / robustness | Builds one caption per human label. Useful for scale and hard negatives, but not a clean song-caption benchmark. |

### Image-text datasets

| Dataset | Main use | Notes |
|---|---|---|
| COCO 2014 | Image-text benchmark | General image-caption alignment using Karpathy caption metadata. |
| Flickr30k | Clean image-text evaluation | Copies the dataset caption list into `captions`. |
| EmoSet | Image mood alignment | Converts visual emotion into music-oriented captions. |

All normal metadata JSONL rows use a small generic schema:

```json
{
  "audio_id": "...",
  "audio_path": "...",
  "captions": ["..."],
  "split": "...",
  "source_dataset": "..."
}
```

For image datasets, `audio_id/audio_path` are replaced by `image_id/image_path`. Metadata loaders keep captions grouped on each media item; torch datasets combine the `captions` list into one training text when reading a sample.

---

## Installation

This project uses `pyproject.toml`, so install it as a package from the project root.

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install package


```bash
pip install -e .
```


### 3. Run setup script

```bash
bash scripts/setup.sh
```

The setup script creates project directories, writes `configs/reflectra.toml` if missing, installs the package in editable mode, pulls the Qdrant Docker image, and starts a local Qdrant container.

---

## Qdrant Setup

Qdrant is used to store CLAP audio embeddings and perform fast approximate nearest-neighbor search.

Default local URL:

```text
http://localhost:6333
```

Start Qdrant manually:

```bash
docker run -d \
  --name reflectra-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest
```

Stop Qdrant:

```bash
docker stop reflectra-qdrant
```

Start an existing Qdrant container again:

```bash
docker start reflectra-qdrant
```

---

## Scripts

The `scripts/` folder contains runnable shortcuts for common workflows:

```bash
bash scripts/setup.sh
```

Creates local data/config folders, installs the project in editable mode, and starts Qdrant with Docker when available.

```bash
bash scripts/generate_clap_dataset.sh --audio-samples 100 --max-audios 6
```

Builds or resumes the CLAP caption-to-audio LLM benchmark using Song Describer metadata and a local `llama-server`.

```bash
bash scripts/generate_dataset.sh --max-samples 6
```

Builds or resumes the image-to-audio Reflectra benchmark from Flickr30k images and Song Describer audio. Use `-s INDEX` for sharded runs.

```bash
bash scripts/train_proj.sh -n 1000
```

Trains the image-to-CLAP projection on a sampled Flickr30k metadata subset.

```bash
bash scripts/evaluate_reflectra.sh --checkpoint checkpoints/reflectra.pt
```

Downloads and unpacks the Reflectra benchmark dataset, then runs Reflectra evaluation. Omit `--checkpoint` to use the first projection checkpoint found in `checkpoints/`.

```bash
bash scripts/evaluate_clap.sh
```

Downloads and unpacks the CLAP benchmark dataset, then runs only the CLAP caption-to-audio evaluation.

---

## Download Data

Examples:

```bash
python -m src.datasets.downloaders.download_musiccaps --number 100
python -m src.datasets.downloaders.download_audioset --number 100
python -m src.datasets.downloaders.download_mtg_jamendo --split train --number 100
python -m src.datasets.downloaders.download_mtg_jamendo --split validation --number 100
python -m src.datasets.downloaders.download_coco
python -m src.datasets.downloaders.download_flickr30k --number 100
python -m src.datasets.downloaders.download_emoset --split train --number 100
python -m src.datasets.downloaders.download_emoset --split test --number 100
```

Start small first. Verify metadata and file paths before downloading large datasets.

COCO has a large official val2014 zip file. The downloader shows byte progress while downloading and file progress while extracting:

```bash
python -m src.datasets.downloaders.download_coco
python -m src.datasets.downloaders.download_coco --skip-metadata
```

For the CxC SITS CLIP benchmark, prepare the merged Karpathy+CxC val metadata:

```bash
python -m src.datasets.downloaders.download_coco --skip-images --prepare-cxc
```

This writes files such as:

```text
data/metadata/coco_karpathy_cxc_sits_val.json
```

---

## Create Image-Audio Benchmark

The benchmark builder samples image metadata and audio metadata deterministically, asks an OpenAI model to score every assigned image/audio pair from 0 to 10, and writes resumable shard files.

Set your client settings in `configs/reflectra.toml` or pass them on the CLI:

```toml
[llm]
base_url = ""
api_key = ""
api_key_env = "OPENAI_API_KEY"
```

For OpenAI-hosted models, setting the environment variable is usually enough:

```bash
export OPENAI_API_KEY="..."
```

For a local OpenAI-compatible server, set `base_url` in config or pass `--base_url`.

Run a small single-shard benchmark:

```bash
python -m src.benchmark.create_benchmark \
  --mode build \
  --image_samples 20 \
  --audio_samples 20 \
  --batch_size 10
```

For multiple workers or machines, give every run the same metadata paths, sample counts, seed, model, and shard count, changing only `--shard_index`:

```bash
python -m src.benchmark.create_benchmark \
  --mode build \
  --image_samples 100 \
  --audio_samples 100 \
  --num_shards 4 \
  --shard_index 0
```

After all shards finish, merge them into compact Parquet tables:

```bash
python -m src.benchmark.create_benchmark \
  --mode merge \
  --image_samples 100 \
  --audio_samples 100 \
  --num_shards 4
```

By default, metadata is read from `data/metadata` and benchmark outputs are written to `data/benchmark`. The default benchmark model, random seed, output directory, and Hugging Face export setting live in `configs/reflectra.toml` under `[benchmark]`. CLI arguments override the config.

Merge writes the normalized byte tables:

```text
data/benchmark/audio_table.parquet
data/benchmark/image_table.parquet
data/benchmark/image_audio_scores.parquet
```

---

## Create CLAP LLM Benchmark

To check whether CLAP is suitable for music captions before fine-tuning, create a caption-to-audio benchmark with LLM-graded positives and random negatives:

```bash
python -m src.benchmark.create_clap_benchmark \
  --audio_metadata data/metadata/custom_audio_metadata.jsonl \
  --audio_samples 100 \
  --queries_per_audio 1 \
  --max_audios 10
```

Each query uses an audio caption and a deterministic candidate audio set. The
LLM scores every candidate audio from 0 to 10. The builder supports sharded
runs with `--num_shards` and `--shard_index`, so multiple workers can score
different query partitions and resume incomplete shards.

The final Hugging Face transport tables are:

```text
data/clap_benchmark/clap_llm_benchmark.parquet
data/clap_benchmark/audio_table.parquet
```

---

## Evaluation

There are three evaluation styles:

1. CLAP evaluation against the LLM-scored caption-to-audio benchmark.
2. CLIP evaluation against CxC graded image-caption relevance.
3. Reflectra image-to-audio evaluation against the created benchmark.

Evaluation computes model similarities and compares them with sparse relevance labels, so scripts do not create dense zero-filled relevance matrices.

---

### 1. Evaluate CLAP: Text-to-Audio

This evaluates CLAP against the LLM-scored CLAP benchmark created by `src.benchmark.create_clap_benchmark`.

```bash
python -m src.evaluation.evaluate_clap \
  --benchmark_dir data/clap_benchmark
```

The script computes CLAP text-to-audio rankings for each benchmark caption over that query's positive and random-negative candidate audios.

Metrics:

```text
- ndcg@1, ndcg@5, ndcg@max-audios
- mrr
- mAP
- recall@1, recall@5, recall@max-audios
- precision@1, precision@5, precision@max-audios
```

NDCG uses the LLM 0..10 score as graded relevance. MRR, mAP, recall, and precision treat scores above `--relevance-threshold` as relevant. Metrics are computed only over audios scored for each query, so unevaluated audios are not treated as irrelevant.

---

### 2. Evaluate CLIP: Image-to-Caption CxC

This evaluates CLIP with CxC SITS graded image-caption relevance labels. It is the old CxC evaluator, now exposed as `evaluate_clip.py`.

```bash
python -m src.evaluation.evaluate_clip \
  --metadata data/metadata/coco_karpathy_cxc_sits_val.json \
  --image-root data/coco_images \
  --max-images 1000
```

The script computes:

```text
image → caption
caption → image
```

Metrics:

```text
- ndcg@1, ndcg@5, ndcg@10
- mrr
- mAP
- recall@1, recall@5, recall@10
- precision@1, precision@5, precision@10
```

For MRR, mAP, recall, and precision, every CxC score `> 0` is treated as relevant. NDCG uses the raw graded CxC score.

---

### 3. Evaluate Reflectra: Image-to-Audio

This evaluates the full image-to-audio system on the benchmark created by `src.benchmark.create_benchmark`.

```bash
python -m src.datasets.downloaders.download_reflectra_benchmark

python -m src.evaluation.evaluate_reflectra \
  --benchmark data/benchmark \
  --checkpoint checkpoints/reflectra.pt
```

The downloader unpacks media from the Hugging Face Parquet tables, writes JSONL indexes, and removes the local Parquet transport files. The evaluator reads those unpacked files directly.

Metrics:

```text
- ndcg@1, ndcg@5, ndcg@max-audios
- mrr
- mAP
- recall@1, recall@5, recall@max-audios
- precision@1, precision@5, precision@max-audios
```

NDCG uses the benchmark LLM score as graded relevance. MRR, mAP, recall, and precision treat scores above `--relevance-threshold` as relevant. Metrics are computed only over audios scored for each image, so unevaluated audios are not treated as irrelevant.

---

### 4. Index Your Music in Qdrant

Put local music files under `data/music/`, or pass any folder with `--music-dir`.

```bash
python -m src.vector_db.index_clap_audio_qdrant \
  --music-dir data/music
```

Index another folder:

```bash
python -m src.vector_db.index_clap_audio_qdrant \
  --music-dir /path/to/my/music \
  --collection-name reflectra_music_clap
```

Each indexed audio point should contain payload fields like:

```json
{
  "audio_id": "...",
  "audio_path": "...",
  "relative_path": "...",
  "filename": "...",
  "stem": "...",
  "extension": "...",
  "source": "local_music_library"
}
```

Defaults such as CLAP model name, Qdrant URL, collection name, vector size, batch size, and audio extensions come from `configs/reflectra.toml`. CLI arguments override the config.

## Benchmarks Used

Reflectra uses three benchmark sources:

- Reflectra image-to-audio benchmark: https://huggingface.co/datasets/AraNge/reflectra-benchmark
- Reflectra CLAP caption-to-audio benchmark: https://huggingface.co/datasets/AraNge/reflectra-clap-benchmark
- CLIP CxC image-caption benchmark labels: https://github.com/google-research-datasets/Crisscrossed-Captions

The CLIP CxC evaluation uses CxC SITS graded semantic similarity labels on
MS-COCO image/caption pairs. The Reflectra and CLAP benchmark repositories are
distributed as Parquet transport tables and should be unpacked locally with the
downloaders before evaluation.

## Dataset Links

Audio-text datasets:

- MusicCaps: https://huggingface.co/datasets/google/MusicCaps
- Song Describer Dataset: https://huggingface.co/datasets/renumics/song-describer-dataset
- MTG-Jamendo Hugging Face: https://huggingface.co/datasets/rkstgr/mtg-jamendo
- AudioSet Hugging Face: https://huggingface.co/datasets/agkphysics/AudioSet

Image-text / image-mood datasets:

- Flickr30k Hugging Face: https://huggingface.co/datasets/nlphuji/flickr30k
- EmoSet: https://huggingface.co/datasets/LiangJian24/EmoSet
- CxC SITS labels: https://github.com/google-research-datasets/Crisscrossed-Captions
- COCO 2014 validation images: http://images.cocodataset.org/zips/val2014.zip
- Karpathy COCO caption metadata: http://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip
