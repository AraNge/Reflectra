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

The final goal is:

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
ranked songs
```

The first practical MVP can be simpler:

```text
image
↓
image caption / mood query
↓
CLAP text encoder
↓
Qdrant search over CLAP audio embeddings
↓
ranked songs
```

---

## Architecture

### 1. Text-to-Audio Retrieval

Text-to-audio retrieval is the first baseline. It checks whether pretrained CLAP can retrieve the correct audio clip from a natural language music description.

```text
caption: "happy energetic pop song with female vocals"
        ↓
CLAP text encoder
        ↓
text embedding

song.wav
        ↓
CLAP audio encoder
        ↓
audio embedding

similarity = cosine(text embedding, audio embedding)
```

This stage is used to evaluate CLAP before any image-based retrieval is added.

### 2. Image-to-Text Retrieval

Image-to-text retrieval evaluates the visual side using CLIP.

```text
image
↓
CLIP image encoder
↓
image embedding

caption / mood caption
↓
CLIP text encoder
↓
text embedding

similarity = cosine(image embedding, text embedding)
```

This stage answers whether CLIP understands image captions, scenes, and mood-related text well enough for the project.

### 3. Image-to-Audio Retrieval

The advanced system maps images into the CLAP audio-text space.

```text
image
↓
CLIP image encoder
↓
projection layer
↓
CLAP-compatible embedding
↓
Qdrant search over CLAP audio embeddings
↓
songs
```

The projection layer is trained first while CLIP and CLAP remain frozen.

Recommended projection design:

```text
Linear(CLIP dimension → hidden dimension)
GELU
LayerNorm
Dropout
Linear(hidden dimension → CLAP dimension)
L2 normalization
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

## Project Structure

Recommended structure:

```text
reflectra/
├── assets/
│   └── logo.png
├── configs/
│   └── reflectra.toml
├── data/
│   ├── music/
│   ├── audio/
│   ├── images/
│   ├── metadata/
│   ├── embeddings/
│   └── hf_cache/
├── results/
├── scripts/
│   └── setup_environment.sh
├── src/
│   ├── datasets/
│   │   ├── downloaders/
│   │   ├── evaluation_inputs/
│   │   ├── loaders/
│   │   ├── preprocessing/
│   │   └── selection/
│   ├── evaluation/
│   │   ├── evaluate_clap.py
│   │   └── evaluate_clip.py
│   ├── metrics/
│   │   └── retrieval_metrics.py
│   ├── models/
│   │   ├── clap_encoder.py
│   │   ├── clip_encoder.py
│   │   ├── projection_head.py
│   │   └── reflectra_model.py
│   └── vector_db/
│       ├── qdrant_store.py
│       └── index_clap_audio_qdrant.py
├── pyproject.toml
└── README.md
```

Recommended data layout:

```text
data/
├── music/
├── audio/
│   ├── musiccaps/
│   ├── audioset/
│   ├── mtg_jamendo/
│   └── song_describer/
├── images/
│   ├── coco_captions/
│   ├── flickr30k/
│   └── emoset/
├── metadata/
│   ├── musiccaps_metadata.jsonl
│   ├── audioset_metadata.jsonl
│   ├── song_describer_metadata.jsonl
│   ├── mtg_jamendo_train_metadata.jsonl
│   ├── mtg_jamendo_validation_metadata.jsonl
│   ├── coco_captions_metadata.jsonl
│   ├── flickr30k_metadata.jsonl
│   ├── emoset_train_metadata.jsonl
│   └── emoset_test_metadata.jsonl
└── embeddings/
```

The `data/music/` folder is for your own local music library. Dataset downloaders write benchmark/training data under the other `data/` subfolders.

Default model and Qdrant settings live in:

```text
configs/reflectra.toml
```

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

Basic install:

```bash
pip install -e .
```

Install with development/data dependencies:

```bash
pip install -e .[dev]
```

The editable install is important because commands like this use imports from `src/`:

```bash
python -m src.evaluation.evaluate_clap
```

### 3. Run setup script

```bash
bash setup.sh
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

## Create Metadata From Local Media

You can create Reflectra metadata JSONL from your own audio and image folders with an OpenAI-compatible multimodal model:

```bash
python -m src.datasets.create_metadata \
  --audio_path data/music \
  --image_path data/my_images \
  --source_dataset my_local_media
```

This writes:

```text
data/metadata/custom_audio_metadata.jsonl
data/metadata/custom_image_metadata.jsonl
```

Use custom output names when needed:

```bash
python -m src.datasets.create_metadata \
  --audio-path /path/to/music \
  --audio-output my_music_metadata.jsonl
```

The script uses `[benchmark].model` from `configs/reflectra.toml` by default, unless `[metadata].model` or `--model` is set. Images and audio are sent to the LLM, so the selected model/server must support the corresponding media input type.

For long audio files, only a middle excerpt is sent to the LLM by default:

```bash
python -m src.datasets.create_metadata \
  --audio_path data/music \
  --audio_clip_seconds 15
```

Use `--audio_clip_seconds 0` to send full audio files.

To benchmark only this generated metadata, pass the files explicitly:

```bash
python -m src.benchmark.create_benchmark \
  --image_metadata data/metadata/custom_image_metadata.jsonl \
  --audio_metadata data/metadata/custom_audio_metadata.jsonl
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

It also writes `data/benchmark/benchmark_hf.parquet` by default. That table duplicates each scored pair into one row and stores `image` and `audio` with Hugging Face `Image` and `Audio` features, backed by embedded bytes so they can be previewed or played after upload. Use `--no-write_hf` to skip that larger viewer-friendly file.

---

## Create CLAP LLM Benchmark

To check whether CLAP is suitable for music captions before fine-tuning, create a caption-to-audio benchmark with LLM-graded positives and random negatives:

```bash
python -m src.benchmark.create_clap_benchmark \
  --audio_metadata data/metadata/custom_audio_metadata.jsonl \
  --audio_samples 100 \
  --queries_per_audio 1 \
  --num_negatives 9
```

Each query uses a known caption/audio pair as the positive and random audio files as negatives. The LLM scores every candidate audio from 0 to 10 and writes:

```text
data/benchmark/clap_llm_benchmark.jsonl
data/benchmark/clap_llm_benchmark.csv
data/benchmark/clap_llm_benchmark.parquet
data/benchmark/clap_llm_benchmark_manifest.json
```

The audio sent to the LLM is clipped to a middle 15-second excerpt by default. Use `--audio_clip_seconds 0` for full files.

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
  --benchmark data/benchmark/clap_llm_benchmark.jsonl \
  --audio_metadata data/metadata/custom_audio_metadata.jsonl
```

The script computes CLAP text-to-audio rankings for each benchmark caption over that query's positive and random-negative candidate audios.

Metrics:

```text
- ndcg@1, ndcg@5, ndcg@10
- mrr
- mAP
- recall@1, recall@5, recall@10
- precision@1, precision@5, precision@10
```

NDCG uses the LLM 0..10 score as graded relevance. MRR, mAP, recall, and precision treat scores above `--relevance-threshold` as relevant.

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
python -m src.evaluation.evaluate_reflectra \
  --benchmark data/benchmark/image_audio_scores.parquet \
  --image_metadata data/metadata/custom_image_metadata.jsonl \
  --audio_metadata data/metadata/custom_audio_metadata.jsonl \
  --checkpoint checkpoints/reflectra.pt
```

The evaluator loads image/audio paths from metadata, computes Reflectra image-to-audio similarities, and compares them with the LLM benchmark scores.

Metrics:

```text
- ndcg@1, ndcg@5, ndcg@10
- mrr
- mAP
- recall@1, recall@5, recall@10
- precision@1, precision@5, precision@10
```

NDCG uses the benchmark LLM score as graded relevance. MRR, mAP, recall, and precision treat scores above `--relevance-threshold` as relevant.

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

## Metrics

CLAP evaluation uses the LLM-scored caption-to-audio benchmark. CLIP evaluation uses CxC graded sparse relevance. Reflectra evaluation uses the created image/audio benchmark:

```text
python -m src.evaluation.evaluate_clap
python -m src.evaluation.evaluate_clip
python -m src.evaluation.evaluate_reflectra
```

The scripts report:

```text
ndcg@K = ranking quality with relevant items rewarded near the top
mrr = reciprocal rank of the first relevant target
mAP = mean average precision across queries
recall@K = relevant retrieved items in top K / total relevant items
precision@K = relevant retrieved items in top K / K
```

CLAP NDCG uses LLM 0..10 scores as graded relevance. CLIP NDCG uses raw CxC scores as graded relevance. Reflectra NDCG uses the image/audio benchmark LLM scores as graded relevance.

Recommended reporting:

```text
CLAP LLM benchmark text-to-audio:
- ndcg@1
- ndcg@5
- ndcg@10
- mrr
- mAP
- recall@1, recall@5, recall@10
- precision@1, precision@5, precision@10

CLIP CxC:
- ndcg@1
- ndcg@5
- ndcg@10
- mrr
- mAP
- recall@1, recall@5, recall@10
- precision@1, precision@5, precision@10

Reflectra image-to-audio:
- ndcg@1, ndcg@5, ndcg@10
- mrr, mAP
- recall@1, recall@5, recall@10
- precision@1, precision@5, precision@10

Image-to-music final system, later:
- Precision@10 by mood/genre metadata
- NDCG@10
- Human rating from 1 to 5
- Percentage of recommendations rated 4 or 5
```

---

## Recommended Experiments

### Experiment 1: CLAP zero-shot baseline

```bash
python -m src.evaluation.evaluate_clap \
  --benchmark data/benchmark/clap_llm_benchmark.jsonl \
  --audio_metadata data/metadata/custom_audio_metadata.jsonl
```

Purpose:

```text
Measure pretrained CLAP against LLM-scored music-caption relevance before fine-tuning.
```

### Experiment 2: CLIP CxC baseline

```bash
python -m src.evaluation.evaluate_clip \
  --metadata data/metadata/coco_karpathy_cxc_sits_val.json \
  --image-root data/coco_images \
  --max-images 1000
```

Purpose:

```text
Measure whether CLIP aligns images with captions using CxC graded relevance labels.
```

### Experiment 3: Reflectra image-to-audio benchmark

```bash
python -m src.evaluation.evaluate_reflectra \
  --benchmark data/benchmark/image_audio_scores.parquet \
  --image_metadata data/metadata/custom_image_metadata.jsonl \
  --audio_metadata data/metadata/custom_audio_metadata.jsonl \
  --checkpoint checkpoints/reflectra.pt
```

Purpose:

```text
Measure the trained image-to-CLAP projection against LLM-scored image/audio relevance.
```

---

## Development Order

Recommended order:

```text
1. Install the project with pyproject.toml.
2. Start Qdrant.
3. Download a small sample from each dataset.
4. Create local metadata or download dataset metadata.
5. Create the CLAP LLM benchmark and evaluate CLAP.
6. Prepare CxC and evaluate CLIP.
7. Create the image/audio benchmark.
8. Train image-to-CLAP projection only after baselines work.
9. Evaluate Reflectra on the image/audio benchmark.
10. Put local songs in `data/music` and index CLAP audio embeddings in Qdrant.
11. Scale to a larger music library.
12. Build final image-to-song demo.
```

---

## Notes

- Do not train from scratch at the beginning.
- Always evaluate pretrained CLAP first.
- Keep MusicCaps and Song Describer mainly for clean testing.
- Use MTG-Jamendo for training, validation, and scale experiments.
- Use AudioSet carefully because it is weakly labeled general audio, not a clean music-caption dataset.
- Use COCO and Flickr30k for image-caption evaluation.
- Use EmoSet for image mood alignment through music-oriented captions.
- Use Qdrant only after small local dense retrieval works correctly.

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
