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

For image datasets, `audio_id/audio_path` are replaced by `image_id/image_path`. Loaders always expand `captions` into one training/evaluation record per caption.

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
│   │   └── preprocessing/
│   ├── evaluation/
│   │   ├── evaluate_clap.py
│   │   ├── evaluate_clip.py
│   │   └── evaluate_clip_cxc.py
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
python -m src.datasets.downloaders.download_coco --splits val2014
python -m src.datasets.downloaders.download_flickr30k --number 100
python -m src.datasets.downloaders.download_emoset --split train --number 100
python -m src.datasets.downloaders.download_emoset --split test --number 100
```

Start small first. Verify metadata and file paths before downloading large datasets.

COCO has large official zip files. The downloader shows byte progress while downloading and file progress while extracting:

```bash
python -m src.datasets.downloaders.download_coco --splits train2014 val2014
python -m src.datasets.downloaders.download_coco --splits test2014 --skip-metadata
```

For the CxC SITS CLIP benchmark, prepare the merged Karpathy+CxC metadata:

```bash
python -m src.datasets.downloaders.download_coco --skip-images --prepare-cxc --cxc-split all
```

This writes files such as:

```text
data/metadata/coco_karpathy_cxc_sits_val.json
data/metadata/coco_karpathy_cxc_sits_test.json
```

---

## Evaluation

There are two evaluation styles:

1. Dense local evaluation for small and medium benchmark subsets.
2. Qdrant vector search for indexing and retrieving from a local music library.

Dense evaluation computes a similarity matrix between the sampled media items and captions. Relevance is stored sparsely, so the scripts do not create a dense zero-filled relevance matrix.

---

### 1. Evaluate CLAP: Text-to-Audio

This evaluates whether CLAP can retrieve the correct audio from a text description.

```bash
python -m src.evaluation.evaluate_clap --max-samples 1000
```

Use selected dataset fractions:

```bash
python -m src.evaluation.evaluate_clap \
  --dataset-fractions "google/MusicCaps=1.0,agkphysics/AudioSet=0.5,rkstgr/mtg-jamendo=0.8" \
  --max-samples 50000
```

Use exact dataset counts:

```bash
python -m src.evaluation.evaluate_clap \
  --dataset-counts "google/MusicCaps=5000,rkstgr/mtg-jamendo=10000,agkphysics/AudioSet=20000"
```

The script computes:

```text
audio → text
text → audio
```

Typical metrics:

```text
Binary retrieval:
- hit@1, hit@5, hit@10
- recall@1, recall@5, recall@10
- mrr
- median_rank
- mean_rank

Binary NDCG:
- ndcg@1, ndcg@5, ndcg@10

Balanced pairwise:
- hit@1, hit@5, hit@10
- mrr
- median_rank
- mean_rank
```

Each audio file may have multiple captions. All captions attached to the same audio item are treated as relevant positives.

---

### 2. Evaluate CLIP: Image-to-Text

This evaluates whether CLIP retrieves matching captions for images.

```bash
python -m src.evaluation.evaluate_clip --max-samples 1000
```

Use selected dataset fractions:

```bash
python -m src.evaluation.evaluate_clip \
  --dataset-fractions "coco_karpathy=0.5,nlphuji/flickr30k=0.8,LiangJian24/EmoSet=1.0" \
  --max-samples 50000
```

Use exact dataset counts:

```bash
python -m src.evaluation.evaluate_clip \
  --dataset-counts "coco_karpathy=5000,nlphuji/flickr30k=1000,LiangJian24/EmoSet=1000"
```

The script computes:

```text
image → text
text → image
```

Metrics are the same grouped binary metrics used by CLAP:

```text
Binary retrieval:
- hit@1, hit@5, hit@10
- recall@1, recall@5, recall@10
- mrr
- median_rank
- mean_rank

Binary NDCG:
- ndcg@1, ndcg@5, ndcg@10

Balanced pairwise:
- hit@1, hit@5, hit@10
- mrr
- median_rank
- mean_rank
```

Each image may have multiple captions. All captions attached to the same image are treated as relevant positives.

---

### 3. Evaluate CLIP on CxC

CxC is different from the normal CLIP script because it has graded relevance labels. This evaluation stores CxC SITS scores sparsely instead of building a dense zero-filled relevance matrix.

```bash
python -m src.evaluation.evaluate_clip_cxc \
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
Graded:
- ndcg@1, ndcg@5, ndcg@10
- ndcg@1/5/10 with exponential gain

Binary view of CxC labels:
- hit@1, hit@5, hit@10
- recall@1, recall@5, recall@10
- mrr
- median_rank
- mean_rank

Balanced pairwise:
- hit@1, hit@5, hit@10
- mrr
- median_rank
- mean_rank
```

For binary metrics, every CxC score `> 0` is treated as relevant.

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

Normal CLIP and CLAP evaluation uses grouped sparse relevance:

```text
python -m src.evaluation.evaluate_clap
python -m src.evaluation.evaluate_clip
```

In this setup, each media item can have one or more captions. The scripts build unique media targets, unique text targets, and sparse positive links:

```text
hit@K = 1 if at least one relevant target appears in top K, else 0
recall@K = relevant retrieved items in top K / total relevant items
mrr = reciprocal rank of the first relevant target
median_rank = median first-relevant rank
mean_rank = mean first-relevant rank
ndcg@K = ranking quality with relevant items rewarded near the top
```

CxC evaluation uses graded sparse relevance:

```text
python -m src.evaluation.evaluate_clip_cxc
```

CxC SITS has graded scores. For NDCG it uses the raw CxC score as relevance. For binary retrieval metrics, every CxC score `> 0` is treated as relevant.

Balanced pairwise metrics are also reported. Each positive edge is evaluated against one positive plus up to the requested number of sampled negatives. If the target set is smaller than the requested candidate count, the script reports the actual candidate count used.

Recommended reporting:

```text
CLAP grouped text-to-audio / audio-to-text:
- hit@1, hit@5, hit@10
- recall@1, recall@5, recall@10
- ndcg@1, ndcg@5, ndcg@10
- mrr, median_rank, mean_rank

CLIP grouped image-to-text / text-to-image:
- hit@1, hit@5, hit@10
- recall@1, recall@5, recall@10
- ndcg@1, ndcg@5, ndcg@10
- mrr, median_rank, mean_rank

CLIP CxC:
- ndcg@1
- ndcg@5
- ndcg@10
- hit@1, hit@5, hit@10
- recall@1, recall@5, recall@10
- mrr

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
python -m src.evaluation.evaluate_clap --max-samples 1000
```

Purpose:

```text
Measure pretrained CLAP before fine-tuning.
```

### Experiment 2: CLIP image-text baseline

```bash
python -m src.evaluation.evaluate_clip --max-samples 1000
```

Purpose:

```text
Measure whether CLIP aligns images with captions and music-oriented mood captions.
```

---

## Development Order

Recommended order:

```text
1. Install the project with pyproject.toml.
2. Start Qdrant.
3. Download a small sample from each dataset.
4. Run CLAP dense evaluation.
5. Run CLIP dense evaluation.
6. Put local songs in `data/music`.
7. Index local music CLAP audio embeddings in Qdrant.
8. Scale to a larger music library.
9. Train image-to-CLAP projection only after baselines work.
10. Build final image-to-song demo.
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
- MTG-Jamendo Hugging Face mirror: https://huggingface.co/datasets/rkstgr/mtg-jamendo
- MTG-Jamendo original project: https://github.com/MTG/mtg-jamendo-dataset
- AudioSet Hugging Face mirror: https://huggingface.co/datasets/agkphysics/AudioSet
- AudioSet original project: https://research.google.com/audioset/

Image-text / image-mood datasets:

- Flickr30k Hugging Face mirror: https://huggingface.co/datasets/nlphuji/flickr30k
- EmoSet: https://huggingface.co/datasets/LiangJian24/EmoSet
- COCO dataset downloads: https://cocodataset.org/#download

- CxC SITS labels: https://github.com/google-research-datasets/Crisscrossed-Captions
- COCO 2014 train images: http://images.cocodataset.org/zips/train2014.zip
- COCO 2014 validation images: http://images.cocodataset.org/zips/val2014.zip
- COCO 2014 test images: http://images.cocodataset.org/zips/test2014.zip
- Karpathy COCO caption metadata: http://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip
