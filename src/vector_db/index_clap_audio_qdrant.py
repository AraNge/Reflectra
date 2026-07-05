import argparse
from pathlib import Path

from tqdm import tqdm

from src.datasets.loaders.audio_metadata import AudioMetadataLoader
from src.vector_db.qdrant_store import (
    get_qdrant_client,
    create_collection_if_not_exists,
    upsert_vectors,
)
from src.models.clap_encoder import PretrainedCLAPEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


DEFAULT_AUDIO_METADATA_PATHS = [
    DATA_DIR / "musiccaps_metadata.jsonl",
    DATA_DIR / "audioset_metadata.jsonl",
    DATA_DIR / "song_describer_metadata.jsonl",
    DATA_DIR / "mtg_jamendo_metadata" / "mtg_jamendo_train_metadata.jsonl",
    DATA_DIR / "mtg_jamendo_metadata" / "mtg_jamendo_validation_metadata.jsonl",
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--collection-name", type=str, default="reflectra_audio_clap")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--model-name", type=str, default="laion/clap-htsat-unfused")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--vector-size", type=int, default=512)

    args = parser.parse_args()

    loader = AudioMetadataLoader(
        metadata_paths=DEFAULT_AUDIO_METADATA_PATHS,
        project_root=PROJECT_ROOT,
        require_audio_exists=True,
    )

    records = loader.load()

    if args.max_samples is not None:
        records = records[: args.max_samples]

    print(f"[INFO] Audio records loaded: {len(records)}")

    client = get_qdrant_client(url=args.qdrant_url)

    create_collection_if_not_exists(
        client=client,
        collection_name=args.collection_name,
        vector_size=args.vector_size,
    )

    model = PretrainedCLAPEncoder(
        model_name=args.model_name,
        freeze=True,
    )

    for start in tqdm(range(0, len(records), args.batch_size), desc="Indexing audio"):
        end = start + args.batch_size
        batch = records[start:end]

        audio_paths = [r["audio_path"] for r in batch]

        audio_embeddings = model.encode_audio(audio_paths)
        audio_embeddings = audio_embeddings.cpu().numpy().tolist()

        ids = [
            f'{r["source_dataset"]}:{r["audio_id"]}'
            for r in batch
        ]

        payloads = [
            {
                "audio_id": r["audio_id"],
                "audio_path": r["audio_path"],
                "text": r["text"],
                "source_dataset": r["source_dataset"],
                "split": r["split"],
            }
            for r in batch
        ]

        upsert_vectors(
            client=client,
            collection_name=args.collection_name,
            ids=ids,
            vectors=audio_embeddings,
            payloads=payloads,
            batch_size=args.batch_size,
        )

    print("[INFO] Done indexing CLAP audio embeddings.")


"""
python -m src.vector_db.index_clap_audio_qdrant \
  --collection-name reflectra_audio_clap \
  --max-samples 10000

python -m src.vector_db.index_clap_audio_qdrant \
  --collection-name reflectra_audio_clap \
  --max-samples 10000
"""

if __name__ == "__main__":
    main()