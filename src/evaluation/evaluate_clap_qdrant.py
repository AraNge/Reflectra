import argparse
import json
from pathlib import Path
from tqdm import tqdm
from src.datasets.loaders.audio_metadata import AudioMetadataLoader
from src.models.clap_encoder import PretrainedCLAPEncoder
from src.vector_db.qdrant_store import get_qdrant_client, search_vector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


DEFAULT_AUDIO_METADATA_PATHS = [
    DATA_DIR / "musiccaps_metadata.jsonl",
    DATA_DIR / "audioset_metadata.jsonl",
    DATA_DIR / "song_describer_metadata.jsonl",
    DATA_DIR / "mtg_jamendo_metadata" / "mtg_jamendo_validation_metadata.jsonl",
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--collection-name", type=str, default="reflectra_audio_clap")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--model-name", type=str, default="laion/clap-htsat-unfused")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--split", type=str, default=None)

    args = parser.parse_args()

    loader = AudioMetadataLoader(
        metadata_paths=DEFAULT_AUDIO_METADATA_PATHS,
        project_root=PROJECT_ROOT,
        require_audio_exists=True,
    )

    records = loader.load()

    if args.split:
        records = [
            r for r in records
            if r["split"] == args.split
        ]

    if args.max_samples is not None:
        records = records[: args.max_samples]

    print(f"[INFO] Evaluation records: {len(records)}")

    client = get_qdrant_client(url=args.qdrant_url)

    model = PretrainedCLAPEncoder(
        model_name=args.model_name,
        freeze=True,
    )

    eval_results = []

    for start in tqdm(range(0, len(records), args.batch_size), desc="Evaluating CLAP Qdrant"):
        end = start + args.batch_size
        batch = records[start:end]

        texts = [r["text"] for r in batch]
        text_embeddings = model.encode_text(texts)
        text_embeddings = text_embeddings.cpu().numpy().tolist()

        for record, query_vector in zip(batch, text_embeddings):
            query_id = f'{record["source_dataset"]}:{record["audio_id"]}'
            relevant_ids = {query_id}

            points = search_vector(
                client=client,
                collection_name=args.collection_name,
                query_vector=query_vector,
                limit=args.top_k,
            )

    output = {
        "task": "text_to_audio",
        "num_queries": len(eval_results),
        "collection_name": args.collection_name,
        "model_name": args.model_name,
        "metrics": None, # TODO
    }

    print(json.dumps(output, indent=2))

    output_dir = PROJECT_ROOT / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "clap_qdrant_eval.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[INFO] Saved results to: {output_path}")


"""
python -m src.evaluation.evaluate_clap_qdrant \
  --collection-name reflectra_audio_clap \
  --max-samples 1000 \
  --top-k 50
"""

if __name__ == "__main__":
    main()