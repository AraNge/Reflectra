import os
import uuid
import importlib
from typing import Any, Dict, List, Optional

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

_import_module = importlib.import_module


def _protobuf_safe_import(name: str, package: str | None = None):
    if name == "google._upb._message":
        raise ImportError(name)
    return _import_module(name, package)


importlib.import_module = _protobuf_safe_import
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
    )
finally:
    importlib.import_module = _import_module


def get_qdrant_client(
    url: str = "http://localhost:6333",
    api_key: Optional[str] = None,
) -> QdrantClient:
    return QdrantClient(
        url=url,
        api_key=api_key,
    )


def create_collection_if_not_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: Distance = Distance.COSINE,
) -> None:
    if client.collection_exists(collection_name=collection_name):
        print(f"[INFO] Collection already exists: {collection_name}")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance,
        ),
    )

    print(f"[INFO] Created collection: {collection_name}")


def stable_uuid(value: str) -> str:
    """
    Qdrant point ids can be integers or UUID strings.
    This converts any stable string into a deterministic UUID.
    """

    return str(uuid.uuid5(uuid.NAMESPACE_DNS, value))


def upsert_vectors(
    client: QdrantClient,
    collection_name: str,
    ids: List[str],
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
    batch_size: int = 256,
) -> None:
    if len(ids) != len(vectors) or len(ids) != len(payloads):
        raise ValueError(
            "ids, vectors, and payloads must have the same length: "
            f"ids={len(ids)}, vectors={len(vectors)}, payloads={len(payloads)}"
        )

    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))

        points = [
            PointStruct(
                id=stable_uuid(ids[i]),
                vector=vectors[i],
                payload={
                    **payloads[i],
                    "original_id": ids[i],
                },
            )
            for i in range(start, end)
        ]

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

        print(f"[INFO] Upserted {end}/{len(ids)} points")


def search_vector(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int = 10,
    with_vectors: bool = False,
):
    """
    Uses query_points because newer Qdrant client versions prefer it.
    """

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=with_vectors,
    )

    return response.points
