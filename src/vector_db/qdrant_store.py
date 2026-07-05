import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


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
    assert len(ids) == len(vectors) == len(payloads)

    for start in range(0, len(ids), batch_size):
        end = start + batch_size

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
):
    """
    Uses query_points because newer Qdrant client versions prefer it.
    """

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    return response.points