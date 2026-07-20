import argparse
import uuid
from pathlib import Path

from qdrant_client.models import Distance, PointStruct, VectorParams

from src.chunking import chunk_document
from src.clients import qdrant_client
from src.config import settings
from src.embeddings import embed_document
from src.parsers import PARSERS, parse_file
from src.tracing import node_span


def ensure_collection() -> None:
    existing = {c.name for c in qdrant_client.get_collections().collections}
    if settings.qdrant_docs_collection not in existing:
        qdrant_client.create_collection(
            collection_name=settings.qdrant_docs_collection,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )
    if settings.qdrant_cache_collection not in existing:
        qdrant_client.create_collection(
            collection_name=settings.qdrant_cache_collection,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )


def ingest_directory(directory: Path, source_authority: str) -> int:
    count = 0
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in PARSERS:
            continue

        with node_span("ingest_file", path=str(path)):
            text = parse_file(path)
            if not text.strip():
                continue

            chunks = chunk_document(
                text,
                base_metadata={"source_path": str(path), "source_authority": source_authority},
            )

            points = []
            for chunk in chunks:
                vector = embed_document(chunk["text"])
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={"text": chunk["text"], "metadata": chunk["metadata"]},
                    )
                )

            if points:
                qdrant_client.upsert(collection_name=settings.qdrant_docs_collection, points=points)
                count += len(points)

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Terraform docs into Qdrant.")
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--community-dir", type=Path, required=True)
    args = parser.parse_args()

    ensure_collection()

    official_count = ingest_directory(args.official_dir, "official")
    print(f"ingested {official_count} chunks from {args.official_dir} (official)")

    community_count = ingest_directory(args.community_dir, "community")
    print(f"ingested {community_count} chunks from {args.community_dir} (community)")


if __name__ == "__main__":
    main()