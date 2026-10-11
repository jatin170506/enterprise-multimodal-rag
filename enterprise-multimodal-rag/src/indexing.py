import os
import sys
import warnings
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    SparseVectorParams, 
    SparseIndexParams,
    PayloadSchemaType
)

load_dotenv(override=True)

# Suppress local mode payload index warning
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

COLLECTION_NAME = "multimodal_rag"

def get_qdrant_client(storage_path: str = "qdrant_storage") -> QdrantClient:
    """
    Returns a QdrantClient instance using local persistent disk storage.
    No Docker or external server required.
    """
    return QdrantClient(path=storage_path)


def init_qdrant_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME):
    """
    Initializes a Qdrant collection configured for 384-dim local embeddings (bge-small-en-v1.5)
    and Hybrid Search (Dense + Sparse) with metadata payload indexes.
    """
    print(f"[INFO] Initializing Qdrant collection: '{collection_name}'...")

    # Re-create collection if missing or misconfigured
    if client.collection_exists(collection_name):
        print(f"[INFO] Collection '{collection_name}' already exists. Re-creating for clean status...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            # 384 dimensions matches FastEmbed BAAI/bge-small-en-v1.5
            "dense": VectorParams(size=384, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            # Sparse vectors for exact keyword matching
            "sparse": SparseVectorParams(index=SparseIndexParams())
        }
    )
    print(f"[SUCCESS] Collection '{collection_name}' created successfully.")

    # Create payload field indexes for rapid filtering
    print("[INFO] Creating payload indexes for metadata filtering...")
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="type",
            field_schema=PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD
        )
    except Exception:
        pass  # Payload indexes are optional in local storage mode
    
    print("[SUCCESS] Vector storage initialization complete.")


if __name__ == "__main__":
    print("=== Day 3: Qdrant Vector DB Storage Initialization Test ===")
    qdrant_client = None
    try:
        qdrant_client = get_qdrant_client()
        init_qdrant_collection(qdrant_client)
        
        # Verify collection status
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        print("\n--- Collection Status ---")
        print(f"Collection Name: {COLLECTION_NAME}")
        print(f"Status: {collection_info.status}")
        # Updated to points_count for current Qdrant API
        print(f"Points Count: {collection_info.points_count}")
    except Exception as e:
        print(f"[ERROR] Qdrant initialization failed: {e}", file=sys.stderr)
    finally:
        if qdrant_client:
            qdrant_client.close()