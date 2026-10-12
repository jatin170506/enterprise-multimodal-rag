import os
import sys
import uuid
from typing import List, Dict, Any

# Ensure project root is in Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.indexing import get_qdrant_client, COLLECTION_NAME

load_dotenv(override=True)

class DocumentIndexer:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self.client: QdrantClient = get_qdrant_client()
        # BAAI/bge-small-en-v1.5 generates 384-dim dense vectors locally at zero cost
        print("[INFO] Initializing FastEmbed model (BAAI/bge-small-en-v1.5)...")
        self.embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings locally using FastEmbed."""
        embeddings = list(self.embedding_model.embed(texts))
        return [e.tolist() for e in embeddings]

    def index_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embeds and indexes document chunks into Qdrant.
        Expects chunk structure:
        {
            "content": str,
            "type": "text" | "table" | "diagram_summary",
            "source": str,
            "page_number": int,
            "image_path": Optional[str]
        }
        """
        if not chunks:
            print("[NOTICE] No chunks provided for indexing.")
            return 0

        texts = [c["content"] for c in chunks]
        print(f"[INFO] Generating 384-dim embeddings for {len(texts)} chunk(s)...")
        vectors = self.generate_embeddings(texts)

        points = []
        for idx, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            payload = {
                "content": chunk["content"],
                "type": chunk.get("type", "text"),
                "source": chunk.get("source", "unknown"),
                "page_number": chunk.get("page_number", 1),
                "image_path": chunk.get("image_path", "")
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector={"dense": vectors[idx]},
                    payload=payload
                )
            )

        print(f"[INFO] Upserting {len(points)} point(s) into Qdrant collection '{self.collection_name}'...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print("[SUCCESS] Multi-vector document indexing complete.")
        return len(points)


if __name__ == "__main__":
    print("=== Day 4: Multi-Vector Indexing Pipeline Test ===")
    
    sample_chunks = [
        {
            "content": "The system architecture consists of a React frontend, Next.js API gateway, and Qdrant vector database.",
            "type": "text",
            "source": "architecture_doc.pdf",
            "page_number": 1
        },
        {
            "content": "DIAGRAM SUMMARY: Flowchart showing incoming HTTP requests routing through authentication middleware into Qdrant vector storage.",
            "type": "diagram_summary",
            "source": "architecture_doc.pdf",
            "page_number": 2,
            "image_path": "data/extracted_images/sample_diagram.png"
        }
    ]

    indexer = None
    try:
        indexer = DocumentIndexer()
        count = indexer.index_documents(sample_chunks)
        
        # Verify collection contents
        client = indexer.client
        info = client.get_collection(COLLECTION_NAME)
        print("\n--- Updated Collection Status ---")
        print(f"Collection Name: {COLLECTION_NAME}")
        print(f"Status: {info.status}")
        print(f"Total Points Indexed: {info.points_count}")
    except Exception as e:
        print(f"[ERROR] Indexing failed: {e}", file=sys.stderr)
    finally:
        if indexer and hasattr(indexer, "client") and indexer.client:
            indexer.client.close()