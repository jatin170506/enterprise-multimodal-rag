import os
import sys
import requests
from typing import List, Dict, Any

# Ensure project root is in Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from src.indexing import get_qdrant_client, COLLECTION_NAME

load_dotenv(override=True)

class RAGQueryEngine:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self.client: QdrantClient = get_qdrant_client()
        print("[INFO] Loading FastEmbed query encoder (BAAI/bge-small-en-v1.5)...")
        self.embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env file!")

    def embed_query(self, query: str) -> List[float]:
        """Generates 384-dim vector representation for input query."""
        embeddings = list(self.embedding_model.embed([query]))
        return embeddings[0].tolist()

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Searches Qdrant dense vector index for top_k relevant context chunks."""
        query_vector = self.embed_query(query)
        
        # Search Qdrant dense vector space
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            limit=top_k
        ).points

        context_chunks = []
        for point in search_results:
            context_chunks.append({
                "score": point.score,
                "content": point.payload.get("content", ""),
                "type": point.payload.get("type", "text"),
                "source": point.payload.get("source", "unknown"),
                "page_number": point.payload.get("page_number", 1),
                "image_path": point.payload.get("image_path", "")
            })
        
        return context_chunks

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Sends query and retrieved context to Gemini 2.5 Flash via REST API."""
        if not context_chunks:
            return "No relevant context found in database to answer this question."

        formatted_context = ""
        for i, chunk in enumerate(context_chunks, 1):
            formatted_context += (
                f"\n--- Context Item [{i}] (Type: {chunk['type']}, Source: {chunk['source']}, Page: {chunk['page_number']}) ---\n"
                f"{chunk['content']}\n"
            )

        prompt = (
            "You are an expert enterprise technical Q&A assistant.\n"
            "Answer the user query based ONLY on the provided context below. "
            "If the context contains diagram summaries or tables, synthesize them directly.\n"
            "Cite your source document and page numbers in your answer.\n\n"
            f"CONTEXT:\n{formatted_context}\n\n"
            f"USER QUERY: {query}"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }

        response = requests.post(url, headers=headers, json=payload)
        res_json = response.json()

        if response.status_code != 200:
            error_msg = res_json.get("error", {}).get("message", str(res_json))
            raise RuntimeError(f"Gemini API Error ({response.status_code}): {error_msg}")

        return res_json["candidates"][0]["content"]["parts"][0]["text"]

    def query(self, user_query: str, top_k: int = 3) -> Dict[str, Any]:
        """Executes full RAG flow: Retrieve -> Format -> Synthesize."""
        print(f"\n[INFO] Processing Query: '{user_query}'")
        retrieved_chunks = self.retrieve_context(user_query, top_k=top_k)
        print(f"[INFO] Retrieved {len(retrieved_chunks)} context chunk(s) from Qdrant.")
        
        answer = self.generate_answer(user_query, retrieved_chunks)
        return {
            "query": user_query,
            "answer": answer,
            "sources": retrieved_chunks
        }


if __name__ == "__main__":
    print("=== Day 5: Hybrid Search & Reranking Query Engine Test ===")
    
    test_query = "What database is used for storing vectors in the system architecture?"
    
    engine = None
    try:
        engine = RAGQueryEngine()
        result = engine.query(test_query)
        
        print("\n" + "="*50)
        print(f"QUERY: {result['query']}")
        print("="*50)
        print(f"\nGENERATED ANSWER:\n{result['answer']}")
        
        print("\n--- RETRIEVED SOURCES ---")
        for idx, src in enumerate(result['sources'], 1):
            print(f"[{idx}] Score: {src['score']:.4f} | Type: {src['type']} | Source: {src['source']} (Page {src['page_number']})")
            
    except Exception as e:
        print(f"[ERROR] Query Engine execution failed: {e}", file=sys.stderr)
    finally:
        if engine and hasattr(engine, "client") and engine.client:
            engine.client.close()