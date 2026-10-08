import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

def parse_pdf_document(file_path: str) -> List[Dict[str, Any]]:
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        raise ValueError("Missing LLAMA_CLOUD_API_KEY in .env file!")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at: {file_path}")

    print(f"[INFO] Parsing PDF: {file_path}")
    
    parser = LlamaParse(
        result_type="markdown",
        verbose=True,
        extract_charts=True
    )

    documents = parser.load_data(file_path)
    
    results = []
    for idx, doc in enumerate(documents):
        results.append({
            "chunk_id": f"chunk_{idx}",
            "text": doc.text,
            "source": file_path
        })

    print(f"[SUCCESS] Parsed {len(results)} chunks.")
    return results

if __name__ == "__main__":
    test_pdf = "data/sample_docs/sample.pdf"
    if os.path.exists(test_pdf):
        chunks = parse_pdf_document(test_pdf)
        print("\n--- First Chunk Output ---")
        print(chunks[0]["text"][:300] if chunks else "Empty PDF")
    else:
        print(f"[NOTICE] Add a sample PDF at '{test_pdf}' to test.")
