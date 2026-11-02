# Enterprise Multimodal Document & Diagram Q&A Assistant

A zero-cost, local-first Multimodal RAG (Retrieval-Augmented Generation) system built to ingest, index, and query technical documents, tables, and architectural diagrams.

Powered by **Gemini 2.5 Flash**, **Qdrant (Local Disk Storage)**, **FastEmbed (`bge-small-en-v1.5`)**, and **Streamlit**.

---

## 🏗️ Architecture & Data Flow

```text
[ Technical Diagram / Doc ] 
            │
            ▼
 [ Gemini 2.5 Flash Vision ]  ── (Compressed 800px Base64 Payload + thinkingBudget: 0)
            │
            ▼
[ Text / Diagram Summaries ]
            │
            ▼
  [ FastEmbed Model ]         ── (384-dim Dense Vector Embeddings: BAAI/bge-small-en-v1.5)
            │
            ▼
 [ Local Qdrant Storage ]     ── (Single-Process Connection via @st.cache_resource)
            │
            ▼
   [ Streamlit Chat UI ]      ── (Hybrid Retrieval + Context Synthesis + Source Citations)
   