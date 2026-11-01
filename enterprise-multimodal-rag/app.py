import os
import sys
import streamlit as st
from PIL import Image

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
from src.indexing import get_qdrant_client, init_qdrant_collection, COLLECTION_NAME
from src.indexer import DocumentIndexer
from src.query_engine import RAGQueryEngine
from src.summarizer import generate_diagram_summary

load_dotenv(override=True)

st.set_page_config(
    page_title="Multimodal Enterprise RAG",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Enterprise Document & Diagram Q&A Assistant")
st.caption("Powered by Gemini 2.5 Flash, Qdrant, and FastEmbed (Local & Zero Cost)")

# --- 1. Persistent Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "latest_summary" not in st.session_state:
    st.session_state.latest_summary = None

# --- 2. Cache Heavy Resources in Memory ---
@st.cache_resource
def load_qdrant_client():
    return get_qdrant_client()

client = load_qdrant_client()

@st.cache_resource
def load_indexer():
    return DocumentIndexer(client=client)

@st.cache_resource
def load_query_engine():
    return RAGQueryEngine(client=client)

# --- 3. SIDEBAR: Controls & Artifact Ingestion ---
with st.sidebar:
    st.header("⚙️ System Control & Ingestion")
    
    try:
        col_info = client.get_collection(COLLECTION_NAME)
        st.success(f"**Collection Status:** {col_info.status}")
        st.info(f"**Total Indexed Points:** {col_info.points_count}")
    except Exception:
        st.warning("Collection not initialized.")

    if st.button("🗑️ Reset Qdrant Database", use_container_width=True):
        init_qdrant_collection(client)
        st.session_state.latest_summary = None
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("📥 Upload & Index Artifacts")
    
    uploaded_file = st.file_uploader(
        "Upload Diagram / Schematic (PNG, JPG)", 
        type=["png", "jpg", "jpeg"],
        key="diagram_uploader"
    )
    
    doc_source_name = st.text_input("Source Document Name", value="manual_upload.pdf")
    page_num = st.number_input("Page Number", min_value=1, value=1)

    # Ingestion Trigger
    if uploaded_file and st.button("Process & Index Artifact", use_container_width=True):
        os.makedirs("data/extracted_images", exist_ok=True)
        save_path = os.path.join("data/extracted_images", uploaded_file.name)
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Analyzing diagram with Gemini 2.5 Flash..."):
            summary_text = generate_diagram_summary(save_path)
            
        with st.spinner("Embedding & indexing into Qdrant..."):
            indexer = load_indexer()
            chunk = [{
                "content": f"DIAGRAM SUMMARY: {summary_text}",
                "type": "diagram_summary",
                "source": doc_source_name,
                "page_number": int(page_num),
                "image_path": save_path
            }]
            indexer.index_documents(chunk)

        # Store in session state so it stays visible permanently after rerun
        st.session_state.latest_summary = {
            "summary": summary_text,
            "source": doc_source_name,
            "page": int(page_num),
            "image_path": save_path
        }
        st.success("Indexing complete!")
        st.rerun()

    # --- Persistent Display Outside Button Execution ---
    if st.session_state.latest_summary:
        st.divider()
        st.subheader("📋 Last Indexed Summary")
        st.caption(f"Source: `{st.session_state.latest_summary['source']}` (Page {st.session_state.latest_summary['page']})")
        st.info(st.session_state.latest_summary["summary"])

# --- 4. MAIN PANEL: Chat Interface ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📌 Retrieved Sources & Metadata"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.write(f"**[{idx}] Score:** `{src['score']:.4f}` | **Type:** `{src['type']}` | **Source:** `{src['source']}` (Page {src['page_number']})")
                    st.text(src["content"])

if user_query := st.chat_input("Ask a question about your documents or diagrams..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching vector index & generating response..."):
            try:
                engine = load_query_engine()
                response = engine.query(user_query)
                
                answer_text = response["answer"]
                sources = response["sources"]

                st.markdown(answer_text)
                
                if sources:
                    with st.expander("📌 Retrieved Sources & Metadata"):
                        for idx, src in enumerate(sources, 1):
                            st.write(f"**[{idx}] Score:** `{src['score']:.4f}` | **Type:** `{src['type']}` | **Source:** `{src['source']}` (Page {src['page_number']})")
                            st.text(src["content"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Failed to generate response: {e}")
                