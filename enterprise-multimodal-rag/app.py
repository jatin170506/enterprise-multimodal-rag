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

# Page Configuration
st.set_page_config(
    page_title="Enterprise Multimodal Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM TECH THEME CSS ---
st.markdown("""
<style>
    /* Dark Tech Command Center Styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0px;
    }
    
    /* Custom Tech Badges */
    .tech-badge {
        display: inline-block;
        background: rgba(0, 242, 254, 0.1);
        border: 1px solid rgba(0, 242, 254, 0.3);
        color: #00f2fe;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-right: 8px;
    }
    
    /* Card Glassmorphism */
    .css-card {
        background: #161b26;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #111622;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Input Styling */
    .stTextInput input, .stNumberInput input {
        background-color: #1a202c !important;
        border: 1px solid #2d3748 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* Quick Prompt Chips */
    div.stButton > button {
        border-radius: 20px;
        border: 1px solid #2d3748;
        background-color: #1a202c;
        color: #cbd5e0;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        border-color: #00f2fe;
        color: #00f2fe;
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Persistent Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "latest_summary" not in st.session_state:
    st.session_state.latest_summary = None

if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = ""

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
    st.markdown("<h2 style='margin-bottom: 0px;'>⚙️ Command Center</h2>", unsafe_allow_html=True)
    st.caption("Vector DB & Ingestion Engine")
    
    st.divider()
    
    # DB Status Metric Cards
    try:
        col_info = client.get_collection(COLLECTION_NAME)
        st.markdown(f"""
        <div class="css-card">
            <div style="font-size:0.8rem; color:#a0aec0;">VECTOR DB STATUS</div>
            <div style="font-size:1.2rem; font-weight:bold; color:#48bb78;">🟢 {col_info.status.upper()}</div>
            <div style="font-size:0.8rem; color:#a0aec0; margin-top:8px;">INDEXED VECTORS</div>
            <div style="font-size:1.1rem; font-weight:bold; color:#00f2fe;">{col_info.points_count} Points</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.warning("Collection offline or uninitialized.")

    if st.button("🗑️ Reset Qdrant Database", use_container_width=True):
        init_qdrant_collection(client)
        st.session_state.latest_summary = None
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("📥 Ingest Diagram Artifact")
    
    uploaded_file = st.file_uploader(
        "Upload Technical Diagram (PNG, JPG)", 
        type=["png", "jpg", "jpeg"],
        key="diagram_uploader"
    )
    
    doc_source_name = st.text_input("Source Document Tag", value="manual_upload.pdf")
    page_num = st.number_input("Page Reference", min_value=1, value=1)

    if uploaded_file and st.button("⚡ Process & Index Artifact", use_container_width=True):
        os.makedirs("data/extracted_images", exist_ok=True)
        save_path = os.path.join("data/extracted_images", uploaded_file.name)
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Analyzing diagram vision with Gemini 2.5 Flash..."):
            summary_text = generate_diagram_summary(save_path)
            
        with st.spinner("Generating 384-dim FastEmbed vectors..."):
            indexer = load_indexer()
            chunk = [{
                "content": f"DIAGRAM SUMMARY: {summary_text}",
                "type": "diagram_summary",
                "source": doc_source_name,
                "page_number": int(page_num),
                "image_path": save_path
            }]
            indexer.index_documents(chunk)

        st.session_state.latest_summary = {
            "summary": summary_text,
            "source": doc_source_name,
            "page": int(page_num),
            "image_path": save_path
        }
        st.success("Artifact successfully indexed!")
        st.rerun()

    if st.session_state.latest_summary:
        st.divider()
        st.subheader("📋 Active Artifact Context")
        st.caption(f"Source: `{st.session_state.latest_summary['source']}` | P. {st.session_state.latest_summary['page']}")
        with st.expander("View Full Summary", expanded=True):
            st.write(st.session_state.latest_summary["summary"])

# --- 4. MAIN PANEL: HEADER & DASHBOARD ---
st.markdown('<p class="main-header">⚡ Multimodal Enterprise Intelligence</p>', unsafe_allow_html=True)

# Tech Architecture Badges
st.markdown("""
<div style="margin-bottom: 20px;">
    <span class="tech-badge">GEMINI 2.5 FLASH</span>
    <span class="tech-badge">QDRANT LOCAL</span>
    <span class="tech-badge">FASTEMBED BGE-SMALL</span>
    <span class="tech-badge">ZERO COST</span>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Interactive Q&A Engine", "🖼️ Artifact Inspection Hub"])

# --- TAB 1: Chat Engine & Interactive Chips ---
with tab1:
    st.caption("Ask questions across all indexed text, tables, and architectural diagrams.")
    
    # Quick Action Interactive Prompt Chips
    st.markdown("<p style='font-size:0.85rem; color:#a0aec0; margin-bottom:5px;'>Suggested Technical Queries:</p>", unsafe_allow_html=True)
    chip_col1, chip_col2, chip_col3 = st.columns(3)
    
    selected_chip = None
    with chip_col1:
        if st.button("📊 Summarize system architecture", use_container_width=True):
            selected_chip = "What database and components are used in the overall system architecture?"
    with chip_col2:
        if st.button("🔌 Explain data flow", use_container_width=True):
            selected_chip = "Explain the step-by-step data flow shown in the latest indexed diagram."
    with chip_col3:
        if st.button("⚡ List all technical specs", use_container_width=True):
            selected_chip = "List all components, technologies, and specs referenced in the context."

    st.divider()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 Grounded Source Citations & Scores"):
                    for idx, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{idx}] Similarity Score:** `{src['score']:.4f}` | **Type:** `{src['type']}` | **Ref:** `{src['source']}` (P. {src['page_number']})")
                        st.text(src["content"])

    # User Input Processing
    user_query = st.chat_input("Ask a question about your documents or diagrams...")
    
    # If a prompt chip was clicked, override the query input
    if selected_chip:
        user_query = selected_chip

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving dense vectors & synthesizing response..."):
                try:
                    engine = load_query_engine()
                    response = engine.query(user_query)
                    
                    answer_text = response["answer"]
                    sources = response["sources"]

                    st.markdown(answer_text)
                    
                    if sources:
                        with st.expander("🔍 Grounded Source Citations & Scores"):
                            for idx, src in enumerate(sources, 1):
                                st.markdown(f"**[{idx}] Similarity Score:** `{src['score']:.4f}` | **Type:** `{src['type']}` | **Ref:** `{src['source']}` (P. {src['page_number']})")
                                st.text(src["content"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Failed to generate response: {e}")

# --- TAB 2: Artifact Inspection Hub ---
with tab2:
    st.subheader("🖼️ Latest Ingested Artifact Inspection")
    if st.session_state.latest_summary:
        img_col, info_col = st.columns([1, 1])
        with img_col:
            if os.path.exists(st.session_state.latest_summary["image_path"]):
                st.image(
                    st.session_state.latest_summary["image_path"], 
                    caption=f"Source File: {st.session_state.latest_summary['source']}", 
                    use_container_width=True
                )
        with info_col:
            st.markdown("### Generated Vision Summary")
            st.info(st.session_state.latest_summary["summary"])
    else:
        st.info("No diagram artifacts uploaded in this session yet. Upload a diagram in the sidebar to inspect its visual representation and generated summary.")

if uploaded_file and st.button("⚡ Process & Index Artifact", use_container_width=True):
    os.makedirs("data/extracted_images", exist_ok=True)
    save_path = os.path.join("data/extracted_images", uploaded_file.name)
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        with st.spinner("Analyzing diagram with Gemini 2.5 Flash..."):
            summary_text = generate_diagram_summary(save_path)
            
        with st.spinner("Generating FastEmbed vectors..."):
            indexer = load_indexer()
            chunk = [{
                "content": f"DIAGRAM SUMMARY: {summary_text}",
                "type": "diagram_summary",
                "source": doc_source_name,
                "page_number": int(page_num),
                "image_path": save_path
            }]
            indexer.index_documents(chunk)

        st.session_state.latest_summary = {
            "summary": summary_text,
            "source": doc_source_name,
            "page": int(page_num),
            "image_path": save_path
        }
        st.success("Artifact successfully indexed!")
        st.rerun()

    except Exception as e:
        st.error(f"⚠️ Service temporary busy: {e}")