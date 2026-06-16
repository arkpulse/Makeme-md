"""
DocFlow — Streamlit Dashboard
Enterprise-grade document intelligence UI with drag & drop upload,
Markdown preview, OCR viewer, AI summaries, chunk explorer, and semantic search.
"""
from __future__ import annotations

import asyncio
import io
import sys
import tempfile
from pathlib import Path

import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocFlow — Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    :root {
        --primary: #00d4ff;
        --accent: #ff6b35;
        --bg: #0a0e1a;
        --surface: #111827;
        --surface2: #1a2233;
        --text: #e2e8f0;
        --muted: #64748b;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
    }

    .stApp { background: var(--bg); color: var(--text); font-family: 'IBM Plex Sans', sans-serif; }

    .metric-card {
        background: var(--surface);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-card h3 { color: var(--primary); font-size: 2rem; margin: 0; font-family: 'IBM Plex Mono', monospace; }
    .metric-card p { color: var(--muted); margin: 4px 0 0; font-size: 0.85rem; }

    .chunk-card {
        background: var(--surface2);
        border-left: 3px solid var(--primary);
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
    }

    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin: 2px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .tag-blue { background: #1e3a5f; color: var(--primary); }
    .tag-orange { background: #3d1a0a; color: var(--accent); }
    .tag-green { background: #064e3b; color: var(--success); }

    div[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid #1e293b; }
    .stButton button { background: var(--primary); color: #000; font-weight: 600; border: none; border-radius: 8px; }
    .stButton button:hover { background: #00b8db; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_async(coro):
    """Run async coroutine from Streamlit's sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@st.cache_resource
def get_pipeline():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.pipeline import DocFlowPipeline
    return DocFlowPipeline()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🧠 DocFlow")
    st.markdown("*Document Intelligence Platform*")
    st.divider()

    st.markdown("### ⚙️ Processing Options")
    enable_ocr = st.toggle("Enable OCR", value=True)
    enable_summarize = st.toggle("AI Summarization", value=False)
    enable_embed = st.toggle("Generate Embeddings", value=False)

    st.markdown("### 📐 Chunking")
    chunk_strategy = st.selectbox(
        "Strategy",
        ["recursive", "heading", "token", "semantic"],
        help="recursive: split by paragraphs/sentences | heading: split at Markdown headings | token: split by token count | semantic: split by meaning",
    )
    chunk_size = st.slider("Chunk Size (chars)", 128, 2048, 512, 64)
    chunk_overlap = st.slider("Overlap (chars)", 0, 256, 64, 16)

    st.divider()
    st.markdown("### 📊 Stats")
    if "result" in st.session_state:
        r = st.session_state.result
        st.markdown(f"⏱ `{r.processing_time_s:.2f}s`")
        st.markdown(f"📄 `{len(r.chunks)} chunks`")
        st.markdown(f"🗂 `{len(r.tables)} tables`")


# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown("## 📂 Upload & Convert")

tab_upload, tab_search, tab_about = st.tabs(["🔄 Convert", "🔍 Semantic Search", "ℹ️ About"])

# ─── Convert Tab ─────────────────────────────────────────────────────────────
with tab_upload:
    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        uploaded = st.file_uploader(
            "Drop your document here",
            type=["pdf", "docx", "pptx", "xlsx", "csv", "txt", "html", "md",
                  "json", "xml", "png", "jpg", "jpeg", "mp3", "wav", "zip"],
            help="Supports PDF, Word, Excel, PowerPoint, images, audio, and more",
        )

        # YouTube URL input
        yt_url = st.text_input("…or paste a YouTube URL", placeholder="https://youtube.com/watch?v=...")

        process_btn = st.button("⚡ Process Document", use_container_width=True, type="primary")

        if process_btn and (uploaded or yt_url):
            from core.pipeline import ProcessingOptions

            opts = ProcessingOptions(
                enable_ocr=enable_ocr,
                enable_summarization=enable_summarize,
                enable_embeddings=enable_embed,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            pipeline = get_pipeline()

            with st.spinner("🔄 Processing document…"):
                if yt_url:
                    from parsers.youtube_parser import YouTubeParser
                    result_data = run_async(YouTubeParser().parse_url(yt_url, opts))
                    # Wrap in a mock result
                    from core.pipeline import ProcessingResult
                    result = ProcessingResult(
                        file_id="yt",
                        source_path=yt_url,
                        file_name=yt_url[:50],
                        file_type="youtube",
                        processing_time_s=0,
                        markdown=result_data.get("markdown", ""),
                        metadata=result_data.get("metadata", {}),
                    )
                else:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(uploaded.name).suffix
                    ) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = Path(tmp.name)

                    result = run_async(pipeline.process(tmp_path, opts))
                    tmp_path.unlink(missing_ok=True)

            st.session_state.result = result
            if result.success:
                st.success(f"✅ Processed in {result.processing_time_s:.2f}s")
            else:
                st.error(f"❌ Errors: {result.errors}")

    with col_preview:
        if "result" in st.session_state:
            r = st.session_state.result

            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><h3>{len(r.chunks)}</h3><p>Chunks</p></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><h3>{len(r.tables)}</h3><p>Tables</p></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><h3>{r.processing_time_s:.1f}s</h3><p>Time</p></div>', unsafe_allow_html=True)

    # ── Results Tabs ──────────────────────────────────────────────────────────
    if "result" in st.session_state:
        r = st.session_state.result

        r_md, r_chunks, r_tables, r_summary, r_meta = st.tabs(
            ["📝 Markdown", "🧩 Chunks", "📊 Tables", "🤖 AI Summary", "📋 Metadata"]
        )

        with r_md:
            st.markdown("### Extracted Markdown")
            st.markdown(r.markdown if r.markdown else "*No content extracted.*")
            st.download_button(
                "⬇️ Download Markdown",
                data=r.markdown,
                file_name=f"{r.file_name}.md",
                mime="text/markdown",
            )

        with r_chunks:
            st.markdown(f"### {len(r.chunks)} Semantic Chunks")
            for i, chunk in enumerate(r.chunks):
                with st.expander(f"Chunk {i+1} — {chunk.get('token_count', '?')} tokens"):
                    st.markdown(f'<div class="chunk-card">{chunk["text"]}</div>', unsafe_allow_html=True)
                    meta_cols = st.columns(4)
                    meta_cols[0].markdown(f'<span class="tag tag-blue">ID: {chunk["chunk_id"][:8]}…</span>', unsafe_allow_html=True)
                    if chunk.get("section_title"):
                        meta_cols[1].markdown(f'<span class="tag tag-orange">§ {chunk["section_title"]}</span>', unsafe_allow_html=True)
                    meta_cols[2].markdown(f'<span class="tag tag-green">{chunk.get("token_count", 0)} tokens</span>', unsafe_allow_html=True)

        with r_tables:
            if r.tables:
                for i, tbl in enumerate(r.tables):
                    st.markdown(f"**Table {i+1}**" + (f" (Page {tbl.get('page', '?')})" if tbl.get("page") else ""))
                    if tbl.get("markdown"):
                        st.markdown(tbl["markdown"])
                    elif tbl.get("data"):
                        import pandas as pd
                        try:
                            df = pd.DataFrame(tbl["data"][1:], columns=tbl["data"][0])
                            st.dataframe(df, use_container_width=True)
                        except Exception:
                            st.code(str(tbl["data"]))
            else:
                st.info("No tables found in this document.")

        with r_summary:
            if r.summary:
                st.markdown("### 📄 Summary")
                st.markdown(r.summary)
                if r.key_insights:
                    st.markdown("### 💡 Key Insights")
                    for insight in r.key_insights:
                        st.markdown(f"- {insight}")
            else:
                st.info("Enable 'AI Summarization' in the sidebar and reprocess to generate summaries.")

        with r_meta:
            st.json(r.metadata)

# ─── Search Tab ───────────────────────────────────────────────────────────────
with tab_search:
    st.markdown("### 🔍 Semantic Search")
    st.info("Requires documents to be processed with 'Generate Embeddings' enabled.")

    query = st.text_input("Search query", placeholder="What are the key findings about…")
    top_k = st.slider("Results", 1, 20, 5)

    if st.button("🔍 Search", use_container_width=True) and query:
        from embeddings.engine import EmbeddingEngine
        from vectorstores.base import get_vector_store

        with st.spinner("Searching…"):
            engine = EmbeddingEngine()
            store = get_vector_store()
            q_emb = run_async(engine.embed(query))
            results = run_async(store.search(q_emb, top_k=top_k))

        if results:
            for i, hit in enumerate(results):
                score = hit.get("score", 0)
                score_color = "green" if score > 0.8 else "orange" if score > 0.6 else "red"
                st.markdown(f"**Result {i+1}** — Score: :{score_color}[{score:.3f}]")
                st.markdown(f'<div class="chunk-card">{hit.get("text", "")}</div>', unsafe_allow_html=True)
                st.caption(f"Source: {hit.get('source', 'unknown')} | Chunk: {hit.get('chunk_id', '')[:8]}")
                st.divider()
        else:
            st.warning("No results found. Make sure documents have been indexed with embeddings.")

# ─── About Tab ────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    ## 🧠 DocFlow — Universal AI Document Intelligence Platform

    DocFlow converts any document format into clean Markdown, structured JSON,
    semantic chunks, and vector embeddings — ready for LLMs and RAG pipelines.

    ### Supported Formats
    PDF · DOCX · PPTX · XLSX · CSV · JSON · XML · TXT · HTML · Markdown  
    PNG · JPG · TIFF · BMP · WEBP · MP3 · WAV · FLAC · M4A · ZIP · YouTube URLs

    ### Features
    - 🔍 **Advanced PDF Parsing** with layout detection and multi-column support
    - 🖼️ **GPU OCR** via PaddleOCR, EasyOCR, and Tesseract fallback
    - 🎙️ **Audio Transcription** via Whisper with timestamps
    - 🤖 **AI Summarization** via OpenAI, Anthropic, or Ollama
    - 🧩 **Semantic Chunking** with multiple strategies
    - 📦 **Vector Storage** via ChromaDB or FAISS
    - 🔗 **LangChain & LlamaIndex** compatible
    - 🚀 **FastAPI REST API** with Swagger docs
    """)
