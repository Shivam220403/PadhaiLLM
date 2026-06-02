import streamlit as st
import os
import hashlib
import tempfile
import shutil
import time
from datetime import datetime
import re

# PDF Extraction Libraries
import fitz                     # PyMuPDF — primary extractor
import pdfplumber               # Fallback extractor

# LangChain Core
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# ═══════════════════════════════════════════════════════════
#  1. PAGE CONFIG & PREMIUM UI
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PadhaiLLM — Syllabus Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Import Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 40%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c9d1d9;
    }

    /* ── Chat Messages ── */
    .stChatMessage[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        animation: fadeSlideIn 0.4s ease-out;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background: linear-gradient(135deg, #1e293b 0%, #1a1f36 100%);
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background: linear-gradient(135deg, #0f172a 0%, #141927 100%);
        border-left: 3px solid #6366f1;
    }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Title Area ── */
    .hero-title {
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 40%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 2.4rem;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #64748b;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 8px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #818cf8;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        margin-bottom: 28px;
    }

    /* ── Metric Cards ── */
    .metric-row {
        display: flex;
        gap: 12px;
        margin: 16px 0;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(99,102,241,0.02));
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #818cf8;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card .metric-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 2px;
    }

    /* ── Status Pill ── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #4ade80;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 8px 0;
    }
    .status-pill .pulse-dot {
        width: 8px; height: 8px;
        background: #4ade80;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.5; transform: scale(0.8); }
    }

    /* ── Chat Input ── */
    .stChatInputContainer {
        background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.98)) !important;
        border: 1px solid rgba(99,102,241,0.28) !important;
        border-radius: 20px !important;
        box-shadow: 0 0 0 0 rgba(99,102,241,0);
        backdrop-filter: blur(12px);
        padding: 4px 8px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stChatInputContainer:focus-within {
        border-color: rgba(99,102,241,0.6) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12), 0 0 20px rgba(99,102,241,0.08) !important;
    }
    .stChatInput textarea {
        background: transparent !important;
        border: none !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 10px 4px !important;
        resize: none !important;
    }
    .stChatInput textarea::placeholder {
        color: #475569 !important;
    }
    .stChatInput textarea:focus {
        box-shadow: none !important;
        outline: none !important;
    }
    /* Send button */
    .stChatInput button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border-radius: 12px !important;
        border: none !important;
        color: white !important;
        transition: opacity 0.2s ease, transform 0.15s ease !important;
    }
    .stChatInput button:hover {
        opacity: 0.85 !important;
        transform: scale(1.05) !important;
    }
    /* ── Response timestamp ── */
    .msg-timestamp {
        font-size: 0.7rem;
        color: #334155;
        margin-top: 6px;
        letter-spacing: 0.3px;
        user-select: none;
    }

    /* ── Evidence Expander ── */
    .streamlit-expanderHeader {
        background: rgba(99,102,241,0.06) !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(99,102,241,0.3);
        border-radius: 3px;
    }

    /* ── Welcome Card ── */
    .welcome-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(139,92,246,0.04));
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        margin: 60px auto;
        max-width: 600px;
    }
    .welcome-card .welcome-icon {
        font-size: 3.5rem;
        margin-bottom: 16px;
    }
    .welcome-card h2 {
        color: #e2e8f0;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .welcome-card p {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown('<h1 class="hero-title">PadhaiLLM</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Course Exam Prep Assistant — Powered by Local RAG Architecture</p>', unsafe_allow_html=True)
st.markdown('<span class="hero-badge">LLAMA 3.2 · 3B PARAMS · OFFLINE · PRIVATE</span>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  2. SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None
if "pipeline_stats" not in st.session_state:
    st.session_state.pipeline_stats = {}


# ═══════════════════════════════════════════════════════════
#  3. ROBUST PDF TEXT EXTRACTION (Multi-Strategy)
# ═══════════════════════════════════════════════════════════
def extract_text_pymupdf(pdf_path: str) -> list[Document]:
    """
    Strategy 1 (Primary): PyMuPDF — fast, handles most PDFs well.
    Extracts text page-by-page, preserving layout structure.
    """
    documents = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Use "text" mode for clean extraction
            text = page.get_text("text")
            # Also try "blocks" mode and merge if text mode yields too little
            if len(text.strip()) < 50:
                blocks = page.get_text("blocks")
                block_texts = [b[4] for b in blocks if b[6] == 0]  # type 0 = text
                text = "\n".join(block_texts)
            if text.strip():
                documents.append(Document(
                    page_content=text.strip(),
                    metadata={"page": page_num + 1, "source": os.path.basename(pdf_path), "extractor": "pymupdf"}
                ))
        doc.close()
    except Exception as e:
        st.warning(f"PyMuPDF extraction warning: {e}")
    return documents


def extract_text_pdfplumber(pdf_path: str) -> list[Document]:
    """
    Strategy 2 (Fallback): pdfplumber — better for table-heavy and
    complex-layout PDFs. Also extracts tables as formatted text.
    """
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text_parts = []

                # Extract main text
                main_text = page.extract_text(
                    x_tolerance=3,
                    y_tolerance=3,
                    layout=False
                )
                if main_text:
                    text_parts.append(main_text)

                # Extract tables and convert to text representation
                tables = page.extract_tables()
                for table in tables:
                    table_text = "\n".join(
                        " | ".join(str(cell or "") for cell in row)
                        for row in table
                    )
                    if table_text.strip():
                        text_parts.append(f"\n[Table]\n{table_text}\n[/Table]")

                combined = "\n".join(text_parts)
                if combined.strip():
                    documents.append(Document(
                        page_content=combined.strip(),
                        metadata={"page": page_num + 1, "source": os.path.basename(pdf_path), "extractor": "pdfplumber"}
                    ))
    except Exception as e:
        st.warning(f"pdfplumber extraction warning: {e}")
    return documents


def extract_text_pypdf(pdf_path: str) -> list[Document]:
    """
    Strategy 3 (Last Resort): pypdf — most compatible but least capable.
    """
    from pypdf import PdfReader
    documents = []
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(Document(
                    page_content=text.strip(),
                    metadata={"page": page_num + 1, "source": os.path.basename(pdf_path), "extractor": "pypdf"}
                ))
    except Exception as e:
        st.warning(f"pypdf extraction warning: {e}")
    return documents


def robust_pdf_extract(pdf_path: str) -> list[Document]:
    """
    Master extraction function: tries PyMuPDF first, validates coverage,
    falls back to pdfplumber, then pypdf. Merges best results per page.
    """
    # Strategy 1: PyMuPDF
    docs_mupdf = extract_text_pymupdf(pdf_path)
    total_chars_mupdf = sum(len(d.page_content) for d in docs_mupdf)

    # Count total pages for coverage check
    try:
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
        pdf_doc.close()
    except:
        total_pages = len(docs_mupdf) or 1

    coverage_ratio = len(docs_mupdf) / total_pages if total_pages > 0 else 0

    # If coverage is good (>80% pages extracted, decent char count), use PyMuPDF
    if coverage_ratio >= 0.8 and total_chars_mupdf > 100:
        return docs_mupdf

    # Strategy 2: pdfplumber as fallback/supplement
    docs_plumber = extract_text_pdfplumber(pdf_path)
    total_chars_plumber = sum(len(d.page_content) for d in docs_plumber)

    # Pick the strategy that extracted more content
    if total_chars_plumber > total_chars_mupdf:
        primary_docs = docs_plumber
    else:
        primary_docs = docs_mupdf

    # Check if we still have low coverage
    if len(primary_docs) / total_pages < 0.5:
        # Strategy 3: pypdf as last resort, merge missing pages
        docs_pypdf = extract_text_pypdf(pdf_path)
        extracted_pages = {d.metadata["page"] for d in primary_docs}
        for doc in docs_pypdf:
            if doc.metadata["page"] not in extracted_pages:
                primary_docs.append(doc)
        # Sort by page number
        primary_docs.sort(key=lambda d: d.metadata["page"])

    return primary_docs


def clean_text(text: str) -> str:
    """Post-process extracted text: fix common OCR/extraction artifacts."""
    # Remove null/control characters first
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    # Fix broken hyphenation at line ends (e.g. "net-\nwork" → "network")
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # Collapse 3+ blank lines to 2 — preserve paragraph & bullet structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Ensure bullet characters get a newline prefix so they aren't merged
    text = re.sub(r'([^\n])([•◦▪▸\-])\s', r'\1\n\2 ', text)
    return text.strip()


def format_fragment_for_display(text: str) -> str:
    """Make a raw chunk readable in the source evidence panel."""
    # Ensure bullet chars start on their own line
    text = re.sub(r'([^\n])([•◦▪▸])\s', r'\1\n\2 ', text)
    # Add spacing after numbered list markers like "1." "2."
    text = re.sub(r'([^\n])(\d+\.)\s', r'\1\n\2 ', text)
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════
#  4. RAG PIPELINE BUILDER
# ═══════════════════════════════════════════════════════════
def compute_file_hash(file_bytes: bytes) -> str:
    """SHA-256 hash for cache invalidation on new file upload."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def classify_query(query: str) -> str:
    """
    Classify user query into one of 3 tiers:
      - 'summary'  → read entire PDF (e.g. "summarize the pdf")
      - 'broad'    → MMR retrieval, k=10 diverse chunks (e.g. "explain all routing protocols")
      - 'focused'  → similarity search, k=4 tight chunks (e.g. "what is Go-Back-N?")
    """
    q = query.lower().strip()

    # ── Tier 1: Summary / full document ──
    summary_keywords = [
        "summarize", "summary", "summarise", "overview", "outline",
        "what is this pdf about", "what is this document about",
        "list all topics", "all topics", "entire pdf",
        "full summary", "complete summary", "whole document", "whole pdf",
        "brief the pdf", "brief the document", "tell me about this pdf",
        "explain the pdf", "explain the document", "what does this cover",
        "table of contents", "main topics", "key topics",
    ]
    if any(kw in q for kw in summary_keywords):
        return "summary"

    # ── Tier 2: Broad / multi-topic ──
    broad_keywords = [
        "explain all", "list all", "describe all", "what are the different",
        "compare", "differences between", "types of", "various",
        "how many", "enumerate", "classify", "classification",
        "advantages and disadvantages", "pros and cons",
        "explain the topics", "cover all", "discuss all",
        "entire chapter", "all the", "everything about",
    ]
    if any(kw in q for kw in broad_keywords):
        return "broad"

    # ── Tier 3: Focused / single-topic (default) ──
    return "focused"


def build_rag_pipeline(pdf_path: str):
    """
    Constructs the full RAG pipeline with DUAL-PATH architecture:
      - Path 1 (Summarization): Feeds ALL document pages to LLM
      - Path 2 (Specific Q&A): Uses MMR retrieval for targeted answers
    Returns: (retrieval_chain, raw_docs, llm, stats_dict)
    """

    # ── A. Extraction ──
    raw_docs = robust_pdf_extract(pdf_path)
    if not raw_docs:
        raise ValueError("Could not extract any text from the PDF. The file may be image-only or corrupted.")

    # Clean all documents
    for doc in raw_docs:
        doc.page_content = clean_text(doc.page_content)

    # Remove empty docs after cleaning
    raw_docs = [d for d in raw_docs if len(d.page_content) > 20]

    total_pages = len(raw_docs)
    total_chars = sum(len(d.page_content) for d in raw_docs)
    extractor_used = raw_docs[0].metadata.get("extractor", "unknown") if raw_docs else "none"

    # ── B. Chunking ──
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(raw_docs)

    # ── C. Embeddings & Vector Store ──
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="padhai_collection",
    )

    # Focused retriever: pure similarity, few tight results
    focused_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Broad retriever: MMR for diverse coverage across the document
    broad_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 25, "lambda_mult": 0.7}
    )

    # ── D. LLM for Q&A ──
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
        num_ctx=8192,
        num_predict=4096,
        top_p=0.1,
        top_k=20,
        repeat_penalty=1.15,
    )

    # ── D2. LLM for Summarization — larger context window ──
    summary_llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
        num_ctx=16384,
        num_predict=4096,
        top_p=0.1,
        top_k=20,
        repeat_penalty=1.15,
    )

    # ── E. Prompt for Q&A (shared by focused & broad) ──
    system_prompt = (
        "You are **PadhaiLLM**, a strict, highly accurate academic assistant designed to help students prepare for exams.\n\n"
        "## RULES\n"
        "1. Answer the student's question using **ONLY** the provided context chunks below.\n"
        "2. If the context contains relevant information, provide a comprehensive, well-structured answer.\n"
        "3. Use bullet points, numbered lists, and bold text for clarity when appropriate.\n"
        "4. If you reference specific content, mention which part of the document it comes from.\n"
        "5. If the answer is **NOT** present in the context, respond exactly with:\n"
        "   > ⚠️ The requested information is not present in the uploaded document.\n"
        "6. **NEVER** force connections between unrelated topics. If the prompt asks for broad information or comparisons, do not artificially compare unrelated concepts (e.g., comparing a network topology to a routing protocol). Present them as distinct, separate topics.\n"
        "7. **NEVER** generate information from your own training data. Only use the provided context.\n\n"
        "## CONTEXT CHUNKS\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # ── F. Two Chains — one per retrieval strategy ──
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    focused_chain = create_retrieval_chain(focused_retriever, combine_docs_chain)
    broad_chain = create_retrieval_chain(broad_retriever, combine_docs_chain)

    stats = {
        "pages": total_pages,
        "chunks": len(chunks),
        "characters": total_chars,
        "extractor": extractor_used,
    }

    return focused_chain, broad_chain, raw_docs, llm, summary_llm, stats


def generate_full_summary(summary_llm, raw_docs: list, user_query: str) -> tuple:
    """
    Summarization path: feeds ALL document pages to the LLM in a SINGLE pass.
    Uses a high-context LLM (num_ctx=16384) to avoid multi-chunk slow processing.
    Falls back to 2-chunk approach only for very large documents.
    Returns: (answer_text, source_documents_used)
    """
    SUMMARY_PROMPT = (
        "You are **PadhaiLLM**, a thorough academic assistant.\n\n"
        "The student has uploaded a document and wants a COMPLETE summary.\n"
        "You have been given the ENTIRE document text below.\n\n"
        "## INSTRUCTIONS\n"
        "1. Read through ALL the content carefully from start to end.\n"
        "2. Provide a **detailed, well-organized summary** covering EVERY major topic and subtopic.\n"
        "3. Use **headings (##)**, **bullet points**, and **bold text** for structure.\n"
        "4. Mention page numbers where key topics are discussed.\n"
        "5. Do NOT skip any section — cover the entire document.\n"
        "6. Make it exam-preparation friendly with clear definitions and key points.\n\n"
        "## FULL DOCUMENT TEXT\n"
        "{full_text}"
    )

    # Build full text with page markers
    all_text_parts = []
    for doc in raw_docs:
        page_num = doc.metadata.get("page", "?")
        all_text_parts.append(f"[Page {page_num}]\n{doc.page_content}")
    full_text = "\n\n".join(all_text_parts)

    # ── SINGLE-PASS: threshold ~50K chars covers most academic PDFs ──
    # With num_ctx=16384, we have ~12K tokens for input (50K chars ÷ 4 = 12.5K tokens)
    # This handles up to ~150 pages of typical course notes in one shot.
    if len(full_text) <= 50000:
        messages = [
            ("system", SUMMARY_PROMPT.format(full_text=full_text)),
            ("human", user_query),
        ]
        response = summary_llm.invoke(ChatPromptTemplate.from_messages(messages).format_messages())
        return response.content, raw_docs

    # ── FALLBACK: very large docs — split into exactly 2 halves, 2 LLM calls ──
    mid = len(all_text_parts) // 2
    half_a = "\n\n".join(all_text_parts[:mid])
    half_b = "\n\n".join(all_text_parts[mid:])

    def summarize_half(text, label):
        msgs = [
            ("system",
             f"You are an academic assistant. Summarize this {label} of a document thoroughly.\n"
             f"Cover ALL topics, subtopics, definitions and key points. Use bullet points and headings.\n\n"
             f"## DOCUMENT TEXT ({label})\n{text}"),
            ("human", "Summarize completely."),
        ]
        r = summary_llm.invoke(ChatPromptTemplate.from_messages(msgs).format_messages())
        return r.content

    part_a = summarize_half(half_a, "first half")
    part_b = summarize_half(half_b, "second half")

    merge_msgs = [
        ("system",
         "You are **PadhaiLLM**. Combine these two halves into one comprehensive summary.\n"
         "Use headings, bullet points, bold text. Organize by topic. Cover everything.\n\n"
         f"## FIRST HALF\n{part_a}\n\n## SECOND HALF\n{part_b}"),
        ("human", user_query),
    ]
    final = summary_llm.invoke(ChatPromptTemplate.from_messages(merge_msgs).format_messages())
    return final.content, raw_docs


# ═══════════════════════════════════════════════════════════
#  5. SIDEBAR — FILE UPLOAD & CONTROLS
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Upload Study Material")
    uploaded_file = st.file_uploader(
        "Drop your course PDF here",
        type=["pdf"],
        help="Upload a syllabus, textbook chapter, or notes PDF",
    )

    st.markdown("---")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Pipeline stats display
    if st.session_state.pipeline_stats:
        stats = st.session_state.pipeline_stats
        st.markdown("### Pipeline Metrics")
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{stats['pages']}</div>
                <div class="metric-label">Pages</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['chunks']}</div>
                <div class="metric-label">Chunks</div>
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{stats['characters']:,}</div>
                <div class="metric-label">Characters</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['characters'] // stats['pages'] if stats['pages'] else 0:,}</div>
                <div class="metric-label">Avg Chars/Page</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="status-pill"><span class="pulse-dot"></span> Engine Active</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "**Engine:** Llama 3.2 (3B)  \n"
        "**Embeddings:** all-MiniLM-L6-v2  \n"
        "**Vector DB:** ChromaDB  \n"
        "**Search:** 3-tier (Focused k=4 · Broad k=10 · Full-doc)",
        help="Technical specifications of the RAG pipeline",
    )


# ═══════════════════════════════════════════════════════════
#  6. MAIN LOGIC — PIPELINE INIT & CHAT
# ═══════════════════════════════════════════════════════════
if uploaded_file:
    file_bytes = uploaded_file.getbuffer()
    current_hash = compute_file_hash(bytes(file_bytes))

    # Detect if a new/different PDF was uploaded → rebuild pipeline
    if current_hash != st.session_state.pdf_hash:
        st.session_state.pdf_hash = current_hash
        st.session_state.messages = []  # Clear chat for new document

        # Save to temp file
        temp_path = os.path.join(tempfile.gettempdir(), f"padhai_{current_hash}.pdf")
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # Build pipeline with progress
        with st.status("Initializing RAG Pipeline...", expanded=True) as status:
            st.write("Extracting text from PDF (multi-strategy)...")
            time.sleep(0.3)

            try:
                focused_chain, broad_chain, raw_docs, llm, summary_llm, stats = build_rag_pipeline(temp_path)
                st.session_state.focused_chain = focused_chain
                st.session_state.broad_chain = broad_chain
                st.session_state.raw_docs = raw_docs
                st.session_state.llm = llm
                st.session_state.summary_llm = summary_llm
                st.session_state.pipeline_stats = stats

                st.write(f"Extracted **{stats['pages']} pages** ({stats['characters']:,} chars) via `{stats['extractor']}`")
                st.write(f"Created **{stats['chunks']} semantic chunks**")
                st.write("Embeddings computed & stored in ChromaDB")
                st.write("Llama 3.2 (3B) connected via Ollama")
                status.update(label="Pipeline Ready!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Pipeline Failed", state="error")
                st.error(f"Pipeline initialization error: {e}")
                st.stop()

        st.rerun()  # Rerun to show metrics in sidebar

    # Pipeline is ready — render chat interface
    if "focused_chain" in st.session_state:

        # ── Chat History ──
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    if "sources" in message and message["sources"]:
                        with st.expander("View Source Evidence", expanded=False):
                            for idx, src in enumerate(message["sources"]):
                                st.markdown(f"**Fragment {idx + 1}** — *Page {src['page']}*")
                                display_text = format_fragment_for_display(src["text"])
                                st.markdown(
                                    f'<div style="background:#0f172a;border-left:3px solid #334155;'
                                    f'padding:10px 14px;border-radius:6px;font-size:0.8rem;'
                                    f'color:#94a3b8;white-space:pre-wrap;font-family:monospace;">'
                                    f'{display_text}</div>',
                                    unsafe_allow_html=True
                                )
                                st.markdown("---")
                    if "timestamp" in message:
                        st.markdown(
                            f'<div class="msg-timestamp">⏱ {message["timestamp"]}</div>',
                            unsafe_allow_html=True
                        )

        # ── New Query ──
        if user_query := st.chat_input("Ask a question about your uploaded document..."):
            # Display user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate response
            with st.chat_message("assistant"):
                # ── 3-TIER ROUTING ──
                query_tier = classify_query(user_query)

                if query_tier == "summary":
                    spinner_text = "Reading ENTIRE document for summarization..."
                elif query_tier == "broad":
                    spinner_text = "Broad search — scanning across document..."
                else:
                    spinner_text = "Focused search — finding exact answer..."

                with st.spinner(spinner_text):
                    try:
                        t_start = time.time()

                        if query_tier == "summary":
                            # TIER 1: Full document summary
                            st.toast("Full-document mode", icon="📖")
                            answer, context_docs = generate_full_summary(
                                st.session_state.summary_llm,
                                st.session_state.raw_docs,
                                user_query,
                            )
                        elif query_tier == "broad":
                            # TIER 2: MMR retrieval — diverse chunks (k=10)
                            st.toast("Broad mode: diverse retrieval", icon="🔍")
                            output = st.session_state.broad_chain.invoke({"input": user_query})
                            answer = output["answer"]
                            context_docs = output.get("context", [])
                        else:
                            # TIER 3: Similarity search — tight, relevant chunks (k=4)
                            output = st.session_state.focused_chain.invoke({"input": user_query})
                            answer = output["answer"]
                            context_docs = output.get("context", [])

                        elapsed = time.time() - t_start
                        ts = datetime.now().strftime("%I:%M %p")
                        timestamp_str = f"{ts} · {elapsed:.1f}s"

                        st.markdown(answer)

                        # Source evidence
                        sources = []
                        if context_docs:
                            with st.expander("View Source Evidence", expanded=False):
                                for idx, chunk in enumerate(context_docs):
                                    page_num = chunk.metadata.get("page", "?")
                                    st.markdown(f"**Fragment {idx + 1}** — *Page {page_num}*")
                                    display_text = format_fragment_for_display(chunk.page_content[:800])
                                    st.markdown(
                                        f'<div style="background:#0f172a;border-left:3px solid #334155;'
                                        f'padding:10px 14px;border-radius:6px;font-size:0.8rem;'
                                        f'color:#94a3b8;white-space:pre-wrap;font-family:monospace;">'
                                        f'{display_text}</div>',
                                        unsafe_allow_html=True
                                    )
                                    st.markdown("---")
                                    sources.append({
                                        "page": page_num,
                                        "text": chunk.page_content[:800],
                                    })

                        # Timestamp under answer
                        st.markdown(
                            f'<div class="msg-timestamp">⏱ {timestamp_str}</div>',
                            unsafe_allow_html=True
                        )

                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "timestamp": timestamp_str,
                        })

                    except Exception as e:
                        error_msg = f"Error generating response: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    # ── Welcome Screen ──
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">📖</div>
        <h2>Welcome to PadhaiLLM</h2>
        <p>
            Your personal course exam prep assistant.<br>
            Upload a syllabus PDF, textbook chapter, or lecture notes<br>
            in the sidebar to get started.
        </p>
        <br>
        <p style="color: #475569; font-size: 0.82rem;">
            ✦ 100% offline & private — your data never leaves your machine<br>
            ✦ Multi-strategy PDF extraction for complete coverage<br>
            ✦ Powered by Llama 3.2 (3B) via Ollama
        </p>
    </div>
    """, unsafe_allow_html=True)
