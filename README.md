# PadhaiLLM — Course Exam Prep Assistant

**PadhaiLLM** is an offline, privacy-first, locally hosted Retrieval-Augmented Generation (RAG) assistant designed specifically for students. It allows users to upload course materials (like syllabi, textbook chapters, or lecture notes) in PDF format and intelligently answers questions or generates complete summaries using a powerful local LLM.

![PadhaiLLM App](https://img.shields.io/badge/Status-Active-success) ![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B) ![LLM](https://img.shields.io/badge/LLM-Llama_3.2_(3B)-blueviolet)

---

## Key Features

- **100% Offline & Private:** Your documents and chat data never leave your machine. Everything runs locally.
- **Robust Multi-Strategy PDF Extraction:** Intelligently switches between `PyMuPDF`, `pdfplumber`, and `pypdf` to handle diverse academic PDFs, including table-heavy documents and complex layouts.
- **Dynamic 3-Tier Query Routing:**
  - **Summary Mode:** Automatically detects summarization requests and uses a high-context window to read and summarize the entire document.
  - **Broad Search:** Uses Maximum Marginal Relevance (MMR) retrieval for diverse topic coverage (e.g., "Explain all routing protocols").
  - **Focused Search:** Uses precise similarity search for targeted, specific questions (e.g., "What is Go-Back-N?").
- **Transparent Source Evidence:** View exactly which sections and pages of your PDF the assistant used to generate its answer.
- **Premium UI:** Built with Streamlit, featuring a modern dark theme, animated chat messages, and live pipeline metrics.

---

## Technology Stack

- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **LLM Engine:** [Ollama](https://ollama.com/) (Running **Llama 3.2 3B**)
- **RAG Orchestration:** [LangChain](https://www.langchain.com/)
- **Embeddings:** `all-MiniLM-L6-v2` (via HuggingFace)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **PDF Extraction:** PyMuPDF (`fitz`), `pdfplumber`, `pypdf`

---

## Setup & Installation

### Prerequisites

1. **Python 3.9+** installed on your system.
2. **Ollama** installed and running locally. You can download it from [ollama.com](https://ollama.com/).

### Step 1: Pull the Local LLM via Ollama
Before running the app, ensure you have the required Llama 3.2 model downloaded:
```bash
ollama run llama3.2:3b
```

### Step 2: Install Python Dependencies
It is recommended to use a virtual environment. Install the required packages using the provided `requirements.txt`:
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run the Application
Launch the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

---

## How to Use

1. **Upload a PDF:** Drag and drop your course material (PDF) into the sidebar.
2. **Wait for Pipeline Initialization:** The app will extract the text, chunk it, generate embeddings, and store them in the local vector database. You can monitor the metrics in the sidebar.
3. **Ask Questions:** 
   - Ask specific questions: *"What is the difference between TCP and UDP?"*
   - Ask for summaries: *"Summarize the entire document"*
4. **Review Evidence:** Click on the **View Source Evidence** dropdown under the assistant's replies to see the exact text fragments and page numbers referenced.

---

## Architecture Highlights

- **Extraction Logic:** The system evaluates extraction quality dynamically. It tries `PyMuPDF` first for speed, falls back to `pdfplumber` if coverage is poor or tables are dominant, and uses `pypdf` as a last resort.
- **Chunking Strategy:** Uses LangChain's `RecursiveCharacterTextSplitter` with overlapping chunks to preserve contextual integrity.
- **Dual LLM Configuration:** 
  - Standard Q&A runs with an 8K context window.
  - Summarization routing uses an extended 16K context window to process full documents efficiently.

---
