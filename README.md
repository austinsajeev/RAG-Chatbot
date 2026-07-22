# 📄 RAG Chatbot — Instant AI Chatbot from Your Own Documents

> **Drop in your PDFs, Word files, or CSVs. It becomes a chatbot. No training. No cloud. Just your data.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20Gemma%202B-orange?style=flat-square)](https://ollama.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-green?style=flat-square)](https://faiss.ai/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-purple?style=flat-square)](https://www.langchain.com/)

---

## 🚀 What Is This?

**RAG Chatbot** is a fully local, document-powered AI chatbot. You place your files — PDFs, text files, CSVs — into a folder, and within seconds you have an intelligent chatbot that answers questions based **only** on your documents.

No API keys. No internet connection required. No model fine-tuning. Just drop files and chat.

> **Sister Project**: Looking for a version that works with websites instead of files? Check out the [Webscraping Chatbot](https://github.com/austinsajeev/Webscraping-Chatbot) which scrapes any URL and turns it into a chatbot instantly.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📁 **Multi-Format Support** | Ingests **PDF**, **TXT**, and **CSV** files out of the box |
| ⚡ **Instant Knowledge Base** | No training required — documents are processed into a vector store at startup |
| 🧠 **RAG Architecture** | Retrieval-Augmented Generation grounds answers in your actual document content |
| 🔒 **100% Local & Private** | Runs entirely on your machine — no data leaves your device |
| 💾 **Vectorstore Caching** | Built index is saved as `vectorstore.pkl` — no re-processing on restarts |
| 🤖 **Gemma 2B LLM** | Uses Google's lightweight `gemma:2b` model via Ollama for fast responses |
| 🚀 **Background Initialization** | Server starts instantly; document loading runs in a background thread |
| 🌐 **Built-in Chat UI** | Clean web interface served directly at `localhost:5000` |

---

## 🧠 How It Works — Step by Step

```
  Drop files into /data folder
          │
          ▼
 ┌─────────────────────┐
 │   Document Loader   │  ◄─ Reads PDF (via pdfplumber), TXT, CSV
 │   (backend.py)      │     Pages processed in parallel threads
 └──────────┬──────────┘
            │ Raw text from all files
            ▼
 ┌─────────────────────┐
 │   Text Splitter     │  ◄─ RecursiveCharacterTextSplitter
 │   (LangChain)       │     Splits into overlapping chunks
 └──────────┬──────────┘
            │ Text chunks
            ▼
 ┌─────────────────────┐
 │  Embedding Model    │  ◄─ HuggingFace sentence-transformers
 │  (HuggingFace)      │     Converts each chunk → dense vector
 └──────────┬──────────┘
            │ Vector embeddings
            ▼
 ┌─────────────────────┐
 │   FAISS Index       │  ◄─ Stores all vectors for similarity search
 │   + vectorstore.pkl │     Cached to disk for fast reloads
 └─────────────────────┘

      ── Chatbot is now ready! ──

  User asks a question in the UI
          │
          ▼
 ┌─────────────────────┐
 │  FAISS Retriever    │  ◄─ Finds top-K most relevant chunks
 │  (Semantic Search)  │     using vector similarity
 └──────────┬──────────┘
            │ Relevant context passages
            ▼
 ┌─────────────────────┐
 │  LangChain          │  ◄─ Constructs RAG prompt:
 │  RetrievalQA Chain  │     [Context] + [Question] → LLM
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────┐
 │  Gemma 2B (Ollama)  │  ◄─ Generates answer grounded in
 │  Local LLM          │     your document content
 └──────────┬──────────┘
            │ Final Answer
            ▼
        Chat UI
```

---

## 🏗️ Architecture Overview

The application has two main Python files:

### `backend.py` — The RAG Brain

| Function | Purpose |
|---|---|
| `load_documents(folder)` | Recursively reads all PDF, TXT, CSV files from the `data/` folder using parallel threads |
| `load_pdf(file_path)` | Extracts text page-by-page from PDFs using `pdfplumber`; results are cached in memory |
| `create_or_load_vectorstore(docs)` | Splits text into chunks, generates HuggingFace embeddings, builds a FAISS index, and saves it as `vectorstore.pkl` |
| `get_qa_chain(vectorstore)` | Creates a LangChain `RetrievalQA` chain with a custom prompt template that instructs the LLM to answer only from context |
| `get_llm()` | Singleton initializer for the `gemma:2b` Ollama model — loaded once and reused |
| `test_ollama()` | Sanity-checks that Ollama is running and the model responds before the server starts serving requests |

### `app.py` — The Flask API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | Serves the chat UI (`index.html`) |
| `/ask` *(inferred)* | `POST` | Receives a question, runs it through the QA chain, returns the answer |

**Key design decision**: On startup, `initialize_components()` runs in a **daemon thread** — meaning the Flask server is immediately available, and document loading happens in the background. Once ready, it prints `✅ Backend components initialized and ready`.

---

## 🖥️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.8+, Flask, Flask-CORS |
| **Document Parsing** | `pdfplumber` (PDF), built-in (TXT/CSV), `pandas` (CSV) |
| **Text Processing** | LangChain `RecursiveCharacterTextSplitter` |
| **Embeddings** | `langchain-community` + HuggingFace sentence-transformers |
| **Vector Search** | `FAISS` via LangChain `FAISS` wrapper |
| **LLM** | `Ollama` running `gemma:2b` locally |
| **QA Chain** | LangChain `RetrievalQA` with custom `PromptTemplate` |
| **Caching** | `pickle` for vectorstore persistence |
| **Frontend** | Vanilla HTML + CSS + JavaScript (`index.html`) |

---

## 📋 Prerequisites

- **Python 3.8+**
- **[Ollama](https://ollama.com/)** — local LLM runner
- **Gemma 2B model** pulled via Ollama

---

## 🛠️ Installation & Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/austinsajeev/RAG-Chatbot.git
cd RAG-Chatbot
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install flask flask-cors pdfplumber pandas pdfminer.six langchain langchain-community langchain-ollama faiss-cpu sentence-transformers
```

### Step 4 — Install and start Ollama with Gemma 2B

```bash
# Download Ollama from https://ollama.com/download
# Then pull the Gemma 2B model:
ollama pull gemma:2b

# Start Ollama (runs in background):
ollama serve
```

### Step 5 — Add your documents

Place your files in the `data/` folder:

```
RAG-Chatbot/
└── data/
    ├── my_document.pdf
    ├── knowledge_base.txt
    └── table_data.csv
```

Supported formats: **`.pdf`**, **`.txt`**, **`.csv`**

### Step 6 — Run the application

```bash
python app.py
```

### Step 7 — Open the chatbot

Visit **[http://localhost:5000](http://localhost:5000)** in your browser.

On first run, it will:
1. Load and parse all files from `data/`
2. Split text into chunks and generate embeddings
3. Build a FAISS vector index
4. Save it as `vectorstore.pkl` for faster future startups
5. Set up the LangChain QA chain with Gemma 2B

On subsequent runs, it loads `vectorstore.pkl` directly — skipping the heavy embedding step.

---

## 💬 Using the Chatbot

1. Open `http://localhost:5000`
2. Type any question about your documents
3. Get an accurate answer grounded in your files — with no hallucinations outside your data

**Example questions** (if you loaded a company handbook PDF):
- *"What is the leave policy?"*
- *"How do I submit an expense report?"*
- *"Who do I contact for IT support?"*

---

## 📁 Project Structure

```
RAG-Chatbot/
│
├── app.py              # Flask server — routes, background init, API endpoints
├── backend.py          # RAG logic — document loading, embeddings, QA chain
├── index.html          # Frontend chat interface (served by Flask)
│
├── data/               # ← Put your documents HERE
│   ├── your_file.pdf
│   ├── notes.txt
│   └── data.csv
│
└── vectorstore.pkl     # Auto-generated: cached FAISS index (don't edit manually)
```

---

## ⚙️ How Vectorstore Caching Works

```
First Run:
  data/*.pdf → load → chunk → embed → FAISS index → save to vectorstore.pkl

Subsequent Runs:
  vectorstore.pkl → load directly (skip embedding step) → QA chain ready
```

If you add new documents to the `data/` folder, **delete `vectorstore.pkl`** to force a rebuild:

```bash
del vectorstore.pkl       # Windows
rm vectorstore.pkl        # macOS/Linux
```

Then restart the app.

---

## 🔒 Privacy & Local-First Design

Everything runs on your machine:
- **Gemma 2B** runs locally via Ollama — no calls to OpenAI, Google, or any cloud LLM
- Your documents never leave your device
- No API keys needed
- Works fully offline once Ollama and the model are downloaded

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| `✗ Ollama error` on startup | Run `ollama serve` and ensure `gemma:2b` is pulled with `ollama pull gemma:2b` |
| Chatbot says "not ready yet" | The background thread is still loading documents — wait a few seconds and retry |
| No answers / poor answers | Ensure your files are in the `data/` folder and the formats are PDF, TXT, or CSV |
| `vectorstore.pkl` is stale | Delete it and restart to rebuild with new documents |
| PDF parsing errors | Some scanned PDFs may not work (they contain images, not text); use text-based PDFs |

---

## 🔄 Comparison: RAG Chatbot vs Webscraping Chatbot

| | [RAG Chatbot](https://github.com/austinsajeev/RAG-Chatbot) | [Webscraping Chatbot](https://github.com/austinsajeev/Webscraping-Chatbot) |
|---|---|---|
| **Data Source** | PDF, TXT, CSV files | Any website URL |
| **Input** | Drop files in `/data` folder | Provide a URL |
| **Scraping** | `pdfplumber`, pandas | `requests` + Selenium |
| **LLM** | Gemma 2B | LLaMA 3 |
| **Best For** | Private documents, manuals, reports | Public websites, knowledge bases |

---

## 👤 Author

**Austin Sajeev Abraham**
- GitHub: [@austinsajeev](https://github.com/austinsajeev)
- Email: austinsajeevabraham123@gmail.com

---

## 📄 License

This project is open-source. Feel free to fork, modify, and build upon it.

---

> *Built with ❤️ for CDIT — Center for Development of Imaging Technology*
