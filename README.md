# Grounded PDF RAG Platform

> High-concurrency, strictly grounded Retrieval-Augmented Generation (RAG) system for PDF-based question answering — with expert review, live terminal logging, a full web UI, and a ChromaDB visual inspector.

---

## Features

| Feature | Description |
|---|---|
| PDF Ingestion Pipeline | Upload PDFs via drag-and-drop; text is extracted, chunked, embedded & indexed automatically |
| Strictly Grounded Q&A | Answers are sourced exclusively from uploaded PDFs — no hallucinations |
| Page-Level Citations | Every answer includes exact source document + page number references |
| Rich Terminal Logging | Color-coded, timestamped console output for every pipeline stage |
| Expert Review System | Users can rate answers and escalate to an admin review queue |
| ChromaDB Vector Store | Persistent 384-dimensional dense embeddings via offline semantic hashing |
| Modern Web UI | Dark-mode glassmorphic Next.js frontend with real-time backend status |
| ChromaDB Admin UI | Visual inspection of all vector collections at http://localhost:3434 |
| Standalone Export | Export entire knowledge base (ChromaDB + SQLite + PDFs) as a portable .zip |
| Multi-LLM Support | Supports OpenAI GPT-4o, Anthropic Claude, or fully offline extractive generation |

---

## Architecture

```
User (Browser)
  http://localhost:3001  (Next.js Frontend)
        |
        | REST API calls
        v
FastAPI Backend (:8000)
  POST /api/upload   --> PDFParser --> TextChunker --> VectorStore
  POST /api/query    --> VectorStore.query --> RAGEngine
  GET  /api/admin/reviews --> ReviewDB (SQLite)
        |                          |
        v                          v
ChromaDB (:8001)          SQLite Review DB
pdf_knowledge_base         (data/rag_app.db)
        ^
        | Admin UI proxy
chromadb-admin (:3434)
```

---

## Project Structure

```
RAG-Model-for-pdfbased-answers/
|
+-- api/
|   +-- main.py              # FastAPI app - all REST endpoints
|   +-- export_import.py     # Standalone zip export/import
|
+-- ingestion/
|   +-- parser.py            # PyMuPDF-based PDF page extractor
|   +-- chunker.py           # Sliding-window text chunker
|
+-- vectordb/
|   +-- store.py             # ChromaDB persistent vector store + offline embeddings
|
+-- retrieval/
|   +-- rag_engine.py        # RAG query engine (retrieve -> generate -> cite)
|
+-- admin/
|   +-- review_db.py         # SQLite-backed expert review queue
|
+-- utils/
|   +-- console.py           # Rich ANSI terminal logger (colors, timestamps, tree view)
|
+-- frontend/
|   +-- src/app/
|       +-- page.js          # Main React UI (upload, Q&A, admin, backup)
|       +-- page.module.css  # Glassmorphic dark-mode styles
|       +-- globals.css      # Design system tokens and global styles
|       +-- layout.js        # Next.js root layout
|
+-- data/
|   +-- uploads/             # Stored PDF files
|   +-- chromadb/            # Persistent ChromaDB vector database
|   +-- rag_app.db           # SQLite review database
|   +-- standalone_exports/  # Exported standalone packages
|
+-- config.py                # Centralized settings (env vars, paths)
+-- .env                     # Local environment variables (not committed)
+-- .env.example             # Environment variable template
+-- requirement.txt          # Python dependencies
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ and npm

---

### 1. Clone & Set Up Python Backend

```bash
# Clone the repository
git clone <your-repo-url>
cd RAG-Model-for-pdfbased-answers

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirement.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys (optional — the system works fully offline without any keys):

```env
# Optional: Add your key for GPT-4o generation
OPENAI_API_KEY=sk-...

# Optional: Add your key for Claude generation
ANTHROPIC_API_KEY=sk-ant-...

# Auto-detects provider: openai -> anthropic -> offline
LLM_PROVIDER=auto
```

### 3. Start the FastAPI Backend

```bash
source venv/bin/activate
python -m api.main
```

Backend API will be available at: `http://localhost:8000`

> Watch your terminal! Every PDF upload will display a rich color-coded pipeline log:
> `[FILE UPLOAD]` -> `[PDF PARSER]` -> `[CHUNKING]` -> `[EMBEDDINGS]` -> `[VECTOR DB]`

### 4. Start the Next.js Frontend

In a new terminal tab:

```bash
cd frontend
npm install
npm run dev
```

Frontend UI: `http://localhost:3001`

### 5. (Optional) Start the ChromaDB Visual Inspector

```bash
# Terminal 3 — ChromaDB HTTP Server
source venv/bin/activate
chroma run --path ./data/chromadb --port 8001

# Terminal 4 — ChromaDB Admin UI
chromadb-admin
```

Open `http://localhost:3434` -> Connection string: `http://localhost:8001` -> **No Auth** -> Connect!

---

## Terminal Console Output

When you upload a PDF, your terminal displays a rich structured log:

```
+--------------------------------------------------------------------+
|  >> PDF UPLOAD & INGESTION PIPELINE                                |
+--------------------------------------------------------------------+
[15:04:04] [FILE UPLOAD] Incoming file upload request: 'research_paper.pdf'
[15:04:04] [FILE UPLOAD] Read 2.35 MB (2,465,312 bytes) from network stream.
[15:04:04] [FILE UPLOAD] File successfully written to disk.

-- [ START INGESTION: research_paper.pdf ] --------------------------
[15:04:04] [PDF PARSER]  Opening 'research_paper.pdf' with PyMuPDF parser...
[15:04:04] [PDF PARSER]  Detected 24 total page(s) in document.
   +-- Page 1/24: 3,241 characters extracted
   +-- Page 2/24: 2,880 characters extracted
   +-- Page 24/24: 1,105 characters extracted
[15:04:04] [PDF PARSER]  Extraction completed: 24 page(s), 58,432 chars in 0.31s.
[15:04:04] [CHUNKING]    Generated 112 total chunks (avg: 521 chars) in 0.02s.
[15:04:04] [EMBEDDINGS]  Vectorizing 112 chunks into 384-d dense vectors...
[15:04:04] [EMBEDDINGS]  Embedding batch completed: 112 vectors in 0.011s.
[15:04:04] [VECTOR DB]   Upsert finished: 112 chunk(s) committed. Total: 112 chunks.
[15:04:04] [SUCCESS]     Pipeline complete: 24 pages, 112 chunks indexed in 0.38s.
---------------------------------------------------------------------
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Backend health & stats |
| GET | `/api/stats` | Query & document statistics |
| POST | `/api/upload` | Upload and index a PDF file |
| GET | `/api/documents` | List all indexed documents |
| DELETE | `/api/documents/{filename}` | Delete a document & its vectors |
| POST | `/api/query` | Ask a question (grounded RAG) |
| POST | `/api/review/self` | Submit user rating on an answer |
| POST | `/api/review/request-admin` | Escalate to expert review queue |
| GET | `/api/admin/reviews` | Fetch the admin review queue |
| POST | `/api/admin/reviews/{id}/action` | Approve / Reject / Modify answer |
| GET | `/api/export` | Export standalone knowledge base zip |
| POST | `/api/import` | Import a standalone zip bundle |

### Interactive Docs

Visit `http://localhost:8000/docs` for the full Swagger UI with live API testing.

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "top_k": 4}'
```

Response:
```json
{
  "query_id": "q_1a2b3c",
  "question": "What are the main findings?",
  "answer": "Based on the uploaded documents:\n* Finding 1 (Source: paper.pdf, Page 3)\n* Finding 2 (Source: paper.pdf, Page 7)",
  "provider": "Offline Engine (Extractive Grounded)",
  "sources": [
    {
      "filename": "paper.pdf",
      "page_number": 3,
      "similarity_score": 0.7821,
      "text": "..."
    }
  ],
  "is_grounded": true
}
```

---

## Configuration

All settings are controlled via environment variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `auto`, `openai`, `anthropic`, or `offline` |
| `OPENAI_API_KEY` | (empty) | OpenAI API key for GPT-4o generation |
| `ANTHROPIC_API_KEY` | (empty) | Anthropic API key for Claude generation |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `ANTHROPIC_MODEL` | `claude-3-5-haiku-20241022` | Anthropic model name |
| `CHUNK_SIZE` | `600` | Max characters per text chunk |
| `CHUNK_OVERLAP` | `120` | Overlap between chunks (sliding window) |
| `TOP_K_CHUNKS` | `4` | Number of chunks retrieved per query |
| `HOST` | `0.0.0.0` | Backend server host |
| `PORT` | `8000` | Backend server port |
| `MAX_CONCURRENT_USERS` | `50` | Max concurrent request workers |

---

## LLM Provider Fallback Chain

The system uses an intelligent 3-tier fallback chain:

```
1. OpenAI (gpt-4o-mini)         <-- if OPENAI_API_KEY is set
        |  (on failure or not configured)
        v
2. Anthropic (claude-3-5-haiku) <-- if ANTHROPIC_API_KEY is set
        |  (on failure or not configured)
        v
3. Offline Extractive Engine    <-- always available, zero API calls
```

The offline engine extracts the most relevant sentences directly from the top-matching PDF chunks using keyword overlap scoring — no internet or API keys required.

---

## Standalone Export & Portability

Export your entire indexed knowledge base for offline use or deployment:

```bash
curl http://localhost:8000/api/export --output knowledge_base.zip
```

The `.zip` bundle contains:
- `chromadb/` — All vector embeddings and HNSW index
- `rag_app.db` — SQLite review & query history
- `uploads/` — Original PDF files
- `standalone_runner.py` — Self-contained offline query CLI

---

## ChromaDB Visual Inspector

Inspect your vector database in a browser UI:

| Step | Action |
|---|---|
| 1 | Start ChromaDB server: `chroma run --path ./data/chromadb --port 8001` |
| 2 | Start admin UI: `chromadb-admin` |
| 3 | Open `http://localhost:3434` |
| 4 | Enter connection string: `http://localhost:8001` |
| 5 | Select **No Auth** -> Click Connect |

Browse the `pdf_knowledge_base` collection, inspect chunk documents, filter by filename/page number, and run similarity searches.

---

## Development

### All Services

```bash
# Terminal 1 - FastAPI Backend
source venv/bin/activate && python -m api.main

# Terminal 2 - Next.js Frontend
cd frontend && npm run dev

# Terminal 3 - ChromaDB HTTP Server (for visual inspector)
source venv/bin/activate && chroma run --path ./data/chromadb --port 8001

# Terminal 4 - ChromaDB Admin UI
source venv/bin/activate && chromadb-admin
```

### Service URLs Summary

| Service | URL |
|---|---|
| Frontend UI | http://localhost:3001 |
| FastAPI Backend | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| ChromaDB Server | http://localhost:8001 |
| ChromaDB Admin UI | http://localhost:3434 |

---

## Requirements

### Python (`requirement.txt`)

```
fastapi>=0.110.0
uvicorn>=0.28.0
chromadb>=0.4.22
pymupdf>=1.23.0
sentence-transformers>=2.5.0
python-dotenv>=1.0.0
openai>=1.14.0
anthropic>=0.19.0
python-multipart>=0.0.9
aiofiles>=23.2.0
```

### Node.js (`frontend/package.json`)

```json
{
  "next": "16.3.4",
  "react": "19.2.8",
  "react-dom": "19.2.8"
}
```

---

## Security Notes

- Designed for **local / internal use**. CORS is set to `*` for development.
- For production, restrict CORS origins in `api/main.py` and add authentication.
- API keys are stored in `.env` — **never commit `.env` to version control**.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and test them
4. Commit: `git commit -m "feat: add my feature"`
5. Push and open a Pull Request

---

## License

MIT License — see LICENSE for details.

---

Built with FastAPI • ChromaDB • Next.js • PyMuPDF

Grounded answers. Zero hallucinations. Full citations.
# RAG-Model-for-pdfbased--answers
