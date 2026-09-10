import os
import time
import shutil
import asyncio
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

import config
from utils.console import ConsoleLogger
from ingestion.parser import PDFParser
from ingestion.chunker import TextChunker
from vectordb.store import VectorStore
from admin.review_db import ReviewDB
from retrieval.rag_engine import RAGEngine
from api.export_import import StandaloneManager

app = FastAPI(
    title="Grounded PDF RAG Platform",
    description="High-concurrency PDF-grounded Q&A with Expert Review and Standalone Export",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_banner():
    ConsoleLogger.header("GROUNDED PDF RAG BACKEND RUNNING")
    ConsoleLogger.info(f"API Server: http://{config.HOST}:{config.PORT}")
    ConsoleLogger.info(f"Active LLM Provider: {config.LLM_PROVIDER}")
    ConsoleLogger.info(f"Vector Database: ChromaDB ('{config.CHROMA_COLLECTION_NAME}')")
    ConsoleLogger.info(f"Chunking Config: size={config.CHUNK_SIZE} chars, overlap={config.CHUNK_OVERLAP} chars")
    ConsoleLogger.info("Terminal console streaming active for uploads, chunking, and embeddings.")
    ConsoleLogger.divider()



# Enable CORS for Next.js / React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core singletons
vector_store = VectorStore()
review_db = ReviewDB()
rag_engine = RAGEngine(vector_store)
chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)

# Pydantic Schemas
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = config.TOP_K_CHUNKS
    session_id: Optional[str] = "default_user"

class SelfReviewRequest(BaseModel):
    query_id: str
    rating: Optional[str] = None # 'positive', 'negative', 'neutral'
    notes: Optional[str] = None
    verified: bool = False

class RequestAdminReviewRequest(BaseModel):
    query_id: str
    reason: Optional[str] = ""

class AdminActionRequest(BaseModel):
    action: str # 'APPROVE', 'REJECT', 'MODIFY'
    admin_id: Optional[str] = "expert_reviewer"
    notes: Optional[str] = ""
    edited_answer: Optional[str] = None

# Routes
@app.get("/api/health")
async def health_check():
    stats = await run_in_threadpool(vector_store.get_collection_stats)
    return {
        "status": "healthy",
        "provider": config.LLM_PROVIDER,
        "openai_configured": bool(config.OPENAI_API_KEY),
        "anthropic_configured": bool(config.ANTHROPIC_API_KEY),
        "total_chunks": stats["total_chunks"],
        "max_concurrent_users": config.MAX_CONCURRENT_USERS
    }

@app.get("/api/stats")
async def get_system_stats():
    db_stats = await run_in_threadpool(review_db.get_stats)
    vector_stats = await run_in_threadpool(vector_store.get_collection_stats)
    return {
        **db_stats,
        "total_chunks": vector_stats["total_chunks"],
        "indexed_filenames": vector_stats["documents"]
    }

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    ConsoleLogger.header(f"PDF UPLOAD & INGESTION PIPELINE")
    ConsoleLogger.upload(f"Incoming file upload request: '{file.filename}'")

    if not file.filename.lower().endswith(".pdf"):
        ConsoleLogger.error(f"Rejected non-PDF file upload: '{file.filename}'. Only PDFs are permitted.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in ("-", "_", ".", " ")).strip()
    dest_path = config.UPLOADS_DIR / safe_filename

    # Save uploaded file
    try:
        content = await file.read()
        file_size = len(content)
        size_kb = file_size / 1024
        size_mb = size_kb / 1024
        size_str = f"{size_mb:.2f} MB" if size_mb >= 1.0 else f"{size_kb:.1f} KB"
        
        ConsoleLogger.upload(f"Read {size_str} ({file_size:,} bytes) from network stream.")
        dest_path.write_bytes(content)
        ConsoleLogger.upload(f"File successfully written to disk: {dest_path}")
    except Exception as e:
        ConsoleLogger.error(f"Failed to save uploaded file '{safe_filename}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Process in thread pool to prevent blocking 50 concurrent requests
    def _process_and_index():
        pipe_start = time.time()
        ConsoleLogger.section(f"START INGESTION: {safe_filename}")
        
        # 1. Parse PDF pages
        pages = PDFParser.extract_pages(dest_path)
        if not pages:
            ConsoleLogger.warning(f"No readable text pages found in '{safe_filename}'. Ingestion stopped.")
            return 0, 0
            
        # 2. Chunk pages
        chunks = chunker.chunk_document_pages(pages)
        
        # 3. Embed & Upsert into ChromaDB
        vector_store.add_chunks(chunks)
        
        # 4. Register in SQLite review db
        review_db.register_document(
            filename=safe_filename,
            file_path=str(dest_path),
            file_size=len(content),
            total_pages=pages[0]["total_pages"] if pages else 1,
            chunk_count=len(chunks)
        )
        ConsoleLogger.info(f"Metadata registered in SQLite review database for '{safe_filename}'.")
        
        pipe_elapsed = time.time() - pipe_start
        ConsoleLogger.success(
            f"Pipeline complete for '{safe_filename}': {len(pages)} pages parsed, "
            f"{len(chunks)} chunks embedded & indexed in {pipe_elapsed:.2f}s."
        )
        ConsoleLogger.divider()
        return len(pages), len(chunks)

    try:
        page_count, chunk_count = await run_in_threadpool(_process_and_index)
        return {
            "status": "success",
            "filename": safe_filename,
            "pages_parsed": page_count,
            "chunks_indexed": chunk_count
        }
    except Exception as e:
        ConsoleLogger.error(f"Failed to parse and index PDF '{safe_filename}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse and index PDF: {str(e)}")

@app.get("/api/documents")
async def get_documents():
    docs = await run_in_threadpool(review_db.list_documents)
    return {"documents": docs}

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    ConsoleLogger.header(f"DOCUMENT DELETION: {filename}")
    def _delete():
        vector_store.delete_document(filename)
        review_db.delete_document(filename)
        file_path = config.UPLOADS_DIR / filename
        if file_path.exists():
            file_path.unlink()
            ConsoleLogger.upload(f"Removed physical file from disk: {file_path}")
        ConsoleLogger.success(f"Document '{filename}' fully purged from system.")
        ConsoleLogger.divider()
    
    await run_in_threadpool(_delete)
    return {"status": "deleted", "filename": filename}

@app.post("/api/query")
async def query_rag(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Execute grounded RAG query asynchronously
    result = await run_in_threadpool(rag_engine.answer_question, req.question, req.top_k)

    # Log query and generate query_id in SQLite
    query_id = await run_in_threadpool(
        review_db.log_query,
        question=req.question,
        answer=result["answer"],
        provider=result["provider"],
        sources=result["sources"],
        session_id=req.session_id,
        is_grounded=result["is_grounded"]
    )

    return {
        "query_id": query_id,
        "question": req.question,
        "answer": result["answer"],
        "provider": result["provider"],
        "sources": result["sources"],
        "is_grounded": result["is_grounded"]
    }

@app.post("/api/review/self")
async def self_review(req: SelfReviewRequest):
    updated = await run_in_threadpool(
        review_db.submit_user_review,
        query_id=req.query_id,
        rating=req.rating,
        notes=req.notes,
        verified=req.verified
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Query record not found.")
    return {"status": "success", "review": updated}

@app.post("/api/review/request-admin")
async def request_admin_review(req: RequestAdminReviewRequest):
    updated = await run_in_threadpool(
        review_db.request_admin_review,
        query_id=req.query_id,
        reason=req.reason
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Query record not found.")
    return {"status": "success", "review": updated}

@app.get("/api/admin/reviews")
async def list_admin_reviews(status: Optional[str] = "ALL", limit: int = 100):
    reviews = await run_in_threadpool(review_db.list_reviews, status=status, limit=limit)
    return {"reviews": reviews}

@app.post("/api/admin/reviews/{query_id}/action")
async def admin_review_action(query_id: str, req: AdminActionRequest):
    updated = await run_in_threadpool(
        review_db.submit_admin_action,
        query_id=query_id,
        action=req.action,
        admin_id=req.admin_id,
        notes=req.notes,
        edited_answer=req.edited_answer
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Query record not found.")
    return {"status": "success", "review": updated}

@app.get("/api/export")
async def export_standalone():
    zip_path = await run_in_threadpool(StandaloneManager.create_standalone_archive)
    if not zip_path.exists():
        raise HTTPException(status_code=500, detail="Failed to create standalone export.")
    return FileResponse(
        path=str(zip_path),
        filename="rag_knowledge_base_standalone.zip",
        media_type="application/zip"
    )

@app.post("/api/import")
async def import_standalone(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip standalone bundles are supported.")
    
    temp_zip = config.STANDALONE_DIR / "temp_import.zip"
    try:
        content = await file.read()
        temp_zip.write_bytes(content)
        res = await run_in_threadpool(StandaloneManager.import_standalone_archive, temp_zip)
        # Reload vector store and review DB
        vector_store.reset()
        review_db.init_db()
        return res
    finally:
        if temp_zip.exists():
            temp_zip.unlink()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=config.HOST, port=config.PORT, reload=True)
