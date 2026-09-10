import os
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Any

import config

class StandaloneManager:
    """
    Handles exporting and importing the entire knowledge base, vector database,
    review history, and documents into a self-contained portable bundle.
    """

    @staticmethod
    def create_standalone_archive() -> Path:
        """
        Creates a standalone zip archive containing the ChromaDB data,
        SQLite database, uploaded PDFs, and a standalone runner.
        """
        export_zip_path = config.STANDALONE_DIR / "rag_knowledge_base_standalone.zip"
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = Path(tmp_dir) / "rag_standalone_bundle"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Copy SQLite database if exists
            if config.SQLITE_DB_PATH.exists():
                shutil.copy2(config.SQLITE_DB_PATH, bundle_dir / "rag_app.db")
                
            # 2. Copy ChromaDB directory
            if config.CHROMA_DIR.exists():
                shutil.copytree(config.CHROMA_DIR, bundle_dir / "chromadb", dirs_exist_ok=True)
                
            # 3. Copy uploaded PDFs
            if config.UPLOADS_DIR.exists():
                shutil.copytree(config.UPLOADS_DIR, bundle_dir / "uploads", dirs_exist_ok=True)
                
            # 4. Include standalone runner script
            standalone_runner_code = '''#!/usr/bin/env python3
"""
Standalone Grounded PDF RAG Runner
Run this script to launch a standalone query console or local web server
using the pre-indexed database included in this package.
"""
import sys
import os
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chromadb")

def main():
    print("=======================================================")
    print("   STANDALONE GROUNDED PDF RAG CONSOLE (OFFLINE)      ")
    print("=======================================================")
    
    if not os.path.exists(CHROMA_DIR):
        print(f"Error: Vector database not found at {CHROMA_DIR}")
        sys.exit(1)
        
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="pdf_knowledge_base",
        embedding_function=embedding_fn
    )
    
    count = collection.count()
    print(f"Loaded Vector Database. Total indexed chunks: {count}")
    print("Ask any question based on the included documents (type 'exit' to quit):\\n")
    
    while True:
        try:
            query = input("Query > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
                
            results = collection.query(
                query_texts=[query],
                n_results=min(3, count),
                include=["documents", "metadatas", "distances"]
            )
            
            if not results or not results["documents"] or not results["documents"][0]:
                print("No relevant information found in the database.\\n")
                continue
                
            print("\\n--- GROUNDED SOURCES FOUND ---")
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            
            for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
                sim = max(0.0, min(1.0, 1.0 - dist))
                print(f"[{idx}] Document: {meta.get('filename')} | Page: {meta.get('page_number')} | Similarity: {sim:.2%}")
                print(f"Excerpt: {doc[:300]}...\\n")
                
            print("-------------------------------------------------------\\n")
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
'''
            (bundle_dir / "run_standalone.py").write_text(standalone_runner_code, encoding="utf-8")
            
            # 5. Include README
            readme_text = """# Standalone Grounded PDF Knowledge Base

This package contains the pre-indexed ChromaDB vector database, SQLite review database, and original PDF documents exported from the Grounded PDF RAG Platform.

## Quick Start (Standalone Console)

1. Make sure you have Python 3 installed with chromadb:
   ```bash
   pip install chromadb
   ```
2. Run the standalone query console:
   ```bash
   python run_standalone.py
   ```

## Importing into another instance of the Web App
You can also upload this `.zip` file directly into any running instance of the RAG Web App via the **Standalone Export & Import** tab!
"""
            (bundle_dir / "README.md").write_text(readme_text, encoding="utf-8")
            
            # 6. Create Zip archive
            shutil.make_archive(
                base_name=str(export_zip_path).replace(".zip", ""),
                format="zip",
                root_dir=tmp_dir,
                base_dir="rag_standalone_bundle"
            )
            
        return export_zip_path

    @staticmethod
    def import_standalone_archive(zip_file_path: Path) -> Dict[str, Any]:
        """
        Imports and restores a standalone archive into the current workspace.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
                
            # Locate bundle folder
            extracted_items = list(Path(tmp_dir).iterdir())
            bundle_root = Path(tmp_dir) / "rag_standalone_bundle" if (Path(tmp_dir) / "rag_standalone_bundle").exists() else Path(tmp_dir)
            
            # Restore SQLite db if present
            sqlite_file = bundle_root / "rag_app.db"
            if sqlite_file.exists():
                shutil.copy2(sqlite_file, config.SQLITE_DB_PATH)
                
            # Restore ChromaDB if present
            chroma_src = bundle_root / "chromadb"
            if chroma_src.exists():
                shutil.copytree(chroma_src, config.CHROMA_DIR, dirs_exist_ok=True)
                
            # Restore Uploads if present
            uploads_src = bundle_root / "uploads"
            if uploads_src.exists():
                shutil.copytree(uploads_src, config.UPLOADS_DIR, dirs_exist_ok=True)
                
        return {"status": "success", "message": "Successfully imported standalone knowledge base."}
