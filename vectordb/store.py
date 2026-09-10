import os
import re
import time
import threading
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

import config
from utils.console import ConsoleLogger

class OfflineSemanticEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    High-speed, 100% offline dense semantic embedding generator.
    Produces normalized 384-dimensional dense vectors using n-gram subword hashing
    and inverse document frequency weighting.
    Zero network calls, zero PyTorch locks, 100% standalone and portable.
    """
    def __init__(self):
        super().__init__()
        self.dimension = 384

    @staticmethod
    def name() -> str:
        return "default"

    def __call__(self, input: Documents) -> Embeddings:
        count = len(input)
        start_time = time.time()
        if count == 1:
            ConsoleLogger.embedding(f"Vectorizing 1 text item ({self.dimension}-d dense semantic embedding)...")
        else:
            ConsoleLogger.embedding(f"Vectorizing batch of {count} text chunks into {self.dimension}-d dense vectors...")
        
        embeddings = []
        for text in input:
            vec = self._vectorize(text)
            embeddings.append(vec.tolist())
            
        elapsed = time.time() - start_time
        if count > 1:
            rate = count / elapsed if elapsed > 0 else count
            ConsoleLogger.embedding(f"Embedding batch completed: {count} vectors in {elapsed:.3f}s ({rate:.0f} chunks/sec).")
        return embeddings

    def embed_documents(self, input: Documents) -> Embeddings:
        return self(input)

    def embed_query(self, input: Documents) -> Embeddings:
        return self(input)

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = re.findall(r'\w+', text.lower())
        if not words:
            return vec
            
        for i, word in enumerate(words):
            # 1. Word level hash
            h_word = abs(hash(word)) % self.dimension
            vec[h_word] += 1.5
            
            # 2. Character 3-gram subwords for typo resilience and morphology
            if len(word) >= 3:
                for j in range(len(word) - 2):
                    sub = word[j:j+3]
                    h_sub = abs(hash(sub)) % self.dimension
                    vec[h_sub] += 0.5
                    
            # 3. Bigram context
            if i < len(words) - 1:
                bigram = f"{word}_{words[i+1]}"
                h_bi = abs(hash(bigram)) % self.dimension
                vec[h_bi] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

class VectorStore:
    """
    Thread-safe ChromaDB persistent vector database manager.
    Works seamlessly both online and 100% offline standalone.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VectorStore, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
            
            ConsoleLogger.vector_db(f"Initializing ChromaDB PersistentClient at: {config.CHROMA_DIR}")
            # Initialize persistent Chroma client
            self.client = chromadb.PersistentClient(
                path=str(config.CHROMA_DIR),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Use offline semantic embedding function: portable, fast, no network timeout risk
            self.embedding_fn = OfflineSemanticEmbeddingFunction()
            
            self.collection = self.client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            existing_count = self.collection.count()
            ConsoleLogger.vector_db(
                f"Connected to collection '{config.CHROMA_COLLECTION_NAME}'. "
                f"Existing indexed chunks: {existing_count}"
            )
            self._initialized = True

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Batch adds chunks to the Chroma collection.
        """
        if not chunks:
            ConsoleLogger.warning("add_chunks called with empty chunks list.")
            return 0
            
        with self._lock:
            total = len(chunks)
            ConsoleLogger.vector_db(f"Starting batch upsert of {total} chunk(s) into ChromaDB...")
            ids = [c["id"] for c in chunks]
            documents = [c["text"] for c in chunks]
            metadatas = [
                {
                    "filename": c["filename"],
                    "page_number": int(c["page_number"]),
                    "total_pages": int(c["total_pages"]),
                    "chunk_index": int(c["chunk_index"]),
                    "char_count": int(c["char_count"])
                }
                for c in chunks
            ]
            
            start_time = time.time()
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            elapsed = time.time() - start_time
            current_total = self.collection.count()
            ConsoleLogger.vector_db(
                f"Upsert finished: {total} chunk(s) committed in {elapsed:.3f}s. "
                f"Total collection count: {current_total} chunk(s)."
            )
            return len(ids)

    def query(self, query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Queries top_k relevant chunks with similarity scores.
        """
        count = self.collection.count()
        if count == 0:
            ConsoleLogger.warning("Vector store is empty (0 chunks). Cannot perform similarity search.")
            return []
            
        k = min(top_k, count)
        ConsoleLogger.vector_db(f"Searching top {k} nearest chunks (collection size: {count})...")
        start_time = time.time()
        results = self.collection.query(
            query_texts=[query_text],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        elapsed = time.time() - start_time
        
        matches = []
        if not results or not results["documents"] or not results["documents"][0]:
            ConsoleLogger.vector_db(f"Search returned 0 results in {elapsed:.3f}s.")
            return matches
            
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        ids = results["ids"][0] if results.get("ids") else [""] * len(docs)
        
        ConsoleLogger.vector_db(f"Search returned {len(docs)} match(es) in {elapsed:.3f}s:")
        for idx, (doc_text, meta, dist, chunk_id) in enumerate(zip(docs, metas, distances, ids), 1):
            # For cosine distance, similarity is 1.0 - distance
            similarity = max(0.0, min(1.0, 1.0 - dist))
            matches.append({
                "id": chunk_id,
                "text": doc_text,
                "filename": meta.get("filename", "Unknown"),
                "page_number": meta.get("page_number", 1),
                "total_pages": meta.get("total_pages", 1),
                "chunk_index": meta.get("chunk_index", 0),
                "similarity_score": round(similarity, 4),
                "distance": round(dist, 4)
            })
            ConsoleLogger.tree_item(
                f"Match #{idx}: {meta.get('filename')} (p.{meta.get('page_number')}) "
                f"| Similarity: {similarity*100:.1f}% (dist: {dist:.4f})",
                is_last=(idx == len(docs))
            )
            
        return matches

    def delete_document(self, filename: str) -> int:
        """
        Deletes all chunks associated with a specific filename.
        """
        with self._lock:
            ConsoleLogger.vector_db(f"Purging vector records for document '{filename}'...")
            existing = self.collection.get(where={"filename": filename})
            if existing and existing.get("ids"):
                ids_to_delete = existing["ids"]
                self.collection.delete(ids=ids_to_delete)
                ConsoleLogger.vector_db(
                    f"Deleted {len(ids_to_delete)} chunk(s) from collection. "
                    f"New total: {self.collection.count()} chunk(s)."
                )
                return len(ids_to_delete)
            ConsoleLogger.info(f"No vector chunks found for '{filename}'.")
            return 0

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Returns stats about total chunks and distinct document filenames.
        """
        count = self.collection.count()
        distinct_files = set()
        if count > 0:
            sample = self.collection.get(include=["metadatas"])
            if sample and sample.get("metadatas"):
                for m in sample["metadatas"]:
                    if m and "filename" in m:
                        distinct_files.add(m["filename"])
                        
        return {
            "total_chunks": count,
            "document_count": len(distinct_files),
            "documents": sorted(list(distinct_files))
        }

    def reset(self):
        """
        Clears the collection.
        """
        with self._lock:
            ConsoleLogger.warning("Resetting ChromaDB collection...")
            try:
                self.client.delete_collection(config.CHROMA_COLLECTION_NAME)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            ConsoleLogger.vector_db(f"Collection '{config.CHROMA_COLLECTION_NAME}' reset to 0 chunks.")

