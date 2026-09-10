import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

import config

class ReviewDB:
    """
    SQLite database with WAL mode for high-concurrency logging,
    document tracking, user self-reviews, and expert/admin review queue.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or config.SQLITE_DB_PATH)
        self.init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        # Enable Write-Ahead Logging for high concurrency (handles 50+ simultaneous readers/writers)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self):
        with self.get_connection() as conn:
            # Documents table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER DEFAULT 0,
                total_pages INTEGER DEFAULT 1,
                chunk_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Queries table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id TEXT PRIMARY KEY,
                session_id TEXT DEFAULT 'default',
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                provider TEXT DEFAULT 'offline',
                sources_json TEXT DEFAULT '[]',
                is_grounded INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Reviews table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id TEXT PRIMARY KEY,
                query_id TEXT NOT NULL UNIQUE,
                user_rating TEXT DEFAULT NULL,
                user_notes TEXT DEFAULT NULL,
                user_verified INTEGER DEFAULT 0,
                admin_review_requested INTEGER DEFAULT 0,
                request_reason TEXT DEFAULT NULL,
                status TEXT DEFAULT 'UNREVIEWED',
                admin_id TEXT DEFAULT NULL,
                admin_notes TEXT DEFAULT NULL,
                admin_edited_answer TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES queries (id) ON DELETE CASCADE
            );
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews (status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_created ON queries (created_at DESC);")

    # Document management
    def register_document(self, filename: str, file_path: str, file_size: int, total_pages: int, chunk_count: int) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (filename, file_path, file_size, total_pages, chunk_count, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(filename) DO UPDATE SET
                    file_path=excluded.file_path,
                    file_size=excluded.file_size,
                    total_pages=excluded.total_pages,
                    chunk_count=excluded.chunk_count,
                    created_at=CURRENT_TIMESTAMP
            """, (filename, file_path, file_size, total_pages, chunk_count))
            return cursor.lastrowid

    def list_documents(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def delete_document(self, filename: str):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))

    # Query logging
    def log_query(self, question: str, answer: str, provider: str, sources: List[Dict[str, Any]], session_id: str = "default", is_grounded: bool = True) -> str:
        query_id = str(uuid.uuid4())
        sources_json = json.dumps(sources)
        review_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO queries (id, session_id, question, answer, provider, sources_json, is_grounded, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (query_id, session_id, question, answer, provider, sources_json, 1 if is_grounded else 0))
            
            # Create matching review record
            conn.execute("""
                INSERT INTO reviews (id, query_id, status, created_at, updated_at)
                VALUES (?, ?, 'UNREVIEWED', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (review_id, query_id))
            
        return query_id

    # User self-review
    def submit_user_review(self, query_id: str, rating: Optional[str] = None, notes: Optional[str] = None, verified: bool = False) -> Dict[str, Any]:
        with self.get_connection() as conn:
            status = 'USER_VERIFIED' if verified else ('USER_RATED' if rating else 'UNREVIEWED')
            conn.execute("""
                UPDATE reviews
                SET user_rating = COALESCE(?, user_rating),
                    user_notes = COALESCE(?, user_notes),
                    user_verified = ?,
                    status = CASE WHEN status = 'PENDING_ADMIN' THEN 'PENDING_ADMIN' ELSE ? END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE query_id = ?
            """, (rating, notes, 1 if verified else 0, status, query_id))
            
        return self.get_review_by_query(query_id)

    # Request expert / admin review
    def request_admin_review(self, query_id: str, reason: str = "") -> Dict[str, Any]:
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE reviews
                SET admin_review_requested = 1,
                    request_reason = ?,
                    status = 'PENDING_ADMIN',
                    updated_at = CURRENT_TIMESTAMP
                WHERE query_id = ?
            """, (reason, query_id))
            
        return self.get_review_by_query(query_id)

    # Admin / Expert review actions
    def submit_admin_action(self, query_id: str, action: str, admin_id: str = "expert_reviewer", notes: str = "", edited_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        action: 'APPROVE', 'REJECT', or 'MODIFY'
        """
        status_map = {
            'APPROVE': 'ADMIN_APPROVED',
            'REJECT': 'ADMIN_REJECTED',
            'MODIFY': 'ADMIN_MODIFIED'
        }
        new_status = status_map.get(action.upper(), 'ADMIN_APPROVED')
        
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE reviews
                SET status = ?,
                    admin_id = ?,
                    admin_notes = ?,
                    admin_edited_answer = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE query_id = ?
            """, (new_status, admin_id, notes, edited_answer, query_id))
            
        return self.get_review_by_query(query_id)

    def get_review_by_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT q.id as query_id, q.question, q.answer, q.provider, q.sources_json, q.is_grounded, q.created_at as query_time,
                       r.id as review_id, r.user_rating, r.user_notes, r.user_verified, r.admin_review_requested,
                       r.request_reason, r.status, r.admin_id, r.admin_notes, r.admin_edited_answer, r.updated_at
                FROM queries q
                LEFT JOIN reviews r ON q.id = r.query_id
                WHERE q.id = ?
            """, (query_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["sources"] = json.loads(res.get("sources_json") or "[]")
            return res

    def list_reviews(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT q.id as query_id, q.question, q.answer, q.provider, q.sources_json, q.is_grounded, q.created_at as query_time,
                       r.id as review_id, r.user_rating, r.user_notes, r.user_verified, r.admin_review_requested,
                       r.request_reason, r.status, r.admin_id, r.admin_notes, r.admin_edited_answer, r.updated_at
                FROM queries q
                JOIN reviews r ON q.id = r.query_id
            """
            params = []
            if status and status.upper() != "ALL":
                query += " WHERE r.status = ? "
                params.append(status.upper())
                
            query += " ORDER BY q.created_at DESC LIMIT ? "
            params.append(limit)
            
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["sources"] = json.loads(item.get("sources_json") or "[]")
                results.append(item)
            return results

    def get_stats(self) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM queries")
            query_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM reviews WHERE status = 'PENDING_ADMIN' OR admin_review_requested = 1")
            pending_admin = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM reviews WHERE user_verified = 1 OR status = 'ADMIN_APPROVED'")
            verified_count = cursor.fetchone()[0]
            
            return {
                "total_documents": doc_count,
                "total_queries": query_count,
                "pending_admin_reviews": pending_admin,
                "verified_answers": verified_count
            }
