import time
import hashlib
from typing import List, Dict, Any
from utils.console import ConsoleLogger

class TextChunker:
    """
    Chunks extracted PDF page texts while preserving page numbers and source citations.
    """
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        start_time = time.time()
        ConsoleLogger.chunker(
            f"Starting chunking pipeline: chunk_size={self.chunk_size} chars, "
            f"overlap={self.chunk_overlap} chars on {len(pages_data)} page(s)."
        )
        
        chunks = []
        page_chunk_counts = {}
        
        for page in pages_data:
            text = page["text"]
            filename = page["filename"]
            page_number = page["page_number"]
            total_pages = page["total_pages"]
            chunks_before = len(chunks)
            
            # If page text is shorter than chunk size, it becomes a single chunk
            if len(text) <= self.chunk_size:
                chunk_id = self._generate_id(filename, page_number, 0, text)
                chunks.append({
                    "id": chunk_id,
                    "text": text,
                    "filename": filename,
                    "page_number": page_number,
                    "total_pages": total_pages,
                    "chunk_index": 0,
                    "char_count": len(text)
                })
                page_chunk_counts[page_number] = 1
                continue
                
            # Sliding window with overlap
            start = 0
            chunk_idx = 0
            text_len = len(text)
            
            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                
                # If we're not at the end of the text, try breaking cleanly on sentence or space boundary
                if end < text_len:
                    last_period = text.rfind(". ", start, end)
                    if last_period != -1 and last_period > start + (self.chunk_size // 2):
                        end = last_period + 1
                    else:
                        last_space = text.rfind(" ", start, end)
                        if last_space != -1 and last_space > start + (self.chunk_size // 2):
                            end = last_space
                            
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunk_id = self._generate_id(filename, page_number, chunk_idx, chunk_text)
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "filename": filename,
                        "page_number": page_number,
                        "total_pages": total_pages,
                        "chunk_index": chunk_idx,
                        "char_count": len(chunk_text)
                    })
                    chunk_idx += 1
                
                if end >= text_len:
                    break
                    
                # Advance start with overlap
                start = max(end - self.chunk_overlap, start + 1)
                
            page_chunk_counts[page_number] = len(chunks) - chunks_before

        elapsed = time.time() - start_time
        
        # Print breakdown for pages
        total_p = len(pages_data)
        for idx, (p_num, count) in enumerate(page_chunk_counts.items()):
            is_last = (idx == total_p - 1)
            ConsoleLogger.tree_item(f"Page {p_num}: {count} chunk(s) generated", is_last=is_last)
            
        if chunks:
            char_lengths = [c["char_count"] for c in chunks]
            avg_len = sum(char_lengths) // len(char_lengths)
            min_len = min(char_lengths)
            max_len = max(char_lengths)
            ConsoleLogger.chunker(
                f"Generated {len(chunks)} total chunks (avg: {avg_len} chars, "
                f"range: [{min_len}, {max_len}] chars) in {elapsed:.2f}s."
            )
            # Preview first chunk
            first_preview = chunks[0]["text"][:60].replace("\n", " ")
            ConsoleLogger.info(f"Sample chunk #1 [{chunks[0]['id']}]: \"{first_preview}...\"")
        else:
            ConsoleLogger.warning("No chunks generated (document pages contained no text).")
            
        return chunks

    def _generate_id(self, filename: str, page_number: int, chunk_idx: int, text: str) -> str:
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        clean_name = "".join(c for c in filename if c.isalnum() or c in ("-", "_")).rstrip()
        return f"{clean_name}_p{page_number}_c{chunk_idx}_{text_hash}"

