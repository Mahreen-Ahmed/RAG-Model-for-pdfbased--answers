import os
import time
from pathlib import Path
from typing import List, Dict, Any
import pymupdf  # PyMuPDF
from utils.console import ConsoleLogger

class PDFParser:
    """
    Extracts text and metadata from PDF files page-by-page.
    """
    
    @staticmethod
    def extract_pages(pdf_path: str | Path) -> List[Dict[str, Any]]:
        """
        Parses a PDF file and returns a list of pages with clean text and metadata.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            ConsoleLogger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
            
        start_time = time.time()
        filename = pdf_path.name
        ConsoleLogger.parser(f"Opening '{filename}' with PyMuPDF parser...")
        
        doc = pymupdf.open(str(pdf_path))
        pages_data = []
        total_chars = 0
        
        try:
            total_pages = len(doc)
            ConsoleLogger.parser(f"Detected {total_pages} total page(s) in document.")
            
            for page_idx in range(total_pages):
                page_num = page_idx + 1
                page = doc.load_page(page_idx)
                # Extract text
                text = page.get_text("text")
                clean_text = " ".join(text.split())
                char_count = len(clean_text)
                
                # Only include pages that have non-trivial text
                if clean_text:
                    total_chars += char_count
                    pages_data.append({
                        "filename": filename,
                        "file_path": str(pdf_path),
                        "page_number": page_num,
                        "total_pages": total_pages,
                        "text": clean_text
                    })
                    ConsoleLogger.tree_item(
                        f"Page {page_num}/{total_pages}: {char_count:,} characters extracted",
                        is_last=(page_idx == total_pages - 1)
                    )
                else:
                    ConsoleLogger.tree_item(
                        f"Page {page_num}/{total_pages}: (Empty or image-only page, skipped)",
                        is_last=(page_idx == total_pages - 1)
                    )
        finally:
            doc.close()
            
        elapsed = time.time() - start_time
        ConsoleLogger.parser(
            f"Extraction completed: {len(pages_data)} active text page(s), "
            f"{total_chars:,} characters in {elapsed:.2f}s."
        )
        return pages_data

