import sys
import time
from datetime import datetime
from typing import Optional, Any

# ANSI color codes for vibrant terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    # Background accents
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"

def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _print(prefix: str, message: str, color: str = Colors.RESET):
    ts = f"{Colors.DIM}[{_timestamp()}]{Colors.RESET}"
    tag = f"{color}{Colors.BOLD}{prefix}{Colors.RESET}"
    print(f"{ts} {tag} {message}", flush=True)

class ConsoleLogger:
    @staticmethod
    def header(title: str):
        line = "━" * 68
        print(f"\n{Colors.CYAN}{Colors.BOLD}┏{line}┓{Colors.RESET}", flush=True)
        print(f"{Colors.CYAN}{Colors.BOLD}┃  🚀 {title.center(64)}  ┃{Colors.RESET}", flush=True)
        print(f"{Colors.CYAN}{Colors.BOLD}┗{line}┛{Colors.RESET}", flush=True)

    @staticmethod
    def section(title: str):
        line = "─" * 64
        print(f"\n{Colors.BLUE}{Colors.BOLD}── [ {title} ] {line[:64 - len(title) - 6]}{Colors.RESET}", flush=True)

    @staticmethod
    def upload(message: str):
        _print("[FILE UPLOAD]", message, Colors.BLUE)

    @staticmethod
    def parser(message: str):
        _print("[PDF PARSER] ", message, Colors.CYAN)

    @staticmethod
    def chunker(message: str):
        _print("[CHUNKING]   ", message, Colors.YELLOW)

    @staticmethod
    def embedding(message: str):
        _print("[EMBEDDINGS] ", message, Colors.MAGENTA)

    @staticmethod
    def vector_db(message: str):
        _print("[VECTOR DB]  ", message, Colors.GREEN)

    @staticmethod
    def query(message: str):
        _print("[RAG QUERY]  ", message, Colors.CYAN)

    @staticmethod
    def success(message: str):
        _print("[SUCCESS]    ", message, Colors.GREEN)

    @staticmethod
    def info(message: str):
        _print("[INFO]       ", message, Colors.WHITE)

    @staticmethod
    def warning(message: str):
        _print("[WARNING]    ", message, Colors.YELLOW)

    @staticmethod
    def error(message: str):
        _print("[ERROR]      ", message, Colors.RED)

    @staticmethod
    def tree_item(message: str, is_last: bool = False, color: str = Colors.WHITE):
        prefix = "   └── " if is_last else "   ├── "
        print(f"{Colors.DIM}{prefix}{Colors.RESET}{color}{message}{Colors.RESET}", flush=True)

    @staticmethod
    def divider():
        print(f"{Colors.DIM}{'─' * 70}{Colors.RESET}", flush=True)
