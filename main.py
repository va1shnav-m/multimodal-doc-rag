"""
Production RAG Unified CLI Entry Point
Run from the root production_rag directory.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_ROOT = PROJECT_ROOT / "PDFtoMarkdown_V3"
QDRANT_ROOT = PROJECT_ROOT / "qdrant_vectordb"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PDF_ROOT) not in sys.path:
    sys.path.insert(0, str(PDF_ROOT))
if str(QDRANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QDRANT_ROOT))

# Forward execution to main CLI
from PDFtoMarkdown_V3.main import main

if __name__ == "__main__":
    main()
