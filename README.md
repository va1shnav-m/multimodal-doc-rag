# Multimodal Document Parsing and RAG Engine

A terminal-based system for converting documents (PDF, DOCX, PPTX, Audio) into structured Markdown and querying them using a local RAG pipeline.

---

## Quick Setup

### 1. Create & Activate Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Linux/macOS: source .venv/bin/activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Pull Ollama Models
Ensure [Ollama](https://ollama.com) is installed and running:
```powershell
ollama pull qwen3-vl:2b-instruct   # For diagram & image analysis
ollama pull qwen2.5:7b             # For terminal chat & Q&A
```

---

## Essential Commands

All commands run through `main.py`:

### Document Conversion (`convert`)
```powershell
# Convert PDF to Markdown + extracted images:
python main.py convert PDFtoMarkdown_V3/input/test1.pdf --out output/

# Fast conversion (skip image analysis):
python main.py convert PDFtoMarkdown_V3/input/test1.pdf --out output/ --no-analysis

# Convert folder or other formats (.docx, .pptx, audio .mp3):
python main.py convert PDFtoMarkdown_V3/input/ --out output/
```

### Convert + RAG Interactive Chat
```powershell
# Convert document, ingest into RAG, and start terminal chat:
python main.py convert PDFtoMarkdown_V3/input/test1.pdf --out output/ --rag --chat

# Ask a direct question on a document:
python main.py convert PDFtoMarkdown_V3/input/test1.pdf --out output/ --rag --query "What is the project budget?"
```

### Query Existing Knowledge Base (`rag`)
```powershell
# Ask a question:
python main.py rag --query "Explain the recruitment workflow"

# Start interactive terminal chat:
python main.py rag --chat

# Ingest an existing markdown folder:
python main.py rag --ingest output/

# Check database status:
python main.py rag --status
```
