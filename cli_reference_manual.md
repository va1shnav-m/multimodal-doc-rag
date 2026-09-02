# Production RAG — CLI Commands & Flags Manual

A focused reference guide detailing every CLI command, argument, flag, and filter implemented in the project.

---

## 1. Command: `convert`

Converts documents (PDF, DOCX, PPTX) and audio files (MP3, WAV, M4A, etc.) to Markdown with embedded OCR extracts, image links, and optional RAG vector ingestion.

### Syntax
```powershell
python main.py convert <inputs...> --out <output_folder> [flags]
```

### Positional Arguments
| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `inputs` | `path ...` | **Yes** | One or more file paths or directory paths. Accepts single files, lists of files, or entire folders. |

### All Flags & Options
| Flag | Values / Choices | Default | Description |
| :--- | :--- | :---: | :--- |
| `--out` | `path` | **Required** | Directory where generated Markdown and `assets/` images are saved. |
| `--pipeline` | `hybrid` \| `docling` | `hybrid` | • `hybrid`: PyMuPDF for fast text pages (0.01s/p) + Docling for complex pages.<br>• `docling`: Docling across all pages. |
| `--whisper-model` | `tiny` \| `base` \| `small` \| `medium` | `base` | Model size used for audio speech-to-text transcription. |
| `--no-summary` | *(flag)* | `False` | Disables LLM summary generation for audio files (outputs only raw transcript). |
| `--no-analysis` | *(flag)* | `False` | Disables VLM and RapidOCR image analysis (saves images to `assets/` without Markdown blockquotes). |
| `--force` | *(flag)* | `False` | Overwrites output directory without prompting (archives prior run to `storage/archive/`). |
| `--rag` | *(flag)* | `False` | Ingests converted Markdown into Qdrant vector database immediately upon conversion. |
| `--append` | *(flag)* | `False` | • **Default (`False`)**: Clears prior database index so queries strictly reference the current upload.<br>• **`--append` (`True`)**: Adds to existing database without deleting earlier documents. |
| `--collection-name` | `string` | `production_rag` | Target Qdrant collection name. |
| `--rag-pipeline` | `hybrid` \| `fast` \| `adaptive` | `hybrid` | Retrieval strategy for immediate query or chat:<br>• `hybrid`: Dense + BM25 + Rerank + LLMLingua compression.<br>• `fast`: Direct dense vector search.<br>• `adaptive`: Dynamic query classifier and router. |
| `--llm` | `gemini` \| `openai` \| `qwen` | `gemini` | LLM backend for generation (`gemini` via Google AI Studio, `openai`, or `qwen` via local Ollama). |
| `--query` | `string` | `None` | Asks a single one-shot question immediately after conversion and ingestion. |
| `--chat` | *(flag)* | `False` | Launches an interactive terminal chat session immediately after conversion and ingestion. |

### `convert` Usage Examples
```powershell
# Convert a single PDF to Markdown
python main.py convert PDFtoMarkdown_V3/input/test3.pdf --out output/

# Convert all documents in a folder
python main.py convert PDFtoMarkdown_V3/input/ --out output/ --pipeline hybrid

# Convert an audio recording with small whisper model
python main.py convert recording.mp3 --out output/ --whisper-model small

# Convert document, replace RAG knowledge base with it, and launch Chat
python main.py convert PDFtoMarkdown_V3/input/test3.pdf --out output/ --rag --chat

# Convert and append to existing knowledge base, then run one-shot query
python main.py convert PDFtoMarkdown_V3/input/test2.pdf --out output/ --rag --append --query "What are the deliverables?"
```

---

## 2. Command: `rag`

Manages, inspects, queries, and chats with the Qdrant Vector Knowledge Base independently of the conversion step.

### Syntax
```powershell
python main.py rag [flags]
```

### All Flags & Options
| Flag | Values / Choices | Default | Description |
| :--- | :--- | :---: | :--- |
| `--ingest` | `path ...` | `None` | Ingests one or more existing Markdown (`.md`) files or directories into Qdrant & BM25 index. |
| `--append` | *(flag)* | `False` | • **Default (`False`)**: Clears prior database index so queries strictly reference the ingested file(s).<br>• **`--append` (`True`)**: Preserves earlier documents and adds new files cumulatively. |
| `--query` | `string` | `None` | Asks a single question against indexed documents and prints the answer with sources, evidence, and diagram paths. |
| `--chat` | *(flag)* | `False` | Starts an interactive multi-turn terminal conversation with session memory. |
| `--status` | *(flag)* | `False` | Displays Qdrant vector statistics, collection name, vector count, dimension, distance metric, and indexed document list. |
| `--clear` | *(flag)* | `False` | Clears all vectors, node chunks, metadata store, and BM25 index. |
| `--collection-name` | `string` | `production_rag` | Target Qdrant collection name. |
| `--pipeline` | `hybrid` \| `fast` \| `adaptive` | `hybrid` | Retrieval engine mode (`hybrid`, `fast`, `adaptive`). |
| `--llm` | `gemini` \| `openai` \| `qwen` | `gemini` | LLM generation backend (`gemini`, `openai`, or `qwen`). |

### `rag` Usage Examples
```powershell
# Ingest converted Markdown (Clean Slate) and start Chat
python main.py rag --ingest output/document_0001.md --chat

# Ingest converted Markdown and run a one-shot query
python main.py rag --ingest output/document_0001.md --query "What is the project scope?"

# Start interactive chat against currently indexed documents
python main.py rag --chat

# Run a one-shot query using Fast retrieval mode
python main.py rag --query "Summarize the key requirements" --pipeline fast

# Inspect database metrics and indexed files
python main.py rag --status

# Clear and reset the database index
python main.py rag --clear
```

---

## 3. Built-In System Filters

These internal filters control what files are processed, which images are analyzed, and how outputs are structured.

### Supported File Types Filter
Only files matching these extensions are processed; all others are ignored:
- **Documents**: `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`
- **Audio**: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.wma`

### Image Quality & Rejection Filters
Images extracted from documents pass through these checks before VLM/OCR:

| Filter | Rule / Threshold | Action if Triggered |
| :--- | :--- | :--- |
| **Small Dimension** | `width < 60` and `height < 60` | Skipped (tiny icon / bullet) |
| **Thin Strip** | `min(width, height) < 30` | Skipped (divider line / border) |
| **Small Surface Area** | `width * height < 3000` px | Skipped (small graphic) |
| **Extreme Aspect Ratio** | `aspect_ratio > 6.0` | Skipped (horizontal banner / vertical bar) |
| **Solid Color** | Single unique color detected | Skipped (blank background block) |

### Deduplication & Label Filters
- **Logo Deduplication**: Images matching an earlier page's logo with perceptual hash distance $\le 5$ are tagged as duplicate headers and suppressed from repeating across Markdown pages.
- **Fast-Path Filter**: Images with $\le 2$ short OCR words (e.g. badges, button labels) bypass VLM inference and produce structured output in **0.01s**.

---

## 4. Master Flag Cheat Sheet

| Command | Flag | Type | Default | What It Does |
| :--- | :--- | :---: | :---: | :--- |
| `convert` | `inputs` | `path ...` | *Required* | File or directory paths to convert |
| `convert` | `--out` | `path` | *Required* | Output directory for markdown & assets |
| `convert` | `--pipeline` | `hybrid` \| `docling` | `hybrid` | Page layout routing engine |
| `convert` | `--whisper-model` | `tiny` \| `base` \| `small` \| `medium` | `base` | Audio transcription model size |
| `convert` | `--no-summary` | `flag` | `False` | Skip LLM summary for audio |
| `convert` | `--no-analysis` | `flag` | `False` | Skip image VLM & RapidOCR analysis |
| `convert` | `--force` | `flag` | `False` | Overwrite output directory |
| `convert` | `--rag` | `flag` | `False` | Ingest into Qdrant after conversion |
| `convert` | `--append` | `flag` | `False` | Append to RAG (default: replaces old docs) |
| `convert` | `--collection-name` | `string` | `production_rag` | Qdrant collection name |
| `convert` | `--rag-pipeline` | `hybrid` \| `fast` \| `adaptive` | `hybrid` | RAG retrieval strategy |
| `convert` | `--llm` | `gemini` \| `openai` \| `qwen` | `gemini` | LLM model for generation |
| `convert` | `--query` | `string` | `None` | Ask single question after conversion |
| `convert` | `--chat` | `flag` | `False` | Start interactive terminal chat |
| `rag` | `--ingest` | `path ...` | `None` | Ingest markdown file(s) into RAG |
| `rag` | `--append` | `flag` | `False` | Append to RAG (default: replaces old docs) |
| `rag` | `--query` | `string` | `None` | Query indexed documents |
| `rag` | `--chat` | `flag` | `False` | Start interactive terminal chat |
| `rag` | `--status` | `flag` | `False` | Show database statistics & doc list |
| `rag` | `--clear` | `flag` | `False` | Clear all vectors, nodes, and BM25 index |
| `rag` | `--collection-name` | `string` | `production_rag` | Qdrant collection name |
| `rag` | `--pipeline` | `hybrid` \| `fast` \| `adaptive` | `hybrid` | RAG retrieval mode |
| `rag` | `--llm` | `gemini` \| `openai` \| `qwen` | `gemini` | LLM model for generation |
