from pathlib import Path
import json
import re
import tempfile
import time

import imagehash
import ollama
from PIL import Image

from modules.image_filter import should_process
from modules.ollama_utils import ensure_ollama_running

try:
    from rapidocr_onnxruntime import RapidOCR
    _ocr_engine = RapidOCR()
except Exception:
    _ocr_engine = None

# ------------------------------------
# Configuration
# ------------------------------------

CACHE_FILE = Path("image_analysis_cache.json")

MAX_IMAGE_SIZE = 1024

PHASH_THRESHOLD = 4

MODEL = "qwen3-vl:2b-instruct"

# ------------------------------------
# Cache
# ------------------------------------


def load_cache():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=4, ensure_ascii=False)


# ------------------------------------
# Perceptual Hash
# ------------------------------------


def image_hash(image_path):
    return imagehash.phash(Image.open(image_path))


def find_similar_hash(current_hash, cache):
    for cached_hash in cache.keys():
        cached = imagehash.hex_to_hash(cached_hash)
        distance = current_hash - cached
        if distance <= PHASH_THRESHOLD:
            return cached_hash
    return None


# ------------------------------------
# Resize
# ------------------------------------


def resize_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(temp.name)
    return temp.name


# ------------------------------------
# OCR Text Extraction
# ------------------------------------


def extract_raw_ocr(image_path):
    """Extract raw text tokens using RapidOCR engine."""
    if _ocr_engine is None:
        return []
    try:
        results, _ = _ocr_engine(str(image_path))
        if not results:
            return []
        tokens = [r[1].strip() for r in results if len(r) > 1 and r[1].strip()]
        return tokens
    except Exception:
        return []


def format_ocr_fallback(tokens):
    """Format raw OCR tokens into structured clean lines."""
    if not tokens:
        return ""
    lines = []
    for tok in tokens:
        cleaned = tok.strip()
        if cleaned:
            lines.append(f"- {cleaned}")
    return "\n".join(lines)


# ------------------------------------
# Parse VLM Response
# ------------------------------------

CONVERSATIONAL_PREFIXES = (
    "got it", "let's", "let us", "first,", "first ", "now,", "now ",
    "let me", "wait,", "so category", "first check", "first identify",
    "identify the", "structure the", "now structure", "so structure",
    "the problem is to", "need to list", "so the image shows", "okay, let's", "okay,"
)


def _filter_conversational_lines(text: str) -> str:
    """Strip conversational thoughts, backticks, and monologue artifacts."""
    # Strip triple backtick fences
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = text.replace("```", "")

    cleaned = []
    seen = set()
    for line in text.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if any(low.startswith(p) for p in CONVERSATIONAL_PREFIXES):
            continue
        if stripped and stripped not in seen:
            seen.add(stripped)
            cleaned.append(line)
        elif not stripped and cleaned and cleaned[-1] != "":
            cleaned.append("")
    return "\n".join(cleaned).strip()


def _is_text_extract_header(line: str) -> bool:
    """Check if line matches any text extract heading pattern."""
    clean = re.sub(r"[*#_`:]", "", line).strip().lower()
    return clean in {
        "text extract", "extracted text", "text", "labels",
        "raw text", "detected text", "entities"
    }


def _is_description_header(line: str) -> bool:
    """Check if line matches any description heading pattern."""
    clean = re.sub(r"[*#_`:]", "", line).strip().lower()
    return clean in {
        "description", "summary", "overview", "explanation", "details"
    }


def parse_analysis_response(response_text, raw_tokens=None):
    """
    Parse the VLM response into text extract and description.
    Falls back to structured raw OCR tokens if VLM output lacks text extract.
    """
    text = response_text.strip()
    lines = text.splitlines()
    current_section = None
    text_extract_lines = []
    description_lines = []

    for line in lines:
        stripped = line.strip()

        if _is_text_extract_header(stripped):
            current_section = "text_extract"
            continue
        elif _is_description_header(stripped):
            current_section = "description"
            continue
        elif stripped.lower().startswith("**category:**") or stripped.lower().startswith("category:"):
            continue

        if current_section == "text_extract":
            text_extract_lines.append(line)
        elif current_section == "description":
            description_lines.append(line)

    text_extract = _filter_conversational_lines("\n".join(text_extract_lines))
    description = _filter_conversational_lines("\n".join(description_lines))

    # Fallback to bullet formats (- Labels: ... / - Description: ...)
    if not text_extract or not description:
        for l in text.splitlines():
            s = l.strip()
            low = s.lower()
            if low.startswith("- labels:") or low.startswith("- text:") or low.startswith("- text extract:"):
                text_extract = _filter_conversational_lines(s.split(":", 1)[1].strip())
            elif low.startswith("- description:") or low.startswith("- summary:"):
                description = _filter_conversational_lines(s.split(":", 1)[1].strip())

    # Fallback if headings were omitted
    if not description and not text_extract and text:
        description = _filter_conversational_lines(text)

    # If VLM emitted placeholder rejection strings, clear it
    if text_extract.lower() in {"no text content.", "no text content", "none", "n/a", "no visible text"}:
        text_extract = ""

    # GUARANTEE: If text_extract is empty or minimal, use formatted OCR tokens
    if (not text_extract or len(text_extract.strip()) < 10) and raw_tokens:
        text_extract = format_ocr_fallback(raw_tokens)

    # Guarantee description is not empty
    if not description and raw_tokens:
        description = "Diagram / graphic containing visible technical entities, labels, and structured data flows."

    return {
        "text_extract": text_extract,
        "description": description,
    }


# ------------------------------------
# Image Analyzer
# ------------------------------------


def analyze_images(assets_dir):
    """
    Analyze all images inside assets folder:
    1. Perceptual hash deduplication (suppresses repeated logos across pages)
    2. High-precision OCR text extraction
    3. Fast-path labeling for small text badges
    4. Relational VLM structuring and description for complex diagrams & flowcharts
    """
    assets_dir = Path(assets_dir)
    cache = load_cache()

    results = {}
    analysis_times = {}
    seen_doc_hashes = {}

    image_files = sorted(
        [
            *assets_dir.glob("*.png"),
            *assets_dir.glob("*.jpg"),
            *assets_dir.glob("*.jpeg"),
            *assets_dir.glob("*.webp"),
        ]
    )

    total_start = time.perf_counter()

    skipped = 0
    cached = 0
    generated = 0
    failed = 0

    if not ensure_ollama_running():
        raise RuntimeError(
            "Ollama could not be started. "
            "Please make sure Ollama is installed correctly."
        )

    for image_path in image_files:
        if not image_path.exists():
            skipped += 1
            continue

        image_start = time.perf_counter()

        # ------------------------------------
        # Image Filter
        # ------------------------------------
        if not should_process(image_path):
            skipped += 1
            continue

        # ------------------------------------
        # Document-Level Deduplication (Repeated Logos across pages)
        # ------------------------------------
        current_hash = image_hash(image_path)
        matched_seen = find_similar_hash(current_hash, seen_doc_hashes)

        if matched_seen:
            # Same logo seen on previous page of this document -> mark as duplicate
            print(f"{image_path.name}: Repeated logo/header graphic -> Suppress duplicate insertion")
            results[image_path.name] = {
                "is_duplicate": True,
                "text_extract": "",
                "description": "",
            }
            skipped += 1
            continue

        seen_doc_hashes[str(current_hash)] = image_path.name

        # ------------------------------------
        # Cache Check
        # ------------------------------------
        matched_hash = find_similar_hash(current_hash, cache)

        if matched_hash:
            cached_result = cache[matched_hash]
            if cached_result.get("text_extract") or cached_result.get("description"):
                if "raw_ocr" not in cached_result:
                    cached_result["raw_ocr"] = extract_raw_ocr(image_path)
                results[image_path.name] = cached_result
                cached += 1
                image_end = time.perf_counter()
                analysis_time = image_end - image_start
                analysis_times[image_path.name] = analysis_time
                print(
                    f"{image_path.name} cached in {analysis_time:.2f} seconds"
                )
                continue

        # ------------------------------------
        # Stage 1: Fast OCR Extraction
        # ------------------------------------
        ocr_tokens = extract_raw_ocr(image_path)

        # ------------------------------------
        # Fast-Path for Simple Badges / Small Labels (< 3 tokens)
        # ------------------------------------
        if 0 < len(ocr_tokens) <= 2 and all(len(t) < 30 for t in ocr_tokens):
            label_text = " - ".join(ocr_tokens)
            parsed = {
                "text_extract": "\n".join(f"- {tok}" for tok in ocr_tokens),
                "description": f"Graphic badge / label displaying: {label_text}.",
                "raw_ocr": ocr_tokens,
            }
            results[image_path.name] = parsed
            cache[str(current_hash)] = parsed
            generated += 1
            image_end = time.perf_counter()
            analysis_time = image_end - image_start
            analysis_times[image_path.name] = analysis_time
            print(f"{image_path.name} [Fast Label] processed in {analysis_time:.2f}s")
            continue

        if ocr_tokens:
            ocr_block = "\n".join(f"- {tok}" for tok in ocr_tokens)
        else:
            ocr_block = "No raw text detected."

        # ------------------------------------
        # Stage 2: Relational VLM Structuring
        # ------------------------------------
        resized = resize_image(image_path)

        prompt = f"""You are an expert document visual analyst.
Below is the exact text detected from this image via high-precision OCR:

--- DETECTED TEXT ---
{ocr_block}
---------------------

Analyze this image carefully. You MUST provide two distinct sections using these exact markdown headings (do NOT wrap your answer in code fences):

## Text Extract
Transcribe, organize, and structure ALL visible text, entities, relationships, labels, and UI elements:
- If Flowchart / Process / Architecture:
  Trace the complete sequential flow, branching decisions, and system connections:
  - `[Start Action / Event]` --> `[Next Process]`
  - `[Decision / Condition?]`:
    - If Yes / True --> `[Target Node]`
    - If No / False --> `[Target Node]`
  - System interactions: `[Actor / Component]` <--> `[System DB / Database]`
  - `[Final Action]` --> `[End]`
- If Hub-and-Spoke / Ingestion / Multi-Source Architecture:
  List all incoming feeds pointing to the central system:
  - `[Source 1]` --> `[Central System / Platform]`
  - `[Source 2]` --> `[Central System / Platform]`
- If Database / ER Diagram / Schema:
  Group by Table/Entity name, list all columns/fields, and specify foreign key / relational links (`[TABLE_A] (col1, col2) --> [TABLE_B] (fk)`).
- If UI Mockup / Form / Web Portal:
  Group by window/section, listing every button, input box, label, radio option, and header verbatim.
- If Table / Comparison:
  Output as a clean Markdown table with headers and rows.
- If General Diagram / Document:
  Transcribe all visible titles, text blocks, and bullet points verbatim.

## Description
Provide 2-4 clear, concise sentences explaining the technical architecture, data model, execution workflow, or user interaction depicted in this image."""

        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [resized],
                    }
                ],
                options={
                    "temperature": 0,
                    "top_p": 0.9,
                    "repeat_penalty": 1.05,
                    "num_predict": 1200,
                    "num_thread": 8,
                }
            )

        except Exception as e:
            failed += 1
            print(f"Analysis failed for {image_path.name}: {e}")
            fallback_extract = format_ocr_fallback(ocr_tokens)
            results[image_path.name] = {
                "text_extract": fallback_extract,
                "description": "Image containing extracted text elements.",
                "failed": True,
            }
            continue

        raw_content = response["message"]["content"].strip()
        raw_thinking = getattr(response["message"], "thinking", "")
        if isinstance(raw_thinking, str):
            raw_thinking = raw_thinking.strip()
        else:
            raw_thinking = ""

        raw_response = raw_content if raw_content else raw_thinking

        parsed = parse_analysis_response(raw_response, raw_tokens=ocr_tokens)
        parsed["raw_ocr"] = ocr_tokens
        results[image_path.name] = parsed
        cache[str(current_hash)] = parsed
        generated += 1

        image_end = time.perf_counter()
        analysis_time = image_end - image_start
        analysis_times[image_path.name] = analysis_time

        print(
            f"{image_path.name} analyzed in {analysis_time:.2f} seconds"
        )

    save_cache(cache)
    total_end = time.perf_counter()

    print("\n==============================")
    print("Image Analysis Summary")
    print("==============================")
    print(f"Total Images     : {len(image_files)}")
    print(f"Analyzed         : {generated}")
    print(f"Cached           : {cached}")
    print(f"Skipped          : {skipped}")
    print(f"Failed           : {failed}")
    print(f"Total Time       : {total_end-total_start:.2f} sec")
    print("==============================")

    return {
        "results": results,
        "generated": generated,
        "cached": cached,
        "skipped": skipped,
        "failed": failed,
        "analysis_times": analysis_times,
    }
