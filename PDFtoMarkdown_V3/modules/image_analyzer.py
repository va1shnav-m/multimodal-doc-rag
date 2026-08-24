from pathlib import Path
import json
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

MAX_IMAGE_SIZE = 512

PHASH_THRESHOLD = 4

MODEL = "qwen3-vl:2b-instruct"

# ------------------------------------
# Cache
# ------------------------------------


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
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
# PaddleOCR Extraction
# ------------------------------------


def extract_raw_ocr(image_path):
    """Extract raw text tokens using PaddleOCR / RapidOCR engine."""
    if _ocr_engine is None:
        return []
    try:
        results, _ = _ocr_engine(str(image_path))
        if not results:
            return []
        return [r[1].strip() for r in results if len(r) > 1 and r[1].strip()]
    except Exception:
        return []


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
    """Strip conversational thoughts and monologue artifacts."""
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


def parse_analysis_response(response_text):
    """
    Parse the VLM response into category, text extract,
    and description.
    """
    text = response_text.strip()
    category = "general"
    text_extract = ""
    description = ""

    lines = text.splitlines()

    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("**category:**"):
            value = stripped.split(":", 1)[1].strip().strip("*").strip()
            if "technical" in value:
                category = "technical"
            else:
                category = "general"
            break

    current_section = None
    text_extract_lines = []
    description_lines = []

    for line in lines:
        stripped = line.strip().lower()

        if stripped.startswith("## text extract") or stripped.startswith("## text:"):
            current_section = "text_extract"
            if ":" in line and not stripped.startswith("## text extract"):
                after_col = line.split(":", 1)[1].strip()
                if after_col:
                    text_extract_lines.append(after_col)
            continue

        elif stripped.startswith("## description"):
            current_section = "description"
            if ":" in line and not stripped.startswith("## description"):
                after_col = line.split(":", 1)[1].strip()
                if after_col:
                    description_lines.append(after_col)
            continue

        elif stripped.startswith("**category:**"):
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
            elif low.startswith("- description:"):
                description = _filter_conversational_lines(s.split(":", 1)[1].strip())

    # Fallback if headings were omitted by the model
    if not description and not text_extract and text:
        description = _filter_conversational_lines(text)

    if text_extract.lower() in {"no text content.", "no text content", "none", "n/a"}:
        text_extract = ""

    return {
        "category": category,
        "text_extract": text_extract,
        "description": description,
    }


# ------------------------------------
# Image Analyzer
# ------------------------------------


def analyze_images(assets_dir):
    """
    Analyze all images inside assets folder using 2-Stage Pipeline:
    Stage 1: High-speed PaddleOCR text extraction (< 0.2s)
    Stage 2: Relational VLM structuring and contextual description (~10-15s)
    """
    assets_dir = Path(assets_dir)
    cache = load_cache()

    results = {}
    analysis_times = {}

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
        # Cache Check
        # ------------------------------------
        current_hash = image_hash(image_path)
        matched_hash = find_similar_hash(current_hash, cache)

        if matched_hash:
            cached_result = cache[matched_hash]
            results[image_path.name] = cached_result
            cached += 1
            image_end = time.perf_counter()
            analysis_time = image_end - image_start
            analysis_times[image_path.name] = analysis_time
            print(
                f"{image_path.name} [{cached_result['category']}] "
                f"cached in {analysis_time:.2f} seconds"
            )
            continue

        # ------------------------------------
        # Stage 1: PaddleOCR Extraction
        # ------------------------------------
        ocr_tokens = extract_raw_ocr(image_path)
        if ocr_tokens:
            ocr_block = "\n".join(f"- {tok}" for tok in ocr_tokens)
        else:
            ocr_block = "No raw text detected."

        # ------------------------------------
        # Stage 2: Relational VLM Structuring
        # ------------------------------------
        resized = resize_image(image_path)

        prompt = f"""You are a technical document analyst.
Below are the exact OCR text tokens extracted from this image:

--- RAW OCR TOKENS ---
{ocr_block}
-----------------------

Using both the image and the raw OCR tokens above, output the information in the exact format below:

## Text Extract
[Structure the relationships between the OCR tokens:
- IF Flowcharts/Pipelines: `[Source Node] --(Connector Label)--> [Destination Node]`
-IF Decisions: `[Decision Question?] -> If Yes: [Target Node] | If No: [Target Node]`
-IF Architecture: `[Component A] → [Component B] (Protocol/Data)`
-IF Tables: Table wise OCR text as same as in the given image.
-IF UI / Forms / Screenshot: List sections with field labels and buttons
- If no text exists in the image, output: No text content.]

## Description
[2-3 concise sentences explaining the technical context, workflow, and purpose of the image.]"""

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
                    "num_predict": 1000,
                    "num_thread": 8,
                }
            )

        except Exception as e:
            failed += 1
            print(f"Analysis failed for {image_path.name}: {e}")
            results[image_path.name] = {
                "category": "general",
                "text_extract": "",
                "description": "",
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

        parsed = parse_analysis_response(raw_response)
        parsed["raw_ocr"] = ocr_tokens
        results[image_path.name] = parsed
        cache[str(current_hash)] = parsed
        generated += 1

        image_end = time.perf_counter()
        analysis_time = image_end - image_start
        analysis_times[image_path.name] = analysis_time

        print(
            f"{image_path.name} [{parsed['category']}] "
            f"analyzed in {analysis_time:.2f} seconds"
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
