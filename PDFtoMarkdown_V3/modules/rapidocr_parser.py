from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
from modules.ocr_cache import (
    load_cache,
    save_cache,
)
from modules.image_filter import should_process

from modules.image_analyzer import (
    image_hash,
    find_similar_hash,
)

def reconstruct_ocr_structure(ocr_result):
    """
    Reconstruct OCR text using bounding-box positions.

    Groups nearby OCR detections into lines and orders
    them from top-to-bottom and left-to-right.
    """

    if not ocr_result:
        return ""

    items = []

    for line in ocr_result:
        bbox = line[0]
        text = line[1]

        if not text or not text.strip():
            continue

        # Get bounding box coordinates
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        height = y_max - y_min
        center_y = (y_min + y_max) / 2

        items.append({
            "text": text.strip(),
            "x": x_min,
            "y": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "height": height,
            "center_y": center_y,
        })

    if not items:
        return ""

    # ---------------------------------------
    # Group detections into visual lines
    # ---------------------------------------

    items.sort(key=lambda item: item["center_y"])

    lines = []

    for item in items:

        placed = False

        for current_line in lines:

            # Average vertical position of existing line
            avg_y = sum(
                x["center_y"]
                for x in current_line
            ) / len(current_line)

            avg_height = sum(
                x["height"]
                for x in current_line
            ) / len(current_line)

            # Adaptive vertical tolerance
            tolerance = max(
                avg_height * 0.5,
                item["height"] * 0.5,
                5
            )

            if abs(item["center_y"] - avg_y) <= tolerance:

                current_line.append(item)
                placed = True
                break

        if not placed:
            lines.append([item])

    # ---------------------------------------
    # Sort each line left → right
    # ---------------------------------------

    for current_line in lines:
        current_line.sort(
            key=lambda item: item["x"]
        )

    # ---------------------------------------
    # Sort lines top → bottom
    # ---------------------------------------

    lines.sort(
        key=lambda line: min(
            item["y"] for item in line
        )
    )

    # ---------------------------------------
    # Build structured text
    # ---------------------------------------

    output_lines = []

    for current_line in lines:

        text = " ".join(
            item["text"]
            for item in current_line
        )

        output_lines.append(text)

    return "\n".join(output_lines)


def extract_ocr_text(assets_dir):
    """
    Extract OCR text from all images inside assets folder.

    Parameters
    ----------
    assets_dir : str | Path

    Returns
    -------
    dict

    {
        "image_1.png":
        {
            "text": "...",
            "char_count": 120,
            "line_count": 8
        },

        ...
    }
    """

    assets_dir = Path(assets_dir)

    ocr_engine = RapidOCR()
    cache = load_cache()

    cache_hits = 0
    ocr_skipped = 0
    ocr_processed = 0
    ocr_failed = 0
    total_characters = 0

    results = {}

    image_files = sorted(
        [
            *assets_dir.glob("*.png"),
            *assets_dir.glob("*.jpg"),
            *assets_dir.glob("*.jpeg"),
            *assets_dir.glob("*.webp"),
        ]
    )

    for image_path in image_files:

        # ---------------------------------------
        # Image Filter
        # ---------------------------------------

        if not should_process(image_path):

            ocr_skipped += 1

            continue


        # ---------------------------------------
        # pHash Cache
        # ---------------------------------------

        current_hash = image_hash(image_path)

        similar_hash = find_similar_hash(
            current_hash,
            cache
        )

        if similar_hash is not None:

            results[image_path.name] = cache[similar_hash]

            cache_hits += 1

            print(f"Using cached OCR for {image_path.name}")

            continue

        ocr_processed += 1

        try:
            ocr_result, _ = ocr_engine(
                str(image_path)
            )

        except Exception as e:
            ocr_failed += 1

            print(
                f"OCR failed for {image_path.name}: {e}"
            )

            continue

        extracted_text = ""

        if ocr_result:

            extracted_text = reconstruct_ocr_structure(
                ocr_result
            )
        total_characters += len(extracted_text.strip())    

        results[image_path.name] = {

            "text": extracted_text,

            "char_count": len(
                extracted_text.strip()
            ),

            "line_count": len(
                extracted_text.splitlines()
            )

        }

        cache[str(current_hash)] = results[image_path.name]

    # ---------------------------------------
    # Count images that contain OCR text
    # ---------------------------------------

    ocr_image_count = sum(
        1
        for item in results.values()
        if item["char_count"] > 0
    )
    save_cache(cache)
    return {
        "results": results,
        "ocr_image_count": ocr_image_count,
        "cache_hits": cache_hits,
        "skipped": ocr_skipped,
        "ocr_failed": ocr_failed,
        "total_characters": total_characters,
    }