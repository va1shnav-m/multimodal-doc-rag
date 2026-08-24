from paddleocr import PaddleOCR
from pathlib import Path
import time


# --------------------------------------------------
# CHANGE THIS TO YOUR TEST IMAGE
# --------------------------------------------------
IMAGE_PATH = Path(
    r"Screenshot 2026-08-14 121427.png"
)

def main():
    if not IMAGE_PATH.exists():
        print(f"Image not found: {IMAGE_PATH}")
        return

    print(f"Image: {IMAGE_PATH}")
    print("Initializing PaddleOCR...")

    start_init = time.perf_counter()

    ocr = PaddleOCR(
        lang="en",
        enable_mkldnn=False,
    )

    init_time = time.perf_counter() - start_init

    print(f"OCR initialized in {init_time:.2f}s")
    print("Running OCR...")

    start = time.perf_counter()

    result = ocr.predict(str(IMAGE_PATH))

    elapsed = time.perf_counter() - start

    print(f"\nOCR processing time: {elapsed:.2f}s")
    print("=" * 70)

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------
    for page_result in result:

        data = page_result.json

        # PaddleOCR 3.x stores the actual OCR result
        # inside the "res" key.
        if "res" in data:
            data = data["res"]

        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        boxes = data.get("rec_polys", [])

        print(f"\nDetected text items: {len(texts)}")
        print("-" * 70)

        for i, text in enumerate(texts):

            score = scores[i] if i < len(scores) else None
            box = boxes[i] if i < len(boxes) else None

            print(f"\n[{i + 1}]")
            print(f"Text       : {text}")
            print(f"Confidence : {score}")
            print(f"Box        : {box}")

    print("\n" + "=" * 70)
    print("Finished.")


if __name__ == "__main__":
    main()