from paddleocr import PPStructureV3
from pathlib import Path
import time
import json

IMAGE_PATH = Path(r"Screenshot 2026-08-14 121515.png")
OUTPUT_DIR = Path("ppstructure_test_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():

    print("=" * 70)
    print("PP-StructureV3 TEST")
    print("=" * 70)

    print(f"\nImage: {IMAGE_PATH}")

    # -----------------------------
    # Initialize
    # -----------------------------

    print("\nInitializing PP-StructureV3...")

    start = time.perf_counter()

    pipeline = PPStructureV3(
        lang="en",
        use_table_recognition=False,
        enable_mkldnn=False,
    )

    init_time = time.perf_counter() - start

    print(f"Initialization time: {init_time:.2f}s")

    # -----------------------------
    # Run
    # -----------------------------

    print("\nRunning PP-StructureV3...")

    start = time.perf_counter()

    results = pipeline.predict(str(IMAGE_PATH))

    processing_time = time.perf_counter() - start

    print(f"Processing time: {processing_time:.2f}s")

    # -----------------------------
    # Results
    # -----------------------------

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    for i, result in enumerate(results):

        print(f"\n--- Result {i + 1} ---")

        result.print()

        # Save PP-Structure JSON
        result.save_to_json(
            save_path=str(OUTPUT_DIR)
        )

        # Find the generated JSON file
        json_files = list(OUTPUT_DIR.glob("*.json"))

        if not json_files:
            print("JSON file was not created.")
            continue

        json_path = json_files[-1]

        # Read the actual JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract OCR text
        parsing_results = data.get("parsing_res_list", [])

        markdown_lines = [
            "# Image Analysis",
            ""
        ]

        for item in parsing_results:

            content = item.get("block_content", "")

            if content:
                markdown_lines.append(content.strip())
                markdown_lines.append("")

        # Save Markdown
        md_path = OUTPUT_DIR / f"result_{i + 1}.md"

        md_path.write_text(
            "\n".join(markdown_lines),
            encoding="utf-8"
        )

        print(f"Markdown saved: {md_path}")
    print("\n" + "=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(f"\nOutput: {OUTPUT_DIR.resolve()}")

def create_markdown(result, output_path):
    """Create Markdown from PP-StructureV3 OCR output."""

    lines = []

    lines.append("# Image Analysis")
    lines.append("")

    parsing_results = result.get("parsing_res_list", [])

    for item in parsing_results:

        label = item.get("block_label", "")
        content = item.get("block_content", "")

        if not content:
            continue

        # Ignore image blocks as image references.
        # We only want their extracted textual content.
        if label == "image":
            lines.append("## Extracted Text")
            lines.append("")
            lines.append(content.strip())
            lines.append("")

        else:
            lines.append(f"## {label.title()}")
            lines.append("")
            lines.append(content.strip())
            lines.append("")

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()