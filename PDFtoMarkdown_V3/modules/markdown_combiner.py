from pathlib import Path


def combine_markdowns(output_dir, output_name="document.md", pattern="*chunk_*.md"):

    output_dir = Path(output_dir)

    markdown_files = sorted(
        output_dir.glob(pattern)
    )
    
    
    combined_markdown = []

    for markdown_file in markdown_files:

        markdown = markdown_file.read_text(
            encoding="utf-8"
        ).strip()

        combined_markdown.append(markdown)
        combined_markdown.append("\n\n")

    final_markdown = output_dir / output_name

    final_markdown.write_text(
        "".join(combined_markdown),
        encoding="utf-8"
    )

    return final_markdown

def combine_final_documents(output_dir):

    output_dir = Path(output_dir)

    final_documents = sorted(
        output_dir.glob("document_*.md")
    )

    combined_markdown = []

    for markdown_file in final_documents:

        # Skip intermediate raw markdown files
        if markdown_file.stem.endswith("_raw"):
            continue

        markdown = markdown_file.read_text(
            encoding="utf-8"
        ).strip()

        combined_markdown.append(markdown)
        combined_markdown.append("\n\n")

    final_output = output_dir / "document.md"

    final_output.write_text(
        "".join(combined_markdown),
        encoding="utf-8"
    )

    return final_output
