import re
from pathlib import Path
from typing import Dict, Any, Optional


def _format_caption_blocks(image_name: str, analysis: Optional[Dict[str, Any]], alt_text: str = "") -> str:
    """Format markdown image tag with blockquote text extract and description."""
    alt = alt_text if alt_text else image_name
    lines = [f"![{alt}](assets/{image_name})", ""]

    if analysis:
        text_extract = analysis.get("text_extract", "").strip()
        if text_extract and text_extract.lower() not in {"no text content.", "no text content", "none", "n/a"}:
            lines.append("> **Text Extract:**")
            lines.append(">")
            for line in text_extract.splitlines():
                lines.append(f"> {line}")
            lines.append("")

        description = analysis.get("description", "").strip()
        if description:
            lines.append("> **Description:**")
            lines.append(">")
            for line in description.splitlines():
                lines.append(f"> {line}")
            lines.append("")

    return "\n".join(lines).strip()


def merge_markdown(
    markdown_path: str | Path,
    assets_dir: str | Path,
    image_analysis: Dict[str, Any],
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Merge Markdown with structured image analysis (Text Extract and Description).
    Handles both `<!-- image -->` placeholders (Docling) and direct `![alt](img.png)` tags (PPTX).

    Parameters
    ----------
    markdown_path : str | Path
    assets_dir : str | Path
    image_analysis : dict
    output_path : str | Path | None

    Returns
    -------
    Path
    """
    markdown_path = Path(markdown_path)
    assets_dir = Path(assets_dir)

    if output_path is None:
        output_path = markdown_path
    else:
        output_path = Path(output_path)

    markdown = markdown_path.read_text(encoding="utf-8")

    # ----------------------------------------------------
    # 1. Handle `<!-- image -->` comment placeholders (Docling PDF)
    # ----------------------------------------------------
    if "<!-- image -->" in markdown:
        image_files = sorted(
            [
                *assets_dir.glob("*.png"),
                *assets_dir.glob("*.jpg"),
                *assets_dir.glob("*.jpeg"),
                *assets_dir.glob("*.webp"),
            ]
        )

        figure_number = 1
        for image_path in image_files:
            if "<!-- image -->" not in markdown:
                break

            image_name = image_path.name
            analysis = image_analysis.get(image_name)

            replacement = [f"### Figure {figure_number}", ""]
            replacement.append(_format_caption_blocks(image_name, analysis, alt_text=image_name))
            replacement.append("")

            markdown = markdown.replace("<!-- image -->", "\n".join(replacement), 1)
            figure_number += 1

    # ----------------------------------------------------
    # 2. Handle direct markdown image tags `![alt](filename)` (PPTX, PyMuPDF)
    # ----------------------------------------------------
    else:
        def replace_img_tag(match: re.Match) -> str:
            alt = match.group(1)
            full_path = match.group(2)
            filename = full_path.replace("\\", "/").split("/")[-1]

            analysis = image_analysis.get(filename) if image_analysis else None
            return _format_caption_blocks(filename, analysis, alt_text=alt)

        img_tag_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        markdown = img_tag_pattern.sub(replace_img_tag, markdown)

    output_path.write_text(markdown, encoding="utf-8")
    return output_path