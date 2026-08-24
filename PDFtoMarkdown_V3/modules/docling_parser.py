import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from docling_core.types.doc import PictureItem

# ----------------------------------------
# Create Docling Converter ONCE
# ----------------------------------------

pipeline_options = PdfPipelineOptions()

pipeline_options.do_ocr = False
pipeline_options.images_scale = 2
pipeline_options.generate_picture_images = True
pipeline_options.generate_page_images = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

def parse_document(input_path, output_dir, assets_dir, page_name=None):
    """
    Parse PDF/DOCX using Docling.

    Generates:
        - Markdown
        - Extracted images

    Returns
    -------
    dict
    """
    print("Using Docling Parser")
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    assets_dir = Path(assets_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # Convert Document
    # ----------------------------------------

    result = converter.convert(input_path)

    # ----------------------------------------
    # Export Markdown
    # ----------------------------------------

    if page_name:

        markdown_path = output_dir / f"{page_name}.md"

    else:

        markdown_path = output_dir / "document.md"

    markdown_path.write_text(

        result.document.export_to_markdown(),

        encoding="utf-8"

    )

    # ----------------------------------------
    # Extract Images
    # ----------------------------------------

    image_count = 0

    for item, _ in result.document.iterate_items():

        if isinstance(item, PictureItem):

            image = item.get_image(result.document)

            if image is None:

                continue

            image_count += 1

            

            if page_name:

                image_filename = (
                    f"{page_name}_image_{image_count:04d}.png"
                )

            else:

                image_filename = (
                    f"image_{image_count:04d}.png"
                )

            image.save(
                assets_dir / image_filename
            )

    return {

        "markdown": markdown_path,

        "assets": assets_dir,

        "image_count": image_count,

    }