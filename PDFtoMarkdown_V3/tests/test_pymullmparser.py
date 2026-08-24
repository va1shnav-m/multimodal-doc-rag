from pathlib import Path
import shutil
from . import pymupdf4llm_parser

pdf_path = r"C:\Users\Vaishnav M\Projects\PDFtoMarkdown_V3\srs_example_2010_group2 (1)-1-36.pdf"

output_dir = Path("test_output")
assets_dir = output_dir / "assets"
if output_dir.exists():
    shutil.rmtree(output_dir)

result = pymupdf4llm_parser.parse_document(
    input_path=pdf_path,
    output_dir=output_dir,
    assets_dir=assets_dir,
)

print(result)