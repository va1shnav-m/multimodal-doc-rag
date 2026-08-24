from pathlib import Path
import subprocess

# Path to LibreOffice
SOFFICE_PATH = r"C:\Program Files\LibreOffice\program\soffice.exe"


def convert_doc(file_path):
    """
    Converts legacy .doc to .docx or .ppt to .pptx using LibreOffice Headless.
    If the file is already modern or PDF, it is returned unchanged.

    Parameters:
        file_path (str | Path): Path to the uploaded document.

    Returns:
        str: Path to the converted or original file.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".doc":
        target_format = "docx"
    elif suffix == ".ppt":
        target_format = "pptx"
    else:
        return str(file_path)

    output_dir = file_path.parent

    subprocess.run(
        [
            SOFFICE_PATH,
            "--headless",
            "--convert-to",
            target_format,
            "--outdir",
            str(output_dir),
            str(file_path),
        ],
        check=True,
    )

    converted_file = output_dir / f"{file_path.stem}.{target_format}"
    return str(converted_file)