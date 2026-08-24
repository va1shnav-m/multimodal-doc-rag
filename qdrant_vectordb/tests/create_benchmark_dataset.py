from pathlib import Path
from pypdf import PdfWriter

TARGETS = [10, 25, 50, 75, 100,150]

SOURCE = Path(r"C:\Users\Manasi Sabnis\Desktop\benchmark_docs")
OUTPUT = Path(r"C:\Users\Manasi Sabnis\Desktop\production_rag\benchmark_docs\generated")

OUTPUT.mkdir(parents=True, exist_ok=True)

pdfs = sorted(SOURCE.glob("*.pdf"))

for target in TARGETS:

    writer = PdfWriter()
    total_size = 0

    for pdf in pdfs:

        size = pdf.stat().st_size / (1024 * 1024)

        if total_size >= target:
            break

        writer.append(str(pdf))
        total_size += size

    out = OUTPUT / f"{target}MB.pdf"

    with open(out, "wb") as f:
        writer.write(f)

    print(f"{out.name}: {total_size:.2f} MB")