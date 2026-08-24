from pathlib import Path
import fitz
import time

# Images covering more than 10% of the page
# are considered important.
IMAGE_AREA_THRESHOLD = 0.10


def analyze_pdf(pdf_path):
    """
    Analyze each page and decide which parser to use.

    Docling:
        - Tables
        - Large images / diagrams / screenshots

    PyMuPDF:
        - Plain digital text
    """

    pdf_path = Path(pdf_path)

    doc = fitz.open(pdf_path)

    analysis = []

    print("\n========== PDF ANALYSIS ==========\n")

    for page_index, page in enumerate(doc):

        page_number = page_index + 1

        # ----------------------------------
        # Detect Tables
        # ----------------------------------

        has_table = detect_table_structure(page)

        table_count = 1 if has_table else 0

        # ----------------------------------
        # Detect Images
        # ----------------------------------

        images = page.get_images(full=True)

        image_count = len(images)

        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        largest_image_ratio = 0.0

        for image in images:

            xref = image[0]

            try:

                rects = page.get_image_rects(xref)

            except Exception:

                continue

            for rect in rects:

                image_area = rect.width * rect.height

                ratio = image_area / page_area

                if ratio > largest_image_ratio:

                    largest_image_ratio = ratio

        has_large_image = (
            largest_image_ratio >= IMAGE_AREA_THRESHOLD
        )

        # ----------------------------------
        # Select Parser
        # ----------------------------------

        if has_table:

            parser = "docling"

        elif has_large_image:

            parser = "docling"

        else:

            parser = "pymupdf"

        page_info = {

            "page": page_number,

            "has_table": has_table,

            "table_count": table_count,

            "image_count": image_count,

            "largest_image_ratio": round(
                largest_image_ratio,
                3
            ),

            "parser": parser,

        }

        analysis.append(page_info)

        print(
            f"Page {page_number:03d} | "
            f"Tables: {table_count:<2} | "
            f"Images: {image_count:<2} | "
            f"Largest: {largest_image_ratio:.2f} | "
            f"Parser: {parser}"
        )

    doc.close()

    print("\n========== ANALYSIS COMPLETE ==========\n")

    return analysis

def detect_table_structure(page):
    """
    table detection.

    Detects:
    - Borderless tables (aligned text columns)
    - Ruled tables (horizontal/vertical vector lines)
    - Rectangle-based tables (cell borders drawn as rects)
    - PyMuPDF built-in table finder (fallback)
    """

    # ----------------------------
    # Check vector drawings
    # ----------------------------

    horizontal_lines = 0
    vertical_lines = 0
    rect_count = 0

    try:
        drawings = page.get_drawings()

        for drawing in drawings:

            for item in drawing["items"]:

                # Straight line
                if item[0] == "l":

                    p1 = item[1]
                    p2 = item[2]

                    # Horizontal line
                    if abs(p1.y - p2.y) < 2 and abs(p1.x - p2.x) > 30:
                        horizontal_lines += 1

                    # Vertical line
                    elif abs(p1.x - p2.x) < 2 and abs(p1.y - p2.y) > 30:
                        vertical_lines += 1

                # Rectangle (table cells often drawn as rects)
                elif item[0] == "re":
                    rect_count += 1

    except Exception:
        pass

    # --------------------------------
    # Ruled table detection
    # --------------------------------

    # Mostly horizontal borders
    if horizontal_lines >= 3:
        return True

    # Mostly vertical borders
    if vertical_lines >= 3:
        return True

    # Mixed borders
    if horizontal_lines >= 2 and vertical_lines >= 2:
        return True

    # --------------------------------
    # Rectangle-based table detection
    # --------------------------------

    # Many small rectangles typically indicate table cells
    if rect_count >= 10:
        return True

    # ----------------------------
    # Built-in table finder
    # ----------------------------

    try:
        tables = page.find_tables()
        if tables.tables:
            for table in tables.tables:
                # Only count tables with at least 2 rows and 2 cols
                if table.row_count >= 2 and table.col_count >= 2:
                    return True
    except Exception:
        pass

    # ----------------------------
    # Borderless table detection
    # ----------------------------

    blocks = page.get_text("blocks")

    if len(blocks) < 5:
        return False

    rows = {}

    for block in blocks:

        x0 = round(block[0], 1)
        y0 = round(block[1], 1)

        row_key = round(y0 / 5) * 5

        rows.setdefault(row_key, []).append(x0)

    row_like_count = 0

    for xs in rows.values():

        unique_x = len(set(xs))

        if unique_x >= 2:
            row_like_count += 1

    return row_like_count >= 3