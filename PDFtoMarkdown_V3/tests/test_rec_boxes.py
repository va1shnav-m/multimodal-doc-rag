import json
from pathlib import Path


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

JSON_PATH = Path(
    "ppstructure_test_output/Screenshot 2026-08-14 121515_res.json"
)

OUTPUT_MD = Path(
    "ppstructure_test_output/spatial_output.md"
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def get_box_center(box):
    """Return center x/y of [x1, y1, x2, y2]."""

    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return center_x, center_y


def get_box_height(box):
    """Return height of bounding box."""

    return box[3] - box[1]


def get_box_width(box):
    """Return width of bounding box."""

    return box[2] - box[0]


# ---------------------------------------------------------
# Load OCR
# ---------------------------------------------------------

def load_ocr_data():

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    ocr = data["overall_ocr_res"]

    texts = ocr["rec_texts"]
    boxes = ocr["rec_boxes"]
    scores = ocr["rec_scores"]

    items = []

    for text, box, score in zip(
        texts,
        boxes,
        scores
    ):

        text = str(text).strip()

        if not text:
            continue

        x1, y1, x2, y2 = box

        center_x, center_y = get_box_center(box)

        items.append(
            {
                "text": text,
                "box": box,
                "score": float(score),
                "x": center_x,
                "y": center_y,
                "width": get_box_width(box),
                "height": get_box_height(box),
            }
        )

    return data, items


# ---------------------------------------------------------
# Spatial grouping
# ---------------------------------------------------------

def group_items(items, image_width, image_height):
    """
    Group OCR items that are spatially close.

    This is intentionally generic.
    It does not know anything about ER diagrams.
    """

    if not items:
        return []

    # Average OCR text height.
    average_height = sum(
        item["height"]
        for item in items
    ) / len(items)

    # Relative thresholds.
    #
    # These are based partly on the OCR text size,
    # rather than fixed image pixels.

    vertical_threshold = average_height * 2.2
    horizontal_threshold = image_width * 0.08

    groups = []

    # Sort top-to-bottom first.
    sorted_items = sorted(
        items,
        key=lambda item: (
            item["y"],
            item["x"]
        )
    )

    for item in sorted_items:

        best_group = None
        best_distance = float("inf")

        for group in groups:

            # Group bounding box.
            gx1 = min(
                x["box"][0]
                for x in group
            )

            gy1 = min(
                x["box"][1]
                for x in group
            )

            gx2 = max(
                x["box"][2]
                for x in group
            )

            gy2 = max(
                x["box"][3]
                for x in group
            )

            group_center_x = (gx1 + gx2) / 2
            group_center_y = (gy1 + gy2) / 2

            vertical_distance = abs(
                item["y"] - group_center_y
            )

            horizontal_distance = abs(
                item["x"] - group_center_x
            )

            # Check whether the item is reasonably
            # close to this existing group.

            if (
                vertical_distance <= vertical_threshold
                and horizontal_distance <= horizontal_threshold
            ):

                distance = (
                    vertical_distance
                    + horizontal_distance
                )

                if distance < best_distance:
                    best_distance = distance
                    best_group = group

        if best_group is not None:

            best_group.append(item)

        else:

            groups.append([item])

    # Sort items inside every group.
    for group in groups:

        group.sort(
            key=lambda item: (
                item["y"],
                item["x"]
            )
        )

    # Sort groups by their position.
    groups.sort(
        key=lambda group: (
            min(item["y"] for item in group),
            min(item["x"] for item in group)
        )
    )

    return groups


# ---------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------

def create_markdown(groups):

    lines = []

    lines.append("# Image Analysis")
    lines.append("")

    for index, group in enumerate(groups, start=1):

        lines.append(
            f"## Region {index}"
        )

        lines.append("")

        for item in group:

            lines.append(
                f"- {item['text']}"
            )

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------
# Debug output
# ---------------------------------------------------------

def print_groups(groups):

    print("\n" + "=" * 100)
    print("SPATIAL GROUPS")
    print("=" * 100)

    for index, group in enumerate(
        groups,
        start=1
    ):

        print(
            f"\n--- Region {index} "
            f"({len(group)} items) ---"
        )

        for item in group:

            print(
                f"{item['text']:<35} "
                f"{item['box']}"
            )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("SPATIAL OCR → MARKDOWN")
    print("=" * 70)

    # ---------------------------------------------
    # Load
    # ---------------------------------------------

    data, items = load_ocr_data()

    image_width = data["width"]
    image_height = data["height"]

    print(
        f"\nImage size: "
        f"{image_width} x {image_height}"
    )

    print(
        f"OCR items: "
        f"{len(items)}"
    )

    # ---------------------------------------------
    # Group
    # ---------------------------------------------

    groups = group_items(
        items,
        image_width,
        image_height
    )

    print(
        f"Spatial groups: "
        f"{len(groups)}"
    )

    # ---------------------------------------------
    # Debug
    # ---------------------------------------------

    print_groups(groups)

    # ---------------------------------------------
    # Markdown
    # ---------------------------------------------

    markdown = create_markdown(
        groups
    )

    OUTPUT_MD.write_text(
        markdown,
        encoding="utf-8"
    )

    print(
        f"\nMarkdown saved to:"
        f"\n{OUTPUT_MD.resolve()}"
    )


if __name__ == "__main__":
    main()