from pathlib import Path
from PIL import Image

MIN_WIDTH = 60
MIN_HEIGHT = 60
MIN_SIDE = 30               # Skip if one side is extremely thin (e.g. 10px lines)
MIN_AREA = 3000             # Skip if total area < 3,000 px
MAX_ASPECT_RATIO = 6.0      # Skip long/thin banners and lines


def should_process(image_path):
    """
    Decide whether an image should be sent for VLM analysis.

    Returns
    -------
    bool
    """

    image_path = Path(image_path)

    if not image_path.exists():
        return False

    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # ---------------------------------------------------
            # Rule 1: Skip if BOTH dimensions are too small
            # ---------------------------------------------------
            if width < MIN_WIDTH and height < MIN_HEIGHT:
                print(f"{image_path.name}: Tiny image ({width}x{height}) -> Skip")
                return False

            # ---------------------------------------------------
            # Rule 2: Skip if ONE dimension is too thin (lines/strips)
            # ---------------------------------------------------
            if min(width, height) < MIN_SIDE:
                print(f"{image_path.name}: Too thin ({width}x{height}) -> Skip")
                return False

            # ---------------------------------------------------
            # Rule 3: Skip if total area is very small
            # ---------------------------------------------------
            if width * height < MIN_AREA:
                print(f"{image_path.name}: Small area ({width}x{height} = {width*height}px) -> Skip")
                return False

            # ---------------------------------------------------
            # Rule 4: Skip extreme aspect ratios (horizontal/vertical bars)
            # ---------------------------------------------------
            aspect_ratio = max(width, height) / max(min(width, height), 1)

            if aspect_ratio > MAX_ASPECT_RATIO:
                print(
                    f"{image_path.name}: Extreme aspect ratio "
                    f"({width}x{height}, ratio={aspect_ratio:.1f}) -> Skip"
                )
                return False

            # ---------------------------------------------------
            # Rule 5: Skip solid single-color images
            # ---------------------------------------------------
            try:
                rgb_img = img.convert("RGB")
                colors = rgb_img.getcolors(maxcolors=2)
                if colors is not None and len(colors) == 1:
                    print(f"{image_path.name}: Solid colour image -> Skip")
                    return False
            except Exception:
                pass

    except Exception as e:
        print(f"{image_path.name}: Could not read image ({e}) -> Skip")
        return False

    # ---------------------------------------------------
    # Otherwise: valid informative image
    # ---------------------------------------------------

    print(f"{image_path.name}: Send For Image Processing")

    return True
