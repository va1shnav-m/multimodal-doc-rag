import json
from pathlib import Path

CACHE_FILE = Path("ocr_cache.json")


def load_cache():
    """
    Load OCR cache from disk.
    """

    if CACHE_FILE.exists():

        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def save_cache(cache):
    """
    Save OCR cache.
    """

    with open(CACHE_FILE, "w", encoding="utf-8") as f:

        json.dump(
            cache,
            f,
            indent=4,
            ensure_ascii=False,
        )