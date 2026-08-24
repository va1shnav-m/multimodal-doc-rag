from pathlib import Path
import shutil

def clear_folder(folder_path):
    """
    Deletes all files and subfolders inside the given folder.
    Creates the folder if it doesn't exist.
    """
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    for item in folder.iterdir():
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass