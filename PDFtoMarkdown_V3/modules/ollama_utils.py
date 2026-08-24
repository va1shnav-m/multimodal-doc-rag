import subprocess
import time
import requests


OLLAMA_URL = "http://127.0.0.1:11434"
STARTUP_TIMEOUT = 60


def is_ollama_running():
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=2
        )
        return response.status_code == 200

    except requests.RequestException:
        return False


def ensure_ollama_running():

    # Already running
    if is_ollama_running():
        print("Ollama is already running.")
        return True

    print("Ollama is not running. Starting Ollama...")

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    except Exception as e:
        print(f"Failed to start Ollama: {e}")
        return False

    # Wait for Ollama to become available
    start_time = time.time()

    while time.time() - start_time < STARTUP_TIMEOUT:

        if is_ollama_running():
            print("Ollama started successfully.")
            return True

        time.sleep(1)

    print("Ollama did not start within the timeout.")
    return False