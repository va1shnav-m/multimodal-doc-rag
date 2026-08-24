import base64
import json
import pickle
import sqlite3


def decode_qdrant_data(encoded_str):
    """Decodes the binary/pickle data stored in SQLite."""
    try:
        byte_data = base64.b64decode(encoded_str)
        return pickle.loads(byte_data)
    except Exception:
        return encoded_str


# Connect to your database
conn = sqlite3.connect("storage/qdrant/collection/production_rag/storage.sqlite")
cursor = conn.cursor()

# Query both 'id' and 'payload' columns from the points table
cursor.execute("SELECT id, payload FROM points LIMIT 5")
rows = cursor.fetchall()

print("--- Reading Human Readable Payload Data ---")
for row in rows:
    raw_id, raw_payload = row

    decoded_id = decode_qdrant_data(raw_id)
    decoded_payload = decode_qdrant_data(raw_payload)

    print(f"\nVector ID: {decoded_id}")

    # The payload is usually a dictionary containing your actual text
    if isinstance(decoded_payload, dict):
        # Look for common text keys like 'text', 'page_content', or 'content'
        text_content = (
            decoded_payload.get("text")
            or decoded_payload.get("page_content")
            or decoded_payload
        )
        print(f"Content: {text_content}")
    else:
        print(f"Raw Payload: {decoded_payload}")

conn.close()
