import time
import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)

MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.eval()

IMAGE_PATH = r"chunk_0003_image_0005.png"      # <-- Change this

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image"
            },
            {
                "type": "text",
                "text": """Describe this image in 1-2 concise sentences.
                            Focus only on technical content:
                            - diagrams
                            - flowcharts
                            - architecture
                            - UI/screenshots
                            - tables
                            - graphs
                            - code
                            - forms
                            Mention only the purpose and key components.
                            Ignore colors, styling, decorations, logos, icons, and blank images.
                            If there is no meaningful technical content, respond exactly:SKIP"""
            },
        ],
    }
]

prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
)

start = time.perf_counter()

with torch.inference_mode():

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

end = time.perf_counter()

generated_text = processor.decode(
    generated_ids[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("\n==========================")
print("Caption:")
print(generated_text)
print("==========================")
print(f"Time: {end-start:.2f} sec")