from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import time

MODEL = "microsoft/Florence-2-base"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL,
    trust_remote_code=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float32
)

model.eval()

print("Model loaded!")

image = Image.open("chunk_0001_image_0008.png").convert("RGB")

task = "<CAPTION>"

inputs = processor(
    text=task,
    images=image,
    return_tensors="pt"
)

start = time.perf_counter()

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=64,
)

end = time.perf_counter()

generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=False
)[0]

result = processor.post_process_generation(
    generated_text,
    task=task,
    image_size=image.size,
)

print(result)
print(f"Time: {end-start:.2f} sec")