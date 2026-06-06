import os
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MANIFEST = "/workspace/manifests/visual.txt"
OUTPUT = "/workspace/captions.jsonl"
CHECKPOINT = "/workspace/captions_checkpoint.txt"
MODEL_PATH = "/workspace/models/Qwen2.5-VL-32B-Instruct"
BATCH_SIZE = 8

# Load checkpoint
done = set()
if Path(CHECKPOINT).exists():
    done = set(Path(CHECKPOINT).read_text().splitlines())

paths = [p for p in Path(MANIFEST).read_text().splitlines() if p and p not in done]
print(f"To process: {len(paths)} images ({len(done)} already done)")

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.")

counter = 0

def caption_batch(batch_paths):
    messages_list = []
    for p in batch_paths:
        messages_list.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{p}"},
                {"type": "text", "text": "Describe this image in detail. Include people, objects, setting, activities, text visible, and mood."}
            ]
        }])

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    image_inputs, video_inputs = process_vision_info(messages_list)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)

with open(OUTPUT, "a") as out_f, open(CHECKPOINT, "a") as ckpt_f:
    for i in range(0, len(paths), BATCH_SIZE):
        batch = paths[i:i+BATCH_SIZE]
        try:
            captions = caption_batch(batch)
            for path, caption in zip(batch, captions):
                record = {"path": path, "caption": caption}
                out_f.write(json.dumps(record) + "\n")
                ckpt_f.write(path + "\n")
            out_f.flush()
            ckpt_f.flush()
            counter += len(batch)
            if counter % 100 == 0:
                print(f"  {counter}/{len(paths)} captioned...")
        except Exception as e:
            print(f"  Batch error: {e}")
            for path in batch:
                out_f.write(json.dumps({"path": path, "caption": "", "error": str(e)}) + "\n")
                ckpt_f.write(path + "\n")

print(f"Stage 3 complete. {counter} images captioned -> {OUTPUT}")
