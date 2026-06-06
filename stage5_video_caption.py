import os
import json
from pathlib import Path
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

KEYFRAMES_DIR = "/workspace/keyframes"
OUTPUT = "/workspace/video_captions.jsonl"
CHECKPOINT = "/workspace/video_captions_checkpoint.txt"
MODEL_PATH = "/workspace/models/Qwen2.5-VL-7B-Instruct"

# Load checkpoint
done = set()
if Path(CHECKPOINT).exists():
    done = set(Path(CHECKPOINT).read_text().splitlines())

# For each video folder, pick the middle frame
video_dirs = [d for d in Path(KEYFRAMES_DIR).iterdir() if d.is_dir()]
tasks = []
for vdir in video_dirs:
    frames = sorted(vdir.glob("*.jpg"))
    if not frames:
        continue
    middle = frames[len(frames) // 2]
    if str(middle) not in done:
        tasks.append((str(vdir.name), str(middle)))

print(f"To process: {len(tasks)} videos ({len(done)} already done)")

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.tokenizer.padding_side = 'left'
print("Model loaded.")

counter = 0

with open(OUTPUT, "a") as out_f, open(CHECKPOINT, "a") as ckpt_f:
    for video_name, frame_path in tasks:
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{frame_path}"},
                    {"type": "text", "text": "Describe this video frame in detail. Include people, objects, setting, activities, text visible, and mood."}
                ]
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info([messages])
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=128)
            trimmed = output_ids[0][len(inputs.input_ids[0]):]
            caption = processor.decode(trimmed, skip_special_tokens=True)

            record = {"video": video_name, "frame": frame_path, "caption": caption}
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            ckpt_f.write(frame_path + "\n")
            ckpt_f.flush()
            counter += 1
            if counter % 100 == 0:
                print(f"  {counter}/{len(tasks)} videos captioned...")
        except Exception as e:
            print(f"  Error on {video_name}: {e}")
            out_f.write(json.dumps({"video": video_name, "frame": frame_path, "caption": "", "error": str(e)}) + "\n")
            ckpt_f.write(frame_path + "\n")

print(f"Stage 5 complete. {counter} videos captioned -> {OUTPUT}")
