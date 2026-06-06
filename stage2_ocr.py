import os
import json
from pathlib import Path
from paddleocr import PaddleOCR
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

MANIFEST = "/workspace/manifests/text_heavy.txt"
OUTPUT = "/workspace/ocr_results.jsonl"
CHECKPOINT = "/workspace/ocr_checkpoint.txt"

# Load already-processed files
done = set()
if Path(CHECKPOINT).exists():
    done = set(Path(CHECKPOINT).read_text().splitlines())

# Load manifest
paths = [p for p in Path(MANIFEST).read_text().splitlines() if p and p not in done]
print(f"To process: {len(paths)} images ({len(done)} already done)")

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
lock = threading.Lock()
counter = [0]

def process_image(path):
    try:
        result = ocr.ocr(path, cls=True)
        lines = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text, conf = line[1][0], line[1][1]
                    if conf > 0.5:
                        lines.append(text)
        return path, " ".join(lines), None
    except Exception as e:
        return path, "", str(e)

with open(OUTPUT, "a") as out_f, open(CHECKPOINT, "a") as ckpt_f:
    for path in paths:
        path_str, text, err = process_image(path)
        record = {"path": path_str, "text": text, "error": err}
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()
        ckpt_f.write(path_str + "\n")
        ckpt_f.flush()
        counter[0] += 1
        if counter[0] % 500 == 0:
            print(f"  {counter[0]}/{len(paths)} OCR'd...")

print(f"Stage 2 complete. Results -> {OUTPUT}")
