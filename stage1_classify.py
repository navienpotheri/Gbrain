import os
from pathlib import Path
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

PHOTOS_DIR = "/workspace/Google Photos"
MANIFEST_DIR = "/workspace/manifests"
os.makedirs(MANIFEST_DIR, exist_ok=True)

lock = threading.Lock()
results = {"text_heavy": [], "visual": []}
counter = [0]

def classify_image(path):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((224, 224))
            arr = np.array(img, dtype=np.float32)
            gray = np.mean(arr, axis=2)
            edges = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
            color_var = arr.std(axis=(0,1)).mean()
            is_text = edges > 8.0 and color_var < 60.0
            return str(path), "text_heavy" if is_text else "visual"
    except:
        return str(path), "error"

exts = {".jpg", ".jpeg", ".png", ".webp"}
all_images = [p for p in Path(PHOTOS_DIR).rglob("*") if p.suffix.lower() in exts]
print(f"Found {len(all_images)} images to classify")

with ThreadPoolExecutor(max_workers=16) as ex:
    futures = {ex.submit(classify_image, p): p for p in all_images}
    for f in as_completed(futures):
        path, label = f.result()
        with lock:
            if label in results:
                results[label].append(path)
            counter[0] += 1
            if counter[0] % 5000 == 0:
                print(f"  {counter[0]}/{len(all_images)} classified...")

for label, paths in results.items():
    out = Path(MANIFEST_DIR) / f"{label}.txt"
    out.write_text("\n".join(paths))
    print(f"{label}: {len(paths)} images -> {out}")

print("Stage 1 complete.")
