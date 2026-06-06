import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

PHOTOS_DIR = "/workspace/Google Photos"
KEYFRAMES_DIR = "/workspace/keyframes"
CHECKPOINT = "/workspace/keyframes_checkpoint.txt"
FPS = 0.5  # 1 frame every 2 seconds

os.makedirs(KEYFRAMES_DIR, exist_ok=True)

# Load checkpoint
done = set()
if Path(CHECKPOINT).exists():
    done = set(Path(CHECKPOINT).read_text().splitlines())

# Find all videos
exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
all_videos = [p for p in Path(PHOTOS_DIR).rglob("*") if p.suffix.lower() in exts and str(p) not in done]
print(f"To process: {len(all_videos)} videos ({len(done)} already done)")

lock = threading.Lock()
counter = [0]

def extract_keyframes(video_path):
    try:
        video_name = Path(video_path).stem
        out_dir = Path(KEYFRAMES_DIR) / video_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps={FPS}",
            "-q:v", "2",
            str(out_dir / "frame_%04d.jpg"),
            "-y", "-loglevel", "error"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        frames = list(out_dir.glob("*.jpg"))
        return str(video_path), len(frames), None
    except Exception as e:
        return str(video_path), 0, str(e)

total_frames = [0]

with open(CHECKPOINT, "a") as ckpt_f:
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(extract_keyframes, v): v for v in all_videos}
        for f in as_completed(futures):
            path, n_frames, err = f.result()
            with lock:
                ckpt_f.write(path + "\n")
                ckpt_f.flush()
                counter[0] += 1
                total_frames[0] += n_frames
                if counter[0] % 100 == 0:
                    print(f"  {counter[0]}/{len(all_videos)} videos done, {total_frames[0]} frames extracted...")

print(f"Stage 4 complete. {total_frames[0]} keyframes -> {KEYFRAMES_DIR}")
