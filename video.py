#!/usr/bin/env python3
"""
gbrain VIDEO + AUDIO stage (GPU instance).

For each short clip:
  1. ffmpeg extracts scene-change keyframes (sparse, content-aware).
  2. VLM captions those frames (general description; NO person identification).
  3. Whisper transcribes any speech (supplements the visual; clips are mainly visual).
One combined record per clip in the shared gbrain.db. Fully resumable.

Run on the rented GPU instance after stills are done:
  python video.py
"""

import os, sqlite3, json, subprocess, tempfile, glob, base64

VIDEO_DIR = "/workspace/screenshots"      # videos live alongside stills; we filter by ext
DB_PATH   = "/workspace/gbrain.db"
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".gif"}
SCENE_THRESHOLD = 0.30                    # ffmpeg scene-change sensitivity (0-1); higher = fewer frames
MAX_FRAMES = 4                            # cap keyframes per short clip
WHISPER_MODEL = "base"                    # base is plenty for "less spoken"; runs fast on GPU
VLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"  # local vLLM server on the instance

# Guardrail baked into the captioning prompt — describe, do not identify people.
VLM_PROMPT = (
    "You are extracting content from a video keyframe for a personal knowledge base "
    "used to assist with writing, thinking, and finding unexpected connections. "
    "Output the following sections, each on its own line:\n\n"
    "DESCRIPTION: 1-2 sentences on what the frame shows.\n"
    "CORE: the central claim, data point, framing, or move being made in this frame. "
    "If it's a chart, state what it shows. If it's a moment in an argument or "
    "demonstration, state what point is being made.\n"
    "DOMAIN: the surface domain in 2-4 words.\n"
    "ENTITIES: named people, organizations, products, papers, concepts. Omit if none.\n"
    "WHY-INTERESTING: 1 sentence on what's notable about this.\n"
    "TAGS: 4-8 comma-separated tags.\n\n"
    "Rules: describe people generically only; do NOT identify, name, or guess the "
    "identity of any individual. Use 'n/a' for sections that don't apply."
)


def db():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""CREATE TABLE IF NOT EXISTS items (
        path TEXT PRIMARY KEY, sha1 TEXT, mtime TEXT, tier TEXT,
        ocr_text TEXT, vlm_desc TEXT, tags TEXT, transcript TEXT, stage_done TEXT)""")
    # add columns if upgrading an older db
    cols = {r[1] for r in conn.execute("PRAGMA table_info(items)")}
    if "transcript" not in cols:
        conn.execute("ALTER TABLE items ADD COLUMN transcript TEXT")
    conn.commit()
    return conn


def iter_videos():
    for root, _, files in os.walk(VIDEO_DIR):
        for f in files:
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                yield os.path.join(root, f)


def keyframes(path, outdir):
    # scene-change detection; fall back to a few evenly spaced frames if none found
    out = os.path.join(outdir, "f_%03d.jpg")
    cmd = ["ffmpeg", "-i", path, "-vf",
           f"select='gt(scene,{SCENE_THRESHOLD})',showinfo", "-vsync", "vfr",
           "-frames:v", str(MAX_FRAMES), out, "-y"]
    subprocess.run(cmd, capture_output=True)
    frames = sorted(glob.glob(os.path.join(outdir, "f_*.jpg")))
    if not frames:
        subprocess.run(["ffmpeg", "-i", path, "-vf", "fps=1/3",
                        "-frames:v", "2", out, "-y"], capture_output=True)
        frames = sorted(glob.glob(os.path.join(outdir, "f_*.jpg")))
    return frames[:MAX_FRAMES]


def caption_frame(frame_path):
    import requests
    with open(frame_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    payload = {
        "model": "vlm",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": VLM_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}],
        "max_tokens": 300,
    }
    r = requests.post(VLM_ENDPOINT, json=payload, timeout=120)
    return r.json()["choices"][0]["message"]["content"]


def transcribe(path, model):
    try:
        res = model.transcribe(path)
        return (res.get("text") or "").strip()
    except Exception:
        return ""


def main():
    import whisper
    wmodel = whisper.load_model(WHISPER_MODEL, device="cuda")
    conn = db()
    done = {r[0] for r in conn.execute(
        "SELECT path FROM items WHERE tier='video' AND stage_done='video'")}
    todo = [v for v in iter_videos() if v not in done]
    print(f"{len(todo)} videos to process ({len(done)} done)")

    for i, path in enumerate(todo, 1):
        with tempfile.TemporaryDirectory() as td:
            frames = keyframes(path, td)
            caps, tags = [], []
            for fr in frames:
                try:
                    out = caption_frame(fr)
                    if "TAGS:" in out:
                        d, t = out.rsplit("TAGS:", 1)
                        caps.append(d.strip())
                        tags += [x.strip() for x in t.replace("\n", ",").split(",") if x.strip()]
                    else:
                        caps.append(out.strip())
                except Exception as e:
                    caps.append(f"__VLM_ERROR__ {e}")
            transcript = transcribe(path, wmodel)
        conn.execute("""INSERT INTO items (path, tier, vlm_desc, tags, transcript, stage_done)
                        VALUES (?,?,?,?,?, 'video')
                        ON CONFLICT(path) DO UPDATE SET
                          tier='video', vlm_desc=excluded.vlm_desc, tags=excluded.tags,
                          transcript=excluded.transcript, stage_done='video'""",
                     (path, "video", "\n".join(caps), json.dumps(sorted(set(tags))), transcript))
        conn.commit()
        if i % 25 == 0:
            print(f"  ...{i}/{len(todo)}")
    conn.close()
    print("video+audio stage complete.")


if __name__ == "__main__":
    main()
