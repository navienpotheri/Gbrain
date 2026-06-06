#!/usr/bin/env python3
"""
gbrain pipeline - local, private, CPU-friendly extraction for ~80k screenshots.

Stages handled here:
  TRIAGE  : a fast PaddleOCR pass decides if an image is text-heavy or visual.
  TEXT    : text-heavy images -> store OCR text.
  VISUAL  : visual images -> local VLM (via Ollama) caption + tags.
All results land in one SQLite DB. Fully resumable: rerun anytime, it skips done work.

Run order:
  python extract.py --stage triage      # fast, all images, sorts them
  python extract.py --stage text         # OCR the text tier (fast, parallel)
  python extract.py --stage visual       # VLM the visual tier (slow, runs for days)

Designed for: Intel Core Ultra 9 185H, 32GB RAM, no NVIDIA GPU, Windows.
"""

import argparse, os, sqlite3, json, hashlib, datetime, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- config you may want to tweak --------------------------------------
IMG_DIR   = "/workspace/screenshots"   # where all your images live on the pod
DB_PATH   = "/workspace/gbrain.db"
EXTS      = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
TEXT_THRESHOLD_CHARS = 40              # >= this many OCR chars => "text-heavy"
OCR_WORKERS = 12                       # leave a couple cores for the OS (185H has 16)
VLM_MODEL = "qwen2-vl"                 # pulled via: ollama pull qwen2-vl
# ------------------------------------------------------------------------


def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")   # safe concurrent-ish access, crash resilient
    conn.execute("""
        CREATE TABLE IF NOT EXISTS items (
            path        TEXT PRIMARY KEY,
            sha1        TEXT,
            mtime       TEXT,
            tier        TEXT,      -- 'text' or 'visual', set in triage
            ocr_text    TEXT,
            vlm_desc    TEXT,
            tags        TEXT,      -- JSON list
            stage_done  TEXT       -- 'triage' | 'text' | 'visual'
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tier ON items(tier, stage_done);")
    conn.commit()
    return conn


def iter_images():
    for root, _, files in os.walk(IMG_DIR):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                yield os.path.join(root, f)


def file_meta(path):
    st = os.stat(path)
    mtime = datetime.datetime.fromtimestamp(st.st_mtime).isoformat()
    # cheap partial hash for dedup verification (first 1MB) - full dedup done by czkawka earlier
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read(1024 * 1024))
    return h.hexdigest(), mtime


# ---- TRIAGE + TEXT (PaddleOCR) -----------------------------------------
# PaddleOCR is imported lazily inside workers so the parent stays light.

def _ocr_one(path):
    from paddleocr import PaddleOCR
    global _OCR
    try:
        _OCR
    except NameError:
        _OCR = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    try:
        result = _OCR.ocr(path, cls=True)
        lines = []
        for block in (result or []):
            for line in (block or []):
                txt = line[1][0]
                if txt:
                    lines.append(txt)
        return path, "\n".join(lines)
    except Exception as e:
        return path, f"__OCR_ERROR__ {e}"


def stage_triage_and_text(only_triage):
    conn = db_connect()
    done = {r[0] for r in conn.execute(
        "SELECT path FROM items WHERE stage_done IS NOT NULL").fetchall()}
    todo = [p for p in iter_images() if p not in done]
    print(f"{len(todo)} images to process (skipping {len(done)} already done)")

    batch = []
    with ProcessPoolExecutor(max_workers=OCR_WORKERS) as ex:
        futs = {ex.submit(_ocr_one, p): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            path, text = fut.result()
            sha1, mtime = file_meta(path)
            tier = "text" if len(text.strip()) >= TEXT_THRESHOLD_CHARS else "visual"
            stage = "text" if (tier == "text" and not only_triage) else "triage"
            batch.append((path, sha1, mtime, tier, text, stage))
            if len(batch) >= 200:
                _flush_text(conn, batch); batch.clear()
                print(f"  ...{i}/{len(todo)}")
        if batch:
            _flush_text(conn, batch)
    conn.close()
    print("triage/text stage complete.")


def _flush_text(conn, batch):
    conn.executemany("""
        INSERT INTO items (path, sha1, mtime, tier, ocr_text, stage_done)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(path) DO UPDATE SET
            sha1=excluded.sha1, mtime=excluded.mtime, tier=excluded.tier,
            ocr_text=excluded.ocr_text, stage_done=excluded.stage_done
    """, batch)
    conn.commit()


# ---- VISUAL (local VLM via Ollama) -------------------------------------

def stage_visual():
    import base64, requests
    conn = db_connect()
    rows = conn.execute(
        "SELECT path FROM items WHERE tier='visual' AND stage_done!='visual'").fetchall()
    print(f"{len(rows)} visual images to caption with {VLM_MODEL} (slow; resumable)")

    prompt = (
        "You are extracting content from an image for a personal knowledge base used "
        "to assist with writing, thinking, and finding unexpected connections across "
        "a large archive. Output the following sections, each on its own line:\n\n"
        "DESCRIPTION: 1-2 sentences on what the image shows (scene, source, format).\n"
        "CORE: the central claim, data point, framing, or move being made. Be specific. "
        "If it's a chart, state what it shows and the key value or trend. If it's an "
        "argument, state the argument in one sentence. If it's an anecdote or example, "
        "state what it's an example of.\n"
        "DOMAIN: the surface domain in 2-4 words (e.g. 'cognitive science', "
        "'AI infrastructure', 'product design', 'cultural criticism').\n"
        "ENTITIES: comma-separated named people, organizations, products, papers, or "
        "concepts referenced. Omit if none.\n"
        "WHY-INTERESTING: 1 sentence on what's notable, counterintuitive, or worth "
        "remembering about this. Skip generic observations.\n"
        "TAGS: 4-8 comma-separated topic tags useful for later retrieval.\n\n"
        "Rules: If people appear, describe only generic context-relevant attributes "
        "(e.g. 'a person presenting at a conference'). Do NOT identify, name, or guess "
        "the identity of any individual. If a section doesn't apply, write 'n/a'."
    )

    for i, (path,) in enumerate(rows, 1):
        try:
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            r = requests.post("http://localhost:11434/api/generate", json={
                "model": VLM_MODEL, "prompt": prompt,
                "images": [b64], "stream": False
            }, timeout=300)
            out = r.json().get("response", "")
            desc, tags = out, []
            if "TAGS:" in out:
                desc, tagline = out.rsplit("TAGS:", 1)
                tags = [t.strip() for t in tagline.replace("\n", ",").split(",") if t.strip()]
            conn.execute("UPDATE items SET vlm_desc=?, tags=?, stage_done='visual' WHERE path=?",
                         (desc.strip(), json.dumps(tags), path))
            conn.commit()
        except Exception as e:
            print(f"  ! error on {path}: {e}")
            continue
        if i % 25 == 0:
            print(f"  ...{i}/{len(rows)}")
    conn.close()
    print("visual stage complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["triage", "text", "visual"])
    args = ap.parse_args()
    if args.stage == "triage":
        stage_triage_and_text(only_triage=True)
    elif args.stage == "text":
        stage_triage_and_text(only_triage=False)
    elif args.stage == "visual":
        stage_visual()
