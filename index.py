#!/usr/bin/env python3
"""
gbrain index + search - local embeddings for semantic search & connection discovery.

  python index.py --build        # embed every item's text+desc (one-time, resumable)
  python index.py --search "..." # semantic search from the terminal
  python index.py --neighbors PATH  # find items most related to a given screenshot

Uses a local sentence-transformers model (downloaded once, then fully offline).
Vectors stored in the same SQLite DB via sqlite-vec. No data leaves the machine.
"""

import argparse, sqlite3, json, struct, sys

DB_PATH = "/workspace/gbrain.db"   # on the pod; change to local path when running search at home
MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast on CPU, 384-dim; good enough for this
DIM = 384


def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)   # caches locally after first download


def connect():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_items
        USING vec0(path TEXT PRIMARY KEY, embedding FLOAT[{DIM}]);
    """)
    return conn


def text_for(row):
    # combine OCR text and VLM description so both tiers are searchable together
    parts = [row["ocr_text"] or "", row["vlm_desc"] or "", row["tags"] or ""]
    return "\n".join(p for p in parts if p).strip()


def build():
    conn = connect()
    conn.row_factory = sqlite3.Row
    done = {r[0] for r in conn.execute("SELECT path FROM vec_items").fetchall()}
    rows = [r for r in conn.execute("SELECT * FROM items WHERE stage_done IS NOT NULL")
            if r["path"] not in done]
    print(f"embedding {len(rows)} items ({len(done)} already embedded)")
    model = get_model()
    buf = []
    for i, r in enumerate(rows, 1):
        txt = text_for(r)
        if not txt:
            continue
        buf.append((r["path"], txt))
        if len(buf) >= 256:
            _embed_flush(conn, model, buf); buf.clear()
            print(f"  ...{i}/{len(rows)}")
    if buf:
        _embed_flush(conn, model, buf)
    conn.close()
    print("index built.")


def _embed_flush(conn, model, buf):
    vecs = model.encode([t for _, t in buf], normalize_embeddings=True)
    for (path, _), v in zip(buf, vecs):
        blob = struct.pack(f"{DIM}f", *v.tolist())
        conn.execute("INSERT OR REPLACE INTO vec_items(path, embedding) VALUES (?, ?)",
                     (path, blob))
    conn.commit()


def search(query, k=15):
    conn = connect()
    conn.row_factory = sqlite3.Row
    model = get_model()
    qv = model.encode([query], normalize_embeddings=True)[0]
    blob = struct.pack(f"{DIM}f", *qv.tolist())
    rows = conn.execute("""
        SELECT v.path, v.distance, i.ocr_text, i.vlm_desc, i.tags
        FROM vec_items v JOIN items i ON i.path = v.path
        WHERE v.embedding MATCH ? ORDER BY v.distance LIMIT ?
    """, (blob, k)).fetchall()
    for r in rows:
        snippet = (r["ocr_text"] or r["vlm_desc"] or "")[:160].replace("\n", " ")
        print(f"\n[{r['distance']:.3f}] {r['path']}\n   {snippet}")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--search")
    args = ap.parse_args()
    if args.build:
        build()
    elif args.search:
        search(args.search)
