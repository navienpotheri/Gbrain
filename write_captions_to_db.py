"""
write_captions_to_db.py
Reads captions.jsonl and writes vlm_desc + stage_done into gbrain.db items table.

Usage:
    python /workspace/write_captions_to_db.py [--captions /workspace/captions.jsonl] [--db /workspace/gbrain.db]

Expected JSONL format (one JSON object per line):
    {"path": "/workspace/Google Photos/...", "caption": "..."}
    or
    {"path": "...", "vlm_desc": "..."}   (alternate key name)
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Write captions into gbrain.db")
    p.add_argument("--captions", default="/workspace/captions.jsonl",
                   help="Path to merged captions JSONL file (default: /workspace/captions.jsonl)")
    p.add_argument("--db", default="/workspace/gbrain.db",
                   help="Path to gbrain.db (default: /workspace/gbrain.db)")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse and count without writing to DB")
    p.add_argument("--batch-size", type=int, default=500,
                   help="DB commit interval (default: 500)")
    return p.parse_args()


def iter_captions(captions_path: str):
    """Yield (path, caption) tuples from JSONL, skipping bad lines."""
    skipped = 0
    with open(captions_path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {lineno}: JSON parse error — {e}", file=sys.stderr)
                skipped += 1
                continue

            path = obj.get("path")
            # Accept either 'caption' or 'vlm_desc' as the text key
            caption = obj.get("caption") or obj.get("vlm_desc")

            if not path or not caption:
                print(f"  [WARN] Line {lineno}: missing 'path' or caption key — skipping", file=sys.stderr)
                skipped += 1
                continue

            yield path, caption

    if skipped:
        print(f"  [WARN] Skipped {skipped} malformed lines total", file=sys.stderr)


def write_captions(captions_path: str, db_path: str, dry_run: bool, batch_size: int):
    if not Path(captions_path).exists():
        print(f"[ERROR] Captions file not found: {captions_path}", file=sys.stderr)
        sys.exit(1)
    if not Path(db_path).exists():
        print(f"[ERROR] DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading captions from : {captions_path}")
    print(f"Target DB             : {db_path}")
    print(f"Dry run               : {dry_run}")
    print(f"Batch size            : {batch_size}")
    print()

    total = 0
    updated = 0
    not_found = 0
    batch = []

    if dry_run:
        for path, caption in iter_captions(captions_path):
            total += 1
        print(f"[DRY RUN] Would attempt to write {total} captions.")
        return

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    cur = con.cursor()

    for path, caption in iter_captions(captions_path):
        total += 1
        batch.append((caption, path))

        if len(batch) >= batch_size:
            cur.executemany(
                "UPDATE items SET vlm_desc=?, stage_done='visual' WHERE path=?",
                batch
            )
            updated += cur.rowcount
            con.commit()
            batch.clear()
            print(f"  Written {total} so far ...", end="\r")

    # Flush remainder
    if batch:
        cur.executemany(
            "UPDATE items SET vlm_desc=?, stage_done='visual' WHERE path=?",
            batch
        )
        updated += cur.rowcount
        con.commit()

    # Quick sanity check: how many paths were in the JSONL but not in the DB?
    not_found = total - updated
    con.close()

    print()
    print(f"Done.")
    print(f"  JSONL records read   : {total}")
    print(f"  DB rows updated      : {updated}")
    if not_found > 0:
        print(f"  Paths not in DB      : {not_found}  (paths in JSONL not matching any items row)")


if __name__ == "__main__":
    args = parse_args()
    write_captions(
        captions_path=args.captions,
        db_path=args.db,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )
