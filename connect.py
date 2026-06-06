#!/usr/bin/env python3
"""
gbrain connection-mining pass.

Runs ONCE after extract + video + index are done. Pre-computes three kinds of
high-value pairings across the whole archive and stores them in the same DB:

  bridges       - items semantically close but from different surface domains
                  (the cross-domain insight surface)
  clusters      - groups of items circling the same latent idea
                  (themes you didn't consciously curate around)
  contradictions - items that make competing claims on the same topic
                   (where your thinking has room to move)

Each pairing/cluster includes a short LLM-written "why this might matter" note
with explicit confidence-hedging baked into the prompt so speculative bridges
are flagged as such.

Run on the GPU instance:
    python connect.py --bridges
    python connect.py --clusters
    python connect.py --contradictions
or:
    python connect.py --all
"""

import argparse, sqlite3, json, struct, random
from collections import defaultdict

DB_PATH = "/workspace/gbrain.db"
DIM = 384
VLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"   # reused for text-only LLM calls

# Tunables (sane defaults; raise for more output, at proportional cost)
BRIDGE_SAMPLE = 5000          # how many anchor items to mine bridges from
BRIDGE_NEIGHBORS = 20         # candidates considered per anchor
BRIDGE_KEEP = 3               # bridges kept per anchor after domain-difference filter
CLUSTER_K = 80                # rough number of latent clusters to surface
CLUSTER_MIN_SIZE = 5
CONTRADICTION_PAIRS = 500     # candidate close-pairs to LLM-check for contradiction


def connect():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS bridges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path_a TEXT, path_b TEXT, distance REAL,
            domain_a TEXT, domain_b TEXT,
            note TEXT, confidence TEXT
        );
        CREATE TABLE IF NOT EXISTS clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT, summary TEXT, size INTEGER
        );
        CREATE TABLE IF NOT EXISTS cluster_members (
            cluster_id INTEGER, path TEXT,
            FOREIGN KEY(cluster_id) REFERENCES clusters(id)
        );
        CREATE TABLE IF NOT EXISTS contradictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path_a TEXT, path_b TEXT,
            claim_a TEXT, claim_b TEXT, note TEXT
        );
    """)
    return conn


def parse_section(text, section):
    """Pull a single 'SECTION:' line out of the structured VLM output."""
    if not text:
        return ""
    for line in text.splitlines():
        if line.upper().startswith(section + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def llm_text(prompt, max_tokens=400):
    """Text-only LLM call (reuses the same VLM endpoint)."""
    import requests
    r = requests.post(VLM_ENDPOINT, json={
        "model": "vlm",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }, timeout=120)
    return r.json()["choices"][0]["message"]["content"].strip()


# ---- BRIDGES -----------------------------------------------------------

def mine_bridges():
    conn = connect()
    print("mining cross-domain bridges...")
    anchors = conn.execute(
        "SELECT path FROM items WHERE stage_done IS NOT NULL ORDER BY RANDOM() LIMIT ?",
        (BRIDGE_SAMPLE,)).fetchall()
    print(f"  anchor pool: {len(anchors)}")

    for i, (anchor_path,) in enumerate(anchors, 1):
        # get anchor's vector + domain
        v_row = conn.execute("SELECT embedding FROM vec_items WHERE path=?",
                             (anchor_path,)).fetchone()
        if not v_row:
            continue
        anchor_item = conn.execute(
            "SELECT vlm_desc, ocr_text FROM items WHERE path=?",
            (anchor_path,)).fetchone()
        anchor_text = (anchor_item["vlm_desc"] or "") + "\n" + (anchor_item["ocr_text"] or "")
        anchor_domain = parse_section(anchor_item["vlm_desc"] or "", "DOMAIN")

        # find near-neighbors
        neighbors = conn.execute("""
            SELECT v.path, v.distance, i.vlm_desc, i.ocr_text
            FROM vec_items v JOIN items i ON i.path = v.path
            WHERE v.embedding MATCH ? AND v.path != ?
            ORDER BY v.distance LIMIT ?
        """, (v_row["embedding"], anchor_path, BRIDGE_NEIGHBORS)).fetchall()

        # keep only neighbors from a DIFFERENT domain (that's the "bridge" part)
        kept = []
        for nb in neighbors:
            nb_domain = parse_section(nb["vlm_desc"] or "", "DOMAIN")
            if nb_domain and anchor_domain and nb_domain.lower() != anchor_domain.lower():
                kept.append((nb, nb_domain))
            if len(kept) >= BRIDGE_KEEP:
                break

        for nb, nb_domain in kept:
            nb_text = (nb["vlm_desc"] or "") + "\n" + (nb["ocr_text"] or "")
            note_prompt = (
                "Two items from a personal knowledge archive ended up semantically "
                "close despite being from different domains. Write 2-3 sentences "
                "on what they might have in common at a deeper level, and whether "
                "the connection is substantive or surface-level. End with one of "
                "exactly these tokens on its own line: CONFIDENCE: high | medium | low.\n\n"
                f"ITEM A (domain: {anchor_domain}):\n{anchor_text[:1200]}\n\n"
                f"ITEM B (domain: {nb_domain}):\n{nb_text[:1200]}"
            )
            try:
                note = llm_text(note_prompt)
                conf = "medium"
                for line in note.splitlines():
                    if line.upper().startswith("CONFIDENCE:"):
                        conf = line.split(":", 1)[1].strip().lower()
                        note = note.replace(line, "").strip()
                        break
                conn.execute("""INSERT INTO bridges
                    (path_a, path_b, distance, domain_a, domain_b, note, confidence)
                    VALUES (?,?,?,?,?,?,?)""",
                    (anchor_path, nb["path"], nb["distance"],
                     anchor_domain, nb_domain, note, conf))
                conn.commit()
            except Exception as e:
                print(f"  ! bridge error: {e}")
        if i % 50 == 0:
            print(f"  ...{i}/{len(anchors)} anchors processed")
    conn.close()
    print("bridges done.")


# ---- CLUSTERS ----------------------------------------------------------

def mine_clusters():
    """K-means over the embedding space to surface latent themes."""
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    conn = connect()
    print(f"mining {CLUSTER_K} latent clusters...")

    rows = conn.execute(
        "SELECT v.path, v.embedding, i.vlm_desc FROM vec_items v "
        "JOIN items i ON i.path = v.path").fetchall()
    if not rows:
        print("  no embeddings found; run index.py --build first")
        return
    paths = [r["path"] for r in rows]
    X = np.array([struct.unpack(f"{DIM}f", r["embedding"]) for r in rows], dtype="float32")
    print(f"  clustering {len(paths)} items...")

    km = MiniBatchKMeans(n_clusters=CLUSTER_K, batch_size=2048, n_init=3, random_state=42)
    labels = km.fit_predict(X)

    groups = defaultdict(list)
    for path, lab in zip(paths, labels):
        groups[int(lab)].append(path)

    for lab, members in groups.items():
        if len(members) < CLUSTER_MIN_SIZE:
            continue
        # sample up to 8 member descriptions for the LLM to summarize
        sample = random.sample(members, min(8, len(members)))
        descs = []
        for p in sample:
            row = conn.execute("SELECT vlm_desc, ocr_text FROM items WHERE path=?",
                               (p,)).fetchone()
            if row:
                txt = (row["vlm_desc"] or "")[:400] + " | " + (row["ocr_text"] or "")[:200]
                descs.append(txt)
        cluster_prompt = (
            "These items from a personal archive ended up in the same latent cluster. "
            "Write a short LABEL (4-8 words) naming the underlying theme, then a "
            "2-3 sentence SUMMARY describing what unites them and what's interesting "
            "about the user having collected so much in this area. Format:\n"
            "LABEL: ...\nSUMMARY: ...\n\n"
            "ITEMS:\n" + "\n---\n".join(descs)
        )
        try:
            out = llm_text(cluster_prompt)
            label = parse_section(out, "LABEL") or f"cluster {lab}"
            summary = parse_section(out, "SUMMARY") or out
            cur = conn.execute("INSERT INTO clusters (label, summary, size) VALUES (?,?,?)",
                               (label, summary, len(members)))
            cid = cur.lastrowid
            conn.executemany("INSERT INTO cluster_members (cluster_id, path) VALUES (?,?)",
                             [(cid, p) for p in members])
            conn.commit()
        except Exception as e:
            print(f"  ! cluster error: {e}")
    conn.close()
    print("clusters done.")


# ---- CONTRADICTIONS ----------------------------------------------------

def mine_contradictions():
    """Look at close pairs and let the LLM judge whether they make competing claims."""
    conn = connect()
    print("mining productive contradictions...")
    # candidate pool: random items, find each one's nearest neighbor
    anchors = conn.execute(
        "SELECT path FROM items WHERE stage_done IS NOT NULL ORDER BY RANDOM() LIMIT ?",
        (CONTRADICTION_PAIRS,)).fetchall()
    found = 0
    for i, (a_path,) in enumerate(anchors, 1):
        v = conn.execute("SELECT embedding FROM vec_items WHERE path=?",
                         (a_path,)).fetchone()
        if not v:
            continue
        nb = conn.execute("""
            SELECT v.path, i.vlm_desc, i.ocr_text FROM vec_items v
            JOIN items i ON i.path=v.path
            WHERE v.embedding MATCH ? AND v.path != ?
            ORDER BY v.distance LIMIT 1
        """, (v["embedding"], a_path)).fetchone()
        if not nb:
            continue
        a_row = conn.execute("SELECT vlm_desc, ocr_text FROM items WHERE path=?",
                             (a_path,)).fetchone()
        a_text = ((a_row["vlm_desc"] or "") + "\n" + (a_row["ocr_text"] or ""))[:1500]
        b_text = ((nb["vlm_desc"] or "") + "\n" + (nb["ocr_text"] or ""))[:1500]

        prompt = (
            "Two items from a personal archive are semantically close. Determine "
            "whether they make COMPETING or CONTRADICTORY claims about the same "
            "topic (not merely different topics, and not minor variations). If yes, "
            "output:\n"
            "CONTRADICTION: yes\n"
            "CLAIM_A: <one sentence>\n"
            "CLAIM_B: <one sentence>\n"
            "NOTE: <2-3 sentences on the tension and why it matters>\n"
            "If no real contradiction, output exactly:\n"
            "CONTRADICTION: no\n\n"
            f"ITEM A:\n{a_text}\n\nITEM B:\n{b_text}"
        )
        try:
            out = llm_text(prompt)
            if parse_section(out, "CONTRADICTION").lower().startswith("yes"):
                conn.execute("""INSERT INTO contradictions
                    (path_a, path_b, claim_a, claim_b, note) VALUES (?,?,?,?,?)""",
                    (a_path, nb["path"],
                     parse_section(out, "CLAIM_A"),
                     parse_section(out, "CLAIM_B"),
                     parse_section(out, "NOTE")))
                conn.commit()
                found += 1
        except Exception as e:
            print(f"  ! contradiction error: {e}")
        if i % 50 == 0:
            print(f"  ...{i}/{len(anchors)} pairs checked, {found} contradictions found")
    conn.close()
    print(f"contradictions done. {found} found.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridges", action="store_true")
    ap.add_argument("--clusters", action="store_true")
    ap.add_argument("--contradictions", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all or args.bridges:        mine_bridges()
    if args.all or args.clusters:       mine_clusters()
    if args.all or args.contradictions: mine_contradictions()
