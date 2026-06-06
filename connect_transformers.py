#!/usr/bin/env python3
"""
connect_transformers.py  —  gbrain connection-mining pass (full edition)
Runs ONCE after extract + index are done.

LLM stages (32B via transformers, 4-bit):
  --bridges         cross-domain semantic bridges
  --clusters        latent k-means themes
  --contradictions  competing-claim pairs

Pure-math stages (no LLM, fast):
  --anchors         best representative item per cluster
  --orphans         items with no cluster membership
  --duplicates      near-identical embedding pairs
  --strong-weak     core vs peripheral members per cluster
  --overlap         items on the boundary between two clusters
  --chains          two-hop bridge chains
  --hubs            most-connected items in similarity graph
  --isolation       per-item isolation score (avg dist to k-NN)
  --drift           embedding spread per folder/date group

  --all-llm         run all LLM stages
  --all-math        run all pure-math stages
  --all             run everything
"""

import sqlite3, random, argparse, struct, json
from collections import defaultdict

# ── config ────────────────────────────────────────────────────────────────────
DB_PATH   = "/workspace/gbrain.db"
MODEL_DIR = "/workspace/models/Qwen2.5-VL-32B-Instruct"
DIM       = 384

BRIDGE_SAMPLE        = 5000
BRIDGE_NEIGHBORS     = 20
BRIDGE_KEEP          = 3
CLUSTER_K            = 80
CLUSTER_MIN_SIZE     = 5
CONTRADICTION_PAIRS  = 500
DUPLICATE_THRESHOLD  = 0.12   # cosine distance below this = semantic duplicate
HUB_TOPK             = 500    # how many top-hub items to store
ISOLATION_K          = 10     # neighbours for isolation score
CHAIN_MAX            = 3000   # max bridge-chain pairs to store

# ── lazy model loader ─────────────────────────────────────────────────────────
_model     = None
_tokenizer = None

def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    print("Loading 32B model via transformers (4-bit)…")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    _model     = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    _model.eval()
    print("32B model ready.")

def llm_text(prompt, max_tokens=400):
    import torch
    _load_model()
    messages = [{"role": "user", "content": prompt}]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    trimmed = out[0][inputs.input_ids.shape[1]:]
    return _tokenizer.decode(trimmed, skip_special_tokens=True).strip()

# ── db ────────────────────────────────────────────────────────────────────────
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
            strength TEXT DEFAULT 'core',
            dist_to_centroid REAL,
            FOREIGN KEY(cluster_id) REFERENCES clusters(id)
        );
        CREATE TABLE IF NOT EXISTS contradictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path_a TEXT, path_b TEXT,
            claim_a TEXT, claim_b TEXT, note TEXT
        );
        CREATE TABLE IF NOT EXISTS anchors (
            cluster_id INTEGER PRIMARY KEY,
            path TEXT,
            FOREIGN KEY(cluster_id) REFERENCES clusters(id)
        );
        CREATE TABLE IF NOT EXISTS orphans (
            path TEXT PRIMARY KEY,
            isolation_score REAL
        );
        CREATE TABLE IF NOT EXISTS semantic_duplicates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path_a TEXT, path_b TEXT, distance REAL
        );
        CREATE TABLE IF NOT EXISTS bridge_chains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path_a TEXT, path_mid TEXT, path_b TEXT,
            dist_ab REAL, dist_bc REAL,
            domain_a TEXT, domain_mid TEXT, domain_b TEXT
        );
        CREATE TABLE IF NOT EXISTS hub_items (
            path TEXT PRIMARY KEY,
            hub_score INTEGER
        );
        CREATE TABLE IF NOT EXISTS isolation_scores (
            path TEXT PRIMARY KEY,
            isolation_score REAL
        );
        CREATE TABLE IF NOT EXISTS cluster_overlap (
            path TEXT PRIMARY KEY,
            cluster_id_a INTEGER, cluster_id_b INTEGER,
            dist_a REAL, dist_b REAL
        );
        CREATE TABLE IF NOT EXISTS embedding_drift (
            group_key TEXT PRIMARY KEY,
            item_count INTEGER,
            drift_score REAL,
            sample_paths TEXT
        );
    """)
    conn.commit()
    return conn

def parse_section(text, section):
    if not text:
        return ""
    for line in text.splitlines():
        if line.upper().startswith(section + ":"):
            return line.split(":", 1)[1].strip()
    return ""

def _unpack(blob):
    import struct
    return struct.unpack(f"{DIM}f", blob)

def _all_embeddings(conn):
    """Return {path: np.array} for all items with embeddings."""
    import numpy as np
    rows = conn.execute("SELECT path, embedding FROM vec_items").fetchall()
    return {r["path"]: np.array(_unpack(r["embedding"]), dtype="float32") for r in rows}

# ══════════════════════════════════════════════════════════════════════════════
# LLM STAGES
# ══════════════════════════════════════════════════════════════════════════════

def mine_bridges():
    conn = connect()
    print("mining cross-domain bridges…")
    anchors = conn.execute(
        "SELECT path FROM items WHERE stage_done IS NOT NULL ORDER BY RANDOM() LIMIT ?",
        (BRIDGE_SAMPLE,)
    ).fetchall()
    print(f"  anchor pool: {len(anchors)}")

    for i, (anchor_path,) in enumerate(anchors, 1):
        v_row = conn.execute(
            "SELECT embedding FROM vec_items WHERE path=?", (anchor_path,)
        ).fetchone()
        if not v_row:
            continue
        anchor_item   = conn.execute(
            "SELECT vlm_desc, ocr_text FROM items WHERE path=?", (anchor_path,)
        ).fetchone()
        anchor_text   = (anchor_item["vlm_desc"] or "") + "\n" + (anchor_item["ocr_text"] or "")
        anchor_domain = parse_section(anchor_item["vlm_desc"] or "", "DOMAIN")

        neighbors = conn.execute("""
            SELECT v.path, v.distance, i.vlm_desc, i.ocr_text
            FROM vec_items v JOIN items i ON i.path = v.path
            WHERE v.embedding MATCH ? AND v.path != ?
            ORDER BY v.distance LIMIT ?
        """, (v_row["embedding"], anchor_path, BRIDGE_NEIGHBORS)).fetchall()

        kept = []
        for nb in neighbors:
            nb_domain = parse_section(nb["vlm_desc"] or "", "DOMAIN")
            if nb_domain and anchor_domain and nb_domain.lower() != anchor_domain.lower():
                kept.append((nb, nb_domain))
        if len(kept) >= BRIDGE_KEEP:
            kept = kept[:BRIDGE_KEEP]

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
                conn.execute("""
                    INSERT INTO bridges
                        (path_a, path_b, distance, domain_a, domain_b, note, confidence)
                    VALUES (?,?,?,?,?,?,?)
                """, (anchor_path, nb["path"], nb["distance"],
                      anchor_domain, nb_domain, note, conf))
                conn.commit()
            except Exception as e:
                print(f"  ! bridge error: {e}")

        if i % 50 == 0:
            print(f"  …{i}/{len(anchors)} anchors processed")

    conn.close()
    print("bridges done.")


def mine_clusters():
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    conn = connect()
    print(f"mining {CLUSTER_K} latent clusters…")

    rows = conn.execute(
        "SELECT v.path, v.embedding, i.vlm_desc FROM vec_items v "
        "JOIN items i ON i.path = v.path"
    ).fetchall()
    if not rows:
        print("  no embeddings found; run index.py --build first")
        return

    paths  = [r["path"] for r in rows]
    X      = np.array([_unpack(r["embedding"]) for r in rows], dtype="float32")
    print(f"  clustering {len(paths)} items…")

    km     = MiniBatchKMeans(n_clusters=CLUSTER_K, batch_size=2048, n_init=3, random_state=42)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    groups = defaultdict(list)
    for idx, (path, lab) in enumerate(zip(paths, labels)):
        dist = float(np.linalg.norm(X[idx] - centroids[lab]))
        groups[int(lab)].append((path, idx, dist))

    for lab, members in groups.items():
        if len(members) < CLUSTER_MIN_SIZE:
            continue
        sample_paths = random.sample([m[0] for m in members], min(8, len(members)))
        descs = []
        for p in sample_paths:
            row = conn.execute(
                "SELECT vlm_desc, ocr_text FROM items WHERE path=?", (p,)
            ).fetchone()
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
            out     = llm_text(cluster_prompt)
            label   = parse_section(out, "LABEL") or f"cluster {lab}"
            summary = parse_section(out, "SUMMARY") or out
            cur = conn.execute(
                "INSERT INTO clusters (label, summary, size) VALUES (?,?,?)",
                (label, summary, len(members))
            )
            cid = cur.lastrowid

            # compute median dist for strong/weak threshold
            dists = sorted([m[2] for m in members])
            median_dist = dists[len(dists)//2]

            rows_to_insert = []
            for path, idx, dist in members:
                strength = "core" if dist <= median_dist else "peripheral"
                rows_to_insert.append((cid, path, strength, dist))

            conn.executemany(
                "INSERT INTO cluster_members (cluster_id, path, strength, dist_to_centroid) VALUES (?,?,?,?)",
                rows_to_insert
            )
            conn.commit()
        except Exception as e:
            print(f"  ! cluster error: {e}")

    conn.close()
    print("clusters done.")


def mine_contradictions():
    conn = connect()
    print("mining contradictions…")
    anchors = conn.execute(
        "SELECT path FROM items WHERE stage_done IS NOT NULL ORDER BY RANDOM() LIMIT ?",
        (CONTRADICTION_PAIRS,)
    ).fetchall()
    found = 0

    for i, (a_path,) in enumerate(anchors, 1):
        v = conn.execute(
            "SELECT embedding FROM vec_items WHERE path=?", (a_path,)
        ).fetchone()
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

        a_row  = conn.execute(
            "SELECT vlm_desc, ocr_text FROM items WHERE path=?", (a_path,)
        ).fetchone()
        a_text = ((a_row["vlm_desc"] or "") + "\n" + (a_row["ocr_text"] or ""))[:1500]
        b_text = ((nb["vlm_desc"]   or "") + "\n" + (nb["ocr_text"]    or ""))[:1500]

        prompt = (
            "Two items from a personal archive are semantically close. Determine "
            "whether they make COMPETING or CONTRADICTORY claims about the same "
            "topic (not merely different topics, and not minor variations). If yes, "
            "output:\nCONTRADICTION: yes\nCLAIM_A: <one sentence>\n"
            "CLAIM_B: <one sentence>\nNOTE: <2-3 sentences on the tension>\n"
            "If no real contradiction, output exactly:\nCONTRADICTION: no\n\n"
            f"ITEM A:\n{a_text}\n\nITEM B:\n{b_text}"
        )
        try:
            out = llm_text(prompt)
            if parse_section(out, "CONTRADICTION").lower().startswith("yes"):
                conn.execute("""
                    INSERT INTO contradictions (path_a, path_b, claim_a, claim_b, note)
                    VALUES (?,?,?,?,?)
                """, (a_path, nb["path"],
                      parse_section(out, "CLAIM_A"),
                      parse_section(out, "CLAIM_B"),
                      parse_section(out, "NOTE")))
                conn.commit()
                found += 1
        except Exception as e:
            print(f"  ! contradiction error: {e}")

        if i % 50 == 0:
            print(f"  …{i}/{len(anchors)} pairs checked, {found} found")

    conn.close()
    print(f"contradictions done. {found} found.")


# ══════════════════════════════════════════════════════════════════════════════
# PURE-MATH STAGES
# ══════════════════════════════════════════════════════════════════════════════

def mine_anchors():
    """Best representative item per cluster (closest to centroid)."""
    import numpy as np
    conn = connect()
    print("computing cluster anchors…")

    clusters = conn.execute("SELECT id FROM clusters").fetchall()
    count = 0
    for (cid,) in clusters:
        members = conn.execute(
            "SELECT path, dist_to_centroid FROM cluster_members WHERE cluster_id=? ORDER BY dist_to_centroid ASC LIMIT 1",
            (cid,)
        ).fetchone()
        if members:
            conn.execute(
                "INSERT OR REPLACE INTO anchors (cluster_id, path) VALUES (?,?)",
                (cid, members["path"])
            )
            count += 1

    conn.commit()
    conn.close()
    print(f"anchors done. {count} anchors stored.")


def mine_orphans():
    """Items not in any cluster, with isolation score."""
    import numpy as np
    conn = connect()
    print("finding orphans…")

    clustered = set(r["path"] for r in conn.execute("SELECT path FROM cluster_members").fetchall())
    all_items = set(r["path"] for r in conn.execute(
        "SELECT path FROM items WHERE stage_done IS NOT NULL"
    ).fetchall())
    orphan_paths = [p for p in all_items if p not in clustered]
    print(f"  {len(orphan_paths)} orphans found")

    emb_map = _all_embeddings(conn)

    inserted = 0
    for path in orphan_paths:
        if path not in emb_map:
            continue
        v = conn.execute("SELECT embedding FROM vec_items WHERE path=?", (path,)).fetchone()
        if not v:
            continue
        neighbors = conn.execute("""
            SELECT v.distance FROM vec_items v
            WHERE v.embedding MATCH ? AND v.path != ?
            ORDER BY v.distance LIMIT ?
        """, (v["embedding"], path, ISOLATION_K)).fetchall()
        iso = float(np.mean([n["distance"] for n in neighbors])) if neighbors else 99.0
        conn.execute(
            "INSERT OR REPLACE INTO orphans (path, isolation_score) VALUES (?,?)",
            (path, iso)
        )
        inserted += 1

    conn.commit()
    conn.close()
    print(f"orphans done. {inserted} stored.")


def mine_duplicates():
    """Near-identical embedding pairs (distance < threshold)."""
    import numpy as np
    conn = connect()
    print(f"finding semantic duplicates (threshold={DUPLICATE_THRESHOLD})…")

    paths = [r["path"] for r in conn.execute(
        "SELECT path FROM items WHERE stage_done IS NOT NULL ORDER BY RANDOM() LIMIT 10000"
    ).fetchall()]

    found = 0
    seen  = set()
    for path in paths:
        v = conn.execute("SELECT embedding FROM vec_items WHERE path=?", (path,)).fetchone()
        if not v:
            continue
        neighbors = conn.execute("""
            SELECT v.path, v.distance FROM vec_items v
            WHERE v.embedding MATCH ? AND v.path != ?
            ORDER BY v.distance LIMIT 5
        """, (v["embedding"], path)).fetchall()
        for nb in neighbors:
            if nb["distance"] < DUPLICATE_THRESHOLD:
                key = tuple(sorted([path, nb["path"]]))
                if key not in seen:
                    seen.add(key)
                    conn.execute(
                        "INSERT OR IGNORE INTO semantic_duplicates (path_a, path_b, distance) VALUES (?,?,?)",
                        (path, nb["path"], nb["distance"])
                    )
                    found += 1

    conn.commit()
    conn.close()
    print(f"duplicates done. {found} pairs stored.")


def mine_strong_weak():
    """Already computed during mine_clusters() — just report counts."""
    conn = connect()
    core = conn.execute("SELECT COUNT(*) FROM cluster_members WHERE strength='core'").fetchone()[0]
    peri = conn.execute("SELECT COUNT(*) FROM cluster_members WHERE strength='peripheral'").fetchone()[0]
    conn.close()
    print(f"strong/weak: {core} core, {peri} peripheral members (computed during clustering).")


def mine_overlap():
    """Items near the boundary of two clusters."""
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    conn = connect()
    print("finding cluster overlap items…")

    rows = conn.execute(
        "SELECT v.path, v.embedding FROM vec_items v "
        "JOIN items i ON i.path=v.path WHERE i.stage_done IS NOT NULL"
    ).fetchall()
    if not rows:
        conn.close()
        return

    paths = [r["path"] for r in rows]
    X     = np.array([_unpack(r["embedding"]) for r in rows], dtype="float32")

    km      = MiniBatchKMeans(n_clusters=CLUSTER_K, batch_size=2048, n_init=3, random_state=42)
    labels  = km.fit_predict(X)
    centers = km.cluster_centers_

    # map cluster label → db cluster id via anchor
    label_to_cid = {}
    for lab in range(CLUSTER_K):
        anchor = conn.execute(
            "SELECT cm.cluster_id FROM cluster_members cm "
            "JOIN anchors a ON a.path=cm.path AND a.cluster_id=cm.cluster_id "
            "LIMIT 1"
        ).fetchone()
        if anchor:
            label_to_cid[lab] = anchor["cluster_id"]

    inserted = 0
    for idx, (path, lab) in enumerate(zip(paths, labels)):
        dists = np.linalg.norm(centers - X[idx], axis=1)
        sorted_idx = np.argsort(dists)
        best, second = int(sorted_idx[0]), int(sorted_idx[1])
        ratio = dists[second] / (dists[best] + 1e-9)
        if ratio < 1.4:  # close to two clusters
            cid_a = label_to_cid.get(best)
            cid_b = label_to_cid.get(second)
            if cid_a and cid_b and cid_a != cid_b:
                conn.execute(
                    "INSERT OR REPLACE INTO cluster_overlap "
                    "(path, cluster_id_a, cluster_id_b, dist_a, dist_b) VALUES (?,?,?,?,?)",
                    (path, cid_a, cid_b, float(dists[best]), float(dists[second]))
                )
                inserted += 1

    conn.commit()
    conn.close()
    print(f"overlap done. {inserted} boundary items stored.")


def mine_chains():
    """Two-hop bridge chains: A→B→C where A and C are far apart."""
    conn = connect()
    print("computing bridge chains…")

    bridges = conn.execute(
        "SELECT path_a, path_b, domain_a, domain_b FROM bridges"
    ).fetchall()

    # build adjacency: path → list of (neighbor, domain)
    adj = defaultdict(list)
    for b in bridges:
        adj[b["path_a"]].append((b["path_b"], b["domain_b"]))
        adj[b["path_b"]].append((b["path_a"], b["domain_a"]))

    inserted = 0
    seen = set()
    for b in bridges:
        mid  = b["path_b"]
        a    = b["path_a"]
        d_ab = None
        for (c, domain_c) in adj[mid]:
            if c == a:
                continue
            key = tuple(sorted([a, c]) + [mid])
            if key in seen:
                continue
            seen.add(key)

            # get dist b→c
            nb_row = conn.execute(
                "SELECT distance FROM bridges WHERE (path_a=? AND path_b=?) OR (path_a=? AND path_b=?) LIMIT 1",
                (mid, c, c, mid)
            ).fetchone()
            dist_bc = nb_row["distance"] if nb_row else 0.0

            if d_ab is None:
                ab_row = conn.execute(
                    "SELECT distance FROM bridges WHERE (path_a=? AND path_b=?) OR (path_a=? AND path_b=?) LIMIT 1",
                    (a, mid, mid, a)
                ).fetchone()
                d_ab = ab_row["distance"] if ab_row else 0.0

            conn.execute(
                "INSERT INTO bridge_chains (path_a, path_mid, path_b, dist_ab, dist_bc, domain_a, domain_mid, domain_b) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (a, mid, c, d_ab, dist_bc, b["domain_a"], b["domain_b"], domain_c)
            )
            inserted += 1
            if inserted >= CHAIN_MAX:
                break
        if inserted >= CHAIN_MAX:
            break

    conn.commit()
    conn.close()
    print(f"chains done. {inserted} chains stored.")


def mine_hubs():
    """Items that appear most frequently as neighbors across the archive."""
    conn = connect()
    print(f"computing hub items (top {HUB_TOPK})…")

    paths = [r["path"] for r in conn.execute(
        "SELECT path FROM items WHERE stage_done IS NOT NULL ORDER BY RANDOM() LIMIT 20000"
    ).fetchall()]

    hub_count = defaultdict(int)
    for path in paths:
        v = conn.execute("SELECT embedding FROM vec_items WHERE path=?", (path,)).fetchone()
        if not v:
            continue
        neighbors = conn.execute("""
            SELECT v.path FROM vec_items v
            WHERE v.embedding MATCH ? AND v.path != ?
            ORDER BY v.distance LIMIT 10
        """, (v["embedding"], path)).fetchall()
        for nb in neighbors:
            hub_count[nb["path"]] += 1

    top = sorted(hub_count.items(), key=lambda x: -x[1])[:HUB_TOPK]
    conn.executemany(
        "INSERT OR REPLACE INTO hub_items (path, hub_score) VALUES (?,?)", top
    )
    conn.commit()
    conn.close()
    print(f"hubs done. {len(top)} hub items stored.")


def mine_isolation():
    """Per-item average distance to k nearest neighbors."""
    import numpy as np
    conn = connect()
    print(f"computing isolation scores…")

    paths = [r["path"] for r in conn.execute(
        "SELECT path FROM items WHERE stage_done IS NOT NULL"
    ).fetchall()]

    inserted = 0
    for path in paths:
        v = conn.execute("SELECT embedding FROM vec_items WHERE path=?", (path,)).fetchone()
        if not v:
            continue
        neighbors = conn.execute("""
            SELECT distance FROM vec_items
            WHERE embedding MATCH ? AND path != ?
            ORDER BY distance LIMIT ?
        """, (v["embedding"], path, ISOLATION_K)).fetchall()
        if not neighbors:
            continue
        iso = float(np.mean([n["distance"] for n in neighbors]))
        conn.execute(
            "INSERT OR REPLACE INTO isolation_scores (path, isolation_score) VALUES (?,?)",
            (path, iso)
        )
        inserted += 1
        if inserted % 5000 == 0:
            conn.commit()
            print(f"  …{inserted}/{len(paths)}")

    conn.commit()
    conn.close()
    print(f"isolation done. {inserted} scores stored.")


def mine_drift():
    """Embedding spread (variance) per folder/date group."""
    import numpy as np, os
    conn = connect()
    print("computing embedding drift per folder group…")

    rows = conn.execute(
        "SELECT v.path, v.embedding FROM vec_items v "
        "JOIN items i ON i.path=v.path WHERE i.stage_done IS NOT NULL"
    ).fetchall()

    groups = defaultdict(list)
    for r in rows:
        key = os.path.dirname(r["path"]).rstrip("/").split("/")[-1] or "root"
        groups[key].append((r["path"], np.array(_unpack(r["embedding"]), dtype="float32")))

    inserted = 0
    for key, items in groups.items():
        if len(items) < 3:
            continue
        vecs  = np.array([v for _, v in items])
        drift = float(np.mean(np.var(vecs, axis=0)))
        sample = json.dumps([p for p, _ in items[:5]])
        conn.execute(
            "INSERT OR REPLACE INTO embedding_drift (group_key, item_count, drift_score, sample_paths) VALUES (?,?,?,?)",
            (key, len(items), drift, sample)
        )
        inserted += 1

    conn.commit()
    conn.close()
    print(f"drift done. {inserted} folder groups stored.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # LLM stages
    ap.add_argument("--bridges",       action="store_true")
    ap.add_argument("--clusters",      action="store_true")
    ap.add_argument("--contradictions",action="store_true")
    # pure-math stages
    ap.add_argument("--anchors",       action="store_true")
    ap.add_argument("--orphans",       action="store_true")
    ap.add_argument("--duplicates",    action="store_true")
    ap.add_argument("--strong-weak",   action="store_true", dest="strong_weak")
    ap.add_argument("--overlap",       action="store_true")
    ap.add_argument("--chains",        action="store_true")
    ap.add_argument("--hubs",          action="store_true")
    ap.add_argument("--isolation",     action="store_true")
    ap.add_argument("--drift",         action="store_true")
    # convenience
    ap.add_argument("--all-llm",       action="store_true", dest="all_llm")
    ap.add_argument("--all-math",      action="store_true", dest="all_math")
    ap.add_argument("--all",           action="store_true")
    args = ap.parse_args()

    llm  = args.all or args.all_llm
    math = args.all or args.all_math

    # LLM stages — clusters must run before anchors/strong-weak/overlap
    if llm or args.bridges:        mine_bridges()
    if llm or args.clusters:       mine_clusters()
    if llm or args.contradictions: mine_contradictions()

    # pure-math stages
    if math or args.anchors:       mine_anchors()
    if math or args.strong_weak:   mine_strong_weak()
    if math or args.overlap:       mine_overlap()
    if math or args.orphans:       mine_orphans()
    if math or args.duplicates:    mine_duplicates()
    if math or args.isolation:     mine_isolation()
    if math or args.hubs:          mine_hubs()
    if math or args.drift:         mine_drift()
    if math or args.chains:        mine_chains()

    print("all done.")
