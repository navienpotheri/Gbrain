"""
mine_threads.py
---------------
Narrative threading for gbrain — finds chains of semantically related items
that form coherent story arcs, without needing date metadata.

USAGE (standalone):
    python3 /workspace/mine_threads.py

OR add to connect_transformers.py:
    1. Paste the mine_threads() function into connect_transformers.py
    2. Add --threads flag to argparse
    3. Call mine_threads() in the main block

WHAT IT DOES:
    - Takes each cluster from the clusters table
    - Finds the best "chain" through items in that cluster using
      greedy nearest-neighbor traversal through the embedding space
    - Uses the 32B model to name the thread and write a one-line arc summary
    - Writes results to a new `threads` table

DB SCHEMA ADDED:
    threads (
        id          INTEGER PRIMARY KEY,
        cluster_id  TEXT,
        title       TEXT,       -- LLM-generated thread name
        arc         TEXT,       -- LLM-generated one-line narrative arc
        length      INTEGER,    -- number of items in chain
        coherence   REAL,       -- mean cosine similarity along chain
        path_chain  TEXT        -- JSON list of file paths in order
    )

    thread_items (
        thread_id   INTEGER,
        position    INTEGER,
        path        TEXT,
        description TEXT        -- vlm_desc or ocr_text snippet
    )
"""

import sqlite3
import sqlite_vec
import json
import struct
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

DB_PATH     = '/workspace/gbrain.db'
MODEL_PATH  = '/workspace/models/Qwen2.5-VL-32B-Instruct'
MIN_THREAD_LENGTH = 4    # minimum items to form a thread
MAX_THREAD_LENGTH = 20   # cap chain length
THREADS_PER_CLUSTER = 3  # how many threads to extract per cluster
TOP_CLUSTERS = 30        # only process top N largest clusters


# ── DB setup ─────────────────────────────────────────────────────────────────

def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS threads (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id  TEXT,
            title       TEXT,
            arc         TEXT,
            length      INTEGER,
            coherence   REAL,
            path_chain  TEXT
        );
        CREATE TABLE IF NOT EXISTS thread_items (
            thread_id   INTEGER,
            position    INTEGER,
            path        TEXT,
            description TEXT
        );
    """)
    conn.commit()


# ── Embedding helpers ─────────────────────────────────────────────────────────

def unpack_vec(blob):
    n = len(blob) // 4
    return np.array(struct.unpack(f'{n}f', blob), dtype=np.float32)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Chain extraction ──────────────────────────────────────────────────────────

def extract_chain(items_with_vecs, start_idx, max_len):
    """
    Greedy nearest-neighbor chain starting from start_idx.
    items_with_vecs: list of (path, vec, desc)
    Returns list of indices in chain order.
    """
    remaining = set(range(len(items_with_vecs)))
    chain = [start_idx]
    remaining.remove(start_idx)

    while len(chain) < max_len and remaining:
        current_vec = items_with_vecs[chain[-1]][1]
        best_idx = max(remaining, key=lambda i: cosine(current_vec, items_with_vecs[i][1]))
        best_sim = cosine(current_vec, items_with_vecs[best_idx][1])
        if best_sim < 0.3:  # coherence threshold — stop if too dissimilar
            break
        chain.append(best_idx)
        remaining.remove(best_idx)

    return chain


def chain_coherence(items_with_vecs, chain):
    """Mean cosine similarity between consecutive items in chain."""
    if len(chain) < 2:
        return 0.0
    sims = [
        cosine(items_with_vecs[chain[i]][1], items_with_vecs[chain[i+1]][1])
        for i in range(len(chain) - 1)
    ]
    return float(np.mean(sims))


# ── LLM helpers ──────────────────────────────────────────────────────────────

def load_model():
    print("Loading 32B model for thread naming...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb,
            device_map='auto',
            trust_remote_code=True
        )
    except Exception as e:
        print(f"  ! model load warning: {e}")
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb,
            device_map='auto',
        )
    mdl.eval()
    return tok, mdl


def llm_name_thread(tok, mdl, descriptions, max_new_tokens=80):
    """Ask LLM to name a thread and write a one-line arc from item descriptions."""
    sample = descriptions[:6]  # use first 6 items for context
    joined = '\n'.join(f'- {d[:120]}' for d in sample if d)

    prompt = f"""You are analyzing a personal photo/document archive.
Here are descriptions of items that form a connected narrative thread:

{joined}

Give:
1. A short evocative title for this thread (5 words max)
2. A one-sentence narrative arc describing the story or theme connecting these items

Respond in this exact format:
TITLE: <title>
ARC: <one sentence>"""

    inputs = tok(prompt, return_tensors='pt').to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.eos_token_id
        )
    text = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    title, arc = 'Unnamed Thread', ''
    for line in text.splitlines():
        if line.startswith('TITLE:'):
            title = line[6:].strip()
        elif line.startswith('ARC:'):
            arc = line[4:].strip()
    return title, arc


# ── Main mining function ──────────────────────────────────────────────────────

def mine_threads(use_llm=True):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    init_db(conn)

    # Check if already done
    existing = conn.execute('SELECT COUNT(*) FROM threads').fetchone()[0]
    if existing > 0:
        print(f'threads table already has {existing} rows, skipping.')
        conn.close()
        return

    # Load model if using LLM
    tok, mdl = (None, None)
    if use_llm:
        tok, mdl = load_model()

    # Get top clusters by size
    try:
        clusters_raw = conn.execute("""
            SELECT cluster_id, paths FROM clusters
            ORDER BY json_array_length(paths) DESC
            LIMIT ?
        """, (TOP_CLUSTERS,)).fetchall()
    except Exception:
        # fallback if paths column is different format
        clusters_raw = conn.execute("""
            SELECT cluster_id, paths FROM clusters
            LIMIT ?
        """, (TOP_CLUSTERS,)).fetchall()

    if not clusters_raw:
        print('No clusters found. Run --all-llm first.')
        conn.close()
        return

    print(f'Processing {len(clusters_raw)} clusters for narrative threads...')

    thread_count = 0

    for cluster_row in clusters_raw:
        cluster_id = cluster_row['cluster_id']

        # Parse paths
        try:
            paths = json.loads(cluster_row['paths'])
        except Exception:
            paths = cluster_row['paths'].split(',') if cluster_row['paths'] else []

        if len(paths) < MIN_THREAD_LENGTH:
            continue

        # Fetch embeddings + descriptions for all items in cluster
        items_with_vecs = []
        for path in paths:
            row = conn.execute("""
                SELECT v.embedding, i.vlm_desc, i.ocr_text
                FROM vec_items v JOIN items i ON i.path = v.path
                WHERE v.path = ?
            """, (path,)).fetchone()
            if row and row['embedding']:
                vec = unpack_vec(row['embedding'])
                desc = (row['vlm_desc'] or '') + ' ' + (row['ocr_text'] or '')
                desc = desc.strip()
                items_with_vecs.append((path, vec, desc))

        if len(items_with_vecs) < MIN_THREAD_LENGTH:
            continue

        # Extract THREADS_PER_CLUSTER chains using different start points
        used_starts = set()
        threads_found = 0

        # Try starting from items spread across the cluster
        n = len(items_with_vecs)
        start_candidates = [0, n//4, n//2, 3*n//4, n-1]

        for start_idx in start_candidates:
            if threads_found >= THREADS_PER_CLUSTER:
                break
            if start_idx in used_starts:
                continue

            chain = extract_chain(items_with_vecs, start_idx, MAX_THREAD_LENGTH)

            if len(chain) < MIN_THREAD_LENGTH:
                continue

            coherence = chain_coherence(items_with_vecs, chain)
            if coherence < 0.35:  # skip low-coherence chains
                continue

            chain_paths = [items_with_vecs[i][0] for i in chain]
            chain_descs = [items_with_vecs[i][2] for i in chain]

            # Name the thread
            if use_llm and tok and mdl:
                try:
                    title, arc = llm_name_thread(tok, mdl, chain_descs)
                except Exception as e:
                    print(f'  LLM error: {e}')
                    title = f'Thread in {cluster_id}'
                    arc = ''
            else:
                title = f'Thread in {cluster_id}'
                arc = chain_descs[0][:100] if chain_descs else ''

            # Write thread
            conn.execute("""
                INSERT INTO threads (cluster_id, title, arc, length, coherence, path_chain)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (cluster_id, title, arc, len(chain), coherence, json.dumps(chain_paths)))
            thread_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]

            # Write thread items
            for pos, (path, desc) in enumerate(zip(chain_paths, chain_descs)):
                conn.execute("""
                    INSERT INTO thread_items (thread_id, position, path, description)
                    VALUES (?, ?, ?, ?)
                """, (thread_id, pos, path, desc[:300]))

            conn.commit()
            used_starts.add(start_idx)
            threads_found += 1
            thread_count += 1
            print(f'  [{cluster_id}] Thread {threads_found}: "{title}" ({len(chain)} items, coherence={coherence:.2f})')

    print(f'\nthreads done. {thread_count} threads found across {len(clusters_raw)} clusters.')
    conn.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM naming, use descriptions directly')
    args = parser.parse_args()
    mine_threads(use_llm=not args.no_llm)
