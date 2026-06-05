# Gbrain  
a memory layer built on top of my digital life, transforms digital artifacts into a structured semantic knowledge graph

**What is this?**

A multi-stage pipeline that transforms 100K+ artifacts of a personal digital life into a semantic memory graph — and a personalization substrate for LLM interactions.
Most LLMs know nothing about you. gbrain changes that. It builds a structured personal knowledge graph from your actual digital life —  photos, videos, documents, tweets, research papers, and screenshots etc — and uses it to ground LLM interactions in real personal context. The result is RAG that isn't generic: it's your narrative arcs, your intellectual fingerprint, retrieved and injected at query time.
The deeper insight is that personal digital artifacts contain connections you've never consciously made. A research paper you saved in 2019 and a photo from 2023 could share a latent theme, a cluster of AI tweets could bridge into a cluster of design artifacts. gbrain makes context tangible — as a browsable knowledge graph of recurring themes, cross-domain bridges, and threads.
Using the semantic graph as a dynamic personalization substrate — a structured, queryable model of a person gets injected into LLM context at inference time. The conjecture here is, the gap between a generic LLM response and a genuinely useful one isn't only dependant on model size or reasoning capability — it also heavily relies on personal context. A model that knows your intellectual history, your recurring narratives, your cross-domain connections, and how your thinking has evolved can generate solutions that are calibrated to you specifically: not just accurate, but relevant, resonant, and actionable in your particular situation. The long-term vision is LLMs that don't just answer questions — they find your answer, drawn from a live model of who you are.

**Pipeline**

<img width="1440" height="2658" alt="image" src="https://github.com/user-attachments/assets/ab5a871c-7d96-4df3-ba21-1fdd59d4698b" />

**Database**
gbrain.db is a single SQLite file — ~290MB — that encodes a full semantic graph of a personal digital life. Every item has a caption or extracted text, a 384-dim embedding, a cluster assignment, and a position in the broader graph. The schema is designed to support both structured browsing (clusters, threads, entities) and vector retrieval (knn search over embeddings), making it usable as both a knowledge base and a RAG backend.
The graph layer sits on top: bridges connect semantically related items across different thematic clusters; threads trace narrative arcs through a cluster's items; tensions surface opposing items within a cluster; obsessions and hubs identify the items the archive keeps returning to. Together these structures capture not just what is in the archive, but how it is organized, how it has evolved, and where the unexpected connections live.

<img width="421" height="277" alt="Screenshot 2026-06-06 045800" src="https://github.com/user-attachments/assets/2ab8d941-1837-41bb-a24a-8f0edf2ccfde" />
<img width="419" height="278" alt="Screenshot 2026-06-06 045818" src="https://github.com/user-attachments/assets/35ad4852-ddcb-462f-9fbb-98ade1c67cae" />
<img width="410" height="284" alt="Screenshot 2026-06-06 045835" src="https://github.com/user-attachments/assets/ed7b4e13-17fa-4346-a3c0-ab3d10391f1e" />

**Stack**

<img width="377" height="434" alt="Screenshot 2026-06-06 050548" src="https://github.com/user-attachments/assets/39d5874d-503d-482e-a9c3-5ce5951c6438" />

**Known issues and patches**

sqlite-vec 0.1.9 — all LIMIT ? replaced with hardcoded values; knn queries use AND k=N syntax
vLLM — broken on H100 due to flashinfer ABI mismatch; use transformers only
bitsandbytes — requires ≥ 0.49.2 for stable 4-bit quantization
Qwen2.5-VL — requires Qwen2_5_VLForConditionalGeneration, not AutoModelForCausalLM
Domain filter — Qwen captions have no DOMAIN: tag; filter bypassed in pipeline
Bridge dedup — patch applied: NOT IN (SELECT path_a FROM bridges) to prevent duplicate mining

