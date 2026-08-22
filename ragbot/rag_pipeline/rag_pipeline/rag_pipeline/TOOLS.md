# Tool & Library Audit — Free-Tier / Open-Source Confirmation

Every tool below is free, open-source, or self-hosted. No paid tier is used
anywhere in this pipeline.

| Component | Tool/Library | License / Cost model | Free-tier limit (verify against current provider docs before production use) |
|---|---|---|---|
| STT (primary) | Sarvam AI API | Free developer tier | Request-count cap per API key per day; exact number set at signup — check https://docs.sarvam.ai |
| STT (alternate) | ElevenLabs API (Scribe) | Free tier, shared credit pool with TTS | Limited monthly credits shared across all audio features — check https://elevenlabs.io/pricing |
| STT (fallback) | `faster-whisper` | MIT license, fully local | No quota — bounded only by local CPU time |
| Tokenization | `tiktoken` (optional) | MIT license | N/A — local, free. Falls back to whitespace tokenization if absent |
| Semantic embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Apache 2.0, local model (~22MB) | N/A — runs on CPU, no API key. Falls back to deterministic hashing embeddings if absent |
| Dense vector index | FAISS (`faiss-cpu`) | MIT license, local | N/A — no hosted tier used |
| Sparse keyword index | `rank_bm25` | Apache 2.0, pure Python | N/A — local |
| Orchestration schemas | `pydantic` v2 | MIT license | N/A |
| HTTP client | `requests` | Apache 2.0 | N/A |
| Generation (default) | Local extractive synthesizer (custom code, no external call) | N/A — no license needed, it's ours | No quota — zero external calls |
| Generation (optional upgrade) | Any free-tier LLM API via `make_llm_generate_fn()` adapter | Depends on provider chosen | Not used by default; if wired in, document that provider's specific free-tier limit here |

## Explicit non-use of paid services

- No paid STT tier (Sarvam/ElevenLabs paid plans) — free tier only, with
  local fallback when exhausted.
- No hosted/paid vector database (e.g. Pinecone, Weaviate Cloud, Qdrant
  Cloud paid tiers) — FAISS runs entirely in-process, no network calls.
- No paid moderation API (e.g. OpenAI Moderation, Perspective API paid
  tiers) — local rule-based filter only.
- No paid LLM API for the core generation call — default generator makes
  zero external calls; the adapter for a real LLM is provided but
  unconfigured, and the burden is on whoever wires in a provider to pick
  one with a genuine free tier.

## Known tradeoffs from staying free/local-only

- FAISS `IndexFlatIP` is exact search (no ANN) — fine at demo/small-corpus
  scale, would need swapping for IVF/HNSW (still free/local) at large scale.
- BM25 index is rebuilt on every `add()` call (`rank_bm25` has no
  incremental API) — fine for batch ingestion, would need a proper
  inverted-index engine (e.g. Whoosh, still free) for high-throughput
  streaming ingestion.
- The local rule-based safety filter is a coarse first line of defense,
  not a comprehensive trust & safety system — a production deployment
  should layer a proper local classifier on top.
- The default extractive generator produces stitched-together source text
  rather than fluent synthesized prose — swapping in a real LLM (even a
  free-tier one) via `make_llm_generate_fn()` will substantially improve
  answer quality at the cost of that provider's own latency and quota.
