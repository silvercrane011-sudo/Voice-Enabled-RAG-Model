# Architecture Write-Up: Voice-to-Answer RAG Pipeline

## 1. Speech-to-Text

**Provider: Sarvam AI (free tier), with ElevenLabs as a drop-in alternate and
`faster-whisper` as an always-available offline fallback.**

Sarvam is the default because its free tier is budgeted per *request*, which
fits a query-driven voice pipeline where each turn is a few seconds of
speech — request volume matters more than minutes of audio. ElevenLabs'
free tier pools STT and TTS credits together and is minute-denominated,
which drains faster under multi-turn STT-only usage. Both are viable; the
provider is swappable via `stt.provider.transcribe(..., provider=...)`.

Every call is validated first (`stt/validation.py`: format, size, duration,
sample rate), then attempted with a 2-retry exponential backoff
(`stt/provider.py`). If the cloud call fails for any reason — network error,
timeout, quota exhaustion (HTTP 429) — the pipeline falls back to
`faster-whisper` running fully offline on CPU with int8 quantization
(`stt/local_fallback.py`), so the system degrades rather than crashing. This
was exercised live: with no API keys configured, the pipeline correctly
retried, fell back, and reported a clean error rather than raising an
unhandled exception.

## 2. Chunking

Naive fixed-size chunking loses structure that matters for retrieval
quality — a meeting transcript chunked blindly mixes speakers together; a
markdown doc chunked blindly severs headers from their content. The router
(`chunking/router.py`) auto-detects document type and dispatches:

| Doc type | Strategy | Why |
|---|---|---|
| Transcript (speaker-turn pattern detected) | Structure-aware | Preserves speaker + timestamp as first-class metadata |
| Structured (markdown headers detected) | Structure-aware | Preserves section_title, keeps paragraphs intact |
| Long document (>1500 tokens) | Hierarchical | Small child chunks for precise retrieval, parent chunks for full context expansion |
| Plain prose | Semantic (falls back to fixed-size) | Breaks on topic shifts, not arbitrary token counts |

Fixed-size (400 tok / 15% overlap) is the baseline every other strategy is
compared against. Semantic chunking embeds sentences with a free local
model (`all-MiniLM-L6-v2`) and breaks where cosine distance between
consecutive sentence embeddings exceeds a percentile threshold — a topic
shift. Every chunk, regardless of strategy, carries a `chunk_strategy` field
so retrieval-time hit logging (`HybridRetriever.strategy_hit_counts()`) can
show which strategy is actually producing useful results over time.

## 3. Indexing & Hybrid Retrieval

FAISS (`IndexFlatIP` over L2-normalized vectors, i.e. exact cosine search)
was chosen over an approximate index (IVF/HNSW) deliberately: it needs no
training step, gives exact results, and at RAG-demo corpus scale its search
time is negligible relative to the 200ms budget — approximate indexing is a
scaling lever to pull later, not a default. BM25 (`rank_bm25`) handles
sparse keyword matching, catching exact-term queries dense embeddings can
miss (acronyms, proper nouns, IDs).

The two rankings are fused with **Reciprocal Rank Fusion** rather than
weighted score blending, because cosine similarity and BM25 scores live on
incomparable scales — RRF sidesteps normalization entirely by fusing on
rank position (`1/(k+rank)`, k=60, the standard default from Cormack et
al. 2009). Chunk metadata (speaker, section, source, strategy) is stored
alongside vectors in both stores so retrieval can filter before scoring.

## 4. Latency (<200ms retrieval+generation)

**Explicit assumption, as permitted by the spec:** STT is measured as a
separate upstream stage. It requires a network round-trip (or CPU-bound
local inference) and cannot realistically be folded into a 200ms budget
alongside retrieval and generation. The 200ms-critical section is
**query received (post-STT) → retrieval → context assembly → generation**.

Three optimizations keep this path fast:
1. **Query embedding cache** (`orchestration/perf.py`) — exact-match LRU,
   so repeated queries in a session skip re-embedding entirely.
2. **Parallel dense+sparse search** — FAISS and BM25 searches run
   concurrently on a thread pool instead of sequentially, roughly halving
   the retrieval-stage wall time on cache misses.
3. **Capped `top_k`** — hard-capped at 20 via the `QueryRequest` schema, so
   no query can force unbounded retrieval/rerank work.

Measured on a 50-document benchmark corpus (Part 8): retrieval+rerank+
generation P100 was **~2.2ms**, three orders of magnitude under budget —
headroom exists to add a real LLM call (which will dominate latency) while
staying well inside 200ms for the local stages.

## 5. Orchestration

The harness (`orchestration/harness.py`) treats retrieval, guardrails,
reranking, and generation as **distinct, independently retryable steps**
(`orchestration/steps.py`) rather than one prompt — each is unit-testable
and independently loggable. Every stage's input/output is a Pydantic model
(`orchestration/schemas.py`), so malformed data is caught at the boundary
rather than propagating silently (this caught a real bug during
development: Pydantic v2 doesn't validate field defaults unless the field
is explicitly passed, which meant a derived `all_passed` field silently
stayed `False` until switched to a `model_validator`).

Retrieval and generation calls use the same `retry_with_backoff` decorator
pattern as the STT module (Part 1) for consistency. If retrieval returns
nothing, if generation raises after retries, or if guardrails block the
query, the harness always returns a schema-valid `FinalAnswer` — verified
across all four failure modes in testing — never an unhandled exception or
malformed output.

## 6. Guardrails

Four checks gate every answer, logged to `logs/guardrail_audit.jsonl` for
full auditability:

- **Off-topic detection**: reuses the RRF top-score from retrieval (no
  second pass) against a minimum threshold — cheap because the work is
  already done.
- **Unsafe-input filtering**: local rule-based regex patterns (no paid
  moderation API); documented as a coarse first line of defense, not a
  full trust & safety system.
- **Groundedness check** (post-generation): combines (a) citation-validity
  — every `[source:chunk_id]` in the answer must exist in the retrieved
  set, catching fabricated citations — with (b) lexical word-overlap ratio
  between answer and retrieved context, catching ungrounded free text that
  cites nothing. A pluggable NLI-entailment hook exists for a stronger
  future check.
- **Explicit refusal path**: any failed check returns "I don't have enough
  grounded information to answer this" rather than guessing — verified for
  unsafe input, off-topic queries, empty retrieval, and fabricated
  citations.

**Known limitation, stated plainly:** with a very small corpus (e.g. the
9-chunk smoke-test corpus in Part 9), RRF-rank-based off-topic detection is
weak — almost anything ranks #1 in a tiny index, so the score-threshold
signal is only meaningful at realistic corpus scale (demonstrated
correctly working on the 8-document, 59-query benchmark in Part 8, where
genuinely off-topic queries still retrieved *something* from the topical
corpus since the corpus itself was topically narrow — a proper off-topic
eval would need a corpus paired with genuinely out-of-domain probe queries).

## 7. Generation

The default generator (`orchestration/steps.local_extractive_generate`) is
a **free, local, zero-cost extractive synthesizer** — no LLM API call is
required for the pipeline to function end-to-end, keeping it free-tier-safe
by default. `make_llm_generate_fn()` provides an adapter so any real
free-tier LLM call can be dropped in via the same `generate_fn(query,
chunks) -> str` signature without touching the orchestrator.
