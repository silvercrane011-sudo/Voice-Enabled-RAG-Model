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

Measured on a 50-document benchmark corpus (Part 8, re-verified with fixes
in §8): retrieval+rerank+generation P100 was **~14ms** across 59 real
queries (0.9ms mean), comfortably under budget for this local segment.
**This number does not include STT or real LLM generation — see §8 for
why, and for the honest assessment of what that means for the full 200ms
claim.**

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

- **Off-topic detection**: combines the RRF top-score from retrieval with an
  independent lexical word-overlap check between the query and the top few
  ranked chunks — added after testing showed RRF rank alone isn't
  sufficient (see remediation log, §8).
- **Unsafe-input filtering**: local rule-based verb×noun proximity matching
  across weapons/explosives, violence/self-harm, illegal-hacking, and CSAE
  categories (no paid moderation API); documented as a coarse first line of
  defense, not a full trust & safety system.
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

**Known limitation, stated plainly:** the unsafe-input filter is a rule-based
screen, not a trained classifier — it generalizes across common phrasings
within each category (verified against paraphrase tests, §8) but is not
exhaustive, and the CSAE category deliberately uses broad terms rather than
an exhaustive phrase list, since an exhaustive list would itself be a misuse
risk. A production deployment should layer a small fine-tuned
text-classification model behind the same `GuardrailVerdict` interface.

## 7. Generation

Two generators share one interface (`generate_fn(query, chunks) -> str`):

- **`local_extractive_generate`** — free, local, zero-cost: stitches the
  top retrieved chunks together with citation markers. This is the
  default, so the pipeline runs end-to-end with zero external
  dependencies or API keys.
- **Real LLM generation** (`orchestration/llm_provider.py` +
  `pipeline.llm_generate_fn()`) — opt-in adapter for any OpenAI-compatible
  free tier (Groq by default; OpenRouter/Together work by changing
  `LLM_BASE_URL`/`LLM_MODEL`) or direct Anthropic. Falls back to the
  extractive generator at call time if the LLM errors, so a transient
  quota/network failure degrades gracefully instead of crashing a demo.

Wiring and prompt assembly were verified end-to-end with a mocked LLM call
(§8) — including one genuinely useful finding: a deliberately fabricated
mock answer was correctly caught and refused by the existing groundedness
guardrail, which is real evidence the hallucination check works, not just
a design claim.

## 8. Remediation Log — What Was Actually Verified, and What Wasn't

This section documents a second pass over the pipeline: every claim below
was checked by *running* the code in this session, not by re-reading it.
Several real bugs were found and fixed this way; they're listed here
rather than quietly folded into the sections above, because how they were
found matters as much as the fix.

**Verified working, this session:**
- Full text query → retrieval → guardrails → generation → answer, including
  the `FinalAnswer.display_text` safe accessor added to prevent a `None`
  crash on refusal (previously `answer_text` was `None` on REFUSED/ERROR
  and any naive caller printing it directly would crash).
- Unsafe-input filter rewritten from 4 rigid full-phrase regexes to a
  verb×noun proximity model; a 14-case regression+paraphrase test suite
  caught two real bugs before being called done — a missing "weapon" noun,
  and a word-token filler regex that silently broke on the apostrophe in
  "someone else's" (fixed by switching to a character-window match).
- Off-topic guardrail rewritten to require both an RRF rank floor *and* an
  absolute lexical-overlap floor, because RRF fuses on rank, not absolute
  relevance — any corpus returns *some* rank-1 result for any query, so
  rank alone can't distinguish "relevant" from "least-irrelevant of what's
  available." Iterating against the real 59-query benchmark (Part 8) caught
  three further real bugs in sequence: (1) naive overlap was trivially
  satisfied by shared stopwords like "is"/"the"/"of"; (2) RRF ties caused
  `max()` to arbitrarily pick the wrong top chunk, refusing legitimate
  queries; (3) exact-token overlap missed simple morphological variants
  ("transcript" vs "transcripts"). All three fixed and reverified — final
  state on the 59-query benchmark: 56 correctly answered, exactly 3
  correctly refused (2 genuinely off-topic + 1 gibberish), 0 false
  positives, 0 false negatives.
- LLM generation wiring, using a mocked `call_llm` function (prompt
  assembly, harness integration, extractive-fallback-on-error).
- STT retry/backoff and cloud→local-fallback failover logic, using a
  synthetic test tone (no real speech).

**NOT verified live, and why — this is the honest gap:**
- No real Sarvam/ElevenLabs STT call. No real free-tier LLM call (Groq/
  OpenRouter/Anthropic). No real download of the ai4bharat/MSMARCO-XI
  dataset. All three require network access to hosts (huggingface.co,
  api.sarvam.ai, api.elevenlabs.io, api.groq.com, openrouter.ai) outside
  this build environment's allowlist. Even the "offline" faster-whisper
  fallback isn't truly network-independent until its model weights are
  pre-cached, since first use pulls from huggingface.co too — confirmed
  by an actual failed attempt, not assumed.
- Practical consequence for the **200ms latency claim**: the P100 numbers
  in §4 and the 59-query benchmark (`benchmark/full_path_benchmark.py`)
  measure the **local path only** — retrieval through extractive
  generation. They explicitly exclude STT and real LLM generation, which
  are very likely to dominate real-world latency (typical cloud STT
  round-trips run in the low hundreds of ms; typical small-model LLM
  generation runs from several hundred ms to a few seconds). The honest
  claim is: *the local retrieval/guardrail path is not the bottleneck and
  has generous headroom under 200ms; whether the full voice-in-to-answer-
  out path meets 200ms has not been demonstrated and, based on typical
  STT/LLM latencies, is unlikely without further optimization (streaming
  STT, a smaller/faster model, or a higher latency budget for those
  stages specifically).*
- Ready-to-run scripts exist for closing this gap outside the sandbox:
  `data/load_msmarco_xi.py`, `stt/test_live_stt.py`,
  `orchestration/test_live_llm.py`, and
  `benchmark/full_path_benchmark.py --with-stt --with-llm --audio <clip>`.
  Running these once with real credentials would convert every item in
  this "not verified" list into a verified one.
