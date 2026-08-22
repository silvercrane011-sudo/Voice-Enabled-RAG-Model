# Voice-to-Answer RAG Pipeline (Free-Tier Only)

A speech-to-text → retrieval-augmented-generation pipeline built entirely on
free-tier, open-source, and self-hosted components. See `ARCHITECTURE.md`
for design rationale and `TOOLS.md` for the full tool/license/free-tier-limit
audit.

## Project structure

```
rag_pipeline/
├── config.py                  # central config, all free-tier params (STT + LLM)
├── pipeline.py                 # VoiceRAGPipeline — main entrypoint
├── data/                        # Part 2/3: real dataset loader + ingestion
│   ├── load_msmarco_xi.py         pulls ai4bharat/MSMARCO-XI via load_dataset
│   └── ingest_msmarco_xi.py       ingests the resulting JSONL into the pipeline
├── stt/                        # Part 1/4: speech-to-text
│   ├── validation.py             audio format/duration/size validation
│   ├── provider.py               Sarvam/ElevenLabs + retry/backoff
│   ├── local_fallback.py         faster-whisper offline fallback
│   └── test_live_stt.py          run locally with a real key to verify live STT
├── chunking/                   # Parts 2-3: multi-strategy chunking
│   ├── fixed_size.py             baseline, 400 tok / 15% overlap
│   ├── semantic.py                embedding-breakpoint chunking
│   ├── structure_aware.py         transcript/markdown-aware chunking
│   ├── hierarchical.py            parent/child chunking
│   └── router.py                  auto-detects doc type, dispatches strategy
├── indexing/                   # Part 4: hybrid retrieval
│   ├── dense_store.py             FAISS vector store
│   ├── sparse_store.py            BM25 keyword store
│   └── hybrid_retriever.py        RRF fusion + strategy-hit logging
├── orchestration/               # Parts 5 & 7: harness + latency + LLM
│   ├── schemas.py                 Pydantic I/O models, every stage
│   ├── steps.py                   retrieval/rerank/generation as tool calls
│   ├── harness.py                 RAGOrchestrator, error recovery
│   ├── perf.py                    query cache, parallel dense+sparse search
│   ├── llm_provider.py            real free-tier LLM adapters (Groq/OpenRouter/Anthropic)
│   └── test_live_llm.py           run locally with a real key to verify live generation
├── guardrails/                  # Part 6: refusal logic
│   ├── off_topic.py                RRF rank + lexical-overlap (stemmed) on-topic check
│   ├── safety_filter.py            verb×noun proximity unsafe-input filter
│   ├── groundedness.py             citation + overlap check
│   └── pipeline.py                 combined pre/post gate + audit log
├── benchmark/                   # Part 8: latency benchmarking
│   ├── corpus.py                   59-query test set
│   ├── percentiles.py              P50/P70/P100 calculation
│   ├── run_benchmark.py            local-path runner (retrieval→generation)
│   └── full_path_benchmark.py      adds STT + real-LLM stages, honestly flags what wasn't measured
├── logs/                        # JSONL structured logs + benchmark CSVs
├── ARCHITECTURE.md              # design write-up + §8 remediation log (deliverable #2)
└── TOOLS.md                     # tool/license/free-tier audit (deliverable #4)
```

## Quick start

```bash
pip install -r requirements.txt --break-system-packages

# Optional but recommended for full quality (falls back gracefully if absent):
pip install sentence-transformers faster-whisper tiktoken datasets --break-system-packages

export SARVAM_API_KEY=...        # optional — falls back to faster-whisper if unset/exhausted
export ELEVENLABS_API_KEY=...    # optional alternate STT provider
export LLM_API_KEY=...           # optional — enables real LLM generation instead of extractive

python3 -c "
from pipeline import VoiceRAGPipeline
pipe = VoiceRAGPipeline()
pipe.ingest_text('Your document text here.', source_id='doc1')
answer = pipe.ask_text('Your question here?')
print(answer.status, answer.display_text)
"
```

## Using the real MSMARCO-XI dataset

```bash
pip install datasets --break-system-packages
python data/load_msmarco_xi.py --lang hi --limit 300
python data/ingest_msmarco_xi.py --in data/msmarco_xi_hi.jsonl
```

## Verifying the parts that need real network access

Three things could not be executed inside the build sandbox (no route to
huggingface.co / api.sarvam.ai / api.elevenlabs.io / api.groq.com) and are
documented as an explicit open gap in `ARCHITECTURE.md` §8 rather than
silently assumed. Run these once locally to close it:

```bash
python stt/test_live_stt.py path/to/clip.wav          # real STT call
python orchestration/test_live_llm.py "your question"  # real LLM call
python data/load_msmarco_xi.py --lang hi --limit 300   # real dataset pull
```

## Running the benchmark (deliverable #3)

```bash
# local path only (retrieval → guardrails → generation, no STT/LLM):
PYTHONPATH=. python3 benchmark/run_benchmark.py

# full path — adds STT + real LLM stages when available, explicitly flags
# in its own output whenever a stage couldn't be measured rather than
# silently omitting it:
PYTHONPATH=. python3 benchmark/full_path_benchmark.py --with-stt --with-llm --audio clip.wav
```

Last local-path run (59 queries, post-fixes — see `ARCHITECTURE.md` §8):
**LOCAL_PATH_TOTAL P100 ≈ 13.4ms**, 56/59 answered, 3/59 correctly refused
(0 false positives/negatives). This covers retrieval through generation
only — **it does not include STT or a real LLM call**, both of which are
very likely to dominate real-world latency; see `ARCHITECTURE.md` §8 for
the honest assessment of what that means for the 200ms target.

## Notes on running without optional dependencies

Every optional free/local model (`sentence-transformers`, `faster-whisper`,
`tiktoken`) has a deterministic, dependency-free fallback so the pipeline
never hard-crashes if one is missing — it just degrades in quality
(hashing-based embeddings instead of real semantic embeddings, etc.), with a
warning logged. This was verified throughout development in a sandboxed
environment without those packages installed.
