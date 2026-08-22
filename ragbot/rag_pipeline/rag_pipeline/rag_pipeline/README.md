# Voice-to-Answer RAG Pipeline (Free-Tier Only)

A speech-to-text -> retrieval-augmented-generation pipeline built entirely on
free-tier, open-source, and self-hosted components. See `ARCHITECTURE.md`
for design rationale and `TOOLS.md` for the full tool/license/free-tier-limit
audit.

## Project structure

```
rag_pipeline/
├── config.py                  # central config, all free-tier params
├── pipeline.py                 # VoiceRAGPipeline - main entrypoint
├── demo.py                     # Interactive CLI demo (text + voice input)
├── stt/                        # Part 1: speech-to-text
│   ├── validation.py            audio format/duration/size validation
│   ├── provider.py              Sarvam/ElevenLabs + retry/backoff
│   └── local_fallback.py        faster-whisper offline fallback
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
├── orchestration/               # Parts 5 & 7: harness + latency
│   ├── schemas.py                 Pydantic I/O models, every stage
│   ├── steps.py                   retrieval/rerank/generation as tool calls
│   ├── harness.py                 RAGOrchestrator, error recovery
│   └── perf.py                    query cache, parallel dense+sparse search
├── guardrails/                  # Part 6: refusal logic
│   ├── off_topic.py                similarity-to-corpus threshold
│   ├── safety_filter.py            local rule-based input filter
│   ├── groundedness.py             citation + overlap check
│   └── pipeline.py                 combined pre/post gate + audit log
├── benchmark/                   # Part 8: latency benchmarking
│   ├── corpus.py                   59-query test set
│   ├── percentiles.py              P50/P70/P100 calculation
│   └── run_benchmark.py            runner, prints table + writes CSV
├── logs/                        # JSONL structured logs + benchmark CSV
├── ARCHITECTURE.md              # design write-up
└── TOOLS.md                     # tool/license/free-tier audit
```

## Quick start

**Linux / macOS:**

```bash
pip install -r requirements.txt --break-system-packages

# Optional but recommended for full quality (falls back gracefully if absent):
pip install sentence-transformers faster-whisper tiktoken --break-system-packages

export SARVAM_API_KEY=...        # optional - falls back to faster-whisper if unset/exhausted
export ELEVENLABS_API_KEY=...    # optional alternate provider

PYTHONPATH=. python3 -c "
from pipeline import VoiceRAGPipeline
pipe = VoiceRAGPipeline()
pipe.ingest_text('Your document text here.', source_id='doc1')
answer = pipe.ask_text('Your question here?')
print(answer.status, answer.answer_text)
"
```

**Windows (cmd.exe):**

```bat
pip install -r requirements.txt

REM Optional but recommended for full quality:
pip install sentence-transformers faster-whisper tiktoken

set PYTHONPATH=%~dp0
python -c "from pipeline import VoiceRAGPipeline; pipe = VoiceRAGPipeline(); pipe.ingest_text('Your document text here.', source_id='doc1'); answer = pipe.ask_text('Your question here?'); print(answer.status, answer.answer_text)"
```

## Interactive CLI Demo

**Linux / macOS:**

```bash
# Start with the built-in sample corpus (9 topics, 14 chunks)
PYTHONPATH=. python demo.py

# Start with your own text files ingested
PYTHONPATH=. python demo.py --corpus /path/to/docs/
PYTHONPATH=. python demo.py --file document.txt

# Start in voice mode (microphone recording)
PYTHONPATH=. python demo.py --voice --record-sec 5
```

**Windows (cmd.exe):**

```bat
REM Start with the built-in sample corpus (9 topics, 14 chunks)
run_demo.bat

REM Start with your own text files ingested
run_demo.bat --corpus C:\path\to\docs
run_demo.bat --file document.txt

REM Start in voice mode (microphone recording)
run_demo.bat --voice --record-sec 5
```

**Windows (PowerShell):**

```powershell
$env:PYTHONPATH = "."
python demo.py

python demo.py --corpus C:\path\to\docs
python demo.py --voice --record-sec 5
```

### Demo commands

| Command | Description |
|---------|-------------|
| `:text` | Switch to text input mode (default) |
| `:voice` | Switch to voice mode (records from microphone via sounddevice) |
| `:file <path>` | Use a specific audio file for voice queries |
| `:ingest <path>` | Ingest a .txt or .md file into the corpus |
| `:ingest-dir <dir>` | Ingest all .txt/.md files from a directory |
| `:stats` | Show corpus statistics (chunks, cache, strategies) |
| `:bench` | Run a quick 5-query benchmark with latency display |
| `:help` | Show available commands |
| `:quit` | Exit the demo |

The demo displays answers with per-stage latency breakdown (retrieval,
guardrails, rerank, generation, post-guard) and supports guardrail
refusal display for off-topic or unsafe queries.

## Running the benchmark

**Linux / macOS:**

```bash
# Standard benchmark (local extractive generation)
PYTHONPATH=. python benchmark/run_benchmark.py

# Full-path benchmark with STT and/or real LLM
PYTHONPATH=. python benchmark/full_path_benchmark.py
PYTHONPATH=. python benchmark/full_path_benchmark.py --with-stt --audio sample.wav
PYTHONPATH=. python benchmark/full_path_benchmark.py --with-llm  # requires LLM_API_KEY
```

**Windows (cmd.exe):**

```bat
REM Standard benchmark
run_benchmark.bat

REM Full-path benchmark
run_benchmark.bat full
run_benchmark.bat full --with-stt --audio sample.wav
run_benchmark.bat full --with-llm
```

### Benchmark results (59 queries)

```
Stage                   P50 (ms)    P70 (ms)   P100 (ms)
--------------------------------------------------------
retrieval                 19.927      21.989      38.874
guardrails_pre             1.566       1.661      22.584
rerank                     0.043       0.051       0.184
generation                 0.019       0.022       0.069
guardrails_post            0.816       0.881       1.588
--------------------------------------------------------
LOCAL_PATH_TOTAL          25.979      28.093      58.732
```

All well under the 200ms retrieval+generation budget. STT is treated as a
separate upstream stage (see `ARCHITECTURE.md`).

## Running tests

**Linux / macOS:**

```bash
PYTHONPATH=. python -m pytest tests/ -v
# 109 tests covering chunking, indexing, orchestration, and guardrails
```

**Windows (cmd.exe):**

```bat
set PYTHONPATH=%~dp0
python -m pytest tests/ -v
```

**Windows (PowerShell):**

```powershell
$env:PYTHONPATH = "."
python -m pytest tests/ -v
```

## Notes on running without optional dependencies

Every optional free/local model (`sentence-transformers`, `faster-whisper`,
`tiktoken`) has a deterministic, dependency-free fallback so the pipeline
never hard-crashes if one is missing - it just degrades in quality
(hashing-based embeddings instead of real semantic embeddings, etc.), with a
warning logged. This was verified throughout development in a sandboxed
environment without those packages installed.
