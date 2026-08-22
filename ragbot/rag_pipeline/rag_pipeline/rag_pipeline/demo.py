#!/usr/bin/env python3
"""
Interactive CLI Demo - Voice-to-Answer RAG Pipeline
====================================================

Accepts text or voice queries against an indexed corpus. Supports:
  - Text mode: type questions directly
  - Voice mode: record from microphone or provide an audio file path
  - Inline corpus loading from .txt files
  - Latency breakdown per stage
  - Guardrail status display

Usage:
    python demo.py                          # start with empty corpus
    python demo.py --corpus path/to/docs    # ingest .txt files from a directory
    python demo.py --file doc.txt           # ingest a single text file

Commands inside the demo:
    :text       Switch to text input mode (default)
    :voice      Switch to voice input mode (microphone recording)
    :file <p>   Switch to voice mode using an audio file path
    :ingest <p> Ingest a text file into the corpus
    :stats      Show corpus statistics
    :bench      Run a quick 5-query benchmark
    :help       Show available commands
    :quit       Exit the demo
"""
import argparse
import io
import os
import sys
import tempfile
import wave
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -- Colours (ANSI, disabled if not a TTY) --------------------------------
_IS_TTY = sys.stdout.isatty()

def _c(code, text):
    return "\033[%sm%s\033[0m" % (code, text) if _IS_TTY else text

BOLD    = lambda t: _c("1", t)
DIM     = lambda t: _c("2", t)
GREEN   = lambda t: _c("32", t)
RED     = lambda t: _c("31", t)
YELLOW  = lambda t: _c("33", t)
CYAN    = lambda t: _c("36", t)
BLUE    = lambda t: _c("34", t)
MAGENTA = lambda t: _c("35", t)
WHITE   = lambda t: _c("97", t)
BG_BLUE = lambda t: _c("44", t)


# -- Audio recording via sounddevice --------------------------------------
def record_audio(duration_seconds=5.0, sample_rate=16000):
    """Record from the default microphone and return path to a temporary WAV file."""
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "sounddevice not installed. Install with: pip install sounddevice"
        )

    print(CYAN("  [REC] Recording %.1fs of audio... (speak now)" % duration_seconds), flush=True)
    recording = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()  # block until recording is complete
    print(DIM("  [OK] Recording complete"), flush=True)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(recording.tobytes())
    return tmp.name


# -- Pretty-print helpers -------------------------------------------------
W = 54  # inner width between box borders (exclusive of "║  " and "  ║")

def _box_line(text="", pad=W):
    """Single inner line of a box: ║ <text padded to W> ║"""
    t = text[:pad]
    return "║  " + t.ljust(pad) + "  ║"

def _box_divider(char="═"):
    return "╠" + "═" * (W + 4) + "╣"

def _box_top():
    return "╔" + "═" * (W + 4) + "╗"

def _box_bottom():
    return "╚" + "═" * (W + 4) + "╝"

def _box_section_divider():
    return "╠" + "═" * (W + 4) + "╣"

def _box_print(lines):
    """Print a list of lines inside the box border."""
    for line in lines:
        print("  " + _box_line(line))

def _box_print_raw(text):
    """Print raw text inside the box, word-wrapped."""
    words = text.split()
    line = ""
    for w in words:
        if line and len(line) + len(w) + 1 > W:
            print("  " + _box_line(line))
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        print("  " + _box_line(line))


def _latency_tracker():
    """Return a mutable list used as a latency accumulator for percentile stats."""
    return []


def _banner(llm_active=False):
    print()
    print("  " + _box_top())
    print("  " + _box_line("🎙️  RAG ASSISTANT".center(W)))
    print("  " + _box_line("MSMARCO-XI Knowledge Base".center(W)))
    print("  " + _box_section_divider())
    print("  " + _box_line())
    print("  " + _box_line("🎤  Ask your question"))
    print("  " + _box_line())
    if llm_active:
        from config import CONFIG
        print("  " + _box_line("  🟢 LLM: Groq (%s)" % CONFIG.llm.model))
    else:
        print("  " + _box_line("  🟡 Local extractive (set LLM_API_KEY for LLM mode)"))
    print("  " + _box_line("  Type :help for commands, :quit to exit"))
    print("  " + _box_line())
    print("  " + _box_section_divider())


def _status_line(label, value, ok=True):
    colour = GREEN if ok else RED
    print("  %-20s %s" % (DIM(label + ":"), colour(value)))


def _display_answer(answer, query="", latency_list=None):
    """Pretty-print a FinalAnswer in box-drawn format."""
    from orchestration.schemas import AnswerStatus

    # -- ANSWER section --
    print("  " + _box_section_divider())
    print("  " + _box_line("ANSWER"))
    print("  " + _box_line())

    if answer.status == AnswerStatus.ANSWERED:
        _box_print_raw(answer.answer_text or "")
        print("  " + _box_line())
        grounded_label = "🟢 Grounded Answer" if answer.grounded else "🟡 Partially Grounded"
        conf = answer.confidence if answer.confidence is not None else "—"
        status_line = "%s        Confidence: %s%%" % (grounded_label, conf)
        print("  " + _box_line(status_line))

    elif answer.status == AnswerStatus.REFUSED:
        _box_print_raw("⚠️  REFUSED")
        _box_print_raw(answer.refusal_reason or "")

    else:  # ERROR
        _box_print_raw("❌ ERROR")
        _box_print_raw(answer.refusal_reason or "")

    # -- SOURCES section --
    if answer.source_chunks:
        print("  " + _box_line())
        print("  " + _box_section_divider())
        print("  " + _box_line("📚 SOURCES"))
        print("  " + _box_line())
        for i, src in enumerate(answer.source_chunks, 1):
            print("  " + _box_line("┌" + "─" * (W - 2) + "┐"))
            print("  " + _box_line("│ Document %02d" % i))
            print("  " + _box_line("│ Relevance: %.2f" % src.score))
            print("  " + _box_line("│ Source: %s" % src.source_id))
            print("  " + _box_line("└" + "─" * (W - 2) + "┘"))
            print("  " + _box_line())

    # -- PIPELINE section --
    if answer.stage_latencies_ms:
        print("  " + _box_section_divider())
        print("  " + _box_line("⚡ PIPELINE"))
        print("  " + _box_line())
        stages = [
            ("Query Processing", "input_validation_ms", None),
            ("Retrieval", "retrieval_ms", None),
            ("Reranking", "rerank_ms", None),
            ("LLM", "generation_ms", None),
            ("Guardrails", "guardrails_pre_ms", None),
        ]
        for label, key, _ in stages:
            ms = answer.stage_latencies_ms.get(key)
            if ms is not None:
                print("  " + _box_line("  %-18s ✓  %6.0f ms" % (label, ms)))
        print("  " + _box_line("  " + "─" * (W - 4)))
        total_ms = answer.total_latency_ms
        threshold_colour = "🟢" if total_ms < 200 else ("🟡" if total_ms < 500 else "🔴")
        print("  " + _box_line("  %-18s     %6.0f ms   %s < 200 ms" % ("TOTAL", total_ms, threshold_colour)))

    # -- LATENCY section --
    if latency_list is not None and latency_list:
        print("  " + _box_line())
        print("  " + _box_section_divider())
        print("  " + _box_line("📊 LATENCY"))
        print("  " + _box_line())
        sorted_lat = sorted(latency_list)
        n = len(sorted_lat)
        p50 = sorted_lat[int(n * 0.50)] if n else 0
        p70 = sorted_lat[int(n * 0.70)] if n else 0
        p100 = sorted_lat[-1] if n else 0
        print("  " + _box_line("  P50    %5.0f ms" % p50))
        print("  " + _box_line("  P70    %5.0f ms" % p70))
        print("  " + _box_line("  P100   %5.0f ms" % p100))

    print("  " + _box_bottom())
    print()


# -- Corpus ingestion helper ----------------------------------------------
def ingest_file(pipe, filepath):
    """Read a text file and ingest it into the pipeline."""
    path = Path(filepath)
    if not path.is_file():
        print(RED("  File not found: %s" % filepath))
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        print(YELLOW("  File is empty: %s" % filepath))
        return 0
    n = pipe.ingest_text(text, source_id=path.name)
    print(GREEN("  [OK] Ingested %s -> %d chunks" % (path.name, n)))
    return n


def ingest_directory(pipe, dirpath):
    """Ingest all .txt files in a directory."""
    p = Path(dirpath)
    if not p.is_dir():
        print(RED("  Directory not found: %s" % dirpath))
        return 0
    files = sorted(p.glob("*.txt"))
    if not files:
        files = sorted(p.glob("*.md"))
    if not files:
        print(YELLOW("  No .txt or .md files found in %s" % dirpath))
        return 0
    total = 0
    for f in files:
        total += ingest_file(pipe, str(f))
    return total


def ingest_sample_corpus(pipe):
    """Load a built-in sample corpus so the demo works out of the box."""
    samples = {
        "speech_to_text": (
            "Speech-to-text conversion in this pipeline uses Sarvam AI as the "
            "primary provider with a free tier, and falls back to a local "
            "faster-whisper model if the cloud call fails or the quota is "
            "exhausted. Input audio is validated for format, duration, and "
            "sample rate before any API call. The validation module checks "
            "that the file exists, has an allowed extension (.wav, .mp3, .flac, "
            ".ogg, .m4a), is within the free-tier size limit (25 MB), and has "
            "a duration between 0.3 and 60 seconds."
        ),
        "chunking": (
            "Chunking strategies include fixed-size windows with 15% overlap "
            "(400 tokens per window), semantic chunking based on embedding "
            "similarity breakpoints using all-MiniLM-L6-v2, structure-aware "
            "chunking that respects markdown headers and speaker turns in "
            "transcripts, and hierarchical parent-child chunking for long "
            "documents where small child chunks enable precise retrieval while "
            "parent chunks provide expanded context. The router auto-detects "
            "document type and dispatches to the best strategy."
        ),
        "hybrid_retrieval": (
            "Hybrid retrieval combines dense vector search using FAISS "
            "(IndexFlatIP over L2-normalized vectors for exact cosine search) "
            "with sparse keyword search using BM25 (rank_bm25 library). The "
            "two rankings are fused with Reciprocal Rank Fusion (RRF, k=60) "
            "rather than weighted score blending, because cosine similarity "
            "and BM25 scores live on incomparable scales. RRF fuses on rank "
            "position instead, sidestepping normalization entirely."
        ),
        "orchestration": (
            "The orchestration harness treats retrieval, guardrails, reranking, "
            "and generation as distinct, independently retryable steps rather "
            "than one prompt. Every stage's input/output is a Pydantic model, "
            "so malformed data is caught at the boundary. Retrieval and "
            "generation calls use exponential backoff retry decorators. If "
            "any stage fails, the harness always returns a schema-valid "
            "FinalAnswer - never an unhandled exception."
        ),
        "guardrails": (
            "Four checks gate every answer: off-topic detection using RRF "
            "score thresholds plus lexical word-overlap between query and "
            "retrieved chunks, unsafe-input filtering with local rule-based "
            "regex patterns (no paid moderation API), groundedness checking "
            "via citation validity (every [source:chunk_id] must exist in "
            "retrieved set) plus lexical overlap ratio, and a pluggable NLI "
            "entailment hook for stronger future guarantees. All verdicts "
            "are logged to guardrail_audit.jsonl for full auditability."
        ),
        "latency_optimization": (
            "Latency optimization techniques include an exact-match LRU cache "
            "for query embeddings so repeated queries skip re-embedding, "
            "parallel dense and sparse search running concurrently on a thread "
            "pool to halve retrieval wall-time, and a hard cap of 20 on top_k "
            "to keep retrieval and rerank work bounded. The benchmark shows "
            "P100 under 60ms for the full retrieval-to-generation path."
        ),
        "vector_database": (
            "FAISS and Chroma are free, open-source, local vector databases "
            "with no hosted paid tier. FAISS stores chunk vectors alongside "
            "metadata so retrieval can filter by speaker, section, or time "
            "range. IndexFlatIP provides exact cosine search at demo scale; "
            "IVF/HNSW can be swapped in for larger corpora while staying "
            "free and local."
        ),
        "meeting_transcripts": (
            "Meeting transcripts are chunked by speaker turn, extracting "
            "timestamp ranges and speaker names as metadata so retrieval can "
            "filter conversations by who said what and when. The structure-aware "
            "chunker detects transcript patterns (Name: text) and preserves "
            "speaker identity, timestamps, and turn boundaries as first-class "
            "metadata fields on each chunk."
        ),
        "msmarco_dataset": (
            "The MSMARCO-XI dataset from AI4Bharat contains passage retrieval "
            "queries and corresponding passages in multiple Indian languages. "
            "Each record has a query, a set of passages with passage_text, and "
            "relevance judgments. The data loader streams from HuggingFace "
            "datasets and exports individual passages as JSONL records with "
            "doc_id, text, and source_query fields for ingestion into the "
            "RAG pipeline."
        ),
    }
    total = 0
    for source_id, text in samples.items():
        n = pipe.ingest_text(text, source_id=source_id)
        total += n
    return total


# -- Main interactive loop ------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI Demo for the Voice-to-Answer RAG Pipeline"
    )
    parser.add_argument("--corpus", help="Directory of .txt/.md files to ingest")
    parser.add_argument(
        "--file", help="Single text file to ingest"
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Start with MSMARCO-XI streaming ingestion (downloads samples on-the-fly)"
    )
    parser.add_argument(
        "--stream-limit", type=int, default=5000,
        help="Max samples to stream from MSMARCO-XI (default: 5000)"
    )
    parser.add_argument("--voice", action="store_true", help="Start in voice input mode")
    parser.add_argument(
        "--record-sec", type=float, default=5.0,
        help="Microphone recording duration in seconds (default: 5)"
    )
    args = parser.parse_args()

    # -- Initialize pipeline --------------------------------------------
    print(DIM("  Initializing pipeline..."))
    from pipeline import VoiceRAGPipeline
    pipe = VoiceRAGPipeline()
    print(GREEN("  [OK] Pipeline ready" + (" (LLM mode)" if pipe._llm_active else " (local extractive)")))

    # -- Ingest corpus -------------------------------------------------
    print(DIM("  Loading sample corpus..."))
    n = ingest_sample_corpus(pipe)
    print(GREEN("  [OK] Sample corpus loaded: %d chunks indexed" % n))

    if args.corpus:
        print(DIM("  Ingesting directory: %s" % args.corpus))
        ingest_directory(pipe, args.corpus)
    if args.file:
        print(DIM("  Ingesting file: %s" % args.file))
        ingest_file(pipe, args.file)
    if args.stream:
        print(DIM("  Streaming MSMARCO-XI (%d samples)..." % args.stream_limit))
        try:
            from data.streaming_ingest import run_streaming_ingest
            stream_stats = run_streaming_ingest(
                pipe, limit=args.stream_limit, verbose=True
            )
            print(GREEN(
                "  [OK] Streaming complete: %d samples, %d chunks" % (
                    stream_stats.samples_seen, stream_stats.total_chunks
                )
            ))
        except Exception as e:
            print(RED("  [!!] Streaming failed: %s" % e))

    stats = pipe.stats()
    print(DIM("  Dense vectors: %d, Sparse chunks: %d" % (
        stats["dense_vectors"], stats["sparse_chunks"]
    )))
    print()

    # -- Interactive loop -----------------------------------------------
    _banner(llm_active=pipe._llm_active)

    mode = "text"
    voice_file_path = None  # for :file mode
    latency_list = _latency_tracker()

    while True:
        try:
            if mode == "text":
                prompt = CYAN("you > ")
                raw = input(prompt).strip()
            elif mode == "voice_file" and voice_file_path:
                prompt = CYAN("voice:%s > " % Path(voice_file_path).name)
                raw = input(prompt).strip()
            else:
                prompt = CYAN("voice > ")
                raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(DIM("  Goodbye!"))
            break

        if not raw:
            continue

        # -- Commands ---------------------------------------------------
        if raw.startswith(":"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in (":quit", ":q", ":exit"):
                print(DIM("  Goodbye!"))
                break

            elif cmd == ":help":
                print()
                print(BOLD("  Available commands:"))
                print(DIM("    :text              Switch to text input mode"))
                print(DIM("    :voice             Switch to voice input mode (microphone)"))
                print(DIM("    :file <path>       Use a specific audio file for voice queries"))
                print(DIM("    :ingest <path>     Ingest a .txt or .md file into the corpus"))
                print(DIM("    :ingest-dir <dir>  Ingest all .txt/.md files from a directory"))
                print(DIM("    :streaming [N]     Stream N samples from MSMARCO-XI (default: 5000)"))
                print(DIM("    :stats             Show corpus statistics"))
                print(DIM("    :bench             Run a quick 5-query benchmark"))
                print(DIM("    :help              Show this help"))
                print(DIM("    :quit              Exit the demo"))
                print()

            elif cmd == ":text":
                mode = "text"
                voice_file_path = None
                print(GREEN("  [OK] Switched to text mode"))

            elif cmd == ":voice":
                try:
                    import sounddevice  # noqa: F401
                    mode = "voice"
                    voice_file_path = None
                    print(GREEN("  [OK] Switched to voice mode (%.1fs recording)" % args.record_sec))
                except ImportError:
                    print(RED("  [!!] sounddevice not installed. Install with: pip install sounddevice"))
                    print(DIM("    Falling back to text mode"))
                    mode = "text"

            elif cmd == ":file":
                if not arg:
                    print(YELLOW("  Usage: :file /path/to/audio.wav"))
                elif not os.path.isfile(arg):
                    print(RED("  File not found: %s" % arg))
                else:
                    voice_file_path = arg
                    mode = "voice_file"
                    print(GREEN("  [OK] Voice file set: %s" % arg))

            elif cmd == ":ingest":
                if not arg:
                    print(YELLOW("  Usage: :ingest /path/to/document.txt"))
                else:
                    ingest_file(pipe, arg)

            elif cmd == ":ingest-dir":
                if not arg:
                    print(YELLOW("  Usage: :ingest-dir /path/to/docs/"))
                else:
                    ingest_directory(pipe, arg)

            elif cmd == ":streaming":
                print(DIM("  Starting MSMARCO-XI streaming ingestion..."))
                try:
                    from data.streaming_ingest import run_streaming_ingest
                    stream_limit = int(arg) if arg else 5000
                    stats = run_streaming_ingest(
                        pipe, limit=stream_limit, verbose=True
                    )
                    print(GREEN(
                        f"  [OK] Streaming complete: {stats.samples_seen} samples, "
                        f"{stats.total_chunks} chunks indexed"
                    ))
                except Exception as e:
                    print(RED(f"  [!!] Streaming failed: {e}"))

            elif cmd == ":stats":
                stats = pipe.stats()
                print()
                print(BOLD("  Corpus statistics:"))
                _status_line("Documents ingested", str(stats["documents_ingested"]))
                _status_line("Dense vectors", str(stats["dense_vectors"]))
                _status_line("Sparse chunks", str(stats["sparse_chunks"]))
                if stats.get("strategy_hit_counts"):
                    print(DIM("    Strategy hits: %s" % stats["strategy_hit_counts"]))
                cache = stats.get("cache_stats", {})
                if isinstance(cache, dict) and "hits" in cache:
                    _status_line("Cache hits", str(cache["hits"]))
                    _status_line("Cache misses", str(cache["misses"]))
                print()

            elif cmd == ":bench":
                print(DIM("  Running 5-query benchmark..."))
                from benchmark.corpus import build_benchmark_queries
                queries = build_benchmark_queries(min_count=5)[:5]
                for i, q in enumerate(queries):
                    result = pipe.ask_text(q)
                    if result.status.value == "answered":
                        status = GREEN("[OK]")
                    else:
                        status = YELLOW("[--]")
                    print("    %s [%7.2fms] %s" % (status, result.total_latency_ms, q[:60]))
                print()

            else:
                print(YELLOW("  Unknown command: %s  (type :help for commands)" % cmd))

            continue

        # -- Process query -----------------------------------------------
        if mode == "text":
            result = pipe.ask_text(raw)
            latency_list.append(result.total_latency_ms)
            _display_answer(result, query=raw, latency_list=latency_list)

        elif mode == "voice":
            # Record from microphone
            try:
                audio_path = record_audio(duration_seconds=args.record_sec)
                print(DIM("  Transcribing..."))
                stt_result, answer = pipe.ask_voice(audio_path)
                if stt_result:
                    print(DIM('  STT: "%s"  [%s, %.0fms]' % (
                        stt_result.text[:80], stt_result.provider, stt_result.latency_ms
                    )))
                latency_list.append(answer.total_latency_ms)
                _display_answer(answer, query=stt_result.text if stt_result else "", latency_list=latency_list)
                # Clean up temp file
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass
            except RuntimeError as e:
                print(RED("  [!!] %s" % e))
            except Exception as e:
                print(RED("  [!!] Recording failed: %s" % e))

        elif mode == "voice_file" and voice_file_path:
            # Use the pre-set audio file
            try:
                print(DIM("  Transcribing: %s" % voice_file_path))
                stt_result, answer = pipe.ask_voice(voice_file_path)
                if stt_result:
                    print(DIM('  STT: "%s"  [%s, %.0fms]' % (
                        stt_result.text[:80], stt_result.provider, stt_result.latency_ms
                    )))
                latency_list.append(answer.total_latency_ms)
                _display_answer(answer, query=stt_result.text if stt_result else "", latency_list=latency_list)
            except Exception as e:
                print(RED("  [!!] Voice query failed: %s" % e))

        else:
            print(YELLOW("  No voice source set. Use :voice for microphone or :file <path> for an audio file."))


if __name__ == "__main__":
    main()
