"""
End-to-end entrypoint: Voice-to-Answer RAG Pipeline.

Wires together every module built in Parts 1-8:
  STT (Part 1) -> Chunking router (Parts 2-3) -> Hybrid indexing (Part 4)
  -> Orchestration harness (Part 5) -> Guardrails (Part 6)
  -> Latency-optimized retrieval (Part 7), all measurable via the
     benchmark harness (Part 8).

This is the single object a caller (CLI, API server, etc.) needs to
instantiate: ingest documents/transcripts, then ask questions by text or by
voice.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from chunking.router import route_and_chunk, DocType
from indexing.hybrid_retriever import HybridRetriever
from orchestration.harness import RAGOrchestrator
from orchestration.schemas import FinalAnswer, AnswerStatus
from orchestration.steps import local_extractive_generate, make_llm_generate_fn
from orchestration.llm_provider import call_llm_with_retry
from guardrails.pipeline import GuardrailPipeline
from stt.provider import transcribe, STTResult
from stt.errors import STTAllProvidersExhaustedError, STTValidationError

logger = logging.getLogger("pipeline")


def llm_generate_fn():
    """Opt-in real generator: VoiceRAGPipeline(generate_fn=llm_generate_fn())
    Requires LLM_API_KEY set (see orchestration/llm_provider.py). Falls back
    to local_extractive_generate at call time if the LLM errors, so a demo
    never hard-crashes on a transient quota/network issue."""
    llm_fn = make_llm_generate_fn(call_llm_with_retry)

    def _generate(query, chunks):
        try:
            return llm_fn(query, chunks)
        except Exception as e:
            logger.warning("LLM generation failed (%s) — falling back to extractive", e)
            return local_extractive_generate(query, chunks)

    return _generate


class VoiceRAGPipeline:
    def __init__(self, generate_fn=None, persist_dir: str = None):
        # Auto-detect LLM: use real LLM if API key is set, else local extractive
        if generate_fn is None:
            from config import CONFIG
            if CONFIG.llm.api_key:
                generate_fn = llm_generate_fn()
                self._llm_active = True
            else:
                generate_fn = local_extractive_generate
                self._llm_active = False
        else:
            self._llm_active = True  # user explicitly provided a generator

        # Load from disk if persist_dir exists, otherwise fresh index
        if persist_dir and Path(persist_dir).is_dir():
            logger.info("Loading persisted index from %s", persist_dir)
            self.retriever = HybridRetriever.load(persist_dir)
        else:
            self.retriever = HybridRetriever()
        self.persist_dir = persist_dir
        self.guardrails = GuardrailPipeline()
        self.orchestrator = RAGOrchestrator(
            self.retriever,
            generate_fn=generate_fn,
            guardrail_fn=self.guardrails.pre_generation_check,
            post_guardrail_fn=self.guardrails.post_generation_check,
        )
        self._doc_count = 0

    # -- Ingestion -----------------------------------------------------
    def ingest_text(self, text: str, source_id: str, doc_type: Optional[DocType] = None) -> int:
        """Chunk (auto-routed strategy) and index a text document. Returns
        number of chunks indexed."""
        chunks = route_and_chunk(text, source_id, doc_type=doc_type)
        self.retriever.index(chunks)
        self._doc_count += 1
        logger.info("Ingested source_id=%s -> %d chunks", source_id, len(chunks))
        return len(chunks)

    def save(self, directory: str = None) -> None:
        """Persist the current index to disk for later reuse."""
        d = directory or self.persist_dir
        if not d:
            raise ValueError("No persist_dir specified. Pass directory= or set persist_dir on init.")
        self.retriever.save(d)
        logger.info("Pipeline index saved to %s", d)

    def ingest_audio(self, audio_path: str, source_id: str) -> Tuple[Optional[int], Optional[STTResult], Optional[str]]:
        """Transcribe audio (STT, upstream stage per Part 7 assumption) then
        ingest as a TRANSCRIPT-type document. Returns
        (num_chunks_or_None, stt_result_or_None, error_message_or_None) so
        callers get graceful degradation instead of an exception if both the
        cloud provider AND local fallback are unavailable."""
        try:
            stt_result = transcribe(audio_path)
        except (STTAllProvidersExhaustedError, STTValidationError) as e:
            logger.error("Audio ingestion failed for %s: %s", source_id, e)
            return None, None, str(e)

        n = self.ingest_text(stt_result.text, source_id, doc_type=DocType.TRANSCRIPT)
        return n, stt_result, None

    # -- Querying --------------------------------------------------------
    def ask_text(self, query_text: str, top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> FinalAnswer:
        return self.orchestrator.answer(query_text, top_k=top_k, metadata_filter=metadata_filter)

    def ask_voice(
        self, audio_path: str, top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[STTResult], FinalAnswer]:
        """Full voice-to-answer path. STT failures degrade gracefully to a
        schema-valid FinalAnswer(status=ERROR) instead of raising, keeping
        the same "never crash" guarantee the orchestrator provides for the
        retrieval/generation side (Part 5)."""
        try:
            stt_result = transcribe(audio_path)
        except (STTAllProvidersExhaustedError, STTValidationError) as e:
            logger.error("Voice query failed at STT stage: %s", e)
            return None, FinalAnswer(
                status=AnswerStatus.ERROR,
                refusal_reason=f"Speech-to-text failed: {e}",
            )

        answer = self.ask_text(stt_result.text, top_k=top_k, metadata_filter=metadata_filter)
        return stt_result, answer

    # -- Introspection ---------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "documents_ingested": self._doc_count,
            "dense_vectors": self.retriever.dense.index.ntotal,
            "sparse_chunks": len(self.retriever.sparse.chunks),
            "cache_stats": self.retriever.cache_stats(),
            "strategy_hit_counts": self.retriever.strategy_hit_counts(),
        }
