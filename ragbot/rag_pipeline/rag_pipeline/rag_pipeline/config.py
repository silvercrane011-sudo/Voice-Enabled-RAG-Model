"""
Central config for the pipeline. All values are free-tier / local by design.
Part 1 of 10: this file will grow in later parts (chunking, indexing, guardrails configs).
"""
import os
from pathlib import Path
from dataclasses import dataclass, field

# Load .env file if present (persists API keys between sessions)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed — env vars still work


@dataclass
class STTConfig:
    provider: str = "sarvam"          # "sarvam" | "elevenlabs" | "local_fallback"
    sarvam_api_key: str = os.environ.get("SARVAM_API_KEY", "")
    elevenlabs_api_key: str = os.environ.get("ELEVENLABS_API_KEY", "")
    sarvam_endpoint: str = "https://api.sarvam.ai/speech-to-text"
    elevenlabs_endpoint: str = "https://api.elevenlabs.io/v1/speech-to-text"
    max_duration_seconds: int = 60        # free tier friendly cap
    min_duration_seconds: float = 0.3
    allowed_sample_rates: tuple = (8000, 16000, 22050, 44100, 48000)
    allowed_formats: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm")
    max_file_size_mb: int = 25
    request_timeout_seconds: float = 8.0
    max_retries: int = 2
    backoff_base_seconds: float = 0.5     # exponential: base * 2^attempt
    local_fallback_model: str = "faster-whisper:small"  # runs fully offline/free


@dataclass
class LLMConfig:
    # "openai_compatible" works for Groq, OpenRouter, Together, Fireworks —
    # any free-tier provider exposing the /chat/completions shape.
    # "anthropic" calls api.anthropic.com/v1/messages directly.
    provider: str = os.environ.get("LLM_PROVIDER", "openai_compatible")
    api_key: str = os.environ.get("LLM_API_KEY", "")
    base_url: str = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    model: str = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
    max_tokens: int = 1024                # longer answers for comprehensive responses
    temperature: float = 0.0              # deterministic, grounded answers
    request_timeout_seconds: float = 5.0  # tighter timeout for speed
    max_retries: int = 2
    backoff_base_seconds: float = 0.5


@dataclass
class PipelineConfig:
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    log_dir: str = "logs"
    data_dir: str = "data"


CONFIG = PipelineConfig()
