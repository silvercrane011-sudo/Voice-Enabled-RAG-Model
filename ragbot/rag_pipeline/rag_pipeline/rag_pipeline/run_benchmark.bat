@echo off
REM Voice-to-Answer RAG Pipeline - Benchmark (Windows launcher)
REM Usage:
REM   run_benchmark.bat                Standard benchmark (local extractive generation)
REM   run_benchmark.bat full           Full-path benchmark with STT and/or real LLM

set PYTHONPATH=%~dp0

if "%1"=="full" (
    python "%~dp0benchmark\full_path_benchmark.py" %2 %3 %4
) else (
    python "%~dp0benchmark\run_benchmark.py"
)
