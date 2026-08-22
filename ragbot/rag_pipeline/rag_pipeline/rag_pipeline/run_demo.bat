@echo off
REM Voice-to-Answer RAG Pipeline - Interactive Demo (Windows launcher)
REM Usage:
REM   run_demo.bat                    Start with sample corpus
REM   run_demo.bat --corpus path      Start with your own documents
REM   run_demo.bat --voice            Start in voice input mode

set PYTHONPATH=%~dp0
python "%~dp0demo.py" %*
