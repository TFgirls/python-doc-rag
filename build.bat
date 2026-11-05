@echo off
rem rebuild python-doc-rag knowledge base
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" build.py
pause
