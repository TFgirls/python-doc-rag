@echo off
rem rebuild python-doc-rag knowledge base
rem 切到 UTF-8 代码页，避免中文日志在 CMD 里显示成乱码
chcp 65001 >nul
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" build.py
pause
