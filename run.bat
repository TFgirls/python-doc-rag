@echo off
rem start python-doc-rag
rem 切到 UTF-8 代码页，避免中文日志在 CMD 里显示成乱码
chcp 65001 >nul
cd /d "%~dp0"
set "PY=%~dp0venv\Scripts\python.exe"
set STREAMLIT_SERVER_FILEWATCHER_TYPE=none
"%PY%" -m streamlit run app.py
pause
