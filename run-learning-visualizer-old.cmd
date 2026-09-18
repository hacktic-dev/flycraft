@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0flycraft_learning_loop_v6_old.py"
if errorlevel 1 pause
