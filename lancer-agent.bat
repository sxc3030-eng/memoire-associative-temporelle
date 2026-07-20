@echo off
cd /d "%~dp0"
py start_agent.py
if errorlevel 1 pause
