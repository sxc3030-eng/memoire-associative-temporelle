@echo off
cd /d "%~dp0"
py start_agent.py --async-injection
if errorlevel 1 pause
