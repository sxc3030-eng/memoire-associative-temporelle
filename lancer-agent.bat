@echo off
cd /d "%~dp0"
py start_agent.py --async-injection --enable-matlm ^
  --matlm-python "D:\MAT-LM\.venv\Scripts\python.exe" ^
  --matlm-model "D:\MAT-LM\models\granite-3.3-2b-instruct" ^
  --matlm-adapter "D:\MAT-LM\adapter" ^
  --matlm-load-mode auto ^
  --matlm-max-new-tokens 384 ^
  --matlm-timeout-seconds 180
if errorlevel 1 pause
