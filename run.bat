@echo off
REM PulseOps Windows Batch Launcher
echo ====================================================
echo  PulseOps -- AI Incident Commander
echo  Launching via WSL2...
echo ====================================================
wsl bash -c "cd /mnt/d/PulseOps && ./backend/run.sh"
