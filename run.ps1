<#
.SYNOPSIS
  PulseOps Windows PowerShell Launcher
.DESCRIPTION
  Starts the Strands Decider 2B service and PulseOps FastAPI backend via WSL2.
#>
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " PulseOps — AI Incident Commander" -ForegroundColor Cyan
Write-Host " Launching via WSL2..." -ForegroundColor DarkGray
Write-Host "====================================================" -ForegroundColor Cyan

wsl bash -c "cd /mnt/d/PulseOps && ./backend/run.sh"
