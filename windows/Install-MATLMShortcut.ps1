$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LauncherPath = Join-Path $PSScriptRoot "Start-MATLM.ps1"
$IconPath = Join-Path $ProjectRoot "web\assets\matlm-favicon.ico"
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "MAT-LM.lnk"
$PowerShellPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"

if (-not (Test-Path -LiteralPath $LauncherPath)) {
    throw "Le lanceur MAT-LM est introuvable."
}
if (-not (Test-Path -LiteralPath $IconPath)) {
    throw "L'icone MAT-LM est introuvable."
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($ShortcutPath)
$shortcut.TargetPath = $PowerShellPath
$shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$LauncherPath`""
$shortcut.WorkingDirectory = $ProjectRoot
$shortcut.IconLocation = "$IconPath,0"
$shortcut.Description = "Ouvrir la conversation locale MAT-LM"
$shortcut.Save()

Write-Output $ShortcutPath
