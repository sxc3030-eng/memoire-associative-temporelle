param(
    [switch]$NoWindow
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BaseUrl = "http://127.0.0.1:8765/"
$HealthUrl = "${BaseUrl}api/health"
$StartUrl = "${BaseUrl}api/matlm/start"

function Test-MATLMServer {
    try {
        $response = Invoke-RestMethod -Uri $HealthUrl -Method Get -TimeoutSec 2
        return $response.ok -eq $true -and $response.engine -eq "memory"
    }
    catch {
        return $false
    }
}

function Show-LaunchError([string]$Message) {
    try {
        Add-Type -AssemblyName PresentationFramework
        [System.Windows.MessageBox]::Show(
            $Message,
            "MAT-LM",
            [System.Windows.MessageBoxButton]::OK,
            [System.Windows.MessageBoxImage]::Error
        ) | Out-Null
    }
    catch {
        # Le raccourci reste silencieux si Windows ne peut pas afficher la boite.
    }
}

try {
    if (-not (Test-MATLMServer)) {
        $python = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($null -eq $python) {
            $python = Get-Command python.exe -ErrorAction SilentlyContinue
        }
        if ($null -eq $python) {
            throw "Python est introuvable sur cet ordinateur."
        }

        $serverArguments = @(
            "start_agent.py",
            "--no-browser",
            "--async-injection",
            "--enable-matlm",
            "--matlm-load-mode", "auto",
            "--matlm-max-new-tokens", "384",
            "--matlm-timeout-seconds", "180"
        )
        Start-Process `
            -FilePath $python.Source `
            -ArgumentList $serverArguments `
            -WorkingDirectory $ProjectRoot `
            -WindowStyle Hidden

        $serverReady = $false
        for ($attempt = 0; $attempt -lt 80; $attempt++) {
            Start-Sleep -Milliseconds 250
            if (Test-MATLMServer) {
                $serverReady = $true
                break
            }
        }
        if (-not $serverReady) {
            throw "Le moteur local ne repond pas sur le port 8765."
        }
    }

    try {
        Invoke-RestMethod `
            -Uri $StartUrl `
            -Method Post `
            -ContentType "application/json" `
            -Body "{}" `
            -TimeoutSec 10 | Out-Null
    }
    catch {
        # La page affiche elle-meme l'etat du modele et permet de le relancer.
    }

    if ($NoWindow) {
        exit 0
    }

    $edgeCandidates = @(
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
        "${env:ProgramFiles}\Microsoft\Edge\Application\msedge.exe"
    )
    $edgePath = $edgeCandidates |
        Where-Object { $_ -and (Test-Path -LiteralPath $_) } |
        Select-Object -First 1

    if (-not $edgePath) {
        $edgeCommand = Get-Command msedge.exe -ErrorAction SilentlyContinue
        if ($null -ne $edgeCommand) {
            $edgePath = $edgeCommand.Source
        }
    }

    if ($edgePath) {
        Start-Process `
            -FilePath $edgePath `
            -ArgumentList @("--app=$BaseUrl", "--window-size=1180,820")
    }
    else {
        Start-Process $BaseUrl
    }
}
catch {
    Show-LaunchError $_.Exception.Message
    exit 1
}
