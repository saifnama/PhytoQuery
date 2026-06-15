<#
.SYNOPSIS
    PhytoQuery - Qdrant Server lifecycle helper (Docker or Podman) for Windows.

.DESCRIPTION
    Mirror of scripts/qdrant.sh for PowerShell. Idempotent: start when
    already running is a no-op, start on a stopped container resumes it
    without losing data.

    Runtime selection (env QDRANT_RUNTIME):
      auto    (default) try podman first, fall back to docker
      podman  force podman
      docker  force docker

    Docker and Podman share the same CLI surface for run / ps / start /
    stop / logs / rm / inspect — the only branching is the prerequisite
    check (Docker needs a reachable daemon; Podman is daemonless).

.EXAMPLE
    .\scripts\qdrant.ps1 start      # start (creates container on first run)
    .\scripts\qdrant.ps1 stop       # stop, keep state on disk
    .\scripts\qdrant.ps1 status     # show container + health + URLs
    .\scripts\qdrant.ps1 restart    # stop + start
    .\scripts\qdrant.ps1 logs       # tail container logs
    .\scripts\qdrant.ps1 remove     # stop + delete container; storage preserved

.NOTES
    Settings via environment variables (sensible defaults baked in):
      QDRANT_RUNTIME      auto | docker | podman   default: auto
      QDRANT_CONTAINER    container name           default: phytoquery-qdrant
      QDRANT_STORAGE_DIR  host storage path        default: %LOCALAPPDATA%\phytoquery\qdrant_storage
      QDRANT_VERSION      qdrant/qdrant image tag  default: v1.18.0
      QDRANT_PORT_REST    REST + Web UI port       default: 6333
      QDRANT_PORT_GRPC    gRPC port                default: 6334

    Requires one of:
      * Docker Desktop for Windows (docker.exe on PATH), OR
      * Podman Desktop / podman.exe on PATH
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop', 'status', 'restart', 'logs', 'remove', 'help')]
    [string]$Command = 'help'
)

$ErrorActionPreference = 'Stop'

# ─── settings ───────────────────────────────────────────────────────────────
$Name        = if ($env:QDRANT_CONTAINER)   { $env:QDRANT_CONTAINER }   else { 'phytoquery-qdrant' }
$StorageDir  = if ($env:QDRANT_STORAGE_DIR) { $env:QDRANT_STORAGE_DIR } else { Join-Path $env:LOCALAPPDATA 'phytoquery\qdrant_storage' }
$Version     = if ($env:QDRANT_VERSION)     { $env:QDRANT_VERSION }     else { 'v1.18.0' }
$PortRest    = if ($env:QDRANT_PORT_REST)   { $env:QDRANT_PORT_REST }   else { '6333' }
$PortGrpc    = if ($env:QDRANT_PORT_GRPC)   { $env:QDRANT_PORT_GRPC }   else { '6334' }
$Image       = "qdrant/qdrant:$Version"
$RestUrl     = "http://localhost:$PortRest"

# ─── runtime selection ──────────────────────────────────────────────────────
$RuntimeChoice = if ($env:QDRANT_RUNTIME) { $env:QDRANT_RUNTIME.ToLower() } else { 'auto' }

function Test-Cmd { param([string]$Name) [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

switch ($RuntimeChoice) {
    'auto' {
        if (Test-Cmd 'podman')      { $Runtime = 'podman' }
        elseif (Test-Cmd 'docker')  { $Runtime = 'docker' }
        else {
            Write-Error "Neither podman nor docker found in PATH. Install one of:`n  https://podman.io/docs/installation`n  https://docs.docker.com/desktop/install/windows-install/"
            exit 1
        }
    }
    { $_ -in 'podman', 'docker' } {
        if (-not (Test-Cmd $RuntimeChoice)) {
            Write-Error "$RuntimeChoice not found in PATH (QDRANT_RUNTIME=$RuntimeChoice)."
            exit 1
        }
        $Runtime = $RuntimeChoice
    }
    default {
        Write-Error "QDRANT_RUNTIME must be one of: auto, podman, docker (got '$RuntimeChoice')."
        exit 1
    }
}

# ─── prerequisite check ─────────────────────────────────────────────────────
# Docker needs a reachable daemon; Podman is daemonless so `podman info`
# is the cheapest liveness probe.
try {
    if ($Runtime -eq 'docker') {
        docker ps 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'docker ps failed' }
    } else {
        podman info 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'podman info failed' }
    }
} catch {
    if ($Runtime -eq 'docker') {
        Write-Error "'docker ps' failed. Is Docker Desktop running? Or set `$env:QDRANT_RUNTIME = 'podman'."
    } else {
        Write-Error "'podman info' failed. On first run try: podman machine init; podman machine start"
    }
    exit 1
}

# ─── helpers ────────────────────────────────────────────────────────────────
function Get-ContainerState {
    # ``<runtime> ps -a --filter name=^NAME$`` is more robust than
    # ``<runtime> inspect --format '{{.State.Status}}'`` because:
    #   * It's container-only by definition (no risk of matching an
    #     image/network/volume with the same name → no "map has no
    #     entry for key 'State'" template-parsing error).
    #   * Empty output means "no container" rather than a non-zero
    #     exit code, which sidesteps PowerShell's $ErrorActionPreference
    #     = 'Stop' converting native-command stderr into terminating
    #     errors.
    #   * --format '{{.State}}' returns the plain state string directly
    #     (running / exited / paused / created / restarting / dead).
    $line = & $Runtime ps -a --filter "name=^${Name}$" --format '{{.State}}' 2>$null
    if (-not $line) { return 'absent' }
    return $line.Trim().ToLower()
}

function Show-Urls {
    Write-Host "REST:    $RestUrl"
    Write-Host "Web UI:  $RestUrl/dashboard"
    Write-Host "gRPC:    localhost:$PortGrpc"
    Write-Host ''
    Write-Host 'Point PhytoQuery at it by setting (in .env / .env.<profile> or shell):'
    Write-Host "  RAG_QDRANT_URL=$RestUrl"
}

function Wait-ForHealth {
    # Polls /healthz for up to 15s. Returns $true on success, $false on timeout.
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $r = Invoke-WebRequest -Uri "$RestUrl/healthz" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
            if ($r.StatusCode -eq 200) { return $true }
        } catch {
            # not ready yet — keep polling
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

# ─── commands ───────────────────────────────────────────────────────────────
function Invoke-Start {
    $state = Get-ContainerState
    switch ($state) {
        'running' {
            Write-Host 'Already running.'
        }
        { $_ -in 'exited', 'created', 'paused' } {
            Write-Host "Starting existing container '$Name' (state: $state)..."
            & $Runtime start $Name | Out-Null
        }
        'absent' {
            if (-not (Test-Path $StorageDir)) {
                New-Item -ItemType Directory -Path $StorageDir -Force | Out-Null
            }
            Write-Host "Creating container '$Name' from image $Image (runtime: $Runtime)..."
            Write-Host "  Storage: $StorageDir"
            & $Runtime run -d `
                --name $Name `
                -p "${PortRest}:6333" -p "${PortGrpc}:6334" `
                -v "${StorageDir}:/qdrant/storage" `
                --restart unless-stopped `
                $Image | Out-Null
        }
        default {
            Write-Error "Container in unexpected state '$state'. Try: $Runtime inspect $Name"
            exit 1
        }
    }

    if (Wait-ForHealth) {
        Write-Host 'Qdrant is healthy.'
    } else {
        Write-Warning "/healthz did not respond within 15s. Check '$Runtime logs $Name'."
    }
    Write-Host ''
    Show-Urls
}

function Invoke-Stop {
    $state = Get-ContainerState
    if ($state -eq 'running') {
        & $Runtime stop $Name | Out-Null
        Write-Host "Stopped (state preserved on disk at $StorageDir)."
    } else {
        Write-Host "Not running (state: $state)."
    }
}

function Invoke-Status {
    $state = Get-ContainerState
    Write-Host "Runtime:   $Runtime"
    Write-Host "Container: $Name"
    Write-Host "State:     $state"
    Write-Host "Image:     $Image"
    Write-Host "Storage:   $StorageDir"
    if ($state -eq 'running') {
        Write-Host "Ports:     REST=$PortRest  gRPC=$PortGrpc"
        Write-Host ''
        try {
            $h = Invoke-WebRequest -Uri "$RestUrl/healthz" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($h.StatusCode -eq 200) {
                Write-Host "Health:    OK ($RestUrl/healthz)"
                try {
                    $c = Invoke-RestMethod -Uri "$RestUrl/collections" -TimeoutSec 2 -ErrorAction Stop
                    $count = $c.result.collections.Count
                    Write-Host "Collections: $count"
                } catch { }
            }
        } catch {
            Write-Host "Health:    UNREACHABLE - check '$Runtime logs $Name'"
        }
        Write-Host ''
        Show-Urls
    }
}

function Invoke-Restart {
    Invoke-Stop
    Invoke-Start
}

function Invoke-Logs {
    & $Runtime logs -f --tail 200 $Name
}

function Invoke-Remove {
    Invoke-Stop
    if ((Get-ContainerState) -ne 'absent') {
        & $Runtime rm $Name | Out-Null
        Write-Host "Container '$Name' removed."
    }
    Write-Host "Storage at $StorageDir preserved (delete manually if you want a clean slate)."
}

function Show-Usage {
    @"
Usage: .\scripts\qdrant.ps1 {start|stop|status|restart|logs|remove}

  start    Start the Qdrant container (creates on first run; idempotent)
  stop     Stop the container; on-disk storage preserved
  status   Show container state, health, collection count, URLs
  restart  Stop + start
  logs     Tail container logs (Ctrl+C to exit)
  remove   Stop + delete container; on-disk storage preserved

Settings via environment variables:
  QDRANT_RUNTIME       auto | docker | podman   (default: auto)
  QDRANT_CONTAINER     (default: phytoquery-qdrant)
  QDRANT_STORAGE_DIR   (default: %LOCALAPPDATA%\phytoquery\qdrant_storage)
  QDRANT_VERSION       (default: v1.18.0)
  QDRANT_PORT_REST     (default: 6333)
  QDRANT_PORT_GRPC     (default: 6334)
"@ | Write-Host
}

switch ($Command) {
    'start'   { Invoke-Start }
    'stop'    { Invoke-Stop }
    'status'  { Invoke-Status }
    'restart' { Invoke-Restart }
    'logs'    { Invoke-Logs }
    'remove'  { Invoke-Remove }
    default   { Show-Usage }
}
