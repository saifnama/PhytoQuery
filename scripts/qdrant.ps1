# BloomIndex — Qdrant Server (Docker or Podman), Windows PowerShell
# Mirrors scripts/qdrant.sh subcommands: start | stop | status | restart | logs | remove
# Settings via env: QDRANT_RUNTIME(auto|podman|docker) QDRANT_CONTAINER QDRANT_STORAGE_DIR QDRANT_VERSION QDRANT_PORT_REST QDRANT_PORT_GRPC
param([Parameter(Position=0)][string]$Action = "status")

$ErrorActionPreference = "Stop"
$name = if ($env:QDRANT_CONTAINER) { $env:QDRANT_CONTAINER } else { "bloomindex-qdrant" }
$ver = if ($env:QDRANT_VERSION) { $env:QDRANT_VERSION } else { "v1.18.0" }
$rest = if ($env:QDRANT_PORT_REST) { $env:QDRANT_PORT_REST } else { "6333" }
$grpc = if ($env:QDRANT_PORT_GRPC) { $env:QDRANT_PORT_GRPC } else { "6334" }
$storage = if ($env:QDRANT_STORAGE_DIR) { $env:QDRANT_STORAGE_DIR } else { "$env:LOCALAPPDATA\bloomindex\qdrant_storage" }
$image = "qdrant/qdrant:$ver"

function Get-Runtime {
  $want = if ($env:QDRANT_RUNTIME) { $env:QDRANT_RUNTIME } else { "auto" }
  if ($want -eq "auto") {
    if (Get-Command podman -ErrorAction SilentlyContinue) { return "podman" }
    if (Get-Command docker -ErrorAction SilentlyContinue) { return "docker" }
    throw "Neither podman nor docker found in PATH."
  }
  return $want
}

$rt = Get-Runtime
switch ($Action) {
  "start" {
    New-Item -ItemType Directory -Force -Path $storage | Out-Null
    $state = & $rt ps -a --filter "name=^$name$" --format "{{.State}}" 2>$null
    if ($state -match "running") { Write-Host "already running"; break }
    if ($state -match "exited|created|paused") { & $rt start $name; break }
    & $rt run -d --name $name -p "${rest}:6333" -p "${grpc}:6334" -v "${storage}:/qdrant/storage" --restart unless-stopped $image
  }
  "stop" { & $rt stop $name }
  "restart" { & $rt stop $name; & $rt start $name }
  "logs" { & $rt logs -f $name }
  "remove" { & $rt rm -f $name }
  default {
    & $rt ps -a --filter "name=^$name$" --format "table {{.Names}}\t{{.State}}\t{{.Ports}}"
    try { (Invoke-RestMethod "http://localhost:$rest/healthz" -TimeoutSec 3); Write-Host "health: ok" } catch { Write-Host "health: down" }
  }
}
