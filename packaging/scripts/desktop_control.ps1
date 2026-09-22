[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet('Install', 'Start', 'Stop')][string]$Action,
    [ValidateRange(1024, 65535)][int]$Port = 8765
)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$runtimeRoot = if (Test-Path -LiteralPath (Join-Path $projectRoot 'runtimes\core\python.exe') -PathType Leaf) {
    $projectRoot
} else {
    Join-Path $projectRoot '.runtime-build'
}
$corePython = Join-Path $runtimeRoot 'runtimes\core\python.exe'
$frontendRoot = Join-Path $projectRoot 'frontend\dist'
$resourceRoot = Join-Path $projectRoot 'resource-library'
$launcherRoot = Join-Path $projectRoot '.runtime-build\launcher'
$installStatePath = Join-Path $projectRoot '.runtime-build\manifests\install-state.json'
$hostScript = Join-Path $projectRoot 'packaging\scripts\webui_host.ps1'
# The visible host delegates to the embedded command: -B -I -m anima_core.

function Get-InstallationStateId([string]$InstallRoot) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($InstallRoot.ToLowerInvariant())
    $sha256 = New-Object System.Security.Cryptography.SHA256Managed
    try { $digest = $sha256.ComputeHash($bytes) }
    finally { $sha256.Dispose() }
    return ([BitConverter]::ToString($digest).Replace('-', '')).ToLowerInvariant()
}

function Get-WebUiInstanceId([string]$InstallRoot, [int]$LocalPort) {
    return "{0}-{1}" -f (Get-InstallationStateId $InstallRoot), $LocalPort
}

$installationStateId = Get-InstallationStateId $projectRoot
$instanceId = Get-WebUiInstanceId $projectRoot $Port
$statePath = Join-Path $launcherRoot ("webui-{0}.json" -f $instanceId)

function Assert-ReleaseBundle {
    if (-not (Test-Path -LiteralPath $installStatePath -PathType Leaf)) {
        throw "Source bootstrap installation is incomplete: $installStatePath"
    }
    try {
        $installState = Get-Content -LiteralPath $installStatePath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ([int]$installState.schemaVersion -ne 1 -or [string]::IsNullOrWhiteSpace([string]$installState.installManifestSha256)) {
            throw 'invalid installation state'
        }
    } catch {
        throw "Source bootstrap installation state is invalid: $installStatePath"
    }
    if (-not (Test-Path -LiteralPath $corePython -PathType Leaf)) {
        throw "Distributed core runtime is missing: $corePython"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $runtimeRoot 'manifests\runtimes\core.json') -PathType Leaf)) {
        throw 'Distributed core runtime manifest is missing'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot 'index.html') -PathType Leaf)) {
        throw "Built frontend is missing: $frontendRoot"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $resourceRoot 'defaults.json') -PathType Leaf)) {
        throw "Project-local resource library is missing: $resourceRoot"
    }
    if (-not (Test-Path -LiteralPath $hostScript -PathType Leaf)) {
        throw "WebUI host script is missing: $hostScript"
    }
    & $corePython -B -I -m anima_core --check-runtime
    if ($LASTEXITCODE -ne 0) { throw 'Distributed core runtime verification failed' }
}

function Get-Listener([int]$LocalPort) {
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $LocalPort -ErrorAction SilentlyContinue)
    if ($listeners.Count -gt 0) { return $listeners }
    $pattern = '^\s*TCP\s+127\.0\.0\.1:{0}\s+\S+\s+LISTENING\s+(\d+)\s*$' -f $LocalPort
    @(
        netstat -ano -p tcp 2>$null |
            ForEach-Object {
                if ($_ -match $pattern) { [pscustomobject]@{ OwningProcess = [int]$Matches[1] } }
            }
    )
}

function Test-HealthyWebUi([int]$LocalPort) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$LocalPort/api/health" -TimeoutSec 2
        return $health.status -eq 'ok' -and $health.protocolVersion -eq '1.0'
    } catch {
        return $false
    }
}

function Read-StateFile([string]$Path) {
    try { return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json }
    catch { throw "WebUI state file is invalid: $Path" }
}

function Read-State {
    if (Test-Path -LiteralPath $statePath -PathType Leaf) {
        return (Read-StateFile $statePath)
    }
    return $null
}

function Test-ProcessAlive([int]$ProcessId) {
    try {
        $process = Get-Process -Id $ProcessId -ErrorAction Stop
        return -not $process.HasExited
    } catch {
        return $false
    }
}

function Get-StateBackendPid([object]$State) {
    if ($null -ne $State.PSObject.Properties['backendPid']) { return [int]$State.backendPid }
    if ($null -ne $State.PSObject.Properties['pid']) { return [int]$State.pid }
    return 0
}

function Write-State([object]$State) {
    New-Item -ItemType Directory -Force -Path $launcherRoot | Out-Null
    $temporary = "$statePath.part"
    $encoding = New-Object -TypeName System.Text.UTF8Encoding -ArgumentList $false
    [System.IO.File]::WriteAllText($temporary, ($State | ConvertTo-Json -Compress), $encoding)
    Move-Item -LiteralPath $temporary -Destination $statePath -Force
}

function New-Token {
    $bytes = New-Object byte[] 32
    $rng = New-Object System.Security.Cryptography.RNGCryptoServiceProvider
    try { $rng.GetBytes($bytes) }
    finally { $rng.Dispose() }
    return ([BitConverter]::ToString($bytes).Replace('-', '')).ToLowerInvariant()
}

function Start-WebUi {
    Assert-ReleaseBundle
    $state = Read-State
    $listeners = @(Get-Listener $Port)
    if ($listeners.Count -gt 0) {
        $statePid = if ($null -ne $state) { Get-StateBackendPid $state } else { 0 }
        if ($null -ne $state -and $state.installRoot -eq $projectRoot -and $state.port -eq $Port -and $statePid -gt 0 -and ($listeners | Where-Object { $_.OwningProcess -eq $statePid })) {
            Start-Process "http://127.0.0.1:$Port/"
            return
        }
        if (Test-HealthyWebUi $Port) {
            Start-Process "http://127.0.0.1:$Port/"
            return
        }
        throw "Port $Port is already owned by another process; it will not be stopped automatically"
    }
    if ($null -ne $state) { Remove-Item -LiteralPath $statePath -Force }
    New-Item -ItemType Directory -Force -Path $launcherRoot | Out-Null
    $token = New-Token
    $hostCommand = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{0}" -ProjectRoot "{1}" -RuntimeRoot "{2}" -CorePython "{3}" -FrontendRoot "{4}" -ResourceRoot "{5}" -LauncherRoot "{6}" -StatePath "{7}" -InstanceId "{8}" -HostPid {9} -Port {10} -Token {11}' -f $hostScript, $projectRoot, $runtimeRoot, $corePython, $frontendRoot, $resourceRoot, $launcherRoot, $statePath, $instanceId, 0, $Port, $token
    $hostProcess = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/d', '/c', $hostCommand) -WorkingDirectory $projectRoot -WindowStyle Normal -PassThru
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
            $state = Read-State
            $backendPid = if ($null -ne $state) { Get-StateBackendPid $state } else { 0 }
            if ($health.status -eq 'ok' -and $backendPid -gt 0 -and @(Get-Listener $Port | Where-Object { $_.OwningProcess -eq $backendPid }).Count -gt 0) {
                Start-Process "http://127.0.0.1:$Port/"
                return
            }
        } catch {}
        if (-not (Test-ProcessAlive $hostProcess.Id) -and (Get-Listener $Port).Count -eq 0) { break }
        Start-Sleep -Milliseconds 250
    }
    throw 'WebUI backend did not become healthy within 30 seconds; inspect the visible host CMD and launcher logs'
}

function Stop-WebUi {
    $state = Read-State
    if ($null -eq $state) {
        if ((@(Get-Listener $Port).Count -eq 0)) { return }
        throw "Port $Port is listening but this installation has no matching state file"
    }
    $stateBelongsToInstallation = (
        ([string]$state.installRoot -eq [string]$projectRoot) -and
        (([int]$state.port) -eq $Port) -and
        (([string]$state.token) -match '^[0-9a-f]{64}$') -and
        ((Get-StateBackendPid $state) -gt 0)
    )
    if (-not $stateBelongsToInstallation) {
        throw 'WebUI state file does not belong to this installation'
    }
    $listeners = @(Get-Listener $Port)
    if ($listeners.Count -eq 0) {
        Remove-Item -LiteralPath $statePath -Force
        return
    }
    $backendPid = Get-StateBackendPid $state
    if (-not ($listeners | Where-Object { $_.OwningProcess -eq $backendPid })) {
        throw "Port $Port is no longer owned by this installation"
    }
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/api/application/shutdown" -ContentType 'application/json' -Body (@{ token = [string]$state.token } | ConvertTo-Json -Compress) -TimeoutSec 10 | Out-Null
    $deadline = [DateTime]::UtcNow.AddSeconds(60)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ((@(Get-Listener $Port).Count -eq 0)) {
            Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
            return
        }
        Start-Sleep -Milliseconds 250
    }
    throw "WebUI backend did not release port $Port within 60 seconds; it was not force-terminated"
}

switch ($Action) {
    'Install' {
        Assert-ReleaseBundle
        New-Item -ItemType Directory -Force -Path $launcherRoot | Out-Null
        Write-Output 'Anima source bootstrap installation is ready.'
    }
    'Start' { Start-WebUi }
    'Stop' { Stop-WebUi }
}
