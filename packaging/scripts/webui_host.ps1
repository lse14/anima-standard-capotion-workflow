[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [Parameter(Mandatory = $true)][string]$RuntimeRoot,
    [Parameter(Mandatory = $true)][string]$CorePython,
    [Parameter(Mandatory = $true)][string]$FrontendRoot,
    [Parameter(Mandatory = $true)][string]$ResourceRoot,
    [Parameter(Mandatory = $true)][string]$LauncherRoot,
    [Parameter(Mandatory = $true)][string]$StatePath,
    [Parameter(Mandatory = $true)][string]$InstanceId,
    [Parameter(Mandatory = $true)][int]$HostPid,
    [Parameter(Mandatory = $true)][ValidateRange(1024, 65535)][int]$Port,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$Token
)

$ErrorActionPreference = 'Stop'

if ($HostPid -le 0) {
    $currentProcess = Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f $PID)
    if ($null -eq $currentProcess -or [int]$currentProcess.ParentProcessId -le 0) {
        throw 'Could not determine the visible host CMD process'
    }
    $HostPid = [int]$currentProcess.ParentProcessId
}

function Write-State([object]$State) {
    New-Item -ItemType Directory -Force -Path $LauncherRoot | Out-Null
    $temporary = "$StatePath.part"
    $encoding = New-Object -TypeName System.Text.UTF8Encoding -ArgumentList $false
    [System.IO.File]::WriteAllText($temporary, ($State | ConvertTo-Json -Compress), $encoding)
    Move-Item -LiteralPath $temporary -Destination $StatePath -Force
}

function Test-ProcessAlive([int]$ProcessId) {
    try {
        $process = Get-Process -Id $ProcessId -ErrorAction Stop
        return -not $process.HasExited
    } catch {
        return $false
    }
}

function Read-StateFile {
    if (-not (Test-Path -LiteralPath $StatePath -PathType Leaf)) { return $null }
    try { return Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json }
    catch { return $null }
}

function Request-Shutdown([string]$ShutdownToken) {
    try {
        Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/api/application/shutdown" `
            -ContentType 'application/json' `
            -Body (@{ token = $ShutdownToken } | ConvertTo-Json -Compress) -TimeoutSec 5 | Out-Null
    } catch {
        Write-Host ("WebUI shutdown request was not accepted: {0}" -f $_.Exception.Message)
    }
}

function Remove-OwnedState([int]$BackendPid) {
    $state = Read-StateFile
    if ($null -ne $state -and [int]$state.backendPid -eq $BackendPid -and [int]$state.hostPid -eq $HostPid) {
        Remove-Item -LiteralPath $StatePath -Force -ErrorAction SilentlyContinue
    }
}

$stdout = Join-Path $LauncherRoot ("webui-{0}.stdout.log" -f $InstanceId)
$stderr = Join-Path $LauncherRoot ("webui-{0}.stderr.log" -f $InstanceId)
$arguments = '-B -I -m anima_core --port {0} --static-root "{1}" --resource-root "{2}" --shutdown-token {3} --host-pid {4}' -f $Port, $FrontendRoot, $ResourceRoot, $Token, $HostPid
$backend = $null

try {
    Write-Host ("Starting Anima WebUI backend in host CMD (host PID {0})..." -f $HostPid)
    $backend = Start-Process -FilePath $CorePython -ArgumentList $arguments -WorkingDirectory $RuntimeRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $state = [ordered]@{
        schemaVersion = 2
        installRoot = $ProjectRoot
        runtimeRoot = $RuntimeRoot
        hostPid = $HostPid
        backendPid = $backend.Id
        pid = $backend.Id
        port = $Port
        token = $Token
        instanceId = $InstanceId
    }
    Write-State $state
    Write-Host ("WebUI backend PID {0} is supervised by this visible CMD window." -f $backend.Id)
    while (-not $backend.HasExited) {
        Start-Sleep -Milliseconds 500
        $backend.Refresh()
    }
    if ([int]$backend.ExitCode -ne 0) {
        Write-Host ("WebUI backend exited with code {0}; see {1} and {2}." -f $backend.ExitCode, $stdout, $stderr)
    }
    exit [int]$backend.ExitCode
} catch {
    Write-Host ("WebUI host failed: {0}" -f $_.Exception.Message)
    if ($null -ne $backend -and (Test-ProcessAlive $backend.Id)) { Request-Shutdown $Token }
    exit 1
} finally {
    if ($null -ne $backend -and (Test-ProcessAlive $backend.Id)) {
        Request-Shutdown $Token
        $deadline = [DateTime]::UtcNow.AddSeconds(20)
        while ([DateTime]::UtcNow -lt $deadline -and (Test-ProcessAlive $backend.Id)) {
            Start-Sleep -Milliseconds 250
            $backend.Refresh()
        }
    }
    if ($null -ne $backend) { Remove-OwnedState $backend.Id }
}
