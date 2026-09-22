[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$ProjectRoot,
    [Parameter(Mandatory = $true)][ValidateSet('Cpu', 'Nvidia')][string]$Scenario,
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Get-CommandEvidence([string]$Name) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    return [ordered]@{
        name = $Name
        found = ($null -ne $command)
        path = if ($null -eq $command) { $null } else { [string]$command.Source }
    }
}

function Get-WindowsSdkEvidence {
    $paths = @(
        (Join-Path ${env:ProgramFiles(x86)} 'Windows Kits'),
        (Join-Path $env:ProgramFiles 'Windows Kits')
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique
    $found = @($paths | Where-Object { Test-Path -LiteralPath $_ -PathType Container })
    return [ordered]@{ checkedPaths = $paths; foundPaths = $found; found = ($found.Count -gt 0) }
}

function Write-AcceptanceResult([string]$Root, [hashtable]$Result) {
    $directory = Join-Path $Root '.runtime-build\acceptance'
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    $path = Join-Path $directory ('source-bootstrap-{0}-{1}.json' -f $Result.scenario.ToLowerInvariant(), [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
    [System.IO.File]::WriteAllText($path, ($Result | ConvertTo-Json -Depth 8), $utf8NoBom)
    return $path
}

function Get-InstallStateEvidence([string]$Root) {
    $path = Join-Path $Root '.runtime-build\manifests\install-state.json'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return [ordered]@{ path = $path; present = $false } }
    try {
        $state = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
        return [ordered]@{ path = $path; present = $true; accelerator = [string]$state.accelerator; components = $state.components }
    } catch {
        return [ordered]@{ path = $path; present = $true; parseError = $_.Exception.Message }
    }
}

function Assert-ScenarioInstallState([string]$ScenarioName, [object]$State) {
    if (-not [bool]$State.present -or $null -ne $State.parseError) { throw 'Install state is missing or invalid after installer completion' }
    if ($ScenarioName -eq 'Cpu') {
        if ([string]$State.accelerator -ne 'cpu') { throw 'CPU acceptance requires an install state with accelerator cpu' }
        foreach ($component in $State.components.PSObject.Properties) {
            if ([string]$component.Value.variant -eq 'cuda' -or [string]$component.Name -eq 'ocr-gpu') {
                throw 'CPU acceptance install state contains a GPU component'
            }
        }
        return
    }
    if ([string]$State.accelerator -ne 'nvidia') { throw 'NVIDIA acceptance requires an install state with accelerator nvidia' }
    foreach ($componentId in @('caption-e621', 'policy')) {
        $component = $State.components.PSObject.Properties[$componentId]
        if ($null -eq $component -or [string]$component.Value.variant -ne 'cuda') {
            throw "NVIDIA acceptance is missing CUDA evidence for $componentId"
        }
    }
    $optionalGpu = $State.components.PSObject.Properties['ocr-gpu']
    if ($null -ne $optionalGpu -and [string]$optionalGpu.Value.variant -ne 'cuda') {
        throw 'NVIDIA acceptance contains OCR GPU evidence with a non-CUDA variant'
    }
}

$root = [System.IO.Path]::GetFullPath($ProjectRoot)
$installerInvoked = $false
$installationValidated = $false
$result = [ordered]@{
    schemaVersion = 1
    startedAtUtc = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
    scenario = $Scenario
    status = 'failed'
    projectRoot = $root
    preflightOnly = [bool]$PreflightOnly
    platform = [ordered]@{
        is64BitOperatingSystem = [Environment]::Is64BitOperatingSystem
        osVersion = [Environment]::OSVersion.VersionString
    }
    commands = @('python', 'py', 'node', 'npm', 'nvcc', 'cl') | ForEach-Object { Get-CommandEvidence $_ }
    windowsSdk = Get-WindowsSdkEvidence
    nvidia = Get-CommandEvidence 'nvidia-smi'
    sourceCommit = $null
    installManifestSha256 = $null
    installerExitCode = $null
    webUiStartExitCode = $null
    stopExitCode = $null
    installState = $null
    installerLogPaths = @()
    error = $null
}

try {
    if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw "Project root does not exist: $root" }
    if (Test-Path -LiteralPath (Join-Path $root '.git')) {
        $sourceCommit = git -C $root rev-parse HEAD 2>$null
        if ($LASTEXITCODE -eq 0) { $result.sourceCommit = ([string]$sourceCommit).Trim() }
    }
    $manifest = Join-Path $root 'packaging\installer\install-manifest.json'
    if (Test-Path -LiteralPath $manifest -PathType Leaf) {
        $result.installManifestSha256 = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $developerCommands = @($result.commands | Where-Object { [bool]$_.found })
    $sdkPresent = [bool]$result.windowsSdk.found
    if ($developerCommands.Count -gt 0 -or $sdkPresent) {
        $result.status = 'not-clean'
        $names = @($developerCommands | ForEach-Object { $_.name })
        if ($sdkPresent) { $names += 'Windows Kits' }
        Write-Output ('Clean-host preflight failed: detected ' + ($names -join ', '))
    } else {
        if (-not [Environment]::Is64BitOperatingSystem) { throw 'Clean-host preflight failed: Windows x64 is required' }
        if ($Scenario -eq 'Nvidia' -and -not [bool]$result.nvidia.found) { throw 'Clean-host preflight failed: NVIDIA scenario requires nvidia-smi' }
        if ($PreflightOnly) {
            $result.status = 'failed'
            Write-Output 'Clean-host preflight found no development tools; installer was not invoked because -PreflightOnly was specified.'
        } else {
            $installer = Join-Path $root 'Install-WebUI.bat'
            if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) { throw "Install-WebUI.bat is missing: $installer" }
            $installerInvoked = $true
            & $installer
            $result.installerExitCode = $LASTEXITCODE
            $result.webUiStartExitCode = $LASTEXITCODE
            if ($result.installerExitCode -ne 0) { throw "Install-WebUI.bat failed with exit code $($result.installerExitCode)" }
            $result.installState = Get-InstallStateEvidence $root
            Assert-ScenarioInstallState $Scenario $result.installState
            $result.installerLogPaths = @(Get-ChildItem -LiteralPath (Join-Path $root '.runtime-build\logs') -Filter 'source-bootstrap-*.log' -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
            $installationValidated = $true
        }
    }
} catch {
    $result.error = $_.Exception.Message
    if ($result.status -ne 'not-clean') { $result.status = 'failed' }
    Write-Error $result.error
} finally {
    if ($installerInvoked) {
        $stop = Join-Path $root 'Stop-WebUI.bat'
        if (-not (Test-Path -LiteralPath $stop -PathType Leaf)) {
            $result.error = "Stop-WebUI.bat is missing: $stop"
            $result.status = 'failed'
        } else {
            & $stop
            $result.stopExitCode = $LASTEXITCODE
            if ($result.stopExitCode -ne 0) {
                $result.error = "Stop-WebUI.bat failed with exit code $($result.stopExitCode)"
                $result.status = 'failed'
            } elseif ($installationValidated) {
                $result.status = 'passed'
            }
        }
    }
    $result.completedAtUtc = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
    $resultPath = Write-AcceptanceResult $root $result
    Write-Output ("Acceptance result: {0}" -f $resultPath)
}

if ($result.status -eq 'passed') { exit 0 }
exit 1
