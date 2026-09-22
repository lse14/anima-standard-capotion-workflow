[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$ProjectRoot,
    [switch]$IncludePolicy,
    [switch]$IncludeQualityStack
)

$ErrorActionPreference = 'Stop'

# The release build replaces this only after publishing the exact matching inventory.
$ExpectedInstallManifestSha256 = 'd46ef0719184b6ae9710119850940ed46a11d9259c212634b681fc7a8c05f738'
$script:projectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$script:runtimeBuildRoot = Join-Path $script:projectRoot '.runtime-build'
$script:logPath = $null
$script:utf8NoBom = New-Object -TypeName System.Text.UTF8Encoding -ArgumentList $false
$script:bootstrapComplete = $null
$script:bootstrapStage = $null

function Test-ReparsePoint([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $item = Get-Item -LiteralPath $Path -Force
    return (([System.IO.FileAttributes]$item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)
}

function Get-ProjectPath([string]$Path) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $rootPrefix = $script:projectRoot.TrimEnd('\') + '\'
    if (-not $fullPath.Equals($script:projectRoot, [System.StringComparison]::OrdinalIgnoreCase) -and
        -not $fullPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Installer path escapes the project root: $fullPath"
    }
    $relative = $fullPath.Substring($script:projectRoot.Length).TrimStart('\')
    $current = $script:projectRoot
    if (Test-ReparsePoint $current) { throw "Project root is a reparse point: $current" }
    foreach ($part in ($relative -split '\\')) {
        if ([string]::IsNullOrEmpty($part)) { continue }
        $current = Join-Path $current $part
        if (-not (Test-Path -LiteralPath $current)) { break }
        if (Test-ReparsePoint $current) { throw "Installer path contains a reparse point: $current" }
    }
    return $fullPath
}

function New-ProjectDirectory([string]$Path) {
    $fullPath = Get-ProjectPath $Path
    New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    if (Test-ReparsePoint $fullPath) { throw "Installer directory is a reparse point: $fullPath" }
    return $fullPath
}

function Write-InstallLog([string]$Message) {
    $line = ('{0:u} {1}' -f [DateTime]::UtcNow, $Message)
    if ($null -ne $script:logPath) {
        [System.IO.File]::AppendAllText($script:logPath, $line + [Environment]::NewLine, $script:utf8NoBom)
    }
    Write-Host $Message
}

function Invoke-DesktopControlStart([string]$DesktopControl, [int]$Attempt) {
    $stdoutPath = "$script:logPath.webui-$Attempt.stdout.log"
    $stderrPath = "$script:logPath.webui-$Attempt.stderr.log"
    $arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f $DesktopControl), '-Action', 'Start')
    $process = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    $process.WaitForExit()
    return [pscustomobject]@{ exitCode = $process.ExitCode; stdoutPath = $stdoutPath; stderrPath = $stderrPath }
}

function Start-InstalledWebUi([string]$DesktopControl) {
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        $result = Invoke-DesktopControlStart $DesktopControl $attempt
        if ([int]$result.exitCode -eq 0) {
            Write-InstallLog ("WebUI start attempt {0} succeeded; stdout: {1}; stderr: {2}" -f $attempt, $result.stdoutPath, $result.stderrPath)
            return
        }
        foreach ($path in @($result.stdoutPath, $result.stderrPath)) {
            foreach ($line in @(Get-Content -LiteralPath $path -ErrorAction SilentlyContinue)) {
                Write-InstallLog ("WebUI start attempt {0}: {1}" -f $attempt, [string]$line)
            }
        }
        if ($attempt -lt 2) {
            Write-InstallLog ("WebUI start attempt {0} failed with exit code {1}; retrying" -f $attempt, $result.exitCode)
            Start-Sleep -Seconds 2
        }
    }
    throw "WebUI failed to start after 2 attempts; see $script:logPath"
}

function Get-Sha256Hex([string]$Path) {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return [System.BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose(); $stream.Dispose() }
}

function Get-RequiredProperty([object]$Value, [string]$Name) {
    $property = $Value.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        throw "Frozen install manifest is missing required field: $Name"
    }
    return $property.Value
}

function Assert-SafeRelativePath([string]$RelativePath, [switch]$AllowDirectory) {
    $normalized = $RelativePath.Replace('/', '\')
    if ($AllowDirectory) { $normalized = $normalized.TrimEnd('\') }
    if ([string]::IsNullOrWhiteSpace($normalized) -or
        $normalized.StartsWith('\') -or
        $normalized -match '^[A-Za-z]:' -or
        [System.IO.Path]::IsPathRooted($normalized)) {
        throw "Frozen install manifest has an unsafe relative path: $RelativePath"
    }
    $reserved = @('CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9')
    foreach ($part in ($normalized -split '\\')) {
        $device = ($part -split '\.')[0].ToUpperInvariant()
        if ([string]::IsNullOrEmpty($part) -or $part -eq '.' -or $part -eq '..' -or
            $part.Contains(':') -or $part.EndsWith('.') -or $part.EndsWith(' ') -or $reserved -contains $device) {
            throw "Frozen install manifest has an unsafe relative path: $RelativePath"
        }
    }
    return $normalized
}

function Assert-ApprovedUri([string]$Url, [object[]]$AllowedHosts) {
    try { $uri = New-Object System.Uri($Url) }
    catch { throw "Bootstrap artifact URL is invalid: $Url" }
    $hosts = @($AllowedHosts | ForEach-Object { ([string]$_).ToLowerInvariant() })
    if ($uri.Scheme -ne 'https' -or [string]::IsNullOrEmpty($uri.Host) -or $hosts -notcontains $uri.Host.ToLowerInvariant()) {
        throw "Bootstrap artifact URL is not an approved HTTPS host: $Url"
    }
    return $uri
}

function Format-ManualDownloadMessage([object]$Artifact, [string]$Reason) {
    return "${Reason}`nOfficial URL: $($Artifact.url)`nTarget file: $($Artifact.relativePath)`nExpected size: $($Artifact.sizeBytes)`nSHA-256: $($Artifact.sha256)"
}

function Test-VerifiedArtifact([string]$Path, [object]$Artifact) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Path -Force).Length -ne [Int64]$Artifact.sizeBytes) { return $false }
    return ((Get-Sha256Hex $Path) -eq ([string]$Artifact.sha256).ToLowerInvariant())
}

function Open-ApprovedResponse([string]$Url, [object[]]$AllowedHosts, [Int64]$Offset) {
    $current = Assert-ApprovedUri $Url $AllowedHosts
    for ($redirect = 0; $redirect -lt 6; $redirect++) {
        $request = [System.Net.HttpWebRequest]::Create($current)
        $request.Method = 'GET'
        $request.AllowAutoRedirect = $false
        $request.Timeout = 60000
        $request.ReadWriteTimeout = 60000
        if ($Offset -gt 0) { $request.AddRange($Offset) }
        try {
            $response = [System.Net.HttpWebResponse]$request.GetResponse()
        } catch [System.Net.WebException] {
            if ($null -eq $_.Exception.Response) { throw }
            $response = [System.Net.HttpWebResponse]$_.Exception.Response
        }
        $status = [Int32]$response.StatusCode
        if ($status -in 301, 302, 303, 307, 308) {
            $location = $response.Headers['Location']
            $response.Close()
            if ([string]::IsNullOrWhiteSpace($location)) { throw 'Bootstrap artifact redirect has no Location header' }
            $current = New-Object System.Uri($current, $location)
            [void](Assert-ApprovedUri $current.AbsoluteUri $AllowedHosts)
            continue
        }
        [void](Assert-ApprovedUri $response.ResponseUri.AbsoluteUri $AllowedHosts)
        return $response
    }
    throw 'Bootstrap artifact exceeded the redirect limit'
}

function Get-VerifiedBootstrapArtifact([object]$Artifact, [string]$CacheRoot) {
    $allowedHosts = @((Get-RequiredProperty $Artifact 'allowedHosts'))
    $url = [string](Get-RequiredProperty $Artifact 'url')
    $sha256 = [string](Get-RequiredProperty $Artifact 'sha256')
    $sizeBytes = [Int64](Get-RequiredProperty $Artifact 'sizeBytes')
    [void](Assert-SafeRelativePath ([string](Get-RequiredProperty $Artifact 'relativePath')))
    if ($sha256 -notmatch '^[a-f0-9]{64}$' -or $sizeBytes -le 0) { throw 'Bootstrap artifact identity is invalid' }
    [void](Assert-ApprovedUri $url $allowedHosts)
    $complete = Get-ProjectPath (Join-Path $CacheRoot $sha256)
    $partial = Get-ProjectPath ($complete + '.partial')
    $script:bootstrapComplete = $complete
    if (Test-Path -LiteralPath $complete) {
        if (Test-VerifiedArtifact $complete $Artifact) { return $complete }
        Remove-Item -LiteralPath $complete -Force
    }
    if ((Test-Path -LiteralPath $partial) -and (Get-Item -LiteralPath $partial -Force).Length -ge $sizeBytes) {
        Remove-Item -LiteralPath $partial -Force
    }
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $offset = if (Test-Path -LiteralPath $partial) { [Int64](Get-Item -LiteralPath $partial -Force).Length } else { [Int64]0 }
        $response = $null
        try {
            Write-InstallLog ("Downloading bootstrap artifact (attempt {0}/3, offset {1} bytes)" -f $attempt, $offset)
            $response = Open-ApprovedResponse $url $allowedHosts $offset
            $status = [Int32]$response.StatusCode
            if ($offset -gt 0 -and $status -eq 200) {
                $response.Close()
                Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
                continue
            }
            if ($offset -gt 0 -and $status -ne 206) { throw "Range request returned HTTP $status" }
            if ($offset -eq 0 -and $status -ne 200) {
                if ($status -in 401, 403, 404) { throw (Format-ManualDownloadMessage $Artifact "Bootstrap download failed with HTTP $status") }
                throw "Bootstrap download returned HTTP $status"
            }
            if ($offset -gt 0) {
                $range = [string]$response.Headers['Content-Range']
                if ($range -notmatch ('^bytes {0}-\d+/{1}$' -f $offset, $sizeBytes)) { throw 'Bootstrap Range response does not match the frozen artifact' }
            }
            $mode = if ($offset -gt 0) { [System.IO.FileMode]::Append } else { [System.IO.FileMode]::Create }
            $target = [System.IO.File]::Open($partial, $mode, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
            try {
                $source = $response.GetResponseStream()
                try {
                    $buffer = New-Object byte[] 1048576
                    while (($read = $source.Read($buffer, 0, $buffer.Length)) -gt 0) {
                        $target.Write($buffer, 0, $read)
                        if ($target.Length -gt $sizeBytes) { throw 'Bootstrap download exceeds the frozen artifact size' }
                    }
                } finally { $source.Dispose() }
            } finally { $target.Dispose() }
        } catch {
            if ($null -ne $response) { $response.Close() }
            if ((Test-Path -LiteralPath $partial) -and (Get-Item -LiteralPath $partial -Force).Length -ge $sizeBytes) {
                $actual = Get-Sha256Hex $partial
                $actualSize = (Get-Item -LiteralPath $partial -Force).Length
                Remove-Item -LiteralPath $partial -Force
                throw (Format-ManualDownloadMessage $Artifact "Bootstrap checksum mismatch: received $actualSize bytes with SHA-256 $actual")
            }
            if ($attempt -eq 3) { throw (Format-ManualDownloadMessage $Artifact 'Bootstrap download failed after bounded retries; a resumable partial may remain') }
            continue
        }
        if ($null -ne $response) { $response.Close() }
        if (Test-VerifiedArtifact $partial $Artifact) {
            [System.IO.File]::Move($partial, $complete)
            return $complete
        }
        if ((Test-Path -LiteralPath $partial) -and (Get-Item -LiteralPath $partial -Force).Length -ge $sizeBytes) {
            $actual = Get-Sha256Hex $partial
            $actualSize = (Get-Item -LiteralPath $partial -Force).Length
            Remove-Item -LiteralPath $partial -Force
            throw (Format-ManualDownloadMessage $Artifact "Bootstrap checksum mismatch: received $actualSize bytes with SHA-256 $actual")
        }
    }
    throw (Format-ManualDownloadMessage $Artifact 'Bootstrap download did not produce a complete artifact; a resumable partial may remain')
}

function Expand-SafeBootstrapArchive([string]$ArchivePath, [string]$Destination) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $destination = Get-ProjectPath $Destination
    if (Test-Path -LiteralPath $destination) { throw "Bootstrap staging directory already exists: $destination" }
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    $archive = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        foreach ($entry in $archive.Entries) {
            $isDirectory = $entry.FullName.EndsWith('/')
            $relative = Assert-SafeRelativePath $entry.FullName -AllowDirectory:$isDirectory
            if (-not $seen.Add($relative)) { throw "Bootstrap archive has a case-colliding member: $relative" }
            $entryType = (([Int64]$entry.ExternalAttributes -shr 16) -band 0xF000)
            if ($entryType -eq 0xA000 -or $entryType -notin 0, 0x8000, 0x4000 -or
                ($entryType -eq 0x4000 -and -not $isDirectory) -or ($entryType -eq 0x8000 -and $isDirectory)) {
                throw "Bootstrap archive has an unsupported member type: $relative"
            }
        }
        New-ProjectDirectory $destination | Out-Null
        foreach ($entry in $archive.Entries) {
            $isDirectory = $entry.FullName.EndsWith('/')
            $relative = Assert-SafeRelativePath $entry.FullName -AllowDirectory:$isDirectory
            $target = Get-ProjectPath (Join-Path $destination ($relative.Replace('\', [System.IO.Path]::DirectorySeparatorChar)))
            if ($isDirectory) {
                New-ProjectDirectory $target | Out-Null
                continue
            }
            New-ProjectDirectory (Split-Path -Parent $target) | Out-Null
            $output = [System.IO.File]::Open($target, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
            try {
                $input = $entry.Open()
                try { $input.CopyTo($output) }
                finally { $input.Dispose() }
            } finally { $output.Dispose() }
        }
    } catch {
        if ((Test-Path -LiteralPath $destination) -and -not (Test-ReparsePoint $destination)) {
            [System.IO.Directory]::Delete($destination, $true)
        }
        throw
    } finally { $archive.Dispose() }
}

function Test-NvidiaAvailable {
    $command = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $command) { return $false }
    $result = & $command.Source '--query-gpu=name' '--format=csv,noheader' 2>$null
    return $LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace(($result -join ''))
}

function Get-RequiredPeakBytes([object]$Manifest, [bool]$NvidiaAvailable, [bool]$IncludePolicy, [bool]$IncludeQualityStack) {
    $bootstrap = Get-RequiredProperty $Manifest 'bootstrap'
    $total = [Int64](Get-RequiredProperty $bootstrap 'peakBytes') + [Int64](Get-RequiredProperty (Get-RequiredProperty $bootstrap 'artifact') 'sizeBytes')
    foreach ($component in @((Get-RequiredProperty $Manifest 'components'))) {
        $componentId = [string]$component.componentId
        if ($componentId -in @('ocr-cpu', 'ocr-gpu', 'ocr-models')) { continue }
        if ($componentId -eq 'policy' -and -not $IncludePolicy) { continue }
        if ($componentId -eq 'quality-stack' -and -not $IncludeQualityStack) { continue }
        $variants = Get-RequiredProperty $component 'variants'
        $variantName = if ($NvidiaAvailable -and $null -ne $variants.PSObject.Properties['cuda']) {
            'cuda'
        } elseif ($null -ne $variants.PSObject.Properties['cpu']) {
            'cpu'
        } elseif ($null -ne $variants.PSObject.Properties['shared']) {
            'shared'
        } elseif (-not [bool]$component.required) {
            continue
        } else {
            throw "Frozen install manifest has no usable variant: $($component.componentId)"
        }
        $variant = Get-RequiredProperty $variants $variantName
        $total += [Int64](Get-RequiredProperty $variant 'peakBytes')
        foreach ($artifact in @((Get-RequiredProperty $variant 'artifacts'))) {
            $total += [Int64](Get-RequiredProperty $artifact 'sizeBytes')
        }
    }
    return $total
}

function Assert-FreeSpace([Int64]$RequiredBytes) {
    $driveLetter = ([System.IO.Path]::GetPathRoot($script:projectRoot)).Substring(0, 1)
    $volume = Get-Volume -DriveLetter $driveLetter -ErrorAction Stop
    if ([Int64]$volume.SizeRemaining -lt $RequiredBytes) {
        throw ("Insufficient disk space before download: need {0} bytes, available {1} bytes on {2}:" -f $RequiredBytes, $volume.SizeRemaining, $driveLetter)
    }
}

function Clear-BootstrapFailureArtifacts {
    $installerRoot = Get-ProjectPath (Join-Path $script:runtimeBuildRoot 'source-bootstrap')
    $stagingRoot = Get-ProjectPath (Join-Path $installerRoot 'staging')
    if (Test-Path -LiteralPath $stagingRoot -PathType Container) {
        if (Test-ReparsePoint $stagingRoot) { throw "Bootstrap staging is a reparse point: $stagingRoot" }
        [System.IO.Directory]::Delete($stagingRoot, $true)
    }
    $bootstrapRoot = Get-ProjectPath (Join-Path $installerRoot 'bootstrap')
    if (Test-Path -LiteralPath $bootstrapRoot -PathType Container) {
        if (Test-ReparsePoint $bootstrapRoot) { throw "Bootstrap staging is a reparse point: $bootstrapRoot" }
        [System.IO.Directory]::Delete($bootstrapRoot, $true)
    }
    $transactionsRoot = Get-ProjectPath (Join-Path $installerRoot 'transactions')
    if (Test-Path -LiteralPath $transactionsRoot -PathType Container) {
        if (Test-ReparsePoint $transactionsRoot) { throw "Bootstrap transactions are a reparse point: $transactionsRoot" }
        [System.IO.Directory]::Delete($transactionsRoot, $true)
    }
    $cacheRoot = Get-ProjectPath (Join-Path $installerRoot 'cache')
    if (Test-Path -LiteralPath $cacheRoot -PathType Container) {
        if (Test-ReparsePoint $cacheRoot) { throw "Bootstrap cache is a reparse point: $cacheRoot" }
        [System.IO.Directory]::Delete($cacheRoot, $true)
    }
    # Failed source bootstrap attempts leave only the installed runtimes and log evidence.
}

function Clear-BootstrapSuccessArtifacts {
    $installerRoot = Get-ProjectPath (Join-Path $script:runtimeBuildRoot 'source-bootstrap')
    foreach ($name in @('bootstrap', 'cache', 'staging', 'transactions', 'build-cache')) {
        $target = Get-ProjectPath (Join-Path $installerRoot $name)
        if (-not (Test-Path -LiteralPath $target)) { continue }
        if (-not (Test-Path -LiteralPath $target -PathType Container) -or (Test-ReparsePoint $target)) {
            throw "Bootstrap cleanup target is unsafe: $target"
        }
        [System.IO.Directory]::Delete($target, $true)
    }
}

try {
    if ($env:OS -ne 'Windows_NT') { throw 'Source bootstrap requires Windows 10 or Windows 11' }
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'Source bootstrap requires a 64-bit Windows operating system' }
    if (-not (Test-Path -LiteralPath $script:projectRoot -PathType Container)) { throw "Project root does not exist: $script:projectRoot" }
    if (Test-ReparsePoint $script:projectRoot) { throw "Project root must not be a reparse point: $script:projectRoot" }

    New-ProjectDirectory $script:runtimeBuildRoot | Out-Null
    $logRoot = New-ProjectDirectory (Join-Path $script:runtimeBuildRoot 'logs')
    $script:logPath = Get-ProjectPath (Join-Path $logRoot ('source-bootstrap-{0}.log' -f [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')))
    [System.IO.File]::WriteAllText($script:logPath, '', $script:utf8NoBom)
    $writeProbe = Get-ProjectPath (Join-Path $script:runtimeBuildRoot ('.write-probe-{0}' -f [Guid]::NewGuid().ToString('N')))
    [System.IO.File]::WriteAllText($writeProbe, 'ok', $script:utf8NoBom)
    Remove-Item -LiteralPath $writeProbe -Force

    $manifestPath = Get-ProjectPath (Join-Path $script:projectRoot 'packaging\installer\install-manifest.json')
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "Frozen install manifest is missing: $manifestPath" }
    if ($ExpectedInstallManifestSha256 -notmatch '^[a-f0-9]{64}$') {
        throw 'This source snapshot has no published source-bootstrap manifest identity; obtain a matching released source ZIP.'
    }
    $actualManifestSha256 = Get-Sha256Hex $manifestPath
    if ($actualManifestSha256 -ne $ExpectedInstallManifestSha256) { throw 'Frozen install manifest SHA-256 does not match the source release identity' }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $nvidiaAvailable = Test-NvidiaAvailable
    $accelerator = if ($nvidiaAvailable) { 'nvidia' } else { 'cpu' }
    $requiredBytes = Get-RequiredPeakBytes $manifest $nvidiaAvailable $IncludePolicy.IsPresent $IncludeQualityStack.IsPresent
    Assert-FreeSpace $requiredBytes
    Write-InstallLog ("Selected accelerator route: {0}; required peak space: {1} bytes" -f $accelerator, $requiredBytes)

    $installerRoot = New-ProjectDirectory (Join-Path $script:runtimeBuildRoot 'source-bootstrap')
    $cacheRoot = New-ProjectDirectory (Join-Path $installerRoot 'cache')
    $bootstrapRoot = New-ProjectDirectory (Join-Path $installerRoot 'bootstrap')
    $bootstrap = Get-RequiredProperty $manifest 'bootstrap'
    $archive = Get-VerifiedBootstrapArtifact (Get-RequiredProperty $bootstrap 'artifact') $cacheRoot
    $script:bootstrapStage = Get-ProjectPath (Join-Path $bootstrapRoot ([Guid]::NewGuid().ToString('N')))
    Expand-SafeBootstrapArchive $archive $script:bootstrapStage
    $entryRelativePath = Assert-SafeRelativePath ([string](Get-RequiredProperty $bootstrap 'entryRelativePath'))
    $bootstrapPython = Get-ProjectPath (Join-Path $script:bootstrapStage ($entryRelativePath.Replace('\', [System.IO.Path]::DirectorySeparatorChar)))
    if (-not (Test-Path -LiteralPath $bootstrapPython -PathType Leaf)) { throw "Bootstrap Python entry is missing after extraction: $entryRelativePath" }
    $installerScript = Get-ProjectPath (Join-Path $script:projectRoot 'packaging\installer\install.py')
    if (-not (Test-Path -LiteralPath $installerScript -PathType Leaf)) { throw "Source bootstrap installer is missing: $installerScript" }
    Write-InstallLog 'Starting the standard-library installer.'
    $installerArguments = @('--project-root', $script:projectRoot, '--manifest', $manifestPath, '--manifest-sha256', $actualManifestSha256, '--accelerator', $accelerator, '--bootstrap-runtime', $script:bootstrapStage)
    if ($IncludePolicy) { $installerArguments += '--include-policy' }
    if ($IncludeQualityStack) { $installerArguments += '--include-quality-stack' }
    & $bootstrapPython -B -I -u $installerScript @installerArguments 2>&1 |
        ForEach-Object { Write-InstallLog ([string]$_) }
    $installerExitCode = $LASTEXITCODE
    if ($installerExitCode -ne 0) { throw "Standard-library installer failed with exit code $installerExitCode" }
    $guidePath = Get-ProjectPath (Join-Path $script:projectRoot 'OCR_MODEL_DOWNLOAD.md')
    if (-not (Test-Path -LiteralPath $guidePath -PathType Leaf)) { throw "OCR model download guide is missing: $guidePath" }
    Write-InstallLog ("OCR model guide: {0}" -f $guidePath)
    $desktopControl = Get-ProjectPath (Join-Path $script:projectRoot 'packaging\scripts\desktop_control.ps1')
    if (-not (Test-Path -LiteralPath $desktopControl -PathType Leaf)) { throw "WebUI control script is missing: $desktopControl" }
    Write-InstallLog 'Starting WebUI.'
    Start-InstalledWebUi $desktopControl
    Clear-BootstrapSuccessArtifacts
    Write-InstallLog 'Source bootstrap completed successfully.'
} catch {
    $originalError = $_
    try {
        Clear-BootstrapFailureArtifacts
    } catch {
        if ($null -ne $script:logPath) {
            Write-InstallLog ("Bootstrap failure cleanup also failed: {0}" -f $_.Exception.Message)
        }
    }
    $message = $originalError.Exception.Message
    if ($null -ne $script:logPath) { Write-InstallLog ("FAILED: $message") }
    Write-Error $message
    exit 1
}
