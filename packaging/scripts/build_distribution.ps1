[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PythonSourceRoot,
    [Parameter(Mandatory = $true)][string]$BuildPython,
    [Parameter(Mandatory = $true)][string]$InstallRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$PythonSourceUrl,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PythonSourceSha256,
    [string]$WheelhouseRoot,
    [switch]$ReuseBaseRuntime,
    [ValidateSet('None', 'Cpu', 'Gpu')][string]$OcrComponents = 'None',
    [switch]$IncludeOcrPaddleGpu,
    [ValidateSet('None', 'Cpu', 'Cuda')][string]$PolicyComponents = 'None',
    [switch]$IncludeQualityStack
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$builder = (Resolve-Path -LiteralPath $BuildPython).Path
$source = (Resolve-Path -LiteralPath $PythonSourceRoot).Path
$install = [System.IO.Path]::GetFullPath($InstallRoot)
$requirements = Join-Path $repo 'packaging\requirements'
$wheelhouse = if ([string]::IsNullOrWhiteSpace($WheelhouseRoot)) {
    Join-Path $repo 'packaging\wheelhouse'
}
else {
    $resolvedWheelhouse = Resolve-Path -LiteralPath $WheelhouseRoot
    if (-not $resolvedWheelhouse) { throw "Wheelhouse root does not exist: $WheelhouseRoot" }
    $resolvedWheelhouseItem = Get-Item -LiteralPath $resolvedWheelhouse.Path -Force
    if (-not $resolvedWheelhouseItem.PSIsContainer) { throw "Wheelhouse root must be a directory: $($resolvedWheelhouseItem.FullName)" }
    $resolvedWheelhouseItem.FullName
}
if ($PSBoundParameters.ContainsKey('IncludeOcrPaddleGpu')) {
    if ($PSBoundParameters.ContainsKey('OcrComponents')) {
        throw 'IncludeOcrPaddleGpu and OcrComponents are ambiguous; use OcrComponents only'
    }
    $OcrComponents = 'Gpu'
}
if ($OcrComponents -eq 'Gpu') {
    $gpuInputs = @(
        (Join-Path $requirements 'ocr-paddle-gpu.in'),
        (Join-Path $requirements 'ocr-paddle-gpu.lock'),
        (Join-Path $wheelhouse 'ocr-paddle-gpu')
    )
    if (@($gpuInputs | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -ne 0) {
        throw 'GPU distribution opt-in requires complete ocr-paddle-gpu lock and wheelhouse inputs before assembly'
    }
}
$policyLock = $null
if ($PolicyComponents -ne 'None') {
    $policyLock = @{
        Cpu = 'policy-cpu'
        Cuda = 'policy-cuda'
    }[$PolicyComponents]
    $policyInputs = @(
        (Join-Path $requirements "$policyLock.in"),
        (Join-Path $requirements "$policyLock.lock"),
        (Join-Path $wheelhouse $policyLock)
    )
    if (@($policyInputs | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -ne 0) {
        throw "Policy distribution opt-in requires complete $policyLock lock and wheelhouse inputs before assembly"
    }
}
$nodeVersion = (& node --version).Trim()
$npmVersion = (& npm --version).Trim()
if ($nodeVersion -ne 'v24.18.0' -or $npmVersion -ne '11.16.0') {
    throw "Release build requires Node v24.18.0 and npm 11.16.0; found $nodeVersion / $npmVersion"
}
if ($ReuseBaseRuntime) {
    if (-not (Test-Path -LiteralPath (Join-Path $install 'runtimes\_base\python.exe') -PathType Leaf)) {
        throw "Reusable base runtime is missing under $install"
    }
}
else {
    if (Test-Path -LiteralPath $install) { throw "Install root must not already exist: $install" }
    & (Join-Path $repo 'packaging\scripts\build_cpython311_runtime.ps1') -PythonSourceRoot $source -OutputRoot $install
}

$lockRuntimes = @('core', 'caption-e621', 'classify-e621', 'replace-e621', 'nl', 'export', 'token-budget')
if ($PolicyComponents -ne 'None') { $lockRuntimes += $policyLock }
& $builder -B -I (Join-Path $repo 'packaging\scripts\verify_locks.py') --requirements-root $requirements --wheelhouse-root $wheelhouse @lockRuntimes
if ($LASTEXITCODE -ne 0) { throw 'Dependency lock verification failed' }

$base = Join-Path $install 'runtimes\_base'
$runtimeJobs = @(
    @{ Id = 'core'; Lock = 'core'; Source = (Join-Path $repo 'core\src\anima_core'); SharedSource = (Join-Path $repo 'shared\anima_caption_format\anima_caption_format') },
    @{ Id = 'caption-e621'; Lock = 'caption-e621'; Source = (Join-Path $repo 'workers\caption\src\anima_caption_worker') },
    @{ Id = 'classify-e621'; Lock = 'classify-e621'; Source = (Join-Path $repo 'workers\classify\src\anima_classify_worker') },
    @{ Id = 'replace-e621'; Lock = 'replace-e621'; Source = (Join-Path $repo 'workers\replace\src\anima_replace_worker') },
    @{ Id = 'nl'; Lock = 'nl'; Source = (Join-Path $repo 'workers\nl\src\anima_nl_worker') },
    @{ Id = 'export'; Lock = 'export'; Source = (Join-Path $repo 'workers\export\src\anima_export_worker'); SharedSource = (Join-Path $repo 'shared\anima_caption_format\anima_caption_format') },
    @{ Id = 'token-budget'; Lock = 'token-budget'; Source = (Join-Path $repo 'workers\token_budget\src\anima_token_budget_worker'); SharedSource = (Join-Path $repo 'shared\anima_caption_format\anima_caption_format') }
)
if ($PolicyComponents -ne 'None') {
    $runtimeJobs += @{ Id = 'policy'; Lock = $policyLock; Source = (Join-Path $repo 'workers\policy\src\anima_policy_worker') }
}
foreach ($job in $runtimeJobs) {
    $assemblyArguments = @{
        BaseRuntime = $base
        DestinationRuntime = (Join-Path $install "runtimes\$($job.Id)")
        RequirementsLock = (Join-Path $requirements "$($job.Lock).lock")
        Wheelhouse = (Join-Path $wheelhouse $job.Lock)
        OwnerSource = $job.Source
        BuildPython = $builder
    }
    if ($job.SharedSource) { $assemblyArguments.SharedSource = $job.SharedSource }
    & (Join-Path $repo 'packaging\scripts\assemble_runtime.ps1') @assemblyArguments
}
Remove-Item -LiteralPath $base -Recurse -Force
$manifestArguments = @('--install-root', $install, '--requirements-root', $requirements)
foreach ($job in $runtimeJobs) { $manifestArguments += @('--runtime-id', $job.Id) }
& $builder -B -I (Join-Path $repo 'packaging\scripts\generate_runtime_manifests.py') @manifestArguments
if ($LASTEXITCODE -ne 0) { throw 'Runtime manifest generation failed' }
$resourceLibrarySource = (Resolve-Path -LiteralPath (Join-Path $repo 'resource-library')).Path
$developmentInstall = [System.IO.Path]::GetFullPath((Join-Path $repo '.runtime-build'))
$isDevelopmentInstall = [System.StringComparer]::OrdinalIgnoreCase.Equals($install, $developmentInstall)
if ($isDevelopmentInstall) {
    $resourceLibrary = $resourceLibrarySource
}
else {
    $resourceLibrary = Join-Path $install 'resource-library'
    if (Test-Path -LiteralPath $resourceLibrary) { throw "Resource library destination already exists: $resourceLibrary" }
    & $builder -B -I (Join-Path $repo 'packaging\scripts\copy_resource_library.py') `
        --source $resourceLibrarySource `
        --destination $resourceLibrary
    if ($LASTEXITCODE -ne 0) { throw 'Distributable resource library assembly failed' }
    $optionalOcrResource = Join-Path $resourceLibrary 'ocr-models\ocr-ppocrv5-server-paddle-v1'
    if (Test-Path -LiteralPath $optionalOcrResource) {
        Remove-Item -LiteralPath $optionalOcrResource -Recurse -Force
    }
    if (-not $IncludeQualityStack) {
        $optionalQualityResource = Join-Path $resourceLibrary 'dropout-models\lse14-scorer-5k-v1'
        if (Test-Path -LiteralPath $optionalQualityResource) {
            Remove-Item -LiteralPath $optionalQualityResource -Recurse -Force
        }
    }
}
$resourceValidationArguments = @('--root', $resourceLibrary)
if (-not $isDevelopmentInstall) { $resourceValidationArguments += '--release' }
if ($IncludeQualityStack) { $resourceValidationArguments += '--require-quality-stack' }
& $builder -B -I (Join-Path $repo 'packaging\scripts\validate_resource_library.py') @resourceValidationArguments
if ($LASTEXITCODE -ne 0) { throw 'Project-local resource library validation failed' }
if ($OcrComponents -eq 'None') {
    # OCR stays absent from the base tree; existing development artifacts are never removed.
}
else {
    & $builder -B -I (Join-Path $repo 'packaging\scripts\build_optional_ocr_components.py') `
        --source-root $repo `
        --destination-root (Join-Path $install 'optional-components') `
        --mode $OcrComponents.ToLowerInvariant()
    if ($LASTEXITCODE -ne 0) { throw 'Optional OCR component assembly failed' }
}
& $builder -B -I (Join-Path $repo 'packaging\scripts\assemble_nl_prompt_resource.py') `
    --source (Join-Path $repo 'packaging\resources\nl-default-prompt-v1.txt') `
    --install-root $install
if ($LASTEXITCODE -ne 0) { throw 'NL prompt v1 resource assembly failed' }
& $builder -B -I (Join-Path $repo 'packaging\scripts\assemble_nl_prompt_resource.py') `
    --source (Join-Path $repo 'packaging\resources\nl-default-prompt-v2.txt') `
    --install-root $install
if ($LASTEXITCODE -ne 0) { throw 'NL prompt v2 resource assembly failed' }
& $builder -B -I (Join-Path $repo 'packaging\scripts\assemble_nl_prompt_resource.py') `
    --source (Join-Path $repo 'packaging\resources\nl-default-prompt-v3.txt') `
    --install-root $install
if ($LASTEXITCODE -ne 0) { throw 'NL prompt v3 resource assembly failed' }
& $builder -B -I (Join-Path $repo 'packaging\scripts\assemble_nl_prompt_resource.py') `
    --v4-source-root (Join-Path $repo 'packaging\resources') `
    --install-root $install
if ($LASTEXITCODE -ne 0) { throw 'NL prompt v4 fragment resource assembly failed' }
& $builder -B -I (Join-Path $repo 'packaging\scripts\assemble_nl_prompt_resource.py') `
    --moss-source-root (Join-Path $repo 'packaging\resources') `
    --install-root $install
if ($LASTEXITCODE -ne 0) { throw 'MOSS NL prompt resource assembly failed' }

$profileOutput = Join-Path $install 'manifests\profiles'
New-Item -ItemType Directory -Path $profileOutput -Force | Out-Null
Copy-Item -Path (Join-Path $repo 'profiles\*.profile.json') -Destination $profileOutput

$releaseRootFiles = @(
    'Install-WebUI.bat',
    'Install-Policy.bat',
    'Install-QualityStack.bat',
    'Start-WebUI.bat',
    'Stop-WebUI.bat',
    'Install-Ocr.bat',
    'Install-OcrGpu.bat',
    'Import-OcrResource.bat',
    'Reset-OcrRuntime.bat',
    'Reset-OcrGpuRuntime.bat',
    'Clean-OcrArtifacts.bat',
    'Clean-OcrGpuArtifacts.bat',
    'README.md',
    'OCR_MODEL_DOWNLOAD.md'
)
foreach ($filename in $releaseRootFiles) {
    Copy-Item -LiteralPath (Join-Path $repo $filename) -Destination (Join-Path $install $filename)
}
$releaseControlScripts = @(
    'desktop_control.ps1',
    'webui_host.ps1',
    'Import-OcrResource.ps1',
    'Install-OcrGpu.ps1',
    'Reset-OcrRuntime.ps1',
    'Reset-OcrGpuRuntime.ps1',
    'Clean-OcrArtifacts.ps1',
    'Clean-OcrGpuArtifacts.ps1',
    'ocr_component.py',
    'ocr_resource.py',
    'ocr_gpu_resource.py',
    'ocr_compatibility.py'
)
$releaseControlRoot = Join-Path $install 'packaging\scripts'
New-Item -ItemType Directory -Path $releaseControlRoot -Force | Out-Null
foreach ($filename in $releaseControlScripts) {
    Copy-Item -LiteralPath (Join-Path $repo "packaging\scripts\$filename") -Destination (Join-Path $releaseControlRoot $filename)
}

Push-Location (Join-Path $repo 'frontend')
try {
    & npm ci --ignore-scripts
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
    & npm run build
    if ($LASTEXITCODE -ne 0) { throw 'frontend build failed' }
    $frontendOutput = Join-Path $install 'frontend\dist'
    New-Item -ItemType Directory -Path (Split-Path $frontendOutput) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $repo 'frontend\dist') -Destination $frontendOutput -Recurse
}
finally {
    Pop-Location
}

$compiler = (& cl.exe 2>&1 | Select-Object -First 1).ToString()
$provenance = [ordered]@{
    schemaVersion = 1
    pythonVersion = '3.11.15'
    pythonSourceUrl = $PythonSourceUrl
    pythonSourceSha256 = $PythonSourceSha256
    compiler = $compiler
    nodeVersion = $nodeVersion
    npmVersion = $npmVersion
    buildDistributionScriptSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
    buildCpythonScriptSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $repo 'packaging\scripts\build_cpython311_runtime.ps1')).Hash.ToLowerInvariant()
}
$provenance | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $install 'manifests\build-provenance.json') -Encoding utf8
