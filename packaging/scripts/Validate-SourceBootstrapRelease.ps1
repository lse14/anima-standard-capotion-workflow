[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$ProjectRoot,
    [switch]$VerifyPublishedBootstrap
)

$ErrorActionPreference = 'Stop'

function Stop-ReleaseGate([string]$Message) {
    throw "Source-bootstrap release gate failed: $Message"
}

function Get-Sha256Hex([string]$Path) {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        return [System.BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
        $stream.Dispose()
    }
}

function Assert-ApprovedReleaseUri([string]$Url, [string[]]$AllowedHosts) {
    try { $uri = [System.Uri]$Url } catch { Stop-ReleaseGate "published bootstrap URL is invalid: $Url" }
    if ($uri.Scheme -ne 'https' -or [string]::IsNullOrWhiteSpace($uri.Host) -or $uri.UserInfo -or
        $AllowedHosts -notcontains $uri.Host.ToLowerInvariant()) {
        Stop-ReleaseGate "published bootstrap URL is not an allowed HTTPS host: $Url"
    }
    return $uri
}

function Open-ApprovedReleaseResponse([string]$Url, [string[]]$AllowedHosts) {
    $current = Assert-ApprovedReleaseUri $Url $AllowedHosts
    for ($redirect = 0; $redirect -lt 6; $redirect++) {
        $request = [System.Net.HttpWebRequest]::Create($current)
        $request.Method = 'GET'
        $request.AllowAutoRedirect = $false
        $request.Timeout = 60000
        $request.ReadWriteTimeout = 60000
        $request.AddRange([Int64]0)
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
            if ([string]::IsNullOrWhiteSpace($location)) { Stop-ReleaseGate 'published bootstrap redirect has no Location header' }
            $current = New-Object System.Uri($current, $location)
            [void](Assert-ApprovedReleaseUri $current.AbsoluteUri $AllowedHosts)
            continue
        }
        [void](Assert-ApprovedReleaseUri $response.ResponseUri.AbsoluteUri $AllowedHosts)
        return $response
    }
    Stop-ReleaseGate 'published bootstrap exceeded the redirect limit'
}

function Test-PublishedBootstrapIdentity(
    [object]$BootstrapRecord,
    [string[]]$AllowedHosts,
    [string]$ProjectRoot
) {
    $downloadRoot = Join-Path $ProjectRoot '.release-candidate\published-bootstrap-verify'
    $downloadPath = $null
    $response = $null
    try {
        New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
        $downloadPath = Join-Path $downloadRoot (([Guid]::NewGuid().ToString('N')) + '.partial')
        $response = Open-ApprovedReleaseResponse ([string]$BootstrapRecord.publishedUrl) $AllowedHosts
        $status = [Int32]$response.StatusCode
        if ($status -notin 200, 206) { Stop-ReleaseGate "published bootstrap download returned HTTP $status" }
        if ($status -eq 206) {
            $range = [string]$response.Headers['Content-Range']
            if ($range -notmatch ('^bytes 0-\d+/{0}$' -f [Int64]$BootstrapRecord.sizeBytes)) {
                Stop-ReleaseGate 'published bootstrap Range response does not match release-artifacts.json'
            }
        }
        $target = [System.IO.File]::Open($downloadPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        try {
            $source = $response.GetResponseStream()
            try {
                $buffer = New-Object byte[] 1048576
                while (($read = $source.Read($buffer, 0, $buffer.Length)) -gt 0) {
                    $target.Write($buffer, 0, $read)
                    if ($target.Length -gt [Int64]$BootstrapRecord.sizeBytes) {
                        Stop-ReleaseGate 'published bootstrap download exceeds release-artifacts.json sizeBytes'
                    }
                }
            }
            finally { $source.Dispose() }
        }
        finally { $target.Dispose() }
        if ((Get-Item -LiteralPath $downloadPath -Force).Length -ne [Int64]$BootstrapRecord.sizeBytes -or
            (Get-Sha256Hex $downloadPath) -ne [string]$BootstrapRecord.sha256) {
            Stop-ReleaseGate 'published bootstrap bytes do not match release-artifacts.json'
        }
        Write-Output 'Published bootstrap asset bytes match release-artifacts.json.'
    }
    finally {
        if ($null -ne $response) { $response.Close() }
        if ($null -ne $downloadPath -and (Test-Path -LiteralPath $downloadPath -PathType Leaf)) {
            Remove-Item -LiteralPath $downloadPath -Force -ErrorAction SilentlyContinue
        }
        if ((Test-Path -LiteralPath $downloadRoot -PathType Container) -and
            $null -eq (Get-ChildItem -LiteralPath $downloadRoot -Force -ErrorAction SilentlyContinue | Select-Object -First 1)) {
            Remove-Item -LiteralPath $downloadRoot -Force -ErrorAction SilentlyContinue
        }
    }
}

function Read-JsonFile([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Stop-ReleaseGate "$Label is missing"
    }
    try {
        $json = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
        if ((Get-Command ConvertFrom-Json).Parameters.ContainsKey('DateKind')) {
            return $json | ConvertFrom-Json -DateKind String
        }
        return $json | ConvertFrom-Json
    } catch {
        Stop-ReleaseGate "$Label is not valid JSON"
    }
}

function Get-RequiredProperty([object]$Value, [string]$Name, [string]$Label) {
    $property = $Value.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        Stop-ReleaseGate "$Label is missing required field: $Name"
    }
    return $property.Value
}

function Assert-SafeRelative([string]$Value, [string]$Label) {
    if ([string]::IsNullOrWhiteSpace($Value)) { Stop-ReleaseGate "$Label is empty" }
    $normalized = $Value.Replace('/', '\')
    if ($normalized.StartsWith('\') -or $normalized -match '^[A-Za-z]:' -or [System.IO.Path]::IsPathRooted($normalized)) {
        Stop-ReleaseGate "$Label is not a safe relative path: $Value"
    }
    foreach ($part in ($normalized -split '\\')) {
        $device = ($part -split '\.')[0].ToUpperInvariant()
        if ([string]::IsNullOrEmpty($part) -or $part -in @('.', '..') -or $part.Contains(':') -or
            $part.EndsWith('.') -or $part.EndsWith(' ') -or
            $device -in @('CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9')) {
            Stop-ReleaseGate "$Label is not a safe relative path: $Value"
        }
    }
    return $normalized
}

function Test-ExactPropertySet([object]$Value, [string[]]$Expected, [string]$Label) {
    $actual = @($Value.PSObject.Properties.Name)
    if ($actual.Count -ne $Expected.Count -or @($actual | Where-Object { $Expected -notcontains $_ }).Count -gt 0) {
        Stop-ReleaseGate "$Label fields are invalid"
    }
}

function Assert-ProjectSourceFile([string]$ProjectRoot, [string]$RelativePath, [string]$Label) {
    $safeRelative = Assert-SafeRelative $RelativePath "$Label sourceRelativePath"
    $root = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $root $safeRelative))
    $prefix = $root + [System.IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        Stop-ReleaseGate "$Label source-tree artifact is missing"
    }
    $item = Get-Item -LiteralPath $candidate -Force -ErrorAction Stop
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        Stop-ReleaseGate "$Label source-tree artifact is a reparse point"
    }
    return $candidate
}

function Assert-Artifact([object]$Artifact, [string[]]$TopLevelHosts, [string]$ProjectRoot, [string]$Label) {
    $delivery = if ($null -eq $Artifact.PSObject.Properties['delivery']) { 'remote' } else { [string]$Artifact.delivery }
    if ($delivery -eq 'source-tree') {
        Test-ExactPropertySet $Artifact @('id', 'delivery', 'sourceRelativePath', 'sizeBytes', 'sha256', 'relativePath') $Label
        foreach ($name in @('id', 'sourceRelativePath', 'sizeBytes', 'sha256', 'relativePath')) {
            [void](Get-RequiredProperty $Artifact $name $Label)
        }
        if ([Int64]$Artifact.sizeBytes -le 0) { Stop-ReleaseGate "$Label sizeBytes is invalid" }
        if ([string]$Artifact.sha256 -notmatch '^[a-f0-9]{64}$') { Stop-ReleaseGate "$Label SHA-256 is invalid" }
        [void](Assert-SafeRelative ([string]$Artifact.relativePath) "$Label relativePath")
        $source = Assert-ProjectSourceFile $ProjectRoot ([string]$Artifact.sourceRelativePath) $Label
        $item = Get-Item -LiteralPath $source -Force
        if ($item.Length -ne [Int64]$Artifact.sizeBytes -or (Get-Sha256Hex $source) -ne [string]$Artifact.sha256) {
            Stop-ReleaseGate "$Label source-tree artifact identity does not match"
        }
        return
    }
    if ($delivery -ne 'remote') { Stop-ReleaseGate "$Label delivery is invalid for a production manifest" }
    foreach ($name in @('id', 'url', 'allowedHosts', 'sizeBytes', 'sha256', 'relativePath')) {
        [void](Get-RequiredProperty $Artifact $name $Label)
    }
    $url = [string]$Artifact.url
    try { $parsed = [System.Uri]$url } catch { Stop-ReleaseGate "$Label URL is invalid: $url" }
    if ($parsed.Scheme -ne 'https' -or [string]::IsNullOrEmpty($parsed.Host) -or $parsed.UserInfo) {
        Stop-ReleaseGate "$Label URL must use HTTPS without credentials: $url"
    }
    $artifactHosts = @($Artifact.allowedHosts | ForEach-Object { ([string]$_).ToLowerInvariant() })
    if ($artifactHosts.Count -eq 0 -or $artifactHosts.Count -ne (@($artifactHosts | Select-Object -Unique).Count)) {
        Stop-ReleaseGate "$Label allowed hosts are invalid"
    }
    foreach ($allowedHost in $artifactHosts) {
        if ($TopLevelHosts -notcontains $allowedHost) { Stop-ReleaseGate "$Label allowed host is not in the manifest allowlist: $allowedHost" }
    }
    if ($artifactHosts -notcontains $parsed.Host.ToLowerInvariant()) {
        Stop-ReleaseGate "$Label URL host is not allowed: $($parsed.Host)"
    }
    if ([Int64]$Artifact.sizeBytes -le 0) { Stop-ReleaseGate "$Label sizeBytes is invalid" }
    if ([string]$Artifact.sha256 -notmatch '^[a-f0-9]{64}$') { Stop-ReleaseGate "$Label SHA-256 is invalid" }
    [void](Assert-SafeRelative ([string]$Artifact.relativePath) "$Label relativePath")
    $hasRepository = $null -ne $Artifact.PSObject.Properties['repository']
    $hasRevision = $null -ne $Artifact.PSObject.Properties['revision']
    if ($hasRepository -or $hasRevision) {
        if (-not $hasRepository -or -not $hasRevision -or [string]::IsNullOrWhiteSpace([string]$Artifact.repository) -or
            [string]$Artifact.revision -notmatch '^[a-f0-9]{40}$') {
            Stop-ReleaseGate "$Label Hugging Face identity is incomplete"
        }
        if ($parsed.Host.ToLowerInvariant() -ne 'huggingface.co' -or
            -not $parsed.AbsolutePath.StartsWith('/' + [string]$Artifact.repository + '/resolve/' + [string]$Artifact.revision + '/', [System.StringComparison]::Ordinal)) {
            Stop-ReleaseGate "$Label does not use its full Hugging Face revision"
        }
    }
}

function Assert-LicenseEvidenceUri([string]$Value, [string]$Label) {
    try { $uri = [System.Uri]$Value } catch { Stop-ReleaseGate "$Label is invalid: $Value" }
    if ($uri.Scheme -ne 'https' -or [string]::IsNullOrWhiteSpace($uri.Host) -or $uri.UserInfo) {
        Stop-ReleaseGate "$Label must use HTTPS without credentials: $Value"
    }
    return $uri.AbsoluteUri
}

function Assert-UtcTimestamp([string]$TimestampText, [string]$TimestampLabel) {
    if ($TimestampText -notmatch '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$') {
        Stop-ReleaseGate "$TimestampLabel is not an ISO-8601 UTC timestamp"
    }
    try {
        [void][DateTime]::ParseExact(
            $TimestampText,
            'yyyy-MM-ddTHH:mm:ssZ',
            [Globalization.CultureInfo]::InvariantCulture,
            [Globalization.DateTimeStyles]::AssumeUniversal
        )
    } catch {
        Stop-ReleaseGate "$TimestampLabel is not an ISO-8601 UTC timestamp"
    }
}

function Get-LicenseReferences([object]$Manifest) {
    $references = @{}
    foreach ($component in @($Manifest.components)) {
        $componentId = [string](Get-RequiredProperty $component 'componentId' 'component')
        $reference = [string](Get-RequiredProperty $component 'licenseReference' "component $componentId")
        if ([string]::IsNullOrWhiteSpace($reference)) { Stop-ReleaseGate "component $componentId licenseReference is empty" }
        $references[$reference] = $true
    }
    return $references
}

function Get-SourceArtifactsForLicenseReference([object]$Manifest, [string]$LicenseReference) {
    $artifacts = @()
    foreach ($component in @($Manifest.components)) {
        if ([string]$component.licenseReference -ne $LicenseReference) { continue }
        foreach ($variantProperty in $component.variants.PSObject.Properties) {
            foreach ($artifact in @($variantProperty.Value.artifacts)) {
                if ([string]$artifact.delivery -eq 'source-tree') { $artifacts += $artifact }
            }
        }
    }
    return $artifacts
}

function Assert-SourceRedistributionDecision([object]$Entry, [object[]]$Decisions, [object[]]$Artifacts) {
    $entryId = [string]$Entry.id
    $matches = @($Decisions | Where-Object { [string]$_.licenseReference -eq $entryId })
    if ($matches.Count -ne 1) { Stop-ReleaseGate "source-redistributed approval decision is missing: $entryId" }
    if ($Artifacts.Count -eq 0) { Stop-ReleaseGate "source-redistributed approval has no source-tree artifacts: $entryId" }

    $decision = $matches[0]
    Test-ExactPropertySet $decision @('id', 'licenseReference', 'source', 'decidedAtUtc', 'termsUrl', 'termsSha256', 'approvedArtifacts') "source-redistributed approval decision $entryId"
    foreach ($name in @('id', 'licenseReference', 'source', 'decidedAtUtc', 'termsUrl', 'termsSha256', 'approvedArtifacts')) {
        [void](Get-RequiredProperty $decision $name "source-redistributed approval decision $entryId")
    }
    if ([string]$decision.id -notmatch '^[a-z0-9][a-z0-9-]{0,127}$' -or [string]$decision.source -ne 'user-confirmed-project-owner') {
        Stop-ReleaseGate "source-redistributed approval decision is invalid: $entryId"
    }
    Assert-UtcTimestamp ([string]$decision.decidedAtUtc) "source-redistributed approval decision $entryId decidedAtUtc"
    if ((Assert-LicenseEvidenceUri ([string]$decision.termsUrl) "source-redistributed approval decision $entryId termsUrl") -ne [string]$Entry.licenseEvidenceUrl -or
        [string]$decision.termsSha256 -ne [string]$Entry.evidenceSha256) {
        Stop-ReleaseGate "source-redistributed approval decision does not match Terms evidence: $entryId"
    }

    $expected = @{}
    foreach ($artifact in $Artifacts) {
        $path = [string]$artifact.sourceRelativePath
        $key = $path.Replace('/', '\').ToLowerInvariant()
        if ($expected.ContainsKey($key)) { Stop-ReleaseGate "source-redistributed manifest artifact is duplicated: $path" }
        $expected[$key] = $artifact
    }
    $approved = @{}
    foreach ($artifact in @($decision.approvedArtifacts)) {
        Test-ExactPropertySet $artifact @('sourceRelativePath', 'sizeBytes', 'sha256') "source-redistributed approval artifact $entryId"
        foreach ($name in @('sourceRelativePath', 'sizeBytes', 'sha256')) { [void](Get-RequiredProperty $artifact $name "source-redistributed approval artifact $entryId") }
        $path = Assert-SafeRelative ([string]$artifact.sourceRelativePath) "source-redistributed approval artifact $entryId"
        $key = $path.ToLowerInvariant()
        if ($approved.ContainsKey($key) -or -not $expected.ContainsKey($key) -or [Int64]$artifact.sizeBytes -ne [Int64]$expected[$key].sizeBytes -or
            [string]$artifact.sha256 -ne [string]$expected[$key].sha256) {
            Stop-ReleaseGate "source-redistributed approval does not exactly bind manifest artifacts: $entryId"
        }
        $approved[$key] = $true
    }
    if ($approved.Count -ne $expected.Count) { Stop-ReleaseGate "source-redistributed approval does not exactly bind manifest artifacts: $entryId" }
}

function Assert-LicenseLedger([object]$Ledger, [object]$Manifest) {
    Test-ExactPropertySet $Ledger @('schemaVersion', 'entries', 'decisions') 'license ledger'
    if ([Int32](Get-RequiredProperty $Ledger 'schemaVersion' 'license ledger') -ne 1) { Stop-ReleaseGate 'license ledger schemaVersion is not 1' }
    $entries = @((Get-RequiredProperty $Ledger 'entries' 'license ledger'))
    $decisions = @((Get-RequiredProperty $Ledger 'decisions' 'license ledger'))
    if ($entries.Count -eq 0) { Stop-ReleaseGate 'license ledger has no entries' }

    $byId = @{}
    foreach ($entry in $entries) {
        Test-ExactPropertySet $entry @('id', 'delivery', 'officialSourceUrl', 'licenseEvidenceUrl', 'evidenceRetrievedAtUtc', 'evidenceSha256', 'reviewStatus', 'redistributionStatus') 'license ledger entry'
        foreach ($name in @('id', 'delivery', 'officialSourceUrl', 'licenseEvidenceUrl', 'evidenceRetrievedAtUtc', 'evidenceSha256', 'reviewStatus', 'redistributionStatus')) {
            [void](Get-RequiredProperty $entry $name 'license ledger entry')
        }
        $id = [string]$entry.id
        if ($id -notmatch '^[a-z0-9][a-z0-9-]{0,127}$' -or $byId.ContainsKey($id)) { Stop-ReleaseGate "license ledger entry id is invalid: $id" }
        [void](Assert-LicenseEvidenceUri ([string]$entry.officialSourceUrl) "license ledger entry $id officialSourceUrl")
        [void](Assert-LicenseEvidenceUri ([string]$entry.licenseEvidenceUrl) "license ledger entry $id licenseEvidenceUrl")
        Assert-UtcTimestamp ([string]$entry.evidenceRetrievedAtUtc) "license ledger entry $id evidenceRetrievedAtUtc"
        if ([string]$entry.evidenceSha256 -notmatch '^[a-f0-9]{64}$') { Stop-ReleaseGate "license ledger entry evidence SHA-256 is invalid: $id" }
        if ([string]$entry.reviewStatus -ne 'evidence-collected') { Stop-ReleaseGate "license ledger entry reviewStatus is invalid: $id" }

        switch ([string]$entry.delivery) {
            'direct-upstream-only' {
                if ([string]$entry.redistributionStatus -ne 'not-mirrored') { Stop-ReleaseGate "direct-upstream-only license entry must not be mirrored: $id" }
            }
            'local-only' {
                if ([string]$entry.redistributionStatus -ne 'not-mirrored') { Stop-ReleaseGate "local-only license entry must not be mirrored: $id" }
            }
            'project-source' {
                if ([string]$entry.redistributionStatus -ne 'not-mirrored') { Stop-ReleaseGate "project-source license entry must not be mirrored: $id" }
            }
            'source-redistributed' {
                if ([string]$entry.redistributionStatus -notin @('blocked', 'pending-human-review', 'approved')) {
                    Stop-ReleaseGate "source-redistributed redistribution status is invalid: $id"
                }
                if ([string]$entry.redistributionStatus -ne 'approved') { Stop-ReleaseGate "redistribution is not approved: $id" }
            }
            default { Stop-ReleaseGate "license ledger delivery is invalid: $id" }
        }
        $byId[$id] = $entry
    }

    foreach ($reference in (Get-LicenseReferences $Manifest).Keys) {
        if (-not $byId.ContainsKey($reference)) { Stop-ReleaseGate "license ledger entry is missing: $reference" }
    }
    foreach ($component in @($Manifest.components)) {
        $componentId = [string](Get-RequiredProperty $component 'componentId' 'component')
        $reference = [string](Get-RequiredProperty $component 'licenseReference' "component $componentId")
        $hasPayload = $false
        foreach ($variantProperty in $component.variants.PSObject.Properties) {
            foreach ($artifact in @($variantProperty.Value.artifacts)) {
                if ([string]$artifact.delivery -eq 'source-tree' -and
                    ([string]$artifact.relativePath).Replace('/', '\').ToLowerInvariant() -ne 'resource.json') {
                    $hasPayload = $true
                    break
                }
            }
            if ($hasPayload) { break }
        }
        if (-not $hasPayload) { continue }
        $entry = $byId[$reference]
        if ($null -eq $entry -or [string]$entry.delivery -ne 'source-redistributed' -or
            [string]$entry.redistributionStatus -ne 'approved') {
            Stop-ReleaseGate "source-tree payload requires approved source-redistributed license: $componentId"
        }
    }
    foreach ($entry in $byId.Values) {
        if ([string]$entry.delivery -eq 'source-redistributed' -and [string]$entry.redistributionStatus -eq 'approved') {
            Assert-SourceRedistributionDecision $entry $decisions (Get-SourceArtifactsForLicenseReference $Manifest ([string]$entry.id))
        }
    }
}

function Assert-Manifest([object]$Manifest) {
    $expectedFields = @('schemaVersion', 'releaseVersion', 'sourceCommit', 'allowedHosts', 'bootstrap', 'components', 'cleanup')
    $actualFields = @($Manifest.PSObject.Properties.Name)
    if ($actualFields.Count -ne $expectedFields.Count -or @($actualFields | Where-Object { $expectedFields -notcontains $_ }).Count -gt 0) {
        Stop-ReleaseGate 'install manifest top-level fields are invalid'
    }
    if ([Int32]$Manifest.schemaVersion -ne 1) { Stop-ReleaseGate 'install manifest schemaVersion is not 1' }
    $releaseVersion = [string](Get-RequiredProperty $Manifest 'releaseVersion' 'install manifest')
    if ([string]::IsNullOrWhiteSpace($releaseVersion)) { Stop-ReleaseGate 'install manifest releaseVersion is empty' }
    if ([string]$Manifest.sourceCommit -notmatch '^[a-f0-9]{40}$') { Stop-ReleaseGate 'install manifest sourceCommit is not a full commit SHA' }
    $hosts = @($Manifest.allowedHosts | ForEach-Object { ([string]$_).ToLowerInvariant() })
    if ($hosts.Count -eq 0 -or $hosts.Count -ne (@($hosts | Select-Object -Unique).Count)) { Stop-ReleaseGate 'install manifest allowedHosts are invalid' }

    $bootstrap = Get-RequiredProperty $Manifest 'bootstrap' 'install manifest'
    $bootstrapArtifact = Get-RequiredProperty $bootstrap 'artifact' 'bootstrap'
    Assert-Artifact $bootstrapArtifact $hosts $root 'bootstrap artifact'
    [void](Assert-SafeRelative ([string](Get-RequiredProperty $bootstrap 'entryRelativePath' 'bootstrap')) 'bootstrap entryRelativePath')
    if ([Int64](Get-RequiredProperty $bootstrap 'peakBytes' 'bootstrap') -le 0) { Stop-ReleaseGate 'bootstrap peakBytes is invalid' }

    $mandatory = @('core', 'caption-e621', 'classify-e621', 'replace-e621', 'nl', 'export', 'token-budget', 'e621-indexes', 'e621-replacement-indexes', 'e621-tagger', 'qwen3-tokenizer')
    # The optional OCR runtime and models, the policy scorer and the quality
    # stack may be absent from a valid release; when present they must not
    # claim to be mandatory.
    $optionalComponents = @('ocr-cpu', 'ocr-gpu', 'ocr-models', 'policy', 'quality-stack')
    $components = @($Manifest.components)
    if ($components.Count -eq 0) { Stop-ReleaseGate 'install manifest has no components' }
    $ids = @{}
    $targets = @{}
    $artifactIds = @{}
    foreach ($component in $components) {
        $id = [string](Get-RequiredProperty $component 'componentId' 'component')
        if ($ids.ContainsKey($id)) { Stop-ReleaseGate "duplicate componentId: $id" }
        $ids[$id] = $true
        $kind = [string](Get-RequiredProperty $component 'kind' "component $id")
        $target = Assert-SafeRelative ([string](Get-RequiredProperty $component 'targetRelativePath' "component $id")) "component $id targetRelativePath"
        if ($targets.ContainsKey($target.ToLowerInvariant())) { Stop-ReleaseGate "duplicate component target: $target" }
        $targets[$target.ToLowerInvariant()] = $true
        $variants = Get-RequiredProperty $component 'variants' "component $id"
        foreach ($variantProperty in $variants.PSObject.Properties) {
            $variant = $variantProperty.Name
            if ($variant -notin @('cpu', 'cuda', 'shared')) { Stop-ReleaseGate "component $id has an invalid variant: $variant" }
            $record = $variantProperty.Value
            if ([Int64](Get-RequiredProperty $record 'peakBytes' "component $id/$variant") -le 0) { Stop-ReleaseGate "component $id/$variant peakBytes is invalid" }
            $artifacts = @((Get-RequiredProperty $record 'artifacts' "component $id/$variant"))
            if ($artifacts.Count -eq 0 -and $kind -ne 'runtime') { Stop-ReleaseGate "component $id/$variant has no artifacts" }
            $variantTargets = @{}
            foreach ($artifact in $artifacts) {
                Assert-Artifact $artifact $hosts $root "component $id/$variant artifact"
                $artifactId = ([string]$artifact.id).ToLowerInvariant()
                if ($artifactIds.ContainsKey($artifactId)) { Stop-ReleaseGate "duplicate artifact id: $artifactId" }
                $artifactIds[$artifactId] = $true
                $artifactTarget = ([string]$artifact.relativePath).Replace('/', '\').ToLowerInvariant()
                if ($variantTargets.ContainsKey($artifactTarget)) { Stop-ReleaseGate "duplicate artifact target: $id/$variant/$artifactTarget" }
                $variantTargets[$artifactTarget] = $true
                if ($variant -eq 'cpu' -and ([string]$artifact.id + ' ' + [string]$artifact.url) -match '(?i)(cuda|[+]cu[0-9]*|cudnn|nvidia[-_])') {
                    Stop-ReleaseGate "CPU variant contains a CUDA artifact: $id"
                }
            }
        }
    }
    foreach ($id in $mandatory) { if (-not $ids.ContainsKey($id)) { Stop-ReleaseGate "mandatory component is missing: $id" } }
    foreach ($id in $optionalComponents) {
        if (-not $ids.ContainsKey($id)) { continue }
        $component = $components | Where-Object { [string]$_.componentId -eq $id } | Select-Object -First 1
        if ([bool]$component.required) {
            Stop-ReleaseGate "optional component must set required=false: $id"
        }
    }
    return $releaseVersion
}

function Assert-ReleaseArtifacts([object]$Release, [string]$ManifestVersion, [string[]]$ManifestHosts, [object]$BootstrapArtifact) {
    $expectedFields = @('schemaVersion', 'releaseVersion', 'publicationState', 'artifacts')
    $actualFields = @($Release.PSObject.Properties.Name)
    if ($actualFields.Count -ne $expectedFields.Count -or @($actualFields | Where-Object { $expectedFields -notcontains $_ }).Count -gt 0) {
        Stop-ReleaseGate 'release-artifacts.json top-level fields are invalid'
    }
    if ([Int32]$Release.schemaVersion -ne 1 -or [string]$Release.releaseVersion -ne $ManifestVersion) {
        Stop-ReleaseGate 'release-artifacts.json does not match the install manifest releaseVersion'
    }
    if ([string]$Release.publicationState -eq 'candidate') {
        Stop-ReleaseGate 'release-artifacts.json publicationState is candidate; a published Release identity is required'
    }
    if ([string]$Release.publicationState -ne 'published') {
        Stop-ReleaseGate 'release-artifacts.json publicationState is invalid'
    }
    $records = @($Release.artifacts)
    if ($records.Count -eq 0) { Stop-ReleaseGate 'release-artifacts.json has no published artifacts' }
    $ids = @{}
    $bootstrapRecord = $null
    foreach ($record in $records) {
        $id = [string](Get-RequiredProperty $record 'id' 'release artifact')
        if ($ids.ContainsKey($id)) { Stop-ReleaseGate "duplicate release artifact id: $id" }
        $ids[$id] = $true
        $url = [string](Get-RequiredProperty $record 'publishedUrl' "release artifact $id")
        try { $parsed = [System.Uri]$url } catch { Stop-ReleaseGate "release artifact URL is invalid: $url" }
        if ($parsed.Scheme -ne 'https' -or $parsed.UserInfo) { Stop-ReleaseGate "release artifact URL must use HTTPS: $url" }
        if ($ManifestHosts -notcontains $parsed.Host.ToLowerInvariant()) { Stop-ReleaseGate "release artifact URL host is not allowed: $($parsed.Host)" }
        if ([Int64](Get-RequiredProperty $record 'sizeBytes' "release artifact $id") -le 0 -or [string]$record.sha256 -notmatch '^[a-f0-9]{64}$') {
            Stop-ReleaseGate "release artifact identity is incomplete: $id"
        }
        if ($id -eq [string]$BootstrapArtifact.id) { $bootstrapRecord = $record }
    }
    if ($null -eq $bootstrapRecord) { Stop-ReleaseGate 'bootstrap artifact has no published Release identity' }
    if ([string]$bootstrapRecord.publishedUrl -ne [string]$BootstrapArtifact.url -or
        [Int64]$bootstrapRecord.sizeBytes -ne [Int64]$BootstrapArtifact.sizeBytes -or
        [string]$bootstrapRecord.sha256 -ne [string]$BootstrapArtifact.sha256) {
        Stop-ReleaseGate 'bootstrap artifact does not match release-artifacts.json'
    }
    return $bootstrapRecord
}

try {
    $root = [System.IO.Path]::GetFullPath($ProjectRoot)
    $manifestPath = Join-Path $root 'packaging\installer\install-manifest.json'
    $releasePath = Join-Path $root 'packaging\installer\release-artifacts.json'
    $ledgerPath = Join-Path $root 'packaging\installer\license-ledger.json'
    $frontendIndex = Join-Path $root 'frontend\dist\index.html'
    $noticesPath = Join-Path $root 'docs\THIRD_PARTY_NOTICES.md'
    $manifest = Read-JsonFile $manifestPath 'install-manifest.json'
    $release = Read-JsonFile $releasePath 'release-artifacts.json'
    $ledger = Read-JsonFile $ledgerPath 'license-ledger.json'
    $manifestVersion = Assert-Manifest $manifest
    $manifestHosts = @($manifest.allowedHosts | ForEach-Object { ([string]$_).ToLowerInvariant() })
    $bootstrapReleaseRecord = Assert-ReleaseArtifacts $release $manifestVersion $manifestHosts $manifest.bootstrap.artifact
    Assert-LicenseLedger $ledger $manifest
    if (-not (Test-Path -LiteralPath $frontendIndex -PathType Leaf)) { Stop-ReleaseGate 'frontend/dist/index.html is missing' }
    if (-not (Test-Path -LiteralPath $noticesPath -PathType Leaf)) { Stop-ReleaseGate 'third-party notices are missing' }
    $notices = [System.IO.File]::ReadAllText($noticesPath, [System.Text.Encoding]::UTF8)
    if ($notices -match '(?i)license\s+unverified') { Stop-ReleaseGate 'third-party notices still contain an unverified license status' }
    if ($VerifyPublishedBootstrap) {
        Test-PublishedBootstrapIdentity $bootstrapReleaseRecord $manifestHosts $root
        Write-Output ("Source-bootstrap release gate passed for {0}; frontend/dist is present and published bootstrap identity was byte-verified." -f $manifestVersion)
    } else {
        Write-Output ("Source-bootstrap release gate passed for {0}; frontend/dist is present and published bootstrap metadata matched without byte verification." -f $manifestVersion)
    }
    exit 0
} catch {
    Write-Output $_.Exception.Message
    exit 1
}
