from __future__ import annotations

import subprocess
import tempfile
import unittest
import json
import hashlib
import shutil
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "packaging" / "scripts" / "bootstrap_install.ps1"
RELEASE_VALIDATOR = ROOT / "packaging" / "scripts" / "Validate-SourceBootstrapRelease.ps1"
ACCEPTANCE_RUNNER = ROOT / "packaging" / "scripts" / "Invoke-SourceBootstrapAcceptance.ps1"
ACCEPTANCE_GUIDE = ROOT / "docs" / "SOURCE_BOOTSTRAP_ACCEPTANCE.md"


class SourceBootstrapPowerShellTests(unittest.TestCase):
    @staticmethod
    def _sha256(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def _write_published_release_fixture(self, root: Path) -> Path:
        source_artifacts = {
            "resource-library/classification-indexes/e621-classify-20260724-v1/resource.json": b"classification index\n",
            "resource-library/replacement-indexes/e621-replace-20260726-v2/resource.json": b"replacement index\n",
        }
        for relative, payload in source_artifacts.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

        remote_payload = b"fixture remote artifact"
        bootstrap_payload = b"fixture bootstrap artifact"
        mandatory = (
            "core", "caption-e621", "classify-e621", "replace-e621", "nl",
            "export", "token-budget", "e621-tagger", "qwen3-tokenizer",
        )
        optional = ("policy", "quality-stack")
        components: list[dict[str, object]] = []
        for component_id in mandatory + optional:
            components.append(
                {
                    "componentId": component_id,
                    "kind": "runtime",
                    "required": component_id not in optional,
                    "licenseReference": "fixture-license",
                    "targetRelativePath": f".runtime-build/runtimes/{component_id}",
                    "variants": {
                        "cpu": {
                            "artifacts": [{
                                "id": component_id + "-artifact",
                                "url": "https://downloads.example.test/artifacts/" + component_id + ".bin",
                                "allowedHosts": ["downloads.example.test"],
                                "sizeBytes": len(remote_payload),
                                "sha256": self._sha256(remote_payload),
                                "relativePath": "artifacts/" + component_id + ".bin",
                            }],
                            "peakBytes": 1,
                            "probe": "fixture",
                        }
                    },
                }
            )
        for component_id, target, source in (
            (
                "e621-indexes",
                "resource-library/classification-indexes/e621-classify-20260724-v1",
                "resource-library/classification-indexes/e621-classify-20260724-v1/resource.json",
            ),
            (
                "e621-replacement-indexes",
                "resource-library/replacement-indexes/e621-replace-20260726-v2",
                "resource-library/replacement-indexes/e621-replace-20260726-v2/resource.json",
            ),
        ):
            payload = source_artifacts[source]
            components.append(
                {
                    "componentId": component_id,
                    "kind": "resource",
                    "required": True,
                    "licenseReference": "fixture-license",
                    "targetRelativePath": target,
                    "variants": {
                        "shared": {
                            "artifacts": [{
                                "id": component_id + "-artifact",
                                "delivery": "source-tree",
                                "sourceRelativePath": source,
                                "sizeBytes": len(payload),
                                "sha256": self._sha256(payload),
                                "relativePath": "resource.json",
                            }],
                            "peakBytes": 1,
                            "probe": "indexes",
                        }
                    },
                }
            )

        manifest = {
            "schemaVersion": 1,
            "releaseVersion": "source-bootstrap-fixture-v1",
            "sourceCommit": "a" * 40,
            "allowedHosts": ["downloads.example.test"],
            "bootstrap": {
                "artifact": {
                    "id": "cpython311-base",
                    "url": "https://downloads.example.test/bootstrap/cpython.zip",
                    "allowedHosts": ["downloads.example.test"],
                    "sizeBytes": len(bootstrap_payload),
                    "sha256": self._sha256(bootstrap_payload),
                    "relativePath": "bootstrap/cpython.zip",
                },
                "entryRelativePath": "python.exe",
                "peakBytes": 1,
            },
            "components": components,
            "cleanup": {"successRelativePaths": [".runtime-build/source-bootstrap"]},
        }
        release = {
            "schemaVersion": 1,
            "releaseVersion": manifest["releaseVersion"],
            "publicationState": "published",
            "artifacts": [{
                "id": "cpython311-base",
                "publishedUrl": manifest["bootstrap"]["artifact"]["url"],
                "sizeBytes": len(bootstrap_payload),
                "sha256": self._sha256(bootstrap_payload),
            }],
        }
        installer = root / "packaging" / "installer"
        installer.mkdir(parents=True, exist_ok=True)
        (installer / "install-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (installer / "release-artifacts.json").write_text(json.dumps(release), encoding="utf-8")
        frontend = root / "frontend" / "dist"
        frontend.mkdir(parents=True, exist_ok=True)
        (frontend / "index.html").write_text("<!doctype html>", encoding="ascii")
        notices = root / "docs"
        notices.mkdir(parents=True, exist_ok=True)
        (notices / "THIRD_PARTY_NOTICES.md").write_text("fixture notices\n", encoding="utf-8")
        return root / "resource-library" / "classification-indexes" / "e621-classify-20260724-v1" / "resource.json"

    def _write_license_ledger_fixture(self, root: Path) -> Path:
        ledger_path = root / "packaging" / "installer" / "license-ledger.json"
        ledger = {
            "schemaVersion": 1,
            "entries": [{
                "id": "fixture-license",
                "delivery": "direct-upstream-only",
                "officialSourceUrl": "https://downloads.example.test/licenses/fixture",
                "licenseEvidenceUrl": "https://downloads.example.test/licenses/fixture",
                "evidenceRetrievedAtUtc": "2026-08-12T00:00:00Z",
                "evidenceSha256": self._sha256(b"fixture evidence"),
                "reviewStatus": "evidence-collected",
                "redistributionStatus": "not-mirrored",
            }],
            "decisions": [],
        }
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
        return ledger_path

    def _bootstrap_text(self) -> str:
        self.assertTrue(BOOTSTRAP.is_file(), "source bootstrap PowerShell script must exist")
        return BOOTSTRAP.read_text(encoding="utf-8")

    def test_source_entrypoint_uses_bootstrap_not_legacy_ocr_prompt(self) -> None:
        batch = (ROOT / "Install-WebUI.bat").read_text(encoding="ascii")
        script = self._bootstrap_text()

        self.assertIn("bootstrap_install.ps1", batch)
        self.assertNotIn("desktop_control.ps1", batch)
        self.assertNotIn("OcrMode", batch)
        self.assertIn("[Environment]::Is64BitOperatingSystem", script)
        self.assertIn("Get-Volume", script)
        self.assertIn("install-manifest.json", script)
        self.assertIn("Get-Sha256Hex", script)
        self.assertNotIn("Get-FileHash", script)
        self.assertIn("Range", script)
        self.assertIn("install.py", script)
        self.assertIn("--bootstrap-runtime", script)
        self.assertNotIn("LOCALAPPDATA", script)

    def test_bootstrap_streams_unbuffered_installer_progress(self) -> None:
        script = self._bootstrap_text()

        self.assertIn("-B -I -u $installerScript", script)
        self.assertRegex(script, r"2>&1\s*\|\s*ForEach-Object")
        self.assertNotIn("$output = & $bootstrapPython", script)

    def test_bootstrap_manifest_identity_matches_checked_in_release_manifest(self) -> None:
        script = self._bootstrap_text()
        match = re.search(r"\$ExpectedInstallManifestSha256\s*=\s*'([0-9a-f]{64})'", script)
        self.assertIsNotNone(match, "bootstrap must pin a published install manifest identity")
        manifest_path = ROOT / "packaging" / "installer" / "install-manifest.json"
        self.assertEqual(match.group(1), self._sha256(manifest_path.read_bytes()))
        self.assertEqual("source-bootstrap-e621-v8", json.loads(manifest_path.read_text(encoding="utf-8"))["releaseVersion"])

    def test_start_batch_recovers_missing_installation_state_through_the_bootstrap(self) -> None:
        batch = (ROOT / "Start-WebUI.bat").read_text(encoding="ascii")

        self.assertIn('if not exist "%~dp0.runtime-build\\manifests\\install-state.json" goto :bootstrap', batch)
        self.assertIn(":bootstrap", batch)
        self.assertIn('call "%~dp0Install-WebUI.bat"', batch)
        self.assertIn("resuming source bootstrap", batch)

    def test_start_batch_propagates_bootstrap_failure_when_installation_state_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            project_root = Path(temporary_name)
            start = project_root / "Start-WebUI.bat"
            start.write_bytes((ROOT / "Start-WebUI.bat").read_bytes())
            (project_root / "Install-WebUI.bat").write_text(
                "@echo off\r\nexit /b 7\r\n", encoding="ascii",
            )

            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", str(start)],
                cwd=project_root,
                input="\n",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="cp936",
                errors="replace",
                check=False,
                timeout=10,
            )

            self.assertEqual(7, completed.returncode, completed.stdout + completed.stderr)

    def test_bootstrap_parses_in_windows_powershell(self) -> None:
        self.assertTrue(BOOTSTRAP.is_file(), "source bootstrap PowerShell script must exist")
        command = (
            "$tokens=$null; $errors=$null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            "(Resolve-Path 'packaging\\scripts\\bootstrap_install.ps1'),[ref]$tokens,[ref]$errors) | Out-Null; "
            "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)

    def test_bootstrap_parenthesizes_test_path_before_boolean_operators(self) -> None:
        script = self._bootstrap_text()
        invalid = re.findall(r"if \(Test-Path[^\r\n]+ -(?:and|or)\b", script)
        self.assertEqual([], invalid)

    def test_bootstrap_normalizes_safe_relative_paths_without_accepting_absolute_paths(self) -> None:
        self.assertTrue(BOOTSTRAP.is_file(), "source bootstrap PowerShell script must exist")
        command = (
            "$tokens=$null; $errors=$null; "
            "$ast=[System.Management.Automation.Language.Parser]::ParseFile("
            "(Resolve-Path 'packaging\\scripts\\bootstrap_install.ps1'),[ref]$tokens,[ref]$errors); "
            "if ($errors.Count) { exit 1 }; "
            "$node=$ast.FindAll({ param($item) $item -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $item.Name -eq 'Assert-SafeRelativePath' },$true) | Select-Object -First 1; "
            "Invoke-Expression $node.Extent.Text; "
            "$rejected=$false; try { Assert-SafeRelativePath 'C:\\outside' | Out-Null } catch { $rejected=$true }; "
            "$trailingRejected=$false; try { Assert-SafeRelativePath 'Lib/' | Out-Null } catch { $trailingRejected=$true }; "
            "[ordered]@{value=(Assert-SafeRelativePath 'Lib/python.exe');directory=(Assert-SafeRelativePath 'Lib/' -AllowDirectory);rejected=$rejected;trailingRejected=$trailingRejected} | ConvertTo-Json -Compress"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        value = json.loads(completed.stdout)
        self.assertEqual(r"Lib\python.exe", value["value"])
        self.assertEqual("Lib", value["directory"])
        self.assertTrue(value["rejected"])
        self.assertTrue(value["trailingRejected"])

    def test_bootstrap_peak_calculation_skips_delayed_and_unavailable_optional_components(self) -> None:
        command = (
            "$tokens=$null; $errors=$null; "
            "$ast=[System.Management.Automation.Language.Parser]::ParseFile("
            "(Resolve-Path 'packaging\\scripts\\bootstrap_install.ps1'),[ref]$tokens,[ref]$errors); "
            "if ($errors.Count) { exit 1 }; "
            "foreach ($name in @('Get-RequiredProperty','Get-RequiredPeakBytes')) { "
            "$node=$ast.FindAll({ param($item) $item -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $item.Name -eq $name },$true) | Select-Object -First 1; "
            "Invoke-Expression $node.Extent.Text }; "
            "$manifest=[pscustomobject]@{bootstrap=[pscustomobject]@{peakBytes=1;artifact=[pscustomobject]@{sizeBytes=2}};components=@("
            "[pscustomobject]@{componentId='core';required=$true;variants=[pscustomobject]@{cpu=[pscustomobject]@{peakBytes=3;artifacts=@([pscustomobject]@{sizeBytes=4})}}},"
            "[pscustomobject]@{componentId='ocr-cpu';required=$false;variants=[pscustomobject]@{cpu=[pscustomobject]@{peakBytes=50;artifacts=@([pscustomobject]@{sizeBytes=60})}}},"
            "[pscustomobject]@{componentId='ocr-gpu';required=$false;variants=[pscustomobject]@{cuda=[pscustomobject]@{peakBytes=100;artifacts=@([pscustomobject]@{sizeBytes=200})}}},"
            "[pscustomobject]@{componentId='fixture-shared';required=$true;variants=[pscustomobject]@{shared=[pscustomobject]@{peakBytes=5;artifacts=@([pscustomobject]@{sizeBytes=6})}}},"
            "[pscustomobject]@{componentId='ocr-models';required=$false;variants=[pscustomobject]@{shared=[pscustomobject]@{peakBytes=300;artifacts=@([pscustomobject]@{sizeBytes=400})}}}"
            ")}; Get-RequiredPeakBytes $manifest $false"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        self.assertEqual("21", completed.stdout.strip())

    def test_bootstrap_missing_manifest_fails_with_a_project_local_utf8_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            project_root = Path(temporary_name)
            completed = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(BOOTSTRAP), "-ProjectRoot", str(project_root),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, completed.returncode)
            logs = list((project_root / ".runtime-build" / "logs").glob("source-bootstrap-*.log"))
            self.assertEqual(1, len(logs))
            self.assertIn("Frozen install manifest is missing", logs[0].read_text(encoding="utf-8"))
            self.assertFalse((project_root / ".runtime-build" / "source-bootstrap").exists())

    def test_install_batch_passes_a_valid_project_root_to_powershell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            project_root = Path(temporary_name)
            script_dir = project_root / "packaging" / "scripts"
            script_dir.mkdir(parents=True)
            (project_root / "Install-WebUI.bat").write_bytes((ROOT / "Install-WebUI.bat").read_bytes())
            (script_dir / "bootstrap_install.ps1").write_bytes(BOOTSTRAP.read_bytes())
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", str(project_root / "Install-WebUI.bat")],
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="cp936",
                errors="replace",
                check=False,
            )
            output = completed.stdout + completed.stderr
            self.assertNotEqual(0, completed.returncode)
            self.assertIn("Frozen install manifest is missing", output)
            self.assertNotIn("Illegal characters in path", output)

    def test_bootstrap_has_a_private_success_cleanup_path(self) -> None:
        script = self._bootstrap_text()

        self.assertIn("function Clear-BootstrapSuccessArtifacts", script)
        self.assertIn("Clear-BootstrapSuccessArtifacts", script)
        self.assertIn("Join-Path $script:runtimeBuildRoot 'source-bootstrap'", script)

    def test_failure_cleanup_removes_bootstrap_cache_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            installer = root / ".runtime-build" / "source-bootstrap"
            cache = installer / "cache"
            cache.mkdir(parents=True)
            complete = cache / ("a" * 64)
            partial = cache / (("b" * 64) + ".partial")
            root_literal = str(root).replace("'", "''")
            complete_literal = str(complete).replace("'", "''")
            partial_literal = str(partial).replace("'", "''")
            complete.write_bytes(b"verified bootstrap archive")
            partial.write_bytes(b"resumable bootstrap archive")
            for name in ("staging", "bootstrap", "transactions"):
                (installer / name / "temporary.txt").parent.mkdir(parents=True, exist_ok=True)
                (installer / name / "temporary.txt").write_text("temporary", encoding="ascii")
            command = (
                "$tokens=$null; $errors=$null; "
                "$ast=[System.Management.Automation.Language.Parser]::ParseFile("
                "(Resolve-Path 'packaging\\scripts\\bootstrap_install.ps1'),[ref]$tokens,[ref]$errors); "
                "if($errors.Count){exit 1}; "
                "$wanted=@('Test-ReparsePoint','Get-ProjectPath','Clear-BootstrapFailureArtifacts'); "
                "$nodes=$ast.FindAll({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $wanted -contains $node.Name},$true); "
                "Invoke-Expression (($nodes | ForEach-Object {$_.Extent.Text}) -join \"`n\"); "
                f"$script:projectRoot='{root_literal}'; "
                "$script:runtimeBuildRoot=Join-Path $script:projectRoot '.runtime-build'; "
                f"$script:bootstrapComplete='{complete_literal}'; "
                "Clear-BootstrapFailureArtifacts; "
                f"[ordered]@{{complete=(Test-Path -LiteralPath '{complete_literal}');partial=(Test-Path -LiteralPath '{partial_literal}');staging=(Test-Path -LiteralPath (Join-Path $script:runtimeBuildRoot 'source-bootstrap\\staging'));bootstrap=(Test-Path -LiteralPath (Join-Path $script:runtimeBuildRoot 'source-bootstrap\\bootstrap'));transactions=(Test-Path -LiteralPath (Join-Path $script:runtimeBuildRoot 'source-bootstrap\\transactions'))}} | ConvertTo-Json -Compress"
            )
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", command], cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
            )
            self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
            result = json.loads(completed.stdout)
            self.assertFalse(result["complete"])
            self.assertFalse(result["partial"])
            self.assertFalse(result["staging"])
            self.assertFalse(result["bootstrap"])
            self.assertFalse(result["transactions"])

    def test_failure_cleanup_cannot_replace_the_original_installer_error(self) -> None:
        script = self._bootstrap_text()
        outer_catch = script.rfind("} catch {\n    $originalError = $_")
        self.assertNotEqual(-1, outer_catch)
        catch_block = script[outer_catch:]

        self.assertIn("$originalError = $_", catch_block)
        self.assertLess(catch_block.index("$originalError = $_"), catch_block.index("Clear-BootstrapFailureArtifacts"))
        self.assertIn("Bootstrap failure cleanup also failed", catch_block)
        self.assertIn("$originalError.Exception.Message", catch_block)

    def test_bootstrap_starts_webui_and_mentions_the_manual_ocr_guide(self) -> None:
        script = self._bootstrap_text()

        self.assertIn("desktop_control.ps1", script)
        self.assertIn("Invoke-DesktopControlStart", script)
        self.assertIn("'-Action', 'Start'", script)
        self.assertIn("OCR_MODEL_DOWNLOAD.md", script)
        self.assertNotIn("OcrMode", script)

    def test_bootstrap_retries_a_transient_webui_start_and_logs_child_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            log_path = Path(temporary_name) / "bootstrap.log"
            log_literal = str(log_path).replace("'", "''")
            command = (
                "$tokens=$null; $errors=$null; "
                "$ast=[System.Management.Automation.Language.Parser]::ParseFile("
                "(Resolve-Path 'packaging\\scripts\\bootstrap_install.ps1'),[ref]$tokens,[ref]$errors); "
                "if($errors.Count){exit 1}; "
                "$wanted=@('Write-InstallLog','Start-InstalledWebUi'); "
                "$nodes=$ast.FindAll({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $wanted -contains $node.Name},$true); "
                "$start=$nodes | Where-Object Name -eq 'Start-InstalledWebUi'; "
                "if($null -eq $start){Write-Error 'Start-InstalledWebUi is missing'; exit 2}; "
                "Invoke-Expression (($nodes | ForEach-Object {$_.Extent.Text}) -join \"`n\"); "
                f"$script:logPath='{log_literal}'; "
                "$script:utf8NoBom=New-Object System.Text.UTF8Encoding($false); "
                "$script:attempts=0; "
                "function Invoke-DesktopControlStart { "
                "param([string]$DesktopControl,[int]$Attempt); "
                "$script:attempts++; "
                f"$stdout='{log_literal}.attempt-'+$Attempt+'.stdout'; "
                f"$stderr='{log_literal}.attempt-'+$Attempt+'.stderr'; "
                "if($script:attempts -eq 1){[IO.File]::WriteAllText($stdout,'transient core verification failure');$exitCode=1} "
                "else{[IO.File]::WriteAllText($stdout,'webui ready');$exitCode=0}; "
                "[pscustomobject]@{exitCode=$exitCode;stdoutPath=$stdout;stderrPath=$stderr} }; "
                "Start-InstalledWebUi 'C:\\fixture\\desktop_control.ps1'; "
                f"[ordered]@{{attempts=$script:attempts;log=[IO.File]::ReadAllText('{log_literal}')}} | ConvertTo-Json -Compress"
            )
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", command], cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
            result = json.loads(completed.stdout.splitlines()[-1])
            self.assertEqual(2, result["attempts"])
            self.assertIn("transient core verification failure", result["log"])
            self.assertIn("retrying", result["log"].lower())

    def test_bootstrap_webui_start_does_not_wait_for_descendant_output_handles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            desktop_control = temporary / "desktop_control.ps1"
            descendant_pid = temporary / "descendant.pid"
            descendant_pid_literal = str(descendant_pid).replace("'", "''")
            desktop_control.write_text(
                "$info=New-Object System.Diagnostics.ProcessStartInfo\n"
                "$info.FileName='powershell.exe'\n"
                "$info.Arguments='-NoProfile -Command Start-Sleep -Seconds 8'\n"
                "$info.UseShellExecute=$false\n"
                "$child=[System.Diagnostics.Process]::Start($info)\n"
                f"[IO.File]::WriteAllText('{descendant_pid_literal}',[string]$child.Id)\n"
                "Write-Output 'webui ready'\n",
                encoding="utf-8",
            )
            log_path = temporary / "bootstrap.log"
            control_literal = str(desktop_control).replace("'", "''")
            log_literal = str(log_path).replace("'", "''")
            command = (
                "$tokens=$null; $errors=$null; "
                "$ast=[System.Management.Automation.Language.Parser]::ParseFile("
                "(Resolve-Path 'packaging\\scripts\\bootstrap_install.ps1'),[ref]$tokens,[ref]$errors); "
                "if($errors.Count){exit 1}; "
                "$wanted=@('Write-InstallLog','Invoke-DesktopControlStart','Start-InstalledWebUi'); "
                "$nodes=$ast.FindAll({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $wanted -contains $node.Name},$true); "
                "Invoke-Expression (($nodes | ForEach-Object {$_.Extent.Text}) -join \"`n\"); "
                f"$script:logPath='{log_literal}'; "
                "$script:utf8NoBom=New-Object System.Text.UTF8Encoding($false); "
                f"$elapsed=Measure-Command {{ Start-InstalledWebUi '{control_literal}' }}; "
                "[ordered]@{seconds=$elapsed.TotalSeconds} | ConvertTo-Json -Compress"
            )
            try:
                completed = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-Command", command], cwd=ROOT,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=20,
                )

                self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
                result = json.loads(completed.stdout.splitlines()[-1])
                self.assertLess(result["seconds"], 4.0)
            finally:
                if descendant_pid.is_file():
                    pid = int(descendant_pid.read_text(encoding="utf-8"))
                    subprocess.run(
                        [
                            "powershell.exe", "-NoProfile", "-Command",
                            f"$process=Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
                            f"if($process){{Wait-Process -Id {pid} -Timeout 10}}",
                        ],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        check=False, timeout=12,
                    )

    def test_release_validator_treats_ocr_runtime_and_models_as_optional(self) -> None:
        self.assertTrue(RELEASE_VALIDATOR.is_file(), "source bootstrap release validator must exist")
        script = RELEASE_VALIDATOR.read_text(encoding="utf-8")

        self.assertIn("'ocr-cpu'", script)
        self.assertIn("'ocr-gpu'", script)
        self.assertIn("'ocr-models'", script)
        self.assertIn("optional OCR", script)

    def test_release_validator_has_explicit_published_bootstrap_verification_mode(self) -> None:
        self.assertTrue(RELEASE_VALIDATOR.is_file(), "source bootstrap release validator must exist")
        script = RELEASE_VALIDATOR.read_text(encoding="utf-8")
        self.assertIn("VerifyPublishedBootstrap", script)
        self.assertIn("publicationState", script)
        self.assertIn("candidate", script)
        self.assertIn("does not match release-artifacts.json", script)
        self.assertIn("without byte verification", script)
        self.assertNotIn(
            "frontend/dist and published bootstrap identity are present.",
            script,
        )

    def test_release_validator_rejects_source_tree_payload_under_direct_upstream_license(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            self._write_published_release_fixture(root)
            self._write_license_ledger_fixture(root)
            payload = b"redistributed index payload\n"
            relative = "resource-library/replacement-indexes/e621-replace-20260726-v2/e621_tag_replacement_index.csv"
            path = root / relative
            path.write_bytes(payload)
            manifest_path = root / "packaging" / "installer" / "install-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for component in manifest["components"]:
                if component["componentId"] != "e621-replacement-indexes":
                    continue
                component["variants"]["shared"]["artifacts"].append({
                    "id": "e621-replacement-payload",
                    "delivery": "source-tree",
                    "sourceRelativePath": relative,
                    "sizeBytes": len(payload),
                    "sha256": self._sha256(payload),
                    "relativePath": "e621_tag_replacement_index.csv",
                })
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            completed = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(RELEASE_VALIDATOR), "-ProjectRoot", str(root),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, completed.returncode)
            self.assertIn(
                "source-tree payload requires approved source-redistributed license",
                completed.stdout + completed.stderr,
            )

    def test_release_validator_verifies_source_tree_artifact_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            source = self._write_published_release_fixture(root)
            self._write_license_ledger_fixture(root)
            completed = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(RELEASE_VALIDATOR), "-ProjectRoot", str(root),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)

            source.write_bytes(b"tampered source artifact\n")
            tampered = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(RELEASE_VALIDATOR), "-ProjectRoot", str(root),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(0, tampered.returncode)
            self.assertIn("source-tree artifact identity does not match", tampered.stdout + tampered.stderr)

    def test_release_validator_accepts_strict_utc_ledger_timestamps_in_powershell_7(self) -> None:
        pwsh = shutil.which("pwsh")
        if pwsh is None:
            self.skipTest("PowerShell 7 is not available")
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            self._write_published_release_fixture(root)
            self._write_license_ledger_fixture(root)
            completed = subprocess.run(
                [pwsh, "-NoProfile", "-File", str(RELEASE_VALIDATOR), "-ProjectRoot", str(root)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)

    def test_release_validator_rejects_invalid_utc_calendar_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            self._write_published_release_fixture(root)
            ledger_path = self._write_license_ledger_fixture(root)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["entries"][0]["evidenceRetrievedAtUtc"] = "2026-99-99T00:00:00Z"
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RELEASE_VALIDATOR), "-ProjectRoot", str(root)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(0, completed.returncode)
            self.assertIn("is not an ISO-8601 UTC timestamp", completed.stdout + completed.stderr)

    def test_release_validator_rejects_missing_license_ledger_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            self._write_published_release_fixture(root)
            self._write_license_ledger_fixture(root)
            manifest_path = root / "packaging" / "installer" / "install-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["components"][0]["licenseReference"] = "missing-license"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            completed = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(RELEASE_VALIDATOR), "-ProjectRoot", str(root),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, completed.returncode)
            self.assertIn("license ledger entry is missing", completed.stdout + completed.stderr)

    def test_release_validator_rejects_mirrored_local_only_ocr_license_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            self._write_published_release_fixture(root)
            ledger_path = self._write_license_ledger_fixture(root)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["entries"][0]["delivery"] = "local-only"
            ledger["entries"][0]["redistributionStatus"] = "approved"
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

            completed = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(RELEASE_VALIDATOR), "-ProjectRoot", str(root),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, completed.returncode)
            self.assertIn("local-only license entry must not be mirrored", completed.stdout + completed.stderr)

    def test_release_validator_rejects_pending_e621_redistribution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            self._write_published_release_fixture(root)
            ledger_path = self._write_license_ledger_fixture(root)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["entries"][0] = {
                "id": "fixture-license",
                "delivery": "source-redistributed",
                "officialSourceUrl": "https://e621.net/terms_of_use",
                "licenseEvidenceUrl": "https://e621.net/terms_of_use",
                "evidenceRetrievedAtUtc": "2026-08-12T00:00:00Z",
                "evidenceSha256": self._sha256(b"fixture evidence"),
                "reviewStatus": "evidence-collected",
                "redistributionStatus": "pending-human-review",
            }
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

            completed = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(RELEASE_VALIDATOR), "-ProjectRoot", str(root),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, completed.returncode)
            self.assertIn("redistribution is not approved", completed.stdout + completed.stderr)

    def test_release_validator_requires_human_decision_evidence_for_approved_source_redistribution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            self._write_published_release_fixture(root)
            ledger_path = self._write_license_ledger_fixture(root)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["entries"][0] = {
                "id": "fixture-license",
                "delivery": "source-redistributed",
                "officialSourceUrl": "https://e621.net/terms_of_use",
                "licenseEvidenceUrl": "https://e621.net/terms_of_use",
                "evidenceRetrievedAtUtc": "2026-08-13T00:00:00Z",
                "evidenceSha256": self._sha256(b"fixture evidence"),
                "reviewStatus": "evidence-collected",
                "redistributionStatus": "approved",
            }
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

            missing_decision = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(RELEASE_VALIDATOR), "-ProjectRoot", str(root),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, missing_decision.returncode)
            self.assertIn("source-redistributed approval decision is missing", missing_decision.stdout + missing_decision.stderr)

            source_paths = (
                "resource-library/classification-indexes/e621-classify-20260724-v1/resource.json",
                "resource-library/replacement-indexes/e621-replace-20260726-v2/resource.json",
            )
            ledger["decisions"] = [{
                "id": "fixture-e621-project-owner-decision",
                "licenseReference": "fixture-license",
                "source": "user-confirmed-project-owner",
                "decidedAtUtc": "2026-08-13T00:00:00Z",
                "termsUrl": "https://e621.net/terms_of_use",
                "termsSha256": self._sha256(b"fixture evidence"),
                "approvedArtifacts": [{
                    "sourceRelativePath": relative,
                    "sizeBytes": (root / relative).stat().st_size,
                    "sha256": self._sha256((root / relative).read_bytes()),
                } for relative in source_paths],
            }]
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

            approved = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(RELEASE_VALIDATOR), "-ProjectRoot", str(root),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(0, approved.returncode, approved.stdout + approved.stderr)

    def test_acceptance_runner_has_clean_host_and_project_local_evidence_contract(self) -> None:
        self.assertTrue(ACCEPTANCE_RUNNER.is_file(), "source bootstrap acceptance runner must exist")
        script = ACCEPTANCE_RUNNER.read_text(encoding="utf-8")

        self.assertIn("ValidateSet('Cpu', 'Nvidia')", script)
        self.assertIn("PreflightOnly", script)
        self.assertIn(".runtime-build\\acceptance", script)
        for command_name in ("python", "py", "node", "npm", "nvcc", "cl"):
            self.assertIn("'" + command_name + "'", script)
        self.assertIn("Windows Kits", script)
        self.assertIn("Install-WebUI.bat", script)
        self.assertIn("Stop-WebUI.bat", script)
        self.assertIn("webUiStartExitCode", script)
        self.assertIn("installerInvoked", script)
        self.assertIn("finally", script)
        self.assertIn("Clean-host preflight failed", script)
        self.assertIn("not-clean", script)
        self.assertNotIn('"status":"passed"', script)

        self.assertTrue(ACCEPTANCE_GUIDE.is_file(), "source bootstrap acceptance guide must exist")
        guide = ACCEPTANCE_GUIDE.read_text(encoding="utf-8")
        for scenario in (
            "Windows 10 CPU",
            "Windows 11 CPU interrupted-download",
            "Windows 11 NVIDIA",
            "Chinese/space source ZIP path",
        ):
            self.assertIn(scenario, guide)
        self.assertIn("SOURCE_BOOTSTRAP_ACCEPTANCE.md", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn("!docs/SOURCE_BOOTSTRAP_ACCEPTANCE.md", (ROOT / ".gitignore").read_text(encoding="utf-8"))

    def test_acceptance_not_clean_run_does_not_invoke_stop_webui(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            stop = root / "Stop-WebUI.bat"
            marker = root / "stop-called.txt"
            stop.write_text(f'@echo stopped> "{marker}"\r\n', encoding="ascii")
            completed = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(ACCEPTANCE_RUNNER), "-ProjectRoot", str(root), "-Scenario", "Cpu",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            results = list((root / ".runtime-build" / "acceptance").glob("source-bootstrap-cpu-*.json"))
            self.assertEqual(1, len(results), completed.stdout + completed.stderr)
            result = json.loads(results[0].read_text(encoding="utf-8"))
            self.assertEqual("not-clean", result["status"])
            self.assertIsNone(result.get("stopExitCode"))
            self.assertFalse(marker.exists(), completed.stdout + completed.stderr)
            self.assertNotEqual(0, completed.returncode)

    def test_acceptance_preflight_writes_non_passing_project_local_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            completed = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(ACCEPTANCE_RUNNER), "-ProjectRoot", str(root), "-Scenario", "Cpu", "-PreflightOnly",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            results = list((root / ".runtime-build" / "acceptance").glob("source-bootstrap-cpu-*.json"))
            self.assertEqual(1, len(results), completed.stdout + completed.stderr)
            result = json.loads(results[0].read_text(encoding="utf-8"))
            self.assertEqual("not-clean", result["status"])
            self.assertNotEqual(0, completed.returncode, completed.stdout + completed.stderr)

    def test_acceptance_only_reports_passed_after_stop_succeeds(self) -> None:
        script = ACCEPTANCE_RUNNER.read_text(encoding="utf-8")
        passed_index = script.index("$result.status = 'passed'")
        stop_check_index = script.index("$result.stopExitCode -ne 0")
        self.assertGreater(passed_index, stop_check_index)

    def test_acceptance_stop_failure_or_missing_script_cannot_pass(self) -> None:
        script = ACCEPTANCE_RUNNER.read_text(encoding="utf-8")
        self.assertIn("Stop-WebUI.bat is missing", script)
        self.assertIn("$result.stopExitCode -ne 0", script)
        self.assertIn("$result.status = 'failed'", script)


if __name__ == "__main__":
    unittest.main()
