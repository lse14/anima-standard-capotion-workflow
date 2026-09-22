from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class DesktopControlTests(unittest.TestCase):
    def test_start_and_stop_keep_the_control_script_while_install_uses_bootstrap(self) -> None:
        for action, filename in (("Start", "Start-WebUI.bat"), ("Stop", "Stop-WebUI.bat")):
            contents = (ROOT / filename).read_text(encoding="ascii")
            self.assertIn("desktop_control.ps1", contents)
            self.assertIn(f"-Action {action}", contents)
        install = (ROOT / "Install-WebUI.bat").read_text(encoding="ascii")
        self.assertIn("bootstrap_install.ps1", install)
        self.assertNotIn("desktop_control.ps1", install)

    def test_start_batch_keeps_a_double_click_failure_visible_with_log_guidance(self) -> None:
        contents = (ROOT / "Start-WebUI.bat").read_text(encoding="ascii")
        self.assertIn("set \"exitCode=%ERRORLEVEL%\"", contents)
        self.assertIn('if "%exitCode%"=="0" exit /b 0', contents)
        self.assertIn("goto :failed", contents)
        self.assertIn(".runtime-build\\launcher", contents)
        self.assertIn("pause", contents.lower())
        self.assertIn("exit /b %exitCode%", contents)

    def test_launcher_uses_only_the_embedded_core_and_safe_stop_protocol(self) -> None:
        script = (ROOT / "packaging" / "scripts" / "desktop_control.ps1").read_text(encoding="utf-8")
        host = (ROOT / "packaging" / "scripts" / "webui_host.ps1").read_text(encoding="utf-8")
        self.assertIn("runtimes\\core\\python.exe", script)
        self.assertIn("-B -I -m anima_core", host)
        self.assertIn("--check-runtime", script)
        self.assertIn("webui-", script)
        self.assertIn("Get-InstallationStateId", script)
        self.assertIn("Get-WebUiInstanceId", script)
        self.assertIn('("webui-{0}.json" -f $instanceId)', script)
        self.assertIn('("webui-{0}.stdout.log" -f $InstanceId)', host)
        self.assertIn('("webui-{0}.stderr.log" -f $InstanceId)', host)
        self.assertNotIn("legacyStatePath", script)
        self.assertIn("/api/application/shutdown", script)
        self.assertIn("Get-NetTCPConnection", script)
        self.assertIn("RNGCryptoServiceProvider", script)
        self.assertIn("$launcherRoot = Join-Path $projectRoot '.runtime-build\\launcher'", script)
        self.assertIn("cmd.exe", script)
        self.assertIn("hostPid", host)
        self.assertIn("backendPid", host)
        self.assertNotIn("LOCALAPPDATA", script)
        stop_body = script.partition("function Stop-WebUi")[2]
        self.assertNotIn("Stop-Process", stop_body)
        self.assertNotIn("taskkill", script.lower())

    def test_start_recovers_a_healthy_webui_when_the_state_file_is_missing(self) -> None:
        script = (ROOT / "packaging" / "scripts" / "desktop_control.ps1").read_text(encoding="utf-8")
        start_body = script.partition("function Start-WebUi")[2].partition("function Stop-WebUi")[0]
        stop_body = script.partition("function Stop-WebUi")[2]
        self.assertIn("netstat -ano -p tcp", script)
        self.assertIn("function Test-HealthyWebUi", script)
        self.assertIn("$health.protocolVersion -eq '1.0'", script)
        self.assertIn("$listeners = @(Get-Listener $Port)", start_body)
        self.assertIn("Test-HealthyWebUi $Port", start_body)
        self.assertIn('Start-Process "http://127.0.0.1:$Port/"', start_body)
        self.assertIn("(@(Get-Listener $Port).Count -eq 0)", stop_body)

    def test_install_requires_complete_state_and_never_prompts_for_ocr(self) -> None:
        script = (ROOT / "packaging" / "scripts" / "desktop_control.ps1").read_text(encoding="utf-8")
        batch = (ROOT / "Install-WebUI.bat").read_text(encoding="ascii")
        install_body = script.partition("'Install' {")[2].partition("'Start'")[0]
        self.assertIn("install-state.json", script)
        self.assertIn("Assert-ReleaseBundle", install_body)
        self.assertNotIn("Read-Host", script)
        self.assertNotIn("Install-OptionalOcr", script)
        self.assertNotIn("ocr_component.py", script)
        self.assertNotIn("OcrMode", script)
        self.assertNotIn("OcrMode", batch)

    def test_control_script_parses_in_windows_powershell(self) -> None:
        command = (
            "$tokens=$null; $errors=$null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            "(Resolve-Path 'packaging\\scripts\\desktop_control.ps1'),[ref]$tokens,[ref]$errors) | Out-Null; "
            "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)

    def test_same_installation_uses_a_distinct_instance_id_for_each_port(self) -> None:
        command = (
            "$tokens=$null; $errors=$null; "
            "$ast=[System.Management.Automation.Language.Parser]::ParseFile("
            "(Resolve-Path 'packaging\\scripts\\desktop_control.ps1'),[ref]$tokens,[ref]$errors); "
            "$wanted=@('Get-InstallationStateId','Get-WebUiInstanceId'); "
            "$nodes=$ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $wanted -contains $node.Name },$true); "
            "Invoke-Expression (($nodes | ForEach-Object { $_.Extent.Text }) -join \"`n\"); "
            "[ordered]@{first=(Get-WebUiInstanceId 'C:\\Program Files\\Anima Tool' 8765);second=(Get-WebUiInstanceId 'C:\\Program Files\\Anima Tool' 8766)} | ConvertTo-Json -Compress"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        values = json.loads(completed.stdout)
        self.assertNotEqual(values["first"], values["second"])
        self.assertTrue(values["first"].endswith("-8765"))
        self.assertTrue(values["second"].endswith("-8766"))
        self.assertEqual(values["first"].rsplit("-", 1)[0], values["second"].rsplit("-", 1)[0])


if __name__ == "__main__":
    unittest.main()
