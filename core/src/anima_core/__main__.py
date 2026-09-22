"""Minimal localhost-only core entry point for the stage 1 runtime bundle."""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path


def build_app():
    from .api import build_control_app

    return build_control_app()


def verify_embedded_runtime() -> None:
    """Fail closed unless this process is the verified distributed core runtime."""
    from .path_safety import windows_key
    from .pipeline import default_install_root
    from .runtime_manifest import RuntimeBundleManifestV1, RuntimeManifestError

    install_root = default_install_root()
    manifest = RuntimeBundleManifestV1.load(install_root / "manifests" / "runtimes" / "core.json")
    if manifest.runtime.runtimeId != "core" or manifest.runtime.owner != "core":
        raise RuntimeManifestError("core runtime manifest identity is invalid")
    interpreter = manifest.verify_interpreter(install_root)
    if windows_key(interpreter) != windows_key(Path(sys.executable)):
        raise RuntimeManifestError("core must run from its distributed embedded interpreter")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--static-root", type=Path)
    parser.add_argument("--resource-root", type=Path)
    parser.add_argument("--shutdown-token")
    parser.add_argument("--host-pid", type=int)
    args = parser.parse_args()
    try:
        verify_embedded_runtime()
    except Exception as exc:
        parser.error(f"embedded core runtime verification failed: {exc}")
    if args.check_runtime:
        print("anima-core-runtime-ok")
        return 0
    import uvicorn
    from .api import build_control_app

    if (args.static_root is None) != (args.shutdown_token is None):
        parser.error("--static-root and --shutdown-token must be supplied together")
    server: uvicorn.Server | None = None
    from .db import default_state_database_path
    from .pipeline import PipelineService
    from .resource_catalog import ResourceCatalog

    resource_catalog = ResourceCatalog(args.resource_root) if args.resource_root is not None else None
    pipeline = PipelineService(default_state_database_path(), resource_catalog=resource_catalog)

    def request_shutdown() -> None:
        def stop() -> None:
            pipeline.shutdown()
            assert server is not None
            server.should_exit = True
        threading.Thread(target=stop, daemon=True, name="anima-control-shutdown").start()

    if args.host_pid is not None:
        def watch_host_process() -> None:
            if sys.platform != "win32":
                return
            import ctypes

            synchronize = 0x00100000
            infinite = 0xFFFFFFFF
            handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, int(args.host_pid))
            if not handle:
                request_shutdown()
                return
            try:
                if ctypes.windll.kernel32.WaitForSingleObject(handle, infinite) == 0:
                    request_shutdown()
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)

        threading.Thread(target=watch_host_process, daemon=True, name="anima-webui-host-watch").start()

    app = build_control_app(
        pipeline_service=pipeline,
        static_root=args.static_root,
        shutdown_token=args.shutdown_token,
        shutdown_callback=request_shutdown if args.shutdown_token is not None else None,
    )
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="info"))
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
