"""Core-owned, bounded orchestration for Dropout's isolated policy worker."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Protocol

from .classify_overlay import ClassifyJsonError, parse_annotation_json
from .contracts import (
    CURRENT_JOB_CONFIG_SCHEMA_VERSION,
    ProgressEvent,
    SampleIssue,
    WorkLease,
    sha256_json,
)
from .db import StateDatabase
from .overlay import OverlayLayout, WorkingAnnotationView
from .path_safety import PathSafetyError, ensure_within, safe_relative_path, sha256_file
from .scheduler import BoundedScheduler, SchedulerError
from .stdio_transport import StdioJsonlTransportError
from .worker_protocol import ProtocolEnvelopeV1, ProtocolError


RUNTIME_ID = "policy"
OWNER = "policy"
RESOURCE_MANIFEST_PATH = "dropout-models\\lse14-scorer-5k-v1\\resource.json"
PROTECTED_FIELDS = ("count", "character", "series", "tags", "environment")
QUALITY_DIAGNOSTICS = {
    (): "policy_quality_empty",
    ("low quality",): "policy_quality_low",
    ("normal quality",): "policy_quality_normal",
    ("good quality",): "policy_quality_good",
    ("masterpiece", "best quality"): "policy_quality_best",
}


class PolicyTransport(Protocol):
    def exchange(self, request: ProtocolEnvelopeV1) -> ProtocolEnvelopeV1: ...


class PolicyRunnerError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PolicyRunner:
    """The core validates and commits prepared overlay JSON; the worker never owns a lease."""

    def __init__(
        self, database: StateDatabase, scheduler: BoundedScheduler, transport: PolicyTransport,
        view: WorkingAnnotationView, *, job_id: str, worker_instance_id: str, install_root: str | Path,
        resource_manifest_relative_path: str | None = None,
        resource_fingerprint: str | None = None,
    ) -> None:
        if not worker_instance_id or len(worker_instance_id) > 128:
            raise ValueError("policy worker instance id must be a non-empty bounded string")
        self.database = database
        self.scheduler = scheduler
        self.transport = transport
        self.view = view
        self.job_id = job_id
        self.worker_instance_id = worker_instance_id
        self.install_root = Path(install_root)
        self.resource_manifest_relative_path = resource_manifest_relative_path
        self.resource_fingerprint = resource_fingerprint
        self._hello_done = False

    def _publish(self, status: str, attempt: int = 0) -> None:
        summary = self.database.module_summary(self.job_id, "dropout")
        settled = int(summary["completed"] + summary["failed"] + summary["skipped"])
        self.database.append_event(ProgressEvent(
            self.job_id, int(self.database.get_job(self.job_id)["last_event_id"]) + 1, "dropout", status,
            settled, int(summary["total"]), str(self.database.get_job(self.job_id)["config_hash"]), attempt,
        ))

    def _fatal(self, code: str, message: str) -> PolicyRunnerError:
        return PolicyRunnerError(code, message)

    def _config(self) -> tuple[str, dict[str, object], str, str | None]:
        job = self.database.get_job(self.job_id)
        try:
            config = json.loads(str(job["config_json"]))
        except json.JSONDecodeError as exc:
            raise self._fatal("policy_protocol_violation", "frozen JobConfig is invalid JSON") from exc
        schema_version = config.get("schemaVersion") if isinstance(config, dict) else None
        if (
            not isinstance(config, dict)
            or sha256_json(config) != job["config_hash"]
            or schema_version != CURRENT_JOB_CONFIG_SCHEMA_VERSION
            or "profile" in config
            or schema_version != int(job["config_schema_version"])
            or not isinstance(config.get("dropout"), dict)
        ):
            raise self._fatal("policy_protocol_violation", "frozen policy configuration is invalid")
        dropout = dict(config["dropout"])
        if dropout.get("enabled") is not True:
            raise self._fatal("policy_protocol_violation", "policy runner cannot execute a disabled module")
        policy = {key: value for key, value in dropout.items() if key != "enabled"}
        quality = policy.get("quality")
        if not isinstance(quality, dict) or type(quality.get("enabled")) is not bool:
            raise self._fatal("policy_protocol_violation", "frozen policy quality configuration is invalid")
        worker_quality = {
            key: value
            for key, value in quality.items()
            if key not in {"resourceManifestRelativePath", "resourceFingerprint"}
        }
        policy["quality"] = worker_quality
        if quality["enabled"]:
            manifest, fingerprint, resource_id = self._resource()
            configured_id = worker_quality.get("resourceId")
            if configured_id is None:
                worker_quality["resourceId"] = resource_id
            elif configured_id != resource_id:
                raise self._fatal("policy_resource_invalid", "selected policy resource does not match JobConfig")
            return str(job["config_hash"]), policy, manifest, fingerprint
        return str(job["config_hash"]), policy, "", None

    def _manifest_resource_id(self, relative_path: str) -> str:
        try:
            relative = Path(safe_relative_path(relative_path).replace("\\", os.sep))
            path = ensure_within(self.install_root, self.install_root / relative)
            value = json.loads(path.read_text(encoding="utf-8"))
        except (PathSafetyError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise self._fatal("policy_resource_invalid", "policy resource manifest is unreadable") from exc
        resource_id = value.get("resourceId") if isinstance(value, dict) else None
        if (
            not isinstance(resource_id, str)
            or not resource_id
            or len(resource_id) > 128
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in resource_id)
        ):
            raise self._fatal("policy_resource_invalid", "policy resource manifest id is invalid")
        return resource_id

    def _resource(self) -> tuple[str, str, str]:
        if self.resource_manifest_relative_path is not None or self.resource_fingerprint is not None:
            if (
                not isinstance(self.resource_manifest_relative_path, str)
                or not self.resource_manifest_relative_path
                or not isinstance(self.resource_fingerprint, str)
                or len(self.resource_fingerprint) != 64
            ):
                raise self._fatal("policy_resource_invalid", "frozen policy resource reference is invalid")
            return (
                self.resource_manifest_relative_path,
                self.resource_fingerprint,
                self._manifest_resource_id(self.resource_manifest_relative_path),
            )
        path = self.install_root / Path(RESOURCE_MANIFEST_PATH.replace("\\", os.sep))
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise self._fatal("policy_resource_invalid", "policy resource manifest is unreadable") from exc
        fingerprint = value.get("fingerprint") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict) or value.get("schemaVersion") != 1 or value.get("owner") != "policy"
            or value.get("resourceId") != "lse14-scorer-5k-v1" or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
        ):
            raise self._fatal("policy_resource_invalid", "policy resource manifest identity is invalid")
        return RESOURCE_MANIFEST_PATH, fingerprint, str(value["resourceId"])

    def _exchange(self, method: str, payload: dict[str, object], config_hash: str) -> dict[str, object]:
        request = ProtocolEnvelopeV1(
            protocolVersion="1.0", kind="request", messageId=f"policy-{uuid.uuid4().hex}", runtimeId=RUNTIME_ID,
            owner=OWNER, method=method, payload=payload, jobId=self.job_id, configHash=config_hash,
        )
        try:
            response = self.transport.exchange(request)
        except StdioJsonlTransportError:
            raise
        except Exception as exc:
            raise self._fatal("policy_protocol_violation", "policy transport failed") from exc
        if not isinstance(response, ProtocolEnvelopeV1) or (
            response.kind != "response" or response.replyTo != request.messageId or response.runtimeId != RUNTIME_ID
            or response.owner != OWNER or response.jobId != self.job_id or response.configHash != config_hash
        ):
            raise self._fatal("policy_protocol_violation", "policy response envelope identity mismatch")
        if response.method == "error":
            code = response.payload.get("code")
            raise self._fatal(code if isinstance(code, str) and code.startswith("policy_") else "policy_protocol_violation", "policy worker failed")
        expected = "hello" if method == "hello" else "result"
        if response.method != expected:
            raise self._fatal("policy_protocol_violation", "policy response method mismatch")
        return response.payload

    def _hello(self, config_hash: str, policy: dict[str, object], manifest: str, fingerprint: str | None) -> None:
        job = self.database.get_job(self.job_id)
        overlay = job["overlay_root"]
        if not isinstance(overlay, str) or not overlay:
            raise self._fatal("policy_overlay_invalid", "policy job has no annotation overlay")
        payload = self._exchange("hello", {
            "schemaVersion": 1, "payloadType": "policy_hello_request", "jobId": self.job_id,
            "configHash": config_hash, "datasetRoot": str(job["dataset_root"]), "overlayRoot": overlay,
            "artistRootName": Path(str(job["source_root"])).name,
            "resourceManifestRelativePath": manifest if fingerprint is not None else None,
            "resourceFingerprint": fingerprint, "policy": policy,
        }, config_hash)
        expected_loads = 1 if fingerprint is not None else 0
        if (
            payload.get("schemaVersion") != 1 or payload.get("payloadType") != "policy_hello_result"
            or payload.get("ready") is not True or payload.get("modelLoadCount") != expected_loads
            or payload.get("resourceFingerprint") != fingerprint
            or payload.get("qualityEnabled") is not (fingerprint is not None)
            or (fingerprint is None and payload.get("device") is not None)
            or (fingerprint is not None and payload.get("device") not in {"cpu", "cuda"})
        ):
            raise self._fatal("policy_protocol_violation", "policy hello result is invalid")
        self.database.set_runtime_evidence(self.job_id, "dropout", {
            "availability": "available",
            "runtimeId": RUNTIME_ID,
            "qualityEnabled": payload["qualityEnabled"],
            "device": payload["device"],
            "modelLoadCount": payload["modelLoadCount"],
            "resourceFingerprint": payload["resourceFingerprint"],
        })
        self._hello_done = True

    def _work_item(self, lease: WorkLease, row: object) -> dict[str, object]:
        if not lease.leaseId:
            raise self._fatal("policy_protocol_violation", "policy lease id is missing")
        return {
            "schemaVersion": 1, "sampleId": lease.sampleId, "leaseId": lease.leaseId,
            "relativeImagePath": str(row["relative_image_path"]), "annotationKey": str(row["annotation_key"]),
            "imageSize": int(row["image_size"]), "imageMtimeNs": int(row["image_mtime_ns"]),
            "imageFileId": row["image_file_id"],
        }

    def _issue(self, lease: WorkLease, row: object, outcome: dict[str, object]) -> None:
        code, message, retriable, repair = (outcome.get("code"), outcome.get("message"), outcome.get("retriable"), outcome.get("repairStartModule"))
        if (
            not isinstance(code, str) or not code.startswith("policy_") or len(code) > 128
            or not isinstance(message, str) or not isinstance(retriable, bool) or repair not in {"classify", "dropout"}
        ):
            raise self._fatal("policy_protocol_violation", "policy issue outcome is invalid")
        self.scheduler.fail_with_issue(lease, SampleIssue(
            issueId=hashlib.sha256(f"{self.job_id}\0{lease.sampleId}\0dropout\0{code}".encode("utf-8")).hexdigest(),
            jobId=self.job_id, sampleId=lease.sampleId, relativeImagePath=str(row["relative_image_path"]),
            moduleId="dropout", code=code, severity="error", blocking=True, retriable=retriable,
            message=message[:1024], attempt=lease.attempt, repairStartModule=repair,  # type: ignore[arg-type]
        ))

    def _commit_prepared(self, lease: WorkLease, row: object, outcome: dict[str, object]) -> None:
        if not lease.leaseId:
            raise self._fatal("policy_protocol_violation", "policy lease id is missing")
        relative = outcome.get("preparedRelativePath")
        digest = outcome.get("sha256")
        expected = f"prepared\\dropout\\{lease.leaseId}.json"
        if not isinstance(relative, str) or safe_relative_path(relative) != expected or not isinstance(digest, str) or len(digest) != 64:
            raise self._fatal("policy_protocol_violation", "policy prepared artifact identity is invalid")
        prepared = self.view.overlay.resolve_prepared(relative)
        if not prepared.is_file() or sha256_file(prepared) != digest:
            raise self._fatal("policy_result_persistence_failed", "policy prepared artifact digest is invalid")
        try:
            before = parse_annotation_json(self.view.read(str(row["annotation_key"]), ".json"))
            after = parse_annotation_json(prepared.read_bytes())
        except (OSError, ClassifyJsonError) as exc:
            raise self._fatal("policy_result_persistence_failed", "policy JSON cannot be validated") from exc
        if before is None or after is None or any(before.get(field) != after.get(field) for field in PROTECTED_FIELDS):
            raise self._fatal("policy_protocol_violation", "policy output changed a protected field")
        previous_nl = before.get("nl")
        next_nl = after.get("nl")
        if isinstance(previous_nl, str) and previous_nl.strip() and previous_nl != next_nl:
            # Dropout may clear the projection, but the prior text remains a
            # durable, image-identity-bound NL revision.
            self.database.archive_nl_for_sample(
                self.job_id,
                lease.sampleId,
                source="dropout",
                text=previous_nl,
            )
        self.scheduler.stage_prepared(lease, relative_path=relative, sha256=digest)
        self.view.overlay.commit_prepared(relative, digest, str(row["annotation_key"]), ".json")
        quality = after.get("quality")
        if isinstance(quality, list) and all(isinstance(value, str) for value in quality):
            diagnostic = QUALITY_DIAGNOSTICS.get(tuple(quality))
            if diagnostic:
                self.database.increment_module_diagnostic(self.job_id, "dropout", diagnostic, severity="info", amount=1)
        self.scheduler.complete(lease)

    def run(self) -> str:
        config_hash, policy, manifest, fingerprint = self._config()
        active: list[WorkLease] = []
        try:
            self._publish("running")
            while True:
                job = self.database.get_job(self.job_id)
                if job["status"] in {"cancelling", "paused"}:
                    self._publish(str(job["status"]))
                    return str(job["status"])
                if job["status"] != "running" or job["current_module_id"] != "dropout":
                    raise self._fatal("policy_protocol_violation", "policy module is not active")
                active = self.scheduler.claim_batch(self.job_id, "dropout", self.worker_instance_id, config_hash)
                if not active:
                    if self.database.count_module_unsettled(self.job_id, "dropout"):
                        raise self._fatal("policy_protocol_violation", "policy has unsettled but unclaimable work")
                    summary = self.database.module_summary(self.job_id, "dropout")
                    status = self.scheduler.finish_module(self.job_id, "dropout", with_issues=int(summary["issue_count"]) > 0)
                    self._publish(status)
                    return status
                rows = [self.database.get_leased_sample(self.job_id, "dropout", lease.sampleId, lease_id=str(lease.leaseId), worker_instance_id=self.worker_instance_id) for lease in active]
                if not self._hello_done:
                    self._hello(config_hash, policy, manifest, fingerprint)
                payload = self._exchange("process_batch", {
                    "schemaVersion": 1, "payloadType": "policy_process_request",
                    "items": [self._work_item(lease, row) for lease, row in zip(active, rows, strict=True)],
                }, config_hash)
                outcomes = payload.get("outcomes")
                expected_loads = 1 if fingerprint is not None else 0
                if payload.get("schemaVersion") != 1 or payload.get("payloadType") != "policy_batch_result" or payload.get("modelLoadCount") != expected_loads or not isinstance(outcomes, list) or len(outcomes) != len(active):
                    raise self._fatal("policy_protocol_violation", "policy batch result is invalid")
                pending = {(lease.sampleId, str(lease.leaseId)): (lease, row) for lease, row in zip(active, rows, strict=True)}
                for outcome in outcomes:
                    if not isinstance(outcome, dict):
                        raise self._fatal("policy_protocol_violation", "policy outcome is invalid")
                    key = (outcome.get("sampleId"), outcome.get("leaseId"))
                    if key not in pending or outcome.get("relativeImagePath") != pending[key][1]["relative_image_path"]:
                        raise self._fatal("policy_protocol_violation", "policy outcome identity does not match a lease")
                    lease, row = pending.pop(key)
                    if outcome.get("schemaVersion") != 1:
                        raise self._fatal("policy_protocol_violation", "policy outcome schema is invalid")
                    if outcome.get("status") == "prepared":
                        self._commit_prepared(lease, row, outcome)
                    elif outcome.get("status") == "issue":
                        self._issue(lease, row, outcome)
                    else:
                        raise self._fatal("policy_protocol_violation", "policy outcome status is invalid")
                    active.remove(lease)
                    self._publish("running", lease.attempt)
        except (PolicyRunnerError, SchedulerError, ProtocolError):
            for lease in active:
                state = self.database.get_sample_state(self.job_id, lease.sampleId)
                if state["status"] == "leased":
                    self.scheduler.release_unstarted(lease)
            self.database.set_module_summary(self.job_id, "dropout", status="failed", finished=True)
            self.database.set_job_status(self.job_id, "failed", current_module_id="dropout")
            self._publish("failed")
            raise
