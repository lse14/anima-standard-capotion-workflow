from __future__ import annotations

import json
import uuid
from fastapi import APIRouter, Query

from .api_context import ControlPlaneContext, bad_request, conflict, not_found
from .api_models import _TokenBudgetApplyBody, _TokenBudgetRecountBody, _TokenBudgetRewriteShortBody
from .db import DEFAULT_PAGE_SIZE, StateDatabase
from .contracts import NL_MOSS_PROMPT_VERSION, NL_MOSS_RESPONSE_PROTOCOL
from .token_budget_review import (
    IsolatedTokenBudgetCounter,
    TokenBudgetReviewConflictError,
    TokenBudgetReviewError,
    TokenBudgetReviewService,
)
from .nl_protocol import NlOutcomeV1, parse_outcomes
from .nl_runner import DEFAULT_POLICY, WORKER_POLICY_KEYS
from .pipeline import PipelineError
from .stdio_transport import StdioJsonlTransport, StdioJsonlTransportError
from .worker_protocol import ProtocolEnvelopeV1


class _IsolatedShortRewriter:
    def __init__(self, context: ControlPlaneContext, database: StateDatabase, job_id: str) -> None:
        self.context = context
        self.database = database
        self.job_id = job_id

    def __call__(self, item: dict[str, object]) -> NlOutcomeV1:
        return self.with_http_attempt_allowance(item, http_attempt_allowance=None)

    def with_http_attempt_allowance(
        self,
        item: dict[str, object],
        *,
        http_attempt_allowance: int | None,
    ) -> NlOutcomeV1:
        job = self.database.get_job(self.job_id)
        try:
            config = json.loads(str(job["config_json"]))
            nl = config["nl"]
            profile_id = nl.get("apiProfileId", "default")
            profile = next(value for value in self.context.profile_store.load_all() if value.profileId == profile_id)
            api_key = self.context.credential_store.load(profile.apiCredentialRef)
            configured_policy = nl.get("apiPolicy", {})
            if not isinstance(configured_policy, dict):
                raise ValueError("NL API policy is invalid")
            policy = {key: configured_policy.get(key, default) for key, default in DEFAULT_POLICY.items()}
            if type(policy["maxHttpAttempts"]) is not int or policy["maxHttpAttempts"] < 1:
                raise ValueError("NL HTTP attempt budget is invalid")
            if not isinstance(nl.get("systemPrompt"), str):
                raise ValueError("NL supplement is invalid")
            prompt_version = nl.get("promptVersion", "nl-default-prompt-v4")
            response_protocol = NL_MOSS_RESPONSE_PROTOCOL if prompt_version == NL_MOSS_PROMPT_VERSION else "nl-count-v2"
            if prompt_version not in {"nl-default-prompt-v4", NL_MOSS_PROMPT_VERSION}:
                raise ValueError("NL prompt version is invalid")
        except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("NL profile or frozen policy is unavailable") from exc
        allowance = min(160, max(1, int(policy["maxHttpAttempts"])))
        if http_attempt_allowance is not None:
            if type(http_attempt_allowance) is not int or not 1 <= http_attempt_allowance <= 5:
                raise RuntimeError("NL short rewrite allowance is invalid")
            allowance = http_attempt_allowance
        pipeline = self.context.pipeline_service
        launcher = pipeline.launcher_factory(pipeline.install_root)
        request_id = f"nl-review-{uuid.uuid4().hex}"
        try:
            with StdioJsonlTransport(launcher.spawn("nl", expected_owner="nl")) as transport:
                hello = ProtocolEnvelopeV1(
                    "1.0", "request", request_id, "nl", "nl", "hello", {
                        "schemaVersion": 1, "payloadType": "nl_hello_request", "jobId": self.job_id,
                        "configHash": str(job["config_hash"]), "endpoint": profile.endpoint, "model": profile.model,
                        "backupModel": profile.backupModel, "apiKey": api_key,
                        "systemPrompt": nl["systemPrompt"],
                        "apiPolicy": {key: policy[key] for key in WORKER_POLICY_KEYS},
                        "responseProtocol": response_protocol, "promptVersion": prompt_version,
                    }, jobId=self.job_id, configHash=str(job["config_hash"]),
                )
                response = transport.exchange(hello)
                if (
                    response.kind != "response" or response.replyTo != request_id
                    or response.runtimeId != "nl" or response.owner != "nl"
                    or response.jobId != self.job_id or response.configHash != str(job["config_hash"])
                    or response.method != "hello"
                    or response.payload != {"schemaVersion": 1, "payloadType": "nl_hello_result", "ready": True, "concurrency": policy["concurrency"]}
                ):
                    raise RuntimeError("NL worker hello is invalid")
                process_id = f"nl-review-process-{uuid.uuid4().hex}"
                process = ProtocolEnvelopeV1(
                    "1.0", "request", process_id, "nl", "nl", "process_batch", {
                        "schemaVersion": 1, "payloadType": "nl_process_request", "items": [item],
                        "httpAttemptAllowance": allowance,
                    }, jobId=self.job_id, configHash=str(job["config_hash"]),
                )
                result = transport.exchange(process)
                if (
                    result.kind != "response" or result.replyTo != process_id
                    or result.runtimeId != "nl" or result.owner != "nl"
                    or result.jobId != self.job_id or result.configHash != str(job["config_hash"])
                    or result.method != "result"
                ):
                    raise RuntimeError("NL worker result is invalid")
                expected = {(int(item["sampleId"]), str(item["leaseId"])): str(item["relativeImagePath"])}
                outcomes = parse_outcomes(result.payload, expected, response_protocol=response_protocol)
                if len(outcomes) != 1 or outcomes[0].nl is None:
                    raise RuntimeError("NL short rewrite did not produce a caption")
                shutdown_id = f"nl-review-shutdown-{uuid.uuid4().hex}"
                shutdown = ProtocolEnvelopeV1(
                    "1.0", "request", shutdown_id, "nl", "nl", "shutdown", {},
                    jobId=self.job_id, configHash=str(job["config_hash"]),
                )
                shutdown_result = transport.exchange(shutdown)
                if (
                    shutdown_result.kind != "response" or shutdown_result.replyTo != shutdown_id
                    or shutdown_result.runtimeId != "nl" or shutdown_result.owner != "nl"
                    or shutdown_result.jobId != self.job_id or shutdown_result.configHash != str(job["config_hash"])
                    or shutdown_result.method != "result" or shutdown_result.payload != {"accepted": True}
                ):
                    raise RuntimeError("NL worker shutdown is invalid")
                return outcomes[0]
        except (OSError, RuntimeError, StopIteration, StdioJsonlTransportError, ValueError) as exc:
            raise RuntimeError("NL short rewrite failed") from exc


def _service(context: ControlPlaneContext, database: StateDatabase, job_id: str) -> TokenBudgetReviewService:
    pipeline = context.pipeline_service
    launcher = pipeline.launcher_factory(pipeline.install_root)
    return TokenBudgetReviewService(
        database,
        job_id,
        counter=IsolatedTokenBudgetCounter(database, job_id, launcher),
        rewriter=_IsolatedShortRewriter(context, database, job_id),
    )


def build_token_budget_router(context: ControlPlaneContext) -> APIRouter:
    router = APIRouter()

    @router.get("/api/jobs/{job_id}/token-budget/reviews")
    def list_token_budget_reviews(
        job_id: str,
        afterSampleId: int = Query(default=0, ge=0),
        limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=500),
    ) -> dict[str, object]:
        database = StateDatabase.open(context.database_path)
        try:
            try:
                return _service(context, database, job_id).page(
                    after_sample_id=afterSampleId or None,
                    limit=limit,
                )
            except KeyError as exc:
                raise not_found(exc) from exc
            except (TokenBudgetReviewError, ValueError) as exc:
                raise bad_request(exc) from exc
        finally:
            database.close()

    @router.post("/api/jobs/{job_id}/token-budget/recount")
    def recount_token_budget(job_id: str, body: _TokenBudgetRecountBody) -> dict[str, object]:
        database = StateDatabase.open(context.database_path)
        try:
            try:
                proposal = _service(context, database, job_id).recount(
                    body.sampleId,
                    expected_version=body.expectedVersion,
                    annotation=body.annotation,
                )
                return {"sampleId": body.sampleId, "proposal": proposal}
            except KeyError as exc:
                raise not_found(exc) from exc
            except TokenBudgetReviewConflictError as exc:
                raise conflict(exc) from exc
            except (TokenBudgetReviewError, ValueError) as exc:
                raise bad_request(exc) from exc
        finally:
            database.close()

    @router.post("/api/jobs/{job_id}/token-budget/rewrite-short")
    def rewrite_token_budget_short(job_id: str, body: _TokenBudgetRewriteShortBody) -> dict[str, object]:
        # The service validates the mapping before any request may start.  The
        # actual v4 rewrite transport is intentionally injected by its caller.
        database = StateDatabase.open(context.database_path)
        try:
            try:
                return _service(context, database, job_id).rewrite_short(
                    sample_ids=body.sampleIds,
                    expected_versions=body.expectedVersions,
                )
            except KeyError as exc:
                raise not_found(exc) from exc
            except TokenBudgetReviewConflictError as exc:
                raise conflict(exc) from exc
            except (TokenBudgetReviewError, ValueError) as exc:
                raise bad_request(exc) from exc
        finally:
            database.close()

    @router.post("/api/jobs/{job_id}/token-budget/apply")
    def apply_token_budget(job_id: str, body: _TokenBudgetApplyBody) -> dict[str, object]:
        database = StateDatabase.open(context.database_path)
        try:
            try:
                record = _service(context, database, job_id).apply(
                    body.sampleId,
                    expected_version=body.expectedVersion,
                )
            except KeyError as exc:
                raise not_found(exc) from exc
            except TokenBudgetReviewConflictError as exc:
                raise conflict(exc) from exc
            except (TokenBudgetReviewError, ValueError) as exc:
                raise bad_request(exc) from exc
        finally:
            database.close()
        try:
            export_started = context.pipeline_service.resume(job_id)
        except PipelineError:
            # The review apply has already committed atomically.  Surface that
            # durable record instead of turning a later scheduling refusal into
            # a misleading failed apply response.
            export_started = False
        return {"sampleId": body.sampleId, "record": record, "exportStarted": export_started}

    return router
