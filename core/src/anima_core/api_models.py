from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from .ocr_runtime_binding import OcrExecutionRequestV1, normalize_ocr_execution


class _ProfileBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    endpoint: str = ""
    model: str = ""
    backupModel: str | None = None
    apiCredentialRef: str | None = None
    credentialRef: str | None = None
    systemPrompt: str = ""
    apiPolicy: dict[str, object] | None = None
    policy: dict[str, object] | None = None
    name: str | None = None
    protocol: str = "openai_chat_completions"
    providerLabel: str = "gemini-relay"
    jsonMode: str = "prompt_only"
    revision: int | None = None


class _ProfileCopyBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    profileId: str
    name: str | None = None


class _NlApiConfigBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expectedTaskRevision: int = Field(ge=0)
    expectedProfileRevision: int = Field(ge=1)
    profileId: str
    protocol: Literal["openai_chat_completions"] = "openai_chat_completions"
    providerLabel: str = "gemini-relay"
    endpoint: str
    model: str
    backupModel: str | None = None
    credentialRef: str
    jsonMode: Literal["prompt_only", "json_object"] = "prompt_only"
    policy: dict[str, object] = Field(default_factory=dict)


class _SecretBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    secret: str


class _NlPromptPresetBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    type: Literal["general", "style", "character"] = "general"
    promptText: str | None = None
    basePrompt: str | None = None

    @model_validator(mode="after")
    def _require_prompt_text(self) -> "_NlPromptPresetBody":
        if self.promptText is None and self.basePrompt is None:
            raise ValueError("promptText is required")
        if self.promptText is not None and self.basePrompt is not None and self.promptText != self.basePrompt:
            raise ValueError("promptText and basePrompt must match")
        return self


class _NlDiagnosticCredentialsBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    endpoint: str
    apiCredentialRef: str
    apiKey: SecretStr | None = None


class _NlModelDiscoveryBody(_NlDiagnosticCredentialsBody):
    pass


class _NlTestMessageBody(_NlDiagnosticCredentialsBody):
    model: str
    basePrompt: str


class _AmountBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    amount: int = Field(ge=1, le=1_000_000)


class _ConfirmBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    confirmed: bool


class _ExportRecommitBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    confirmed: bool
    inventoryFingerprint: str = Field(min_length=64, max_length=64)


class _NlManualRetryBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sampleId: int | None = Field(default=None, ge=1)
    issueId: str | None = Field(default=None, min_length=1, max_length=128)
    confirmed: bool

    @model_validator(mode="after")
    def _require_one_selector(self) -> "_NlManualRetryBody":
        if (self.sampleId is None) == (self.issueId is None):
            raise ValueError("manual NL retry requires exactly one sampleId or issueId")
        return self


class _NlManualRetryBatchBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    issueIds: list[str] = Field(min_length=1, max_length=1_000)
    confirmed: bool

    @model_validator(mode="after")
    def _require_unique_issue_ids(self) -> "_NlManualRetryBatchBody":
        if any(not 1 <= len(issue_id) <= 128 for issue_id in self.issueIds) or len(set(self.issueIds)) != len(self.issueIds):
            raise ValueError("manual NL retry requires unique issue IDs")
        return self


class _NlManualWriteBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sampleId: int | None = Field(default=None, ge=1)
    issueId: str | None = Field(default=None, min_length=1, max_length=128)
    nl: str = Field(max_length=16_384)
    confirmed: bool

    @model_validator(mode="after")
    def _require_one_selector(self) -> "_NlManualWriteBody":
        if (self.sampleId is None) == (self.issueId is None):
            raise ValueError("manual NL write requires exactly one sampleId or issueId")
        return self


class _ShutdownBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    token: str


class _SelectPathBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    purpose: Literal["source_dataset", "output_dataset", "replacement_csv", "classification_resource_json"]
    currentPath: str | None = Field(default=None, max_length=32_767)


class DeviceRecommendationGpu(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    available: bool
    name: str | None
    totalVramBytes: int | None
    freeVramBytes: int | None
    probeSource: str


class DeviceRecommendationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schemaVersion: int
    baselineVersion: str
    cpuPhysicalCores: int
    cpuLogicalCores: int
    gpu: DeviceRecommendationGpu
    moduleBatchSize: dict[str, int]
    reasons: dict[str, str]
    probeErrors: list[str]


@dataclass(frozen=True)
class SelectPathRequest:
    purpose: Literal["source_dataset", "output_dataset", "replacement_csv", "classification_resource_json"]
    current_path: str | None


def parse_select_path_body(value: object) -> SelectPathRequest:
    body = _SelectPathBody.model_validate(value)
    return SelectPathRequest(body.purpose, body.currentPath)


class _PreflightBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    config: dict[str, object]
    ocrExecution: dict[str, object] | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_bare_config(cls, value: object) -> object:
        if isinstance(value, dict) and "config" not in value:
            return {"config": value}
        return value


@dataclass(frozen=True)
class CreateJobRequest:
    config: dict[str, object]
    ocrExecution: OcrExecutionRequestV1


def parse_create_job_body(value: object) -> CreateJobRequest:
    body = _PreflightBody.model_validate(value)
    return CreateJobRequest(dict(body.config), normalize_ocr_execution(body.ocrExecution))


class _WorkspaceBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    confirmed: bool
    confirmedRebuild: bool = False


class _PinBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    pinned: bool


class _CountReviewDecisionBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expectedVersion: int = Field(ge=1)
    source: Literal["classify", "vlm", "manual"]
    count: Literal["solo", "duo", "trio", "group"] | None = None


class _CountReviewBatchItem(_CountReviewDecisionBody):
    sampleId: int = Field(ge=1)


class _CountReviewBatchBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    updates: list[_CountReviewBatchItem] = Field(min_length=1, max_length=500)


class _TokenBudgetRecountBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sampleId: int = Field(ge=1)
    expectedVersion: int = Field(ge=1)
    annotation: dict[str, object]


class _TokenBudgetRewriteShortBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sampleIds: list[int] = Field(min_length=1, max_length=500)
    expectedVersions: dict[str, int]


class _TokenBudgetApplyBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sampleId: int = Field(ge=1)
    expectedVersion: int = Field(ge=1)
