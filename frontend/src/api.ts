export type PipelineModuleId = "caption" | "classify" | "replace" | "ocr" | "nl" | "count_review" | "dropout" | "token_budget" | "export";
export type AnnotationProfile = "e621" | "danbooru";
export type PathPickerPurpose = "source_dataset" | "output_dataset" | "replacement_csv" | "classification_resource_json";
export type SelectLocalPathResponse = { cancelled: boolean; path: string | null };
export type OcrDevice = "auto" | "cuda" | "cpu";
export type OcrExecutionTuning = { mode: "auto"; value: null } | { mode: "manual"; value: number };
export type OcrExecutionRequest = {
  textDetLimitSideLen: OcrExecutionTuning;
  textBatchSize: OcrExecutionTuning;
};
export type DeviceRecommendationResponse = {
  schemaVersion: number;
  baselineVersion: string;
  cpuPhysicalCores: number;
  cpuLogicalCores: number;
  gpu: {
    available: boolean;
    name: string | null;
    totalVramBytes: number | null;
    freeVramBytes: number | null;
    probeSource: string;
  };
  moduleBatchSize: Record<"caption" | "classify" | "replace" | "ocr" | "nl" | "countReview" | "dropout" | "tokenBudget" | "export", number>;
  reasons: Record<string, string>;
  probeErrors: string[];
};
export type OcrRuntimeStatus = {
  availability: "pending" | "available" | "unavailable";
  runtimeId: "ocr-paddle" | "ocr-paddle-gpu" | null;
  gpuName: string | null;
  totalVramBytes: number | null;
  requestedDevice: OcrDevice | null;
  observedDevice: "cpu" | "cuda" | null;
  recommended: { textDetLimitSideLen: number; textBatchSize: number } | null;
  effective: { textDetLimitSideLen: number; textBatchSize: number } | null;
  startupReason: "gpu_runtime_unavailable" | "binding_invalid" | null;
};
export type DropoutRuntimeStatus = {
  availability: "available";
  runtimeId: "policy";
  qualityEnabled: boolean;
  device: "cpu" | "cuda" | null;
  modelLoadCount: number;
  resourceFingerprint: string | null;
};
export type CreateJobRequest = { config: Record<string, unknown>; ocrExecution?: OcrExecutionRequest };

export type JobSnapshot = {
  job: {
    jobId: string; status: string; currentModuleId?: string; lastEventId: number; apiBudgetExtra: number;
    configSchemaVersion: number;
    apiBudgetRevision: number; pinned: boolean; configHash: string; manifestSchemaVersion: number;
    parentJobId: string | null; createdAt: string; startedAt: string | null; cancelRequestedAt: string | null; finishedAt: string | null;
    frozenConfig?: Record<string, unknown>;
    apiRuntime?: { revision: number; providerLabel: string; endpoint: string; model: string; backupModel: string | null; jsonMode: string; credentialRef: string; policy: Record<string, unknown>; appliedAt: string } | null;
    apiRuntimePending?: { revision: number; providerLabel: string; endpoint: string; model: string; backupModel: string | null; jsonMode: string; credentialRef: string; policy: Record<string, unknown>; appliedAt: string } | null;
    apiConfigRevision?: number;
  };
  moduleOrder: ReadonlyArray<PipelineModuleId>;
  modules: ReadonlyArray<{ module_id: string; status: string; completed: number; failed: number; skipped: number; total: number; issue_count: number }>;
  diagnostics: ReadonlyArray<{ code: string; severity: string; count: number }>;
  captionDiagnostics: ReadonlyArray<{ code: string; severity: string; count: number }>;
  ocrDiagnostics: ReadonlyArray<{ code: string; severity: string; count: number }>;
  events: ReadonlyArray<{ event_id: number; module_id: string; status: string; completed: number; total: number; attempt: number; message: string | null }>;
  issues: ReadonlyArray<{ issue_id: string; sample_id: number; relative_image_path?: string; module_id: string; code: string; severity: string; message: string; retriable: number; attempt: number; repair_start_module?: string; field_errors_json?: string }>;
  exportSummary: null | { format: "json" | "flat_txt" | "both"; commitStatus: string; scanned: number; valid: number; invalid: number; exported: number; skipped: number; issueCount: number; issuesPageEndpoint: string; convertedSamples: number; conversions: Record<string, number>; unlabeled: number; unlabeledFolder: string | null };
  repairPreview: { eligibleTargetCount: number; estimatedApiRequests: number } | null;
  repairChildren: ReadonlyArray<{
    jobId: string; status: string; currentModuleId: string | null; sampleCount: number;
    targetCount: number; createdAt: string; finishedAt: string | null;
  }>;
  ocrRuntime: OcrRuntimeStatus | null;
  dropoutRuntime: DropoutRuntimeStatus | null;
  nlPendingApiDecisions: number;
  nextAfterEventId: number;
  nextIssueAfterSampleId: number;
  nextIssueAfterIssueId: string | null;
  snapshotRequired: boolean;
};

export type JobListEntry = {
  jobId: string; status: string; currentModuleId: string | null; datasetRoot: string;
  sampleCount: number; pinned: boolean; createdAt: string; finishedAt: string | null;
};
export type JobListPage = { jobs: JobListEntry[]; nextAfterCreatedAt: string | null; nextAfterJobId: string | null };
export type DefaultNlPrompt = { promptVersion: string; systemPrompt: string; sha256: string };

export type ResourceKind = "replacement-index" | "classification-index" | "tagging-model" | "dropout-model" | "ocr-model" | "tokenizer";
export type ResourceDistribution = { mode: "bundled" | "local-only"; sourceUrl?: string; licenseUrl?: string; licenseStatus?: "unverified" };
export type ResourceCompatibility = { status: "compatible" | "incompatible" | "unavailable" | "not_applicable"; reason?: string; classificationResourceId?: string };
export type ResourceEntry = {
  schemaVersion: number;
  kind: ResourceKind;
  resourceId: string;
  resourceVersion: string;
  profile: AnnotationProfile | "shared";
  displayName: { "zh-CN": string; en: string };
  description: { "zh-CN": string; en: string };
  runtimeFormat: string;
  distribution: ResourceDistribution;
  fingerprint: string | null;
  metadata: Record<string, unknown>;
  adjustableCategories: string[];
  excludedCategories: string[];
  defaultThresholds: Record<string, number>;
  compatibility: ResourceCompatibility;
  available: boolean;
  default: boolean;
  defaultForProfiles: AnnotationProfile[];
  officialModelId?: string;
  contextLimit?: number;
};
export type ResourceDefaults = {
  taggingModel: string;
  classificationIndex: string;
  dropoutModel: string;
  replacementIndex: string;
};
export type ResourceCatalogResponse = {
  schemaVersion: 3;
  defaultsSchemaVersion: number;
  defaults: ResourceDefaults;
  resources: ResourceEntry[];
  invalidResources: ReadonlyArray<{ relativePath: string; reason: string }>;
  capabilities?: {
    policy: { status: "available" | "missing" | "invalid"; runtimeId: string };
    qualityStack: { status: "available" | "missing" | "invalid"; resourceId: string };
  };
};

export type NlProfile = {
  profileId: string; endpoint: string; model: string; backupModel: string | null;
  apiCredentialRef: string; credentialRef?: string; systemPrompt: string; apiPolicy: Record<string, unknown>;
  name?: string; protocol?: "openai_chat_completions"; providerLabel?: string; jsonMode?: "prompt_only" | "json_object";
  policy?: Record<string, unknown>; revision?: number; updatedAt?: string; hasCredential: boolean;
};
export type NlApiConfigUpdate = {
  expectedTaskRevision: number;
  expectedProfileRevision: number;
  profileId: string;
  protocol?: "openai_chat_completions";
  providerLabel?: string;
  endpoint: string;
  model: string;
  backupModel: string | null;
  credentialRef: string;
  jsonMode?: "prompt_only" | "json_object";
  policy: Record<string, unknown>;
};
export type NlPresetType = "general" | "style" | "character" | "moss-general" | "moss-style" | "moss-character";
export type NlPromptPresetSummary = {
  presetId: string;
  name: string;
  type: NlPresetType;
  builtIn: boolean;
  sha256: string;
  sizeBytes: number;
};
export type NlPromptPresetDetail = NlPromptPresetSummary & { promptText: string; basePrompt?: string };
export type NlPromptPresetWrite = { name: string; type: NlPresetType; promptText: string };
export type NlDiagnosticCredentials = { endpoint: string; apiCredentialRef: string; apiKey?: string };
export type NlModelDiscoveryRequest = NlDiagnosticCredentials;
export type NlTestMessageRequest = NlDiagnosticCredentials & { model: string; basePrompt: string };
export type ModelDiscoveryResult = {
  ok: boolean;
  latencyMs: number;
  models: string[];
  errorCode: string | null;
  errorReason: string | null;
};
export type TestMessageUsage = { promptTokens: number | null; completionTokens: number | null; totalTokens: number | null };
export type TestMessageResult = {
  ok: boolean;
  latencyMs: number;
  actualModel: string | null;
  replyText: string | null;
  usage: TestMessageUsage | null;
  errorCode: string | null;
  errorReason: string | null;
};
export type PreflightSummary = {
  jobId: string; sampleCount: number; inScopeCount: number; outOfScopeCount: number; nonblankTxtCount: number;
  nonblankJsonCount: number; configHash: string; replaceIndex?: { mode: "bundled" | "custom"; path?: string; sha256?: string; ruleCount?: number } | null;
  blankTxtCount: number; blankJsonCount: number; annotationKeyCollisionCount: number; imageIssueCount: number;
  projection: { format?: string; inScopeSamples?: number; retainedSamples?: number; jsonCreate?: number; jsonOverwrite?: number; jsonDelete?: number; txtCreate?: number; txtOverwrite?: number; txtDelete?: number };
  estimate: { existingAnnotationFiles?: number; existingAnnotationBytes?: number; averageAnnotationBytes?: number; backupUpperBoundBytes?: number; incrementalWriteBytes?: number };
  api: { candidateCount?: number; minRequests?: number; maxPrimaryRequests?: number; maxWithBackupRequests?: number; httpAttemptBudget?: number; estimatedUploadBytes?: number };
};

export type PreflightProgressStatus = "queued" | "preflighting" | "completed" | "failed" | "cancelled";
export type PreflightProgress = {
  jobId: string;
  status: PreflightProgressStatus;
  checkedFiles: number;
  totalFiles: number;
  error: string | null;
  summary: PreflightSummary | null;
  updatedAt: string;
  lastEventId: number;
};

export type CountValue = "solo" | "duo" | "trio" | "group";
export type CountReviewDecisionStatus = "pending" | "auto_resolved" | "manual_resolved";
export type CountReviewDecisionSource = "classify" | "vlm" | "manual";
export type CountReviewDecision = {
  status: CountReviewDecisionStatus;
  finalCount: CountValue | null;
  selectedSource: "consensus" | CountReviewDecisionSource | null;
  reviewReasons: string[];
  version: number;
  resolvedAt: string | null;
  appliedAt: string | null;
};
export type CountReviewItem = {
  sampleId: number;
  relativeImagePath: string;
  classify: { count: CountValue | null; warningCodes: string[] };
  vlm: {
    status: "observed" | "invalid" | "not_requested";
    count: CountValue | "unknown" | null;
    layout: "single_scene" | "multi_view" | "character_sheet" | "multi_panel" | "unknown" | null;
    sameCharacterRepeated: boolean | null;
    warningCodes: string[];
    notRequestedReason: string | null;
  };
  decision: CountReviewDecision;
};
export type CountReviewPage = {
  items: CountReviewItem[];
  targetCount: number;
  pendingCount: number;
  nextAfterSampleId: number | null;
};
export type CountReviewFilters = {
  status?: CountReviewDecisionStatus;
  reason?: string;
  classifyCount?: CountValue | "unavailable";
  vlmCount?: CountValue | "unknown" | "unavailable";
  mismatchOnly?: boolean;
};
export type CountReviewUpdate = {
  sampleId: number;
  expectedVersion: number;
  source: CountReviewDecisionSource;
  count?: CountValue;
};

export type TokenBudgetAnnotation = {
  quality: string[];
  count: string;
  character: string;
  series: string;
  artist: string;
  appearance: string[];
  tags: string[];
  environment: string[];
  nl: string;
};
export type TokenBudgetProposal = {
  schemaVersion: number;
  sampleId: number;
  version: number;
  status: "within_budget" | "trimmed" | "overflow" | "failed";
  originalTokens: number;
  finalTokens: number;
  removed: Record<"quality" | "environment" | "tags" | "appearance", string[]>;
  annotation: TokenBudgetAnnotation;
  flatTextSha256: string;
  maxTokens: number;
  resourceId: string;
  resourceFingerprint: string;
};
export type TokenBudgetReviewItem = {
  sampleId: number;
  relativeImagePath: string;
  review: {
    status: "overflow";
    originalTokens: number;
    finalTokens: number;
    removed: Record<"quality" | "environment" | "tags" | "appearance", string[]>;
    maxTokens: number;
    resourceId: string;
    resourceFingerprint: string;
    version: number;
  };
  annotation: TokenBudgetAnnotation;
  proposal: TokenBudgetProposal | null;
  rewriteProposal: { schemaVersion: number; operationId: string; proposal: TokenBudgetProposal } | null;
};
export type TokenBudgetReviewPage = {
  items: TokenBudgetReviewItem[];
  targetCount: number;
  nextAfterSampleId: number | null;
};

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) }, ...init });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(typeof body.detail === "string" ? body.detail : `request failed: ${response.status}`, response.status);
  }
  return response.json() as Promise<T>;
}

export function pollJob(jobId: string, afterEventId: number, issueAfterSampleId = 0, issueAfterIssueId: string | null = null, signal?: AbortSignal): Promise<JobSnapshot> {
  const cursor = issueAfterIssueId ? `&issueAfterIssueId=${encodeURIComponent(issueAfterIssueId)}` : "";
  return request(`/api/jobs/${encodeURIComponent(jobId)}?afterEventId=${afterEventId}&issueAfterSampleId=${issueAfterSampleId}${cursor}`, { signal });
}

export function listJobs(afterCreatedAt: string | null = null, afterJobId: string | null = null, limit = 20): Promise<JobListPage> {
  const cursor = afterCreatedAt && afterJobId ? `&afterCreatedAt=${encodeURIComponent(afterCreatedAt)}&afterJobId=${encodeURIComponent(afterJobId)}` : "";
  return request(`/api/jobs?limit=${limit}${cursor}`);
}
export function fetchDefaultNlPrompt(promptVersion?: string): Promise<DefaultNlPrompt> {
  const query = promptVersion ? `?promptVersion=${encodeURIComponent(promptVersion)}` : "";
  return request(`/api/nl/default-prompt${query}`);
}

export function listNlProfiles(): Promise<{ defaultProfileId: string | null; profiles: NlProfile[] }> { return request("/api/nl/profiles"); }
export function listNlPromptPresets(includeMoss = false): Promise<{ presets: NlPromptPresetSummary[] }> {
  return request(`/api/nl/prompt-presets${includeMoss ? "?includeMoss=true" : ""}`);
}
export function getNlPromptPreset(presetId: string): Promise<NlPromptPresetDetail> {
  return request(`/api/nl/prompt-presets/${encodeURIComponent(presetId)}`);
}
export function createNlPromptPreset(body: NlPromptPresetWrite): Promise<NlPromptPresetDetail> {
  return request("/api/nl/prompt-presets", { method: "POST", body: JSON.stringify(body) });
}
export function updateNlPromptPreset(presetId: string, body: NlPromptPresetWrite): Promise<NlPromptPresetDetail> {
  return request(`/api/nl/prompt-presets/${encodeURIComponent(presetId)}`, { method: "PUT", body: JSON.stringify(body) });
}
export function resetNlPromptPreset(presetId: string): Promise<NlPromptPresetDetail> {
  return request(`/api/nl/prompt-presets/${encodeURIComponent(presetId)}/reset`, { method: "POST" });
}
export function deleteNlPromptPreset(presetId: string): Promise<{ deleted: true }> {
  return request(`/api/nl/prompt-presets/${encodeURIComponent(presetId)}`, { method: "DELETE" });
}
export function discoverNlModels(body: NlModelDiscoveryRequest): Promise<ModelDiscoveryResult> {
  return request("/api/nl/diagnostics/models", { method: "POST", body: JSON.stringify(body) });
}
export function testNlMessage(body: NlTestMessageRequest): Promise<TestMessageResult> {
  return request("/api/nl/diagnostics/test-message", { method: "POST", body: JSON.stringify(body) });
}
export function listResources(): Promise<ResourceCatalogResponse> { return request("/api/resources"); }
export function getDeviceRecommendations(rpm = 60): Promise<DeviceRecommendationResponse> {
  return request(`/api/application/device-recommendations?rpm=${encodeURIComponent(String(rpm))}`);
}
export function selectLocalPath(purpose: PathPickerPurpose, currentPath: string | null): Promise<SelectLocalPathResponse> {
  return request("/api/application/select-path", {
    method: "POST",
    body: JSON.stringify({ purpose, currentPath }),
  });
}
export function saveNlProfile(profile: NlProfile): Promise<NlProfile> {
  // The server body model forbids unknown keys, so response-only fields such as
  // `updatedAt` and `hasCredential` would make every save fail with a 422.
  // Whitelist the writable fields instead of spreading the whole profile.
  const body: Record<string, unknown> = {
    endpoint: profile.endpoint,
    model: profile.model,
    backupModel: profile.backupModel,
    credentialRef: profile.credentialRef ?? profile.apiCredentialRef,
    systemPrompt: profile.systemPrompt,
    policy: profile.policy ?? profile.apiPolicy ?? {},
  };
  if (profile.name !== undefined) body.name = profile.name;
  if (profile.protocol !== undefined) body.protocol = profile.protocol;
  if (profile.providerLabel !== undefined) body.providerLabel = profile.providerLabel;
  if (profile.jsonMode !== undefined) body.jsonMode = profile.jsonMode;
  if (profile.revision !== undefined) body.revision = profile.revision;
  return request(`/api/nl/profiles/${encodeURIComponent(profile.profileId)}`, { method: "PUT", body: JSON.stringify(body) });
}
export function setNlProfileDefault(profileId: string, revision?: number): Promise<{ defaultProfileId: string }> {
  return request(`/api/nl/profiles/${encodeURIComponent(profileId)}/default`, {
    method: "POST", body: JSON.stringify(revision === undefined ? {} : { revision }),
  });
}
export function copyNlProfile(profileId: string, targetProfileId: string, name?: string): Promise<NlProfile> {
  return request(`/api/nl/profiles/${encodeURIComponent(profileId)}/copy`, {
    method: "POST", body: JSON.stringify({ profileId: targetProfileId, ...(name ? { name } : {}) }),
  });
}
export function deleteNlProfile(profileId: string): Promise<{ deleted: boolean }> {
  return request(`/api/nl/profiles/${encodeURIComponent(profileId)}`, { method: "DELETE" });
}
export function updateRunningNlApiConfig(jobId: string, body: NlApiConfigUpdate): Promise<{
  taskRevision: number; profileRevision: number; apiRuntime: NonNullable<JobSnapshot["job"]["apiRuntime"]>; effective: "next_batch";
}> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/nl/api-config`, { method: "PUT", body: JSON.stringify(body) });
}
export function saveNlSecret(reference: string, secret: string): Promise<{ stored: boolean }> {
  return request(`/api/nl/credentials/${encodeURIComponent(reference)}`, { method: "PUT", body: JSON.stringify({ secret }) });
}
export function deleteNlSecret(reference: string): Promise<{ deleted: boolean }> {
  return request(`/api/nl/credentials/${encodeURIComponent(reference)}`, { method: "DELETE" });
}
export function pauseNl(jobId: string): Promise<{ status: string }> { return request(`/api/jobs/${encodeURIComponent(jobId)}/nl/pause`, { method: "POST" }); }
export function resumeNl(jobId: string): Promise<{ status: string }> { return request(`/api/jobs/${encodeURIComponent(jobId)}/nl/resume`, { method: "POST" }); }
export function pausePolicy(jobId: string): Promise<{ status: string }> { return request(`/api/jobs/${encodeURIComponent(jobId)}/policy/pause`, { method: "POST" }); }
export function resumePolicy(jobId: string): Promise<{ status: string }> { return request(`/api/jobs/${encodeURIComponent(jobId)}/policy/resume`, { method: "POST" }); }
export function pauseJob(jobId: string): Promise<{ status: string }> { return request(`/api/jobs/${encodeURIComponent(jobId)}/pause`, { method: "POST" }); }
export function resumeJob(jobId: string): Promise<{ status: string }> { return request(`/api/jobs/${encodeURIComponent(jobId)}/resume`, { method: "POST" }); }
export function pauseModule(jobId: string, moduleId: PipelineModuleId): Promise<JobSnapshot> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/modules/${encodeURIComponent(moduleId)}/pause`, { method: "POST" });
}
export function resumeModule(jobId: string, moduleId: PipelineModuleId): Promise<JobSnapshot> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/modules/${encodeURIComponent(moduleId)}/resume`, { method: "POST" });
}
export function addNlBudget(jobId: string, amount: number): Promise<{ apiBudgetExtra: number; apiBudgetRevision: number }> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/nl/api-budget`, { method: "POST", body: JSON.stringify({ amount }) });
}
export function confirmNlOutcomes(jobId: string): Promise<{ requeued: number }> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/nl/confirm-api-outcomes`, { method: "POST", body: JSON.stringify({ confirmed: true }) });
}
export type ManualNlSelector = { sampleId: number; issueId?: never } | { issueId: string; sampleId?: never };

export function manualNlRetry(jobId: string, body: ManualNlSelector): Promise<{ jobId: string; parentJobId: string; targetCount: number; started: boolean }> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/nl/manual-retry`, {
    method: "POST", body: JSON.stringify({ ...body, confirmed: true }),
  });
}
export function manualNlRetryBatch(jobId: string, issueIds: string[]): Promise<{ jobId: string; parentJobId: string; issueIds: string[]; targetCount: number; started: boolean }> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/nl/manual-retry-batch`, {
    method: "POST", body: JSON.stringify({ issueIds, confirmed: true }),
  });
}
export function manualNlWrite(jobId: string, body: ManualNlSelector & { nl: string }): Promise<{ jobId: string; parentJobId: string; targetCount: number; started: boolean }> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/nl/manual-write`, {
    method: "POST", body: JSON.stringify({ ...body, confirmed: true }),
  });
}
export function preflightJob(body: CreateJobRequest): Promise<PreflightSummary> { return request("/api/jobs/preflight", { method: "POST", body: JSON.stringify(body) }); }
export function startPreflight(body: CreateJobRequest): Promise<PreflightProgress> {
  return request("/api/jobs/preflight/async", { method: "POST", body: JSON.stringify(body) });
}
export function getPreflightProgress(jobId: string, signal?: AbortSignal): Promise<PreflightProgress> {
  return request(`/api/jobs/preflight/${encodeURIComponent(jobId)}/progress`, { signal });
}
export function cancelPreflight(jobId: string): Promise<{ jobId: string; cancelled: boolean }> {
  return request(`/api/jobs/preflight/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
}
export function confirmWorkspace(jobId: string, confirmedRebuild: boolean): Promise<{ jobId: string; status: string; datasetRoot: string; overlayRoot: string }> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/confirm-workspace`, { method: "POST", body: JSON.stringify({ confirmed: true, confirmedRebuild }) });
}
export function startPipeline(jobId: string): Promise<{ jobId: string; started: boolean }> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/start`, { method: "POST" });
}
export function recoverJob(jobId: string): Promise<{ jobId: string; returnedLeases: number; committedPrepared: number; repeatedPrepared: number; pendingApiDecisions: number; started: boolean; status: string }> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/recover`, { method: "POST", body: JSON.stringify({ confirmed: true }) });
}
export function repairJob(jobId: string): Promise<{ jobId: string; parentJobId: string; targetCount: number; preparedTargetCount: number; estimatedApiRequests: number; started: boolean }> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/repair`, { method: "POST" });
}
export function restoreOriginalAnnotations(jobId: string): Promise<{ jobId: string; restored: number; backupZip: string }> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/restore-original-annotations`, { method: "POST", body: JSON.stringify({ confirmed: true }) });
}
export function cancelJob(jobId: string): Promise<{ status: string }> { return request(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" }); }
export function setJobPin(jobId: string, pinned: boolean): Promise<{ pinned: boolean }> { return request(`/api/jobs/${encodeURIComponent(jobId)}/pin`, { method: "PUT", body: JSON.stringify({ pinned }) }); }
export function discardJob(jobId: string): Promise<{ jobId: string; overlayDeleted: boolean }> { return request(`/api/jobs/${encodeURIComponent(jobId)}/discard`, { method: "POST", body: JSON.stringify({ confirmed: true }) }); }
export function deleteCompletedHistory(jobId: string): Promise<{
  jobId: string;
  deletedControlRecord: boolean;
  preserved: { dataset: boolean; exports: boolean; backup: boolean };
}> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/delete-history`, { method: "POST", body: JSON.stringify({ confirmed: true }) });
}

export function listCountReview(
  jobId: string,
  filters: CountReviewFilters = {},
  afterSampleId = 0,
  limit = 50,
): Promise<CountReviewPage> {
  const query = new URLSearchParams({ afterSampleId: String(afterSampleId), limit: String(limit) });
  if (filters.status) query.set("status", filters.status);
  if (filters.reason) query.set("reason", filters.reason);
  if (filters.classifyCount) query.set("classifyCount", filters.classifyCount);
  if (filters.vlmCount) query.set("vlmCount", filters.vlmCount);
  if (filters.mismatchOnly) query.set("mismatchOnly", "true");
  return request(`/api/jobs/${encodeURIComponent(jobId)}/count-review?${query.toString()}`);
}

export function countReviewImageUrl(jobId: string, sampleId: number): string {
  return `/api/jobs/${encodeURIComponent(jobId)}/count-review/${sampleId}/image`;
}

export function updateCountReviewDecision(
  jobId: string,
  update: CountReviewUpdate,
): Promise<{ sampleId: number; decision: CountReviewDecision }> {
  const { sampleId, ...body } = update;
  return request(`/api/jobs/${encodeURIComponent(jobId)}/count-review/${sampleId}`, {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

export function updateCountReviewBatch(
  jobId: string,
  updates: CountReviewUpdate[],
): Promise<{ items: Array<{ sampleId: number; decision: CountReviewDecision }> }> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/count-review/batch`, {
    method: "POST",
    body: JSON.stringify({ updates }),
  });
}

export function confirmCountReview(jobId: string): Promise<{ jobId: string; started: boolean }> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/count-review/confirm`, {
    method: "POST",
    body: JSON.stringify({ confirmed: true }),
  });
}

export function listTokenBudgetReviews(jobId: string, afterSampleId: number | null = null, limit = 50, signal?: AbortSignal): Promise<TokenBudgetReviewPage> {
  const query = new URLSearchParams({ limit: String(limit) });
  if (afterSampleId !== null) query.set("afterSampleId", String(afterSampleId));
  return request(`/api/jobs/${encodeURIComponent(jobId)}/token-budget/reviews?${query.toString()}`, { signal });
}

export function recountTokenBudget(
  jobId: string,
  body: { sampleId: number; expectedVersion: number; annotation: TokenBudgetAnnotation },
  signal?: AbortSignal,
): Promise<TokenBudgetProposal> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/token-budget/recount`, { method: "POST", body: JSON.stringify(body), signal });
}

export function rewriteTokenBudgetShort(
  jobId: string,
  body: { sampleIds: number[]; expectedVersions: Record<string, number> },
): Promise<{ operationId: string; sampleIds: number[]; proposals: TokenBudgetProposal[] }> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/token-budget/rewrite-short`, { method: "POST", body: JSON.stringify(body) });
}

export function applyTokenBudgetProposal(
  jobId: string,
  body: { sampleId: number; expectedVersion: number },
): Promise<{ sampleId: number; version: number; exportStarted: boolean }> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/token-budget/apply`, { method: "POST", body: JSON.stringify(body) });
}
