import { expect, test as base, type Page, type Route } from "@playwright/test";

import type {
  CountReviewDecision,
  CountReviewItem,
  CountReviewPage,
  CountValue,
  DefaultNlPrompt,
  JobListEntry,
  JobListPage,
  JobSnapshot,
  NlProfile,
  NlPresetType,
  NlPromptPresetDetail,
  OcrRuntimeStatus,
  PathPickerPurpose,
  PreflightSummary,
  ResourceCatalogResponse,
  ResourceEntry,
  TokenBudgetAnnotation,
  TokenBudgetProposal,
  TokenBudgetReviewItem,
  TokenBudgetReviewPage,
} from "../../src/api";

const LOCAL_ORIGIN = `http://127.0.0.1:${process.env.ANIMA_E2E_PORT ?? "4173"}`;
const TIMESTAMP = "2026-07-30T00:00:00Z";
const V10_MODULE_ORDER = ["caption", "classify", "replace", "ocr", "nl", "count_review", "dropout", "token_budget", "export"] as const;
const TRANSPARENT_PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL4wQAAAABJRU5ErkJggg==",
  "base64",
);

export const DEFAULT_JOB_ID = "job-e621-characterization";
const BUILTIN_PRESET_ID = "builtin:nl-preset-v1-general";
const BUILTIN_PROMPT = "General task preset prompt.";

type RouteFailure = { status: number; detail: string };
type RouteHold = { promise: Promise<void>; release: () => void };

export type ApiMutation = {
  method: string;
  path: string;
  body: unknown;
};

export type ApiScenario = {
  resources: ResourceCatalogResponse;
  profiles: NlProfile[];
  defaultNlPrompt: DefaultNlPrompt;
  defaultNlPromptV3: DefaultNlPrompt;
  defaultNlPromptV4: DefaultNlPrompt;
  taskNlPromptGeneral: DefaultNlPrompt;
  taskNlPromptStyle: DefaultNlPrompt;
  taskNlPromptCharacter: DefaultNlPrompt;
  promptRequests: Array<string | null>;
  promptPresets: NlPromptPresetDetail[];
  jobs: JobListPage;
  snapshots: Map<string, JobSnapshot>;
  preflight: PreflightSummary;
  preflightCancelled: boolean;
  countReview: CountReviewPage;
  countReviewRequests: number[];
  recommendationRequests: number;
  tokenBudgetReviews: TokenBudgetReviewPage;
  selectedPaths: Record<PathPickerPurpose, string | null>;
  failures: Map<string, RouteFailure>;
  holds: Map<string, RouteHold>;
  mutations: ApiMutation[];
  unexpectedRequests: string[];
  unhandledApiRequests: string[];
  consoleErrors: string[];
  expectedConsoleErrorPatterns: RegExp[];
  pageErrors: string[];
};

export type OpenAppOptions = {
  jobId?: string;
  language?: "zh-CN" | "en";
};

function resource(
  kind: ResourceEntry["kind"],
  resourceId: string,
  displayName: string,
  adjustableCategories: string[] = [],
  options: Partial<Pick<ResourceEntry, "profile" | "available" | "defaultForProfiles" | "distribution" | "officialModelId" | "contextLimit">> = {},
): ResourceEntry {
  return {
    schemaVersion: 1,
    kind,
    resourceId,
    resourceVersion: "test-v1",
    profile: options.profile ?? "e621",
    displayName: { "zh-CN": displayName, en: displayName },
    description: { "zh-CN": "Local browser-test resource", en: "Local browser-test resource" },
    runtimeFormat: "test",
    distribution: options.distribution ?? { mode: "bundled" },
    fingerprint: "test-fingerprint",
    metadata: {},
    adjustableCategories,
    excludedCategories: [],
    defaultThresholds: { general: 0.35 },
    compatibility: { status: "compatible" },
    available: options.available ?? true,
    default: (options.defaultForProfiles ?? ["e621"]).length > 0,
    defaultForProfiles: options.defaultForProfiles ?? ["e621"],
    ...(options.officialModelId ? { officialModelId: options.officialModelId } : {}),
    ...(typeof options.contextLimit === "number" ? { contextLimit: options.contextLimit } : {}),
  };
}

function createResources(): ResourceCatalogResponse {
  return {
    schemaVersion: 3,
    defaultsSchemaVersion: 3,
    defaults: {
      taggingModel: "caption-e621-eva02-large-full-v1",
      classificationIndex: "classify-e621-20260724-v1",
      dropoutModel: "lse14-scorer-5k-v1",
      replacementIndex: "replace-e621-20260726-v2",
    },
    resources: [
      resource("tagging-model", "caption-e621-eva02-large-full-v1", "E621 tagger", ["general"]),
      resource("classification-index", "classify-e621-20260724-v1", "E621 classify index"),
      resource("replacement-index", "replace-e621-20260726-v2", "E621 replacement index"),
      resource("dropout-model", "lse14-scorer-5k-v1", "Quality scorer"),
      resource("tagging-model", "caption-danbooru-cl-tagger-v2-00", "Danbooru tagger", [], { profile: "danbooru", defaultForProfiles: ["danbooru"] }),
      resource("classification-index", "classify-danbooru-test-v1", "Danbooru classify index", [], { profile: "danbooru", defaultForProfiles: ["danbooru"] }),
      resource("ocr-model", "ocr-ppocrv5-server-paddle-v1", "PaddleOCR v5", [], {
        profile: "shared",
        defaultForProfiles: [],
        distribution: {
          mode: "local-only",
          sourceUrl: "https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_recognition.html",
          licenseStatus: "unverified",
        },
      }),
      resource("tokenizer", "tokenizer-qwen3-0.6b-anima-v1", "", [], {
        profile: "shared", defaultForProfiles: [], officialModelId: "Qwen/Qwen3-0.6B", contextLimit: 40960,
        distribution: { mode: "local-only", sourceUrl: "https://huggingface.co/Qwen/Qwen3-0.6B", licenseStatus: "unverified" },
      }),
      resource("tokenizer", "tokenizer-qwen3-vl-4b-krea2-v1", "", [], {
        profile: "shared", defaultForProfiles: [], officialModelId: "Qwen/Qwen3-VL-4B-Instruct", contextLimit: 262144,
        distribution: { mode: "local-only", sourceUrl: "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct", licenseStatus: "unverified" },
      }),
    ],
    invalidResources: [],
  };
}

export function makeSnapshot({
  jobId = DEFAULT_JOB_ID,
  status = "ready",
  currentModuleId,
  parentJobId = null,
  schemaVersion = 10,
  ocrRuntime = null,
  captionDiagnostics = [],
}: {
  jobId?: string;
  status?: string;
  currentModuleId?: string;
  parentJobId?: string | null;
  schemaVersion?: number;
  ocrRuntime?: OcrRuntimeStatus | null;
  captionDiagnostics?: JobSnapshot["captionDiagnostics"];
} = {}): JobSnapshot {
  if (schemaVersion !== 10) throw new Error("mockApi snapshots only supports JobConfig schema v10");
  const moduleOrder = V10_MODULE_ORDER;
  return {
    job: {
      jobId,
      status,
      currentModuleId,
      lastEventId: 7,
      apiBudgetExtra: 0,
      configSchemaVersion: schemaVersion,
      apiBudgetRevision: 0,
      pinned: false,
      configHash: "test-config-hash",
      manifestSchemaVersion: 2,
      createdAt: TIMESTAMP,
      startedAt: null,
      cancelRequestedAt: null,
      finishedAt: null,
      parentJobId,
    },
    moduleOrder: [...moduleOrder],
    modules: moduleOrder.map((moduleId) => ({
      module_id: moduleId,
      status: moduleId === currentModuleId ? (status === "reviewing" ? "running" : status) : "pending",
      completed: moduleId === currentModuleId ? 1 : 0,
      failed: 0,
      skipped: 0,
      total: 3,
      issue_count: 0,
    })),
    diagnostics: [],
    captionDiagnostics,
    ocrDiagnostics: [],
    events: [{ event_id: 7, module_id: currentModuleId ?? "caption", status, completed: 1, total: 3, attempt: 1, message: null }],
    issues: [],
    exportSummary: null,
    repairPreview: { eligibleTargetCount: 2, estimatedApiRequests: 0 },
    repairChildren: [],
    ocrRuntime,
    dropoutRuntime: null,
    nlPendingApiDecisions: 0,
    nextAfterEventId: 7,
    nextIssueAfterSampleId: 0,
    nextIssueAfterIssueId: null,
    snapshotRequired: false,
  };
}

function makePreflight(jobId = DEFAULT_JOB_ID): PreflightSummary {
  return {
    jobId,
    sampleCount: 3,
    inScopeCount: 3,
    outOfScopeCount: 0,
    nonblankTxtCount: 1,
    nonblankJsonCount: 1,
    configHash: "test-config-hash",
    replaceIndex: null,
    blankTxtCount: 0,
    blankJsonCount: 0,
    annotationKeyCollisionCount: 0,
    imageIssueCount: 0,
    projection: {
      format: "both",
      inScopeSamples: 3,
      retainedSamples: 0,
      jsonCreate: 2,
      jsonOverwrite: 1,
      jsonDelete: 0,
      txtCreate: 2,
      txtOverwrite: 1,
      txtDelete: 0,
    },
    estimate: { incrementalWriteBytes: 2048, backupUpperBoundBytes: 4096 },
    api: { minRequests: 0, maxWithBackupRequests: 0, httpAttemptBudget: 0, estimatedUploadBytes: 0 },
  };
}

function pendingReviewItem(
  sampleId: number,
  relativeImagePath: string,
  classifyCount: CountValue,
  vlmCount: CountValue,
): CountReviewItem {
  return {
    sampleId,
    relativeImagePath,
    classify: { count: classifyCount, warningCodes: [] },
    vlm: {
      status: "observed",
      count: vlmCount,
      layout: "single_scene",
      sameCharacterRepeated: false,
      warningCodes: [],
      notRequestedReason: null,
    },
    decision: {
      status: "pending",
      finalCount: null,
      selectedSource: null,
      reviewReasons: ["count_observation_mismatch"],
      version: 1,
      resolvedAt: null,
      appliedAt: null,
    },
  };
}

function makeCountReview(): CountReviewPage {
  return {
    items: [
      pendingReviewItem(1, "sample-a.png", "duo", "trio"),
      pendingReviewItem(2, "sample-b.png", "solo", "duo"),
    ],
    targetCount: 2,
    pendingCount: 2,
    nextAfterSampleId: null,
  };
}

function tokenBudgetAnnotation(sampleId: number): TokenBudgetAnnotation {
  return {
    quality: ["high quality"], count: "solo", character: "", series: "", artist: "",
    appearance: ["red jacket"], tags: [`tag-${sampleId}`], environment: ["street"], nl: `Original caption ${sampleId}.`,
  };
}

function tokenBudgetReviewItem(sampleId: number): TokenBudgetReviewItem {
  return {
    sampleId,
    relativeImagePath: `review/sample-${sampleId}.png`,
    review: {
      status: "overflow", originalTokens: 640 + sampleId, finalTokens: 600 + sampleId,
      removed: { quality: ["high quality"], environment: [], tags: [], appearance: [] }, maxTokens: 512,
      resourceId: "tokenizer-qwen3-0.6b-anima-v1", resourceFingerprint: "a".repeat(64), version: 1,
    },
    annotation: tokenBudgetAnnotation(sampleId), proposal: null, rewriteProposal: null,
  };
}

function makeTokenBudgetReviews(): TokenBudgetReviewPage {
  return {
    items: [tokenBudgetReviewItem(1), tokenBudgetReviewItem(2), tokenBudgetReviewItem(3)],
    targetCount: 3,
    nextAfterSampleId: null,
  };
}

function tokenBudgetProposal(item: TokenBudgetReviewItem, annotation: TokenBudgetAnnotation, status: TokenBudgetProposal["status"] = "within_budget"): TokenBudgetProposal {
  return {
    schemaVersion: 1, sampleId: item.sampleId, version: item.review.version + 1, status,
    originalTokens: item.review.originalTokens, finalTokens: status === "trimmed" ? 480 : 500,
    removed: { quality: [], environment: [], tags: [], appearance: [] }, annotation,
    flatTextSha256: "b".repeat(64), maxTokens: item.review.maxTokens,
    resourceId: item.review.resourceId, resourceFingerprint: item.review.resourceFingerprint,
  };
}

export function createApiScenario(): ApiScenario {
  const defaultProfile: NlProfile = {
    profileId: "default",
    endpoint: "http://127.0.0.1:8999/v1",
    model: "test-model",
    backupModel: null,
    apiCredentialRef: "anima-test",
    systemPrompt: "Describe the image concisely.",
    apiPolicy: { maxRequestsPerMinute: 60 },
    hasCredential: false,
  };
  return {
    resources: createResources(),
    profiles: [defaultProfile],
    defaultNlPrompt: {
      promptVersion: "nl-default-prompt-v2",
      systemPrompt: defaultProfile.systemPrompt,
      sha256: "test-prompt-sha256",
    },
    defaultNlPromptV3: {
      promptVersion: "nl-default-prompt-v3",
      systemPrompt: "Describe visible content, including observable text.",
      sha256: "test-prompt-v3-sha256",
    },
    defaultNlPromptV4: {
      promptVersion: "nl-default-prompt-v4",
      systemPrompt: "Fixed prompt fragments are not editable in the browser.",
      sha256: "test-prompt-v4-sha256",
    },
    taskNlPromptGeneral: {
      promptVersion: "nl-default-prompt-v4-general",
      systemPrompt: "General task preset prompt.",
      sha256: "test-prompt-v4-general-sha256",
    },
    taskNlPromptStyle: {
      promptVersion: "nl-default-prompt-v4-style",
      systemPrompt: "Style task preset prompt.",
      sha256: "test-prompt-v4-style-sha256",
    },
    taskNlPromptCharacter: {
      promptVersion: "nl-default-prompt-v4-character",
      systemPrompt: "Character task preset prompt.",
      sha256: "test-prompt-v4-character-sha256",
    },
    promptRequests: [],
    promptPresets: ([
      ["builtin:nl-preset-v1-general", "General", "general", "General task preset prompt."],
      ["builtin:nl-preset-v1-style", "Style", "style", "Style task preset prompt."],
      ["builtin:nl-preset-v1-character", "Character", "character", "Character task preset prompt."],
    ] as const).map(([presetId, name, type, promptText]) => ({
      presetId, name, type, builtIn: true, sha256: "b".repeat(64),
      sizeBytes: new TextEncoder().encode(promptText).byteLength, promptText, basePrompt: promptText,
    })),
    jobs: { jobs: [], nextAfterCreatedAt: null, nextAfterJobId: null },
    snapshots: new Map([[DEFAULT_JOB_ID, makeSnapshot()]]),
    preflight: makePreflight(),
    preflightCancelled: false,
    countReview: makeCountReview(),
    countReviewRequests: [],
    recommendationRequests: 0,
    tokenBudgetReviews: makeTokenBudgetReviews(),
    selectedPaths: {
      source_dataset: "E:\\picked\\source",
      output_dataset: "E:\\picked\\output",
      replacement_csv: "E:\\picked\\replace.csv",
      classification_resource_json: "E:\\picked\\resource.json",
    },
    failures: new Map(),
    holds: new Map(),
    mutations: [],
    unexpectedRequests: [],
    unhandledApiRequests: [],
    consoleErrors: [],
    expectedConsoleErrorPatterns: [],
    pageErrors: [],
  };
}

export function setJobSnapshot(scenario: ApiScenario, snapshot: JobSnapshot): void {
  scenario.snapshots.set(snapshot.job.jobId, snapshot);
  const entry: JobListEntry = {
    jobId: snapshot.job.jobId,
    status: snapshot.job.status,
    currentModuleId: snapshot.job.currentModuleId ?? null,
    datasetRoot: "E:\\datasets\\browser-characterization",
    sampleCount: 3,
    pinned: snapshot.job.pinned,
    createdAt: snapshot.job.createdAt,
    finishedAt: snapshot.job.finishedAt,
  };
  scenario.jobs = {
    jobs: [entry, ...scenario.jobs.jobs.filter((item) => item.jobId !== entry.jobId)],
    nextAfterCreatedAt: null,
    nextAfterJobId: null,
  };
}

export function failRoute(scenario: ApiScenario, routeKey: string, detail: string, status = 503): void {
  scenario.failures.set(routeKey, { detail, status });
  scenario.expectedConsoleErrorPatterns.push(new RegExp(
    `^Failed to load resource: the server responded with a status of ${status} \\([^)]+\\)$`,
  ));
}

export function clearRouteFailure(scenario: ApiScenario, routeKey: string): void {
  scenario.failures.delete(routeKey);
}

export function holdRoute(scenario: ApiScenario, routeKey: string): () => void {
  let release: () => void = () => undefined;
  const promise = new Promise<void>((resolve) => { release = resolve; });
  scenario.holds.set(routeKey, { promise, release });
  return () => {
    const hold = scenario.holds.get(routeKey);
    if (!hold) return;
    hold.release();
    scenario.holds.delete(routeKey);
  };
}

export function mutationsFor(scenario: ApiScenario, method: string, path: string): ApiMutation[] {
  return scenario.mutations.filter((mutation) => mutation.method === method && mutation.path === path);
}

function routeKey(method: string, pathname: string): string {
  return `${method} ${pathname}`;
}

function snapshotFor(scenario: ApiScenario, jobId: string): JobSnapshot {
  const existing = scenario.snapshots.get(jobId);
  if (existing) return existing;
  const created = makeSnapshot({ jobId });
  scenario.snapshots.set(jobId, created);
  return created;
}

function updateJobState(snapshot: JobSnapshot, status: string, currentModuleId?: string): void {
  snapshot.job.status = status;
  snapshot.job.currentModuleId = currentModuleId;
  const current = snapshot.modules.find((module) => module.module_id === currentModuleId);
  if (current) current.status = status === "reviewing" ? "running" : status;
}

function readBody(route: Route): unknown {
  const raw = route.request().postData();
  if (!raw) return undefined;
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return raw;
  }
}

function recordMutation(scenario: ApiScenario, route: Route, pathname: string): unknown {
  const body = readBody(route);
  const redacted = pathname.startsWith("/api/nl/diagnostics/") && body && typeof body === "object"
    ? Object.fromEntries(Object.entries(body as Record<string, unknown>).filter(([key]) => key !== "apiKey"))
    : body;
  scenario.mutations.push({ method: route.request().method(), path: pathname, body: redacted });
  return body;
}

async function fulfillJson(route: Route, body: unknown, status = 200): Promise<void> {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

function resolveCount(item: CountReviewItem, source: "classify" | "vlm" | "manual", count?: CountValue): CountReviewDecision {
  const selectedCount = count ?? (source === "classify" ? item.classify.count : item.vlm.count);
  if (!selectedCount || selectedCount === "unknown") throw new Error("mock count source has no final value");
  return {
    ...item.decision,
    status: source === "manual" ? "manual_resolved" : "auto_resolved",
    finalCount: selectedCount,
    selectedSource: source,
    version: item.decision.version + 1,
    resolvedAt: TIMESTAMP,
  };
}

function applyCountDecision(
  scenario: ApiScenario,
  sampleId: number,
  source: "classify" | "vlm" | "manual",
  count?: CountValue,
): { sampleId: number; decision: CountReviewDecision } {
  const item = scenario.countReview.items.find((candidate) => candidate.sampleId === sampleId);
  if (!item) throw new Error(`unknown count review sample: ${sampleId}`);
  item.decision = resolveCount(item, source, count);
  scenario.countReview.pendingCount = scenario.countReview.items.filter((candidate) => candidate.decision.status === "pending").length;
  return { sampleId, decision: item.decision };
}

async function handleApiRequest(scenario: ApiScenario, route: Route, pathname: string): Promise<void> {
  const method = route.request().method();
  const key = routeKey(method, pathname);
  const body = method === "GET" ? undefined : recordMutation(scenario, route, pathname);
  // Keep legacy preflight assertions/fixtures working while the production UI
  // uses the asynchronous endpoint.  The actual request remains observable
  // under its new path as well.
  const legacyPreflightPath = pathname === "/api/jobs/preflight/async" ? "/api/jobs/preflight" : pathname;
  if (method === "POST" && pathname === "/api/jobs/preflight/async") {
    scenario.mutations.push({ method, path: legacyPreflightPath, body });
  }
  const hold = scenario.holds.get(key) ?? scenario.holds.get(routeKey(method, legacyPreflightPath));
  if (hold) await hold.promise;
  const failure = scenario.failures.get(key) ?? scenario.failures.get(routeKey(method, legacyPreflightPath));
  if (failure) {
    await fulfillJson(route, { detail: failure.detail }, failure.status);
    return;
  }

  if (method === "GET" && pathname === "/health") return fulfillJson(route, { status: "ok" });
  if (method === "GET" && pathname === "/api/resources") return fulfillJson(route, scenario.resources);
  if (method === "GET" && pathname === "/api/application/device-recommendations") {
    scenario.recommendationRequests += 1;
    return fulfillJson(route, {
      schemaVersion: 1,
      baselineVersion: "module-batching-v1-test",
      cpuPhysicalCores: 8,
      cpuLogicalCores: 16,
      gpu: { available: true, name: "Test GPU", totalVramBytes: 24 * 1024 ** 3, freeVramBytes: 20 * 1024 ** 3, probeSource: "test" },
      moduleBatchSize: { caption: 4, classify: 128, replace: 128, ocr: 8, nl: 3, countReview: 100, dropout: 8, tokenBudget: 128, export: 500 },
      reasons: { caption: "validated caption", classify: "validated classify", replace: "validated replace", ocr: "validated OCR", nl: "API RPM limit", countReview: "validated count review", dropout: "validated dropout", tokenBudget: "validated token budget", export: "validated export" },
      probeErrors: [],
    });
  }
  if (method === "POST" && pathname === "/api/application/select-path") {
    const request = body as { purpose: PathPickerPurpose };
    const path = scenario.selectedPaths[request.purpose];
    return fulfillJson(route, path === null ? { cancelled: true, path: null } : { cancelled: false, path });
  }
  if (method === "GET" && pathname === "/api/nl/default-prompt") {
    const promptVersion = new URL(route.request().url()).searchParams.get("promptVersion");
    scenario.promptRequests.push(promptVersion);
    return fulfillJson(route,
      promptVersion === "nl-default-prompt-v4-general" ? scenario.taskNlPromptGeneral
        : promptVersion === "nl-default-prompt-v4-style" ? scenario.taskNlPromptStyle
          : promptVersion === "nl-default-prompt-v4-character" ? scenario.taskNlPromptCharacter
            : promptVersion === "nl-default-prompt-v4" ? scenario.defaultNlPromptV4
              : promptVersion === "nl-default-prompt-v3" ? scenario.defaultNlPromptV3 : scenario.defaultNlPrompt,
    );
  }
  if (method === "GET" && pathname === "/api/nl/profiles") return fulfillJson(route, { profiles: scenario.profiles });
  if (method === "GET" && pathname === "/api/nl/prompt-presets") {
    return fulfillJson(route, { presets: scenario.promptPresets.map(({ promptText: _promptText, basePrompt: _basePrompt, ...summary }) => summary) });
  }
  const resetMatch = pathname.match(/^\/api\/nl\/prompt-presets\/([^/]+)\/reset$/);
  if (resetMatch && method === "POST") {
    const presetId = decodeURIComponent(resetMatch[1]);
    const index = scenario.promptPresets.findIndex((item) => item.presetId === presetId);
    if (index < 0 || !scenario.promptPresets[index].builtIn) return fulfillJson(route, { detail: "preset not found" }, 404);
    const defaults: Record<string, string> = {
      "builtin:nl-preset-v1-general": "General task preset prompt.",
      "builtin:nl-preset-v1-style": "Style task preset prompt.",
      "builtin:nl-preset-v1-character": "Character task preset prompt.",
    };
    const promptText = defaults[presetId] ?? scenario.promptPresets[index].promptText;
    scenario.promptPresets[index] = { ...scenario.promptPresets[index], promptText, basePrompt: promptText, sizeBytes: new TextEncoder().encode(promptText).byteLength };
    return fulfillJson(route, scenario.promptPresets[index]);
  }
  const presetMatch = pathname.match(/^\/api\/nl\/prompt-presets\/([^/]+)$/);
  if (presetMatch && method === "GET") {
    const preset = scenario.promptPresets.find((item) => item.presetId === decodeURIComponent(presetMatch[1]));
    if (!preset) return fulfillJson(route, { detail: "preset not found" }, 404);
    return fulfillJson(route, preset);
  }
  if (method === "POST" && pathname === "/api/nl/prompt-presets") {
    const request = body as { name: string; type?: NlPresetType; promptText?: string; basePrompt?: string };
    const promptText = request.promptText ?? request.basePrompt ?? "";
    const preset: NlPromptPresetDetail = {
      presetId: `custom:mock-${scenario.promptPresets.length}`,
      name: request.name,
      type: request.type ?? "general",
      builtIn: false,
      sha256: "c".repeat(64),
      sizeBytes: new TextEncoder().encode(promptText).byteLength,
      promptText,
      basePrompt: promptText,
    };
    scenario.promptPresets.push(preset);
    return fulfillJson(route, preset);
  }
  if (presetMatch && method === "PUT") {
    const presetId = decodeURIComponent(presetMatch[1]);
    const index = scenario.promptPresets.findIndex((item) => item.presetId === presetId);
    const request = body as { name: string; type?: NlPresetType; promptText?: string; basePrompt?: string };
    if (index < 0) return fulfillJson(route, { detail: "preset not found" }, 404);
    const promptText = request.promptText ?? request.basePrompt ?? "";
    const updated = { ...scenario.promptPresets[index], name: request.name, type: request.type ?? scenario.promptPresets[index].type, promptText, basePrompt: promptText, sizeBytes: new TextEncoder().encode(promptText).byteLength };
    scenario.promptPresets[index] = updated;
    return fulfillJson(route, updated);
  }
  if (presetMatch && method === "DELETE") {
    const presetId = decodeURIComponent(presetMatch[1]);
    if (presetId === BUILTIN_PRESET_ID) return fulfillJson(route, { detail: "built-in preset cannot be deleted" }, 409);
    const index = scenario.promptPresets.findIndex((item) => item.presetId === presetId);
    if (index < 0) return fulfillJson(route, { detail: "preset not found" }, 404);
    scenario.promptPresets.splice(index, 1);
    return fulfillJson(route, { deleted: true });
  }
  if (method === "POST" && pathname === "/api/nl/diagnostics/models") {
    const request = body as { endpoint?: string };
    if (request.endpoint?.includes("failure")) {
      return fulfillJson(route, { ok: false, latencyMs: 8, models: [], errorCode: "provider_rejected", errorReason: "Provider rejected model discovery." });
    }
    return fulfillJson(route, { ok: true, latencyMs: 12, models: ["provider-model-2", "provider-model-1"], errorCode: null, errorReason: null });
  }
  if (method === "POST" && pathname === "/api/nl/diagnostics/test-message") {
    const request = body as { endpoint?: string; model?: string };
    if (request.endpoint?.includes("failure")) {
      return fulfillJson(route, { ok: false, latencyMs: 9, actualModel: null, replyText: null, usage: null, errorCode: "provider_rejected", errorReason: "Provider rejected the test request." });
    }
    return fulfillJson(route, { ok: true, latencyMs: 17, actualModel: `actual-${request.model ?? ""}`, replyText: "Local diagnostic reply.", usage: { promptTokens: 12, completionTokens: 7, totalTokens: 19 }, errorCode: null, errorReason: null });
  }
  if (method === "GET" && pathname === "/api/jobs") return fulfillJson(route, scenario.jobs);

  if (method === "POST" && (pathname === "/api/jobs/preflight" || pathname === "/api/jobs/preflight/async")) {
    const config = (body as { config?: { ocr?: { enabled?: boolean } } } | undefined)?.config;
    const ocr = scenario.resources.resources.find((item) => item.kind === "ocr-model" && item.resourceId === "ocr-ppocrv5-server-paddle-v1");
    if (config?.ocr?.enabled === true && !ocr?.available) {
      scenario.expectedConsoleErrorPatterns.push(/^Failed to load resource: the server responded with a status of 400 \([^)]+\)$/);
      return fulfillJson(route, { detail: "ocr_resource_install_required: selected OCR resource is unavailable" }, 400);
    }
    const schemaVersion = config && (config as { schemaVersion?: unknown }).schemaVersion;
    const snapshot = schemaVersion === 10
      ? makeSnapshot({
        jobId: scenario.preflight.jobId,
        schemaVersion,
        ocrRuntime: config?.ocr?.enabled === true ? {
          availability: "available", runtimeId: "ocr-paddle-gpu", gpuName: "GPU", totalVramBytes: 24 * 1024 ** 3,
          requestedDevice: "cuda", observedDevice: "cuda",
          recommended: { textDetLimitSideLen: 2560, textBatchSize: 4 },
          effective: { textDetLimitSideLen: 2560, textBatchSize: 4 }, startupReason: null,
        } : null,
      })
      : snapshotFor(scenario, scenario.preflight.jobId);
    scenario.snapshots.set(scenario.preflight.jobId, snapshot);
    updateJobState(snapshot, "ready");
    if (pathname === "/api/jobs/preflight/async") {
      return fulfillJson(route, {
        jobId: scenario.preflight.jobId,
        status: "queued",
        checkedFiles: 0,
        totalFiles: scenario.preflight.sampleCount,
        error: null,
        summary: null,
        updatedAt: TIMESTAMP,
        lastEventId: 0,
      });
    }
    return fulfillJson(route, scenario.preflight);
  }

  const preflightProgressMatch = pathname.match(/^\/api\/jobs\/preflight\/([^/]+)\/progress$/);
  const preflightCancelMatch = pathname.match(/^\/api\/jobs\/preflight\/([^/]+)\/cancel$/);
  if (method === "POST" && preflightCancelMatch) {
    scenario.preflightCancelled = true;
    return fulfillJson(route, { jobId: decodeURIComponent(preflightCancelMatch[1]), cancelled: true });
  }
  if (method === "GET" && preflightProgressMatch) {
    const jobId = decodeURIComponent(preflightProgressMatch[1]);
    const snapshot = snapshotFor(scenario, jobId);
    if (scenario.preflightCancelled) {
      updateJobState(snapshot, "draft");
      return fulfillJson(route, {
        jobId,
        status: "cancelled",
        checkedFiles: 0,
        totalFiles: scenario.preflight.sampleCount,
        error: "preflight cancelled",
        summary: null,
        updatedAt: TIMESTAMP,
        lastEventId: 0,
      });
    }
    updateJobState(snapshot, "ready");
    return fulfillJson(route, {
      jobId,
      status: "completed",
      checkedFiles: scenario.preflight.sampleCount,
      totalFiles: scenario.preflight.sampleCount,
      error: null,
      summary: { ...scenario.preflight, jobId },
      updatedAt: TIMESTAMP,
      lastEventId: 0,
    });
  }

  const tokenBudgetReviewMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/token-budget\/reviews$/);
  if (tokenBudgetReviewMatch && method === "GET") {
    const query = new URL(route.request().url()).searchParams;
    const afterSampleId = Number(query.get("afterSampleId") ?? "0");
    const limit = Number(query.get("limit") ?? "50");
    const items = scenario.tokenBudgetReviews.items.filter((item) => item.sampleId > afterSampleId).slice(0, limit);
    const lastSampleId = items.at(-1)?.sampleId ?? afterSampleId;
    const more = scenario.tokenBudgetReviews.items.some((item) => item.sampleId > lastSampleId);
    return fulfillJson(route, {
      items,
      targetCount: scenario.tokenBudgetReviews.targetCount,
      nextAfterSampleId: more ? lastSampleId : null,
    });
  }

  const tokenBudgetRecountMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/token-budget\/recount$/);
  if (tokenBudgetRecountMatch && method === "POST") {
    const request = body as { sampleId: number; expectedVersion: number; annotation: TokenBudgetAnnotation };
    const item = scenario.tokenBudgetReviews.items.find((candidate) => candidate.sampleId === request.sampleId);
    if (!item || request.expectedVersion !== item.review.version) return fulfillJson(route, { detail: "Token Budget review version conflict" }, 409);
    const proposal = tokenBudgetProposal(item, request.annotation, "trimmed");
    item.review.version = proposal.version;
    item.proposal = proposal;
    return fulfillJson(route, proposal);
  }

  const tokenBudgetRewriteMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/token-budget\/rewrite-short$/);
  if (tokenBudgetRewriteMatch && method === "POST") {
    const request = body as { sampleIds: number[]; expectedVersions: Record<string, number> };
    const selected = request.sampleIds.map((sampleId) => scenario.tokenBudgetReviews.items.find((item) => item.sampleId === sampleId));
    if (new Set(request.sampleIds).size !== request.sampleIds.length || !selected.every(Boolean)
      || selected.some((item) => request.expectedVersions[String(item!.sampleId)] !== item!.review.version)) {
      return fulfillJson(route, { detail: "Token Budget review version conflict" }, 409);
    }
    const proposals = selected.map((item) => {
      const annotation = { ...item!.annotation, nl: `Short rewrite ${item!.sampleId}.` };
      const proposal = tokenBudgetProposal(item!, annotation, "trimmed");
      item!.review.version = proposal.version;
      item!.proposal = proposal;
      item!.rewriteProposal = { schemaVersion: 1, operationId: "rewrite-token-budget-test", proposal };
      return proposal;
    });
    return fulfillJson(route, { operationId: "rewrite-token-budget-test", sampleIds: request.sampleIds, proposals });
  }

  const tokenBudgetApplyMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/token-budget\/apply$/);
  if (tokenBudgetApplyMatch && method === "POST") {
    const request = body as { sampleId: number; expectedVersion: number };
    const index = scenario.tokenBudgetReviews.items.findIndex((item) => item.sampleId === request.sampleId);
    const item = scenario.tokenBudgetReviews.items[index];
    if (!item || !item.proposal || request.expectedVersion !== item.review.version) return fulfillJson(route, { detail: "Token Budget review version conflict" }, 409);
    scenario.tokenBudgetReviews.items.splice(index, 1);
    scenario.tokenBudgetReviews.targetCount = scenario.tokenBudgetReviews.items.length;
    return fulfillJson(route, { sampleId: request.sampleId, version: request.expectedVersion, exportStarted: false });
  }

  const imageMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/count-review\/(\d+)\/image$/);
  if (method === "GET" && imageMatch) {
    await route.fulfill({ status: 200, contentType: "image/png", body: TRANSPARENT_PNG });
    return;
  }

  const countReviewMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/count-review$/);
  if (countReviewMatch && method === "GET") {
    const limit = Number(new URL(route.request().url()).searchParams.get("limit") ?? "50");
    scenario.countReviewRequests.push(limit);
    return fulfillJson(route, scenario.countReview);
  }
  if (countReviewMatch && method === "POST") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(countReviewMatch[1]));
    updateJobState(snapshot, "running", "dropout");
    return fulfillJson(route, { jobId: snapshot.job.jobId, started: true });
  }

  const countConfirmMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/count-review\/confirm$/);
  if (countConfirmMatch && method === "POST") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(countConfirmMatch[1]));
    updateJobState(snapshot, "running", "dropout");
    return fulfillJson(route, { jobId: snapshot.job.jobId, started: true });
  }

  const countItemMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/count-review\/(\d+)$/);
  if (countItemMatch && method === "PUT") {
    const update = body as { source: "classify" | "vlm" | "manual"; count?: CountValue };
    return fulfillJson(route, applyCountDecision(scenario, Number(countItemMatch[2]), update.source, update.count));
  }

  const countBatchMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/count-review\/batch$/);
  if (countBatchMatch && method === "POST") {
    const updates = (body as { updates: Array<{ sampleId: number; source: "classify" | "vlm" | "manual"; count?: CountValue }> }).updates;
    return fulfillJson(route, { items: updates.map((update) => applyCountDecision(scenario, update.sampleId, update.source, update.count)) });
  }

  const jobMatch = pathname.match(/^\/api\/jobs\/([^/]+)$/);
  if (jobMatch && method === "GET") return fulfillJson(route, snapshotFor(scenario, decodeURIComponent(jobMatch[1])));

  const workspaceMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/confirm-workspace$/);
  if (workspaceMatch && method === "POST") {
    const jobId = decodeURIComponent(workspaceMatch[1]);
    const snapshot = snapshotFor(scenario, jobId);
    updateJobState(snapshot, "preparing_workspace");
    return fulfillJson(route, { jobId, status: "preparing_workspace", datasetRoot: "E:\\datasets\\browser-characterization", overlayRoot: "E:\\datasets\\.anima-overlay" });
  }

  const startMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/start$/);
  if (startMatch && method === "POST") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(startMatch[1]));
    updateJobState(snapshot, "running", "caption");
    return fulfillJson(route, { jobId: snapshot.job.jobId, started: true });
  }

  const moduleControlMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/modules\/([^/]+)\/(pause|resume)$/);
  if (moduleControlMatch && method === "POST") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(moduleControlMatch[1]));
    const moduleId = decodeURIComponent(moduleControlMatch[2]);
    const action = moduleControlMatch[3];
    const summary = snapshot.modules.find((module) => module.module_id === moduleId);
    if (action === "pause") {
      if (moduleId === snapshot.job.currentModuleId) updateJobState(snapshot, "paused", moduleId);
      else if (summary) summary.status = "paused";
      else snapshot.modules.push({ module_id: moduleId, status: "paused", completed: 0, failed: 0, skipped: 0, total: 0, issue_count: 0 });
    } else if (moduleId === snapshot.job.currentModuleId) updateJobState(snapshot, "running", moduleId);
    else snapshot.modules = snapshot.modules.filter((module) => module.module_id !== moduleId);
    return fulfillJson(route, snapshot);
  }

  const pauseMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/pause$/);
  if (pauseMatch && method === "POST") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(pauseMatch[1]));
    updateJobState(snapshot, "paused", snapshot.job.currentModuleId);
    return fulfillJson(route, { status: "paused" });
  }

  const resumeMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/resume$/);
  if (resumeMatch && method === "POST") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(resumeMatch[1]));
    updateJobState(snapshot, "running", snapshot.job.currentModuleId);
    return fulfillJson(route, { status: "running" });
  }

  const pauseNlMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/nl\/pause$/);
  if (pauseNlMatch && method === "POST") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(pauseNlMatch[1]));
    updateJobState(snapshot, "paused", "nl");
    return fulfillJson(route, { status: "paused" });
  }

  const resumeNlMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/nl\/resume$/);
  if (resumeNlMatch && method === "POST") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(resumeNlMatch[1]));
    updateJobState(snapshot, "running", "nl");
    return fulfillJson(route, { status: "running" });
  }

  const pausePolicyMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/policy\/pause$/);
  if (pausePolicyMatch && method === "POST") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(pausePolicyMatch[1]));
    updateJobState(snapshot, "paused", "dropout");
    return fulfillJson(route, { status: "paused" });
  }

  const resumePolicyMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/policy\/resume$/);
  if (resumePolicyMatch && method === "POST") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(resumePolicyMatch[1]));
    updateJobState(snapshot, "running", "dropout");
    return fulfillJson(route, { status: "running" });
  }

  const budgetMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/nl\/api-budget$/);
  if (budgetMatch && method === "POST") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(budgetMatch[1]));
    const amount = Number((body as { amount: number }).amount);
    snapshot.job.apiBudgetExtra += amount;
    snapshot.job.apiBudgetRevision += 1;
    return fulfillJson(route, { apiBudgetExtra: snapshot.job.apiBudgetExtra, apiBudgetRevision: snapshot.job.apiBudgetRevision });
  }

  const outcomesMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/nl\/confirm-api-outcomes$/);
  if (outcomesMatch && method === "POST") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(outcomesMatch[1]));
    snapshot.nlPendingApiDecisions = 0;
    return fulfillJson(route, { requeued: 0 });
  }

  const manualNlBatchMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/nl\/manual-retry-batch$/);
  if (manualNlBatchMatch && method === "POST") {
    const parentJobId = decodeURIComponent(manualNlBatchMatch[1]);
    const issueIds = (body as { issueIds: string[] }).issueIds;
    const jobId = `${parentJobId}-manual-retry-batch`;
    scenario.snapshots.set(jobId, makeSnapshot({ jobId, status: "running", currentModuleId: "nl" }));
    return fulfillJson(route, { jobId, parentJobId, issueIds, targetCount: issueIds.length, started: true });
  }

  const manualNlMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/nl\/(manual-retry|manual-write)$/);
  if (manualNlMatch && method === "POST") {
    const parentJobId = decodeURIComponent(manualNlMatch[1]);
    const jobId = `${parentJobId}-${manualNlMatch[2]}`;
    scenario.snapshots.set(jobId, makeSnapshot({ jobId, status: "running", currentModuleId: "nl" }));
    return fulfillJson(route, { jobId, parentJobId, targetCount: 1, started: true });
  }

  const recoverMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/recover$/);
  if (recoverMatch && method === "POST") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(recoverMatch[1]));
    updateJobState(snapshot, "running", "caption");
    return fulfillJson(route, { jobId: snapshot.job.jobId, returnedLeases: 0, committedPrepared: 0, repeatedPrepared: 0, pendingApiDecisions: 0, started: true, status: "running" });
  }

  const repairMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/repair$/);
  if (repairMatch && method === "POST") {
    const parentJobId = decodeURIComponent(repairMatch[1]);
    const jobId = `${parentJobId}-repair`;
    const snapshot = makeSnapshot({ jobId, status: "running", currentModuleId: "caption", parentJobId });
    scenario.snapshots.set(jobId, snapshot);
    return fulfillJson(route, { jobId, parentJobId, targetCount: 2, preparedTargetCount: 2, estimatedApiRequests: 0, started: true });
  }

  const restoreMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/restore-original-annotations$/);
  if (restoreMatch && method === "POST") return fulfillJson(route, { jobId: decodeURIComponent(restoreMatch[1]), restored: 3, backupZip: "E:\\datasets\\backup.zip" });

  const cancelMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/cancel$/);
  if (cancelMatch && method === "POST") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(cancelMatch[1]));
    updateJobState(snapshot, "cancelling", snapshot.job.currentModuleId);
    return fulfillJson(route, { status: "cancelling" });
  }

  const pinMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/pin$/);
  if (pinMatch && method === "PUT") {
    const snapshot = snapshotFor(scenario, decodeURIComponent(pinMatch[1]));
    snapshot.job.pinned = Boolean((body as { pinned: boolean }).pinned);
    return fulfillJson(route, { pinned: snapshot.job.pinned });
  }

  const deleteHistoryMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/delete-history$/);
  if (deleteHistoryMatch && method === "POST") {
    const jobId = decodeURIComponent(deleteHistoryMatch[1]);
    const target = scenario.snapshots.get(jobId);
    scenario.snapshots.delete(jobId);
    scenario.jobs = {
      ...scenario.jobs,
      jobs: scenario.jobs.jobs.filter((item) => item.jobId !== jobId),
    };
    if (target?.job.parentJobId) {
      const parent = scenario.snapshots.get(target.job.parentJobId);
      if (parent) parent.repairChildren = parent.repairChildren.filter((child) => child.jobId !== jobId);
    }
    return fulfillJson(route, {
      jobId,
      deletedControlRecord: true,
      preserved: { dataset: true, exports: true, backup: true },
    });
  }

  const discardMatch = pathname.match(/^\/api\/jobs\/([^/]+)\/discard$/);
  if (discardMatch && method === "POST") {
    const jobId = decodeURIComponent(discardMatch[1]);
    const target = scenario.snapshots.get(jobId);
    scenario.snapshots.delete(jobId);
    if (target?.job.parentJobId) {
      const parent = scenario.snapshots.get(target.job.parentJobId);
      if (parent) parent.repairChildren = parent.repairChildren.filter((child) => child.jobId !== jobId);
    }
    return fulfillJson(route, { jobId, overlayDeleted: true });
  }

  const profileMatch = pathname.match(/^\/api\/nl\/profiles\/([^/]+)$/);
  if (profileMatch && method === "PUT") {
    const profileId = decodeURIComponent(profileMatch[1]);
    const saved = { ...scenario.profiles[0], ...(body as object), profileId } as NlProfile;
    scenario.profiles = [saved];
    return fulfillJson(route, saved);
  }

  const credentialMatch = pathname.match(/^\/api\/nl\/credentials\/([^/]+)$/);
  if (credentialMatch && method === "PUT") return fulfillJson(route, { stored: true });
  if (credentialMatch && method === "DELETE") return fulfillJson(route, { deleted: true });

  scenario.unhandledApiRequests.push(key);
  await fulfillJson(route, { detail: `unhandled mock route: ${key}` }, 500);
}

export async function installApiMock(page: Page, scenario: ApiScenario): Promise<void> {
  await page.addInitScript(() => {
    const disableMotion = () => {
      const style = document.createElement("style");
      style.textContent = "*, *::before, *::after { animation-duration: 0s !important; transition-duration: 0s !important; }";
      document.head.append(style);
    };
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", disableMotion, { once: true });
    else disableMotion();
  });
  page.on("console", (message) => {
    if (message.type() === "error") scenario.consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => scenario.pageErrors.push(error.message));
  await page.route("**/*", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.origin !== LOCAL_ORIGIN) {
      scenario.unexpectedRequests.push(`${request.method()} ${url.href}`);
      await route.abort("blockedbyclient");
      return;
    }
    if (url.pathname !== "/health" && !url.pathname.startsWith("/api/")) {
      await route.continue();
      return;
    }
    await handleApiRequest(scenario, route, url.pathname);
  });
}

export async function openApp(page: Page, options: OpenAppOptions = {}): Promise<void> {
  if (options.jobId !== undefined || options.language !== undefined) {
    await page.addInitScript((values: OpenAppOptions) => {
      if (values.jobId !== undefined) window.localStorage.setItem("anima.ui.jobId.v1", values.jobId);
      if (values.language !== undefined) window.localStorage.setItem("anima.ui.language.v1", values.language);
    }, options);
  }
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Anima Dataset Tool" })).toBeVisible();
}

export async function openPrecheck(page: Page): Promise<void> {
  await page.locator(".workflow-rail").getByRole("button", { name: /Submission checks/ }).click();
}

export async function openSetup(page: Page): Promise<void> {
  await page.locator(".workflow-rail").getByRole("button", { name: /Dataset and configuration/ }).click();
}

export const test = base.extend<{ api: ApiScenario }>({
  api: [async ({ page }, use) => {
    const scenario = createApiScenario();
    await installApiMock(page, scenario);
    await use(scenario);
    expect(scenario.unexpectedRequests, "unexpected external requests").toEqual([]);
    expect(scenario.unhandledApiRequests, "unhandled local API requests").toEqual([]);
    expect(
      scenario.consoleErrors.filter((message) => !scenario.expectedConsoleErrorPatterns.some((pattern) => pattern.test(message))),
      "browser console errors",
    ).toEqual([]);
    expect(scenario.pageErrors, "page errors").toEqual([]);
  }, { auto: true }],
});

export { expect };
