import type { OcrExecutionRequest, OcrExecutionTuning } from "./api";
import { newDraft, newOcrExecution, type Draft } from "./draft";

/** Storage keys are versioned so a future schema can be rejected safely. */
export const draftStorageKey = "anima.ui.draft.v1";
export const annotationPresetStorageKey = "anima.ui.annotation-presets.v1";

const DRAFT_ENVELOPE_VERSION = 1 as const;
const PRESET_ENVELOPE_VERSION = 1 as const;
const PRESET_SCHEMA_VERSION = 1 as const;

export type DraftPersistenceState = {
  draft: Draft;
  ocrExecution: OcrExecutionRequest;
};

export type DraftPersistenceEnvelope = {
  schemaVersion: typeof DRAFT_ENVELOPE_VERSION;
  savedAt: string;
  draft: Draft;
  ocrExecution: OcrExecutionRequest;
};

export type AnnotationPresetConfig = Omit<Draft, "sourceRoot" | "outputRoot">;

export type AnnotationPreset = {
  presetId: string;
  name: string;
  schemaVersion: typeof PRESET_SCHEMA_VERSION;
  savedAt: string;
  config: AnnotationPresetConfig;
  ocrExecution: OcrExecutionRequest;
};

type PresetEnvelope = {
  schemaVersion: typeof PRESET_ENVELOPE_VERSION;
  presets: AnnotationPreset[];
};

const hasOwn = (value: object, key: string): boolean => Object.prototype.hasOwnProperty.call(value, key);

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function exactKeys(value: Record<string, unknown>, required: readonly string[], optional: readonly string[] = []): boolean {
  const allowed = new Set([...required, ...optional]);
  return required.every((key) => hasOwn(value, key)) && Object.keys(value).every((key) => allowed.has(key));
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function integerInRange(value: unknown, minimum: number, maximum: number): value is number {
  return finiteNumber(value) && Number.isInteger(value) && value >= minimum && value <= maximum;
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function stringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function numberRecord(value: unknown, minimum = -Infinity, maximum = Infinity): value is Record<string, number> {
  return record(value) && Object.values(value).every((item) => finiteNumber(item) && item >= minimum && item <= maximum);
}

function isOcrTuning(value: unknown, field: "textDetLimitSideLen" | "textBatchSize"): value is OcrExecutionTuning {
  if (!record(value) || !exactKeys(value, ["mode", "value"])) return false;
  if (value.mode === "auto") return value.value === null;
  if (value.mode !== "manual" || !integerInRange(value.value, field === "textDetLimitSideLen" ? 1920 : 1, field === "textDetLimitSideLen" ? 3840 : 8)) return false;
  return field === "textBatchSize" || value.value % 32 === 0;
}

export function isOcrExecutionRequest(value: unknown): value is OcrExecutionRequest {
  return record(value)
    && exactKeys(value, ["textDetLimitSideLen", "textBatchSize"])
    && isOcrTuning(value.textDetLimitSideLen, "textDetLimitSideLen")
    && isOcrTuning(value.textBatchSize, "textBatchSize");
}

function isDraft(value: unknown): value is Draft {
  if (!record(value) || value.schemaVersion !== 10) return false;
  if (!exactKeys(value, [
    "schemaVersion", "moduleBatchSize", "workMode", "overwriteMode", "sourceRoot", "annotationBackup", "recursive",
    "captionFormat", "imageDecode", "caption", "classify", "replace", "ocr", "nl", "countReview", "dropout", "tokenBudget", "export",
  ], ["outputRoot"])) return false;
  if (typeof value.sourceRoot !== "string" || (hasOwn(value, "outputRoot") && value.outputRoot !== undefined && typeof value.outputRoot !== "string")) return false;
  if (value.workMode !== "in_place" && value.workMode !== "full_copy") return false;
  if (value.overwriteMode !== "incremental" && value.overwriteMode !== "rebuild") return false;
  if (value.annotationBackup !== "required" || typeof value.recursive !== "boolean") return false;

  const batches = value.moduleBatchSize;
  if (!record(batches) || !exactKeys(batches, ["caption", "classify", "replace", "ocr", "nl", "countReview", "dropout", "tokenBudget", "export"])) return false;
  if (!Object.values(batches).every((item) => integerInRange(item, 1, 10_000))) return false;

  const format = value.captionFormat;
  if (!record(format) || !exactKeys(format, ["flatTxtLayout", "replaceUnderscoresWithSpaces", "preserveEscapes", "triggersEnabled", "triggerTerms"])) return false;
  if ((format.flatTxtLayout !== "single_line" && format.flatTxtLayout !== "nl_newline")
    || typeof format.replaceUnderscoresWithSpaces !== "boolean" || typeof format.preserveEscapes !== "boolean"
    || typeof format.triggersEnabled !== "boolean" || !stringArray(format.triggerTerms)) return false;

  const imageDecode = value.imageDecode;
  if (!record(imageDecode) || !exactKeys(imageDecode, ["extensions", "rejectMultiFrame", "applyExifTranspose", "alphaBackground", "invalidImageAction"])) return false;
  if (!stringArray(imageDecode.extensions) || typeof imageDecode.rejectMultiFrame !== "boolean"
    || typeof imageDecode.applyExifTranspose !== "boolean" || typeof imageDecode.alphaBackground !== "string"
    || (imageDecode.invalidImageAction !== "block" && imageDecode.invalidImageAction !== "skip")) return false;

  const caption = value.caption;
  if (!record(caption) || !exactKeys(caption, ["enabled", "thresholdMode", "overwriteTxt", "inputTxtMode", "taggerFallbackOnMissingTxt"], ["resourceId", "uniformThreshold", "categoryThresholds"])) return false;
  if (typeof caption.enabled !== "boolean" || typeof caption.overwriteTxt !== "boolean" || typeof caption.taggerFallbackOnMissingTxt !== "boolean"
    || !["model_default", "uniform", "per_category"].includes(String(caption.thresholdMode))
    || (caption.inputTxtMode !== "tag" && caption.inputTxtMode !== "nl")
    || (hasOwn(caption, "resourceId") && caption.resourceId !== undefined && !nonEmptyString(caption.resourceId))
    || (hasOwn(caption, "uniformThreshold") && caption.uniformThreshold !== undefined && !integerOrFloatInRange(caption.uniformThreshold, 0, 1))
    || (hasOwn(caption, "categoryThresholds") && caption.categoryThresholds !== undefined && !numberRecord(caption.categoryThresholds, 0, 1))) return false;

  const classify = value.classify;
  if (!record(classify) || !exactKeys(classify, ["enabled", "indexMode", "overwriteJson", "overwriteCount"], ["wikiDataSourceId", "resourceId", "customResourcePath"])) return false;
  if (typeof classify.enabled !== "boolean" || (classify.indexMode !== "bundled" && classify.indexMode !== "custom")
    || typeof classify.overwriteJson !== "boolean" || typeof classify.overwriteCount !== "boolean"
    || (hasOwn(classify, "wikiDataSourceId") && classify.wikiDataSourceId !== undefined && !nonEmptyString(classify.wikiDataSourceId))
    || (hasOwn(classify, "resourceId") && classify.resourceId !== undefined && !nonEmptyString(classify.resourceId))
    || (hasOwn(classify, "customResourcePath") && classify.customResourcePath !== undefined && typeof classify.customResourcePath !== "string")) return false;

  const replace = value.replace;
  if (!record(replace) || !exactKeys(replace, ["enabled", "indexMode"], ["resourceId", "customIndexPath"])) return false;
  if (typeof replace.enabled !== "boolean" || (replace.indexMode !== "bundled" && replace.indexMode !== "custom")
    || (hasOwn(replace, "resourceId") && replace.resourceId !== undefined && !nonEmptyString(replace.resourceId))
    || (hasOwn(replace, "customIndexPath") && replace.customIndexPath !== undefined && typeof replace.customIndexPath !== "string")) return false;

  const ocr = value.ocr;
  if (!record(ocr) || !exactKeys(ocr, ["enabled", "device", "llmMinConfidence", "forceReprocess", "resourceId"])) return false;
  if (typeof ocr.enabled !== "boolean" || !["auto", "cuda", "cpu"].includes(String(ocr.device))
    || !integerOrFloatInRange(ocr.llmMinConfidence, 0, 1) || typeof ocr.forceReprocess !== "boolean" || !nonEmptyString(ocr.resourceId)) return false;

  const nl = value.nl;
  if (!record(nl) || !exactKeys(nl, ["enabled", "reuseOriginalNl", "apiEnabled", "useImage", "useFullJson", "systemPrompt", "promptVersion", "captionPreset", "lengthDistribution", "lengthSeed", "apiProfileId", "apiPolicy"])) return false;
  if (typeof nl.enabled !== "boolean" || typeof nl.reuseOriginalNl !== "boolean" || typeof nl.apiEnabled !== "boolean"
    || typeof nl.useImage !== "boolean" || typeof nl.useFullJson !== "boolean" || typeof nl.systemPrompt !== "string"
    || typeof nl.promptVersion !== "string" || !["general", "style", "character", "moss-general", "moss-style", "moss-character"].includes(String(nl.captionPreset))
    || typeof nl.lengthSeed !== "string" || typeof nl.apiProfileId !== "string") return false;
  if (!record(nl.lengthDistribution) || !exactKeys(nl.lengthDistribution, ["short", "medium", "long"])
    || !Object.values(nl.lengthDistribution).every((item) => integerInRange(item, 0, 100))) return false;
  const policy = nl.apiPolicy;
  if (!record(policy) || !exactKeys(policy, ["maxRequestsPerMinute", "backupEnabled"], ["maxHttpAttempts"])) return false;
  if ((policy.maxRequestsPerMinute !== "unlimited" && !integerInRange(policy.maxRequestsPerMinute, 1, 100_000))
    || typeof policy.backupEnabled !== "boolean"
    || (hasOwn(policy, "maxHttpAttempts") && !integerInRange(policy.maxHttpAttempts, 1, 1_000_000))) return false;

  const countReview = value.countReview;
  if (!record(countReview) || !exactKeys(countReview, ["enabled", "protocolVersion"]) || typeof countReview.enabled !== "boolean" || typeof countReview.protocolVersion !== "string") return false;

  const dropout = value.dropout;
  if (!record(dropout) || !exactKeys(dropout, ["enabled", "policyVersion", "seed", "artist", "quality", "appearanceNl"]) || typeof dropout.enabled !== "boolean" || typeof dropout.policyVersion !== "string" || typeof dropout.seed !== "string") return false;
  const artist = dropout.artist;
  if (!record(artist) || !exactKeys(artist, ["enabled", "dropoutProbability"], ["addAtPrefix"])
    || typeof artist.enabled !== "boolean" || !integerOrFloatInRange(artist.dropoutProbability, 0, 1)
    || (hasOwn(artist, "addAtPrefix") && typeof artist.addAtPrefix !== "boolean")) return false;
  const quality = dropout.quality;
  if (!record(quality) || !exactKeys(quality, ["enabled", "dropoutProbability", "device", "batchSize"], ["resourceId"]) || typeof quality.enabled !== "boolean"
    || !integerOrFloatInRange(quality.dropoutProbability, 0, 1) || !["auto", "cuda", "cpu"].includes(String(quality.device))
    || !integerInRange(quality.batchSize, 1, 10_000) || (hasOwn(quality, "resourceId") && quality.resourceId !== undefined && !nonEmptyString(quality.resourceId))) return false;
  const appearanceNl = dropout.appearanceNl;
  if (!record(appearanceNl) || !exactKeys(appearanceNl, ["enabled", "solo", "nonSolo", "unknown"]) || typeof appearanceNl.enabled !== "boolean") return false;
  for (const group of ["solo", "nonSolo", "unknown"] as const) {
    const probabilities = appearanceNl[group];
    if (!record(probabilities) || !exactKeys(probabilities, ["dropNl", "dropAppearance"])
      || !integerOrFloatInRange(probabilities.dropNl, 0, 1) || !integerOrFloatInRange(probabilities.dropAppearance, 0, 1)
      || Number(probabilities.dropNl) + Number(probabilities.dropAppearance) > 1) return false;
  }

  const tokenBudget = value.tokenBudget;
  if (!record(tokenBudget) || !exactKeys(tokenBudget, ["enabled", "maxTokens", "resourceId"]) || typeof tokenBudget.enabled !== "boolean"
    || !integerInRange(tokenBudget.maxTokens, 1, 10_000_000) || !nonEmptyString(tokenBudget.resourceId)) return false;
  const exportConfig = value.export;
  if (!record(exportConfig) || !exactKeys(exportConfig, ["format"]) || !["both", "json", "flat_txt"].includes(String(exportConfig.format))) return false;
  return true;
}

function integerOrFloatInRange(value: unknown, minimum: number, maximum: number): value is number {
  return finiteNumber(value) && value >= minimum && value <= maximum;
}

function storageOrNull(storage?: Storage): Storage | null {
  if (storage) return storage;
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}

function readStorage(storage: Storage | null, key: string): string | null {
  if (!storage) return null;
  try { return storage.getItem(key); } catch { return null; }
}

function writeStorage(storage: Storage | null, key: string, value: string): boolean {
  if (!storage) return false;
  try { storage.setItem(key, value); return true; } catch { return false; }
}

function cloneWithoutPaths(draft: Draft): AnnotationPresetConfig {
  const { sourceRoot: _sourceRoot, outputRoot: _outputRoot, ...config } = draft;
  return config;
}

export function loadDraftState(storage?: Storage): DraftPersistenceState {
  const fallback = { draft: newDraft(), ocrExecution: newOcrExecution() };
  const raw = readStorage(storageOrNull(storage), draftStorageKey);
  if (!raw) return fallback;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!record(parsed) || !exactKeys(parsed, ["schemaVersion", "savedAt", "draft", "ocrExecution"])
      || parsed.schemaVersion !== DRAFT_ENVELOPE_VERSION || typeof parsed.savedAt !== "string"
      || !isDraft(parsed.draft) || !isOcrExecutionRequest(parsed.ocrExecution)) return fallback;
    return { draft: parsed.draft, ocrExecution: parsed.ocrExecution };
  } catch {
    return fallback;
  }
}

export function saveDraftState(state: DraftPersistenceState, storage?: Storage, savedAt = new Date().toISOString()): boolean {
  if (!isDraft(state.draft) || !isOcrExecutionRequest(state.ocrExecution)) return false;
  const envelope: DraftPersistenceEnvelope = { schemaVersion: DRAFT_ENVELOPE_VERSION, savedAt, draft: state.draft, ocrExecution: state.ocrExecution };
  try { return writeStorage(storageOrNull(storage), draftStorageKey, JSON.stringify(envelope)); } catch { return false; }
}

function isPresetConfig(value: unknown): value is AnnotationPresetConfig {
  return record(value) && !hasOwn(value, "sourceRoot") && !hasOwn(value, "outputRoot")
    && isDraft({ ...value, sourceRoot: "" });
}

function isAnnotationPreset(value: unknown): value is AnnotationPreset {
  if (!record(value) || !exactKeys(value, ["presetId", "name", "schemaVersion", "savedAt", "config", "ocrExecution"])) return false;
  return nonEmptyString(value.presetId) && nonEmptyString(value.name) && value.name.length <= 200
    && value.schemaVersion === PRESET_SCHEMA_VERSION && typeof value.savedAt === "string"
    && isPresetConfig(value.config) && isOcrExecutionRequest(value.ocrExecution);
}

export function loadAnnotationPresets(storage?: Storage): AnnotationPreset[] {
  const raw = readStorage(storageOrNull(storage), annotationPresetStorageKey);
  if (!raw) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!record(parsed) || !exactKeys(parsed, ["schemaVersion", "presets"]) || parsed.schemaVersion !== PRESET_ENVELOPE_VERSION || !Array.isArray(parsed.presets)) return [];
    const seen = new Set<string>();
    return parsed.presets.filter((item): item is AnnotationPreset => {
      if (!isAnnotationPreset(item) || seen.has(item.presetId)) return false;
      seen.add(item.presetId);
      return true;
    }).slice(0, 100);
  } catch {
    return [];
  }
}

export function saveAnnotationPresets(presets: readonly AnnotationPreset[], storage?: Storage): boolean {
  const valid = presets.filter(isAnnotationPreset).slice(0, 100);
  if (valid.length !== presets.length) return false;
  const envelope: PresetEnvelope = { schemaVersion: PRESET_ENVELOPE_VERSION, presets: [...valid] };
  try { return writeStorage(storageOrNull(storage), annotationPresetStorageKey, JSON.stringify(envelope)); } catch { return false; }
}

export function createAnnotationPreset(name: string, draft: Draft, ocrExecution: OcrExecutionRequest, savedAt = new Date().toISOString(), presetId = ""): AnnotationPreset | null {
  const trimmedName = name.trim();
  if (!trimmedName || !isDraft(draft) || !isOcrExecutionRequest(ocrExecution)) return null;
  const id = presetId || (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? `annotation:${crypto.randomUUID()}`
    : `annotation:${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`);
  return { presetId: id, name: trimmedName.slice(0, 200), schemaVersion: PRESET_SCHEMA_VERSION, savedAt, config: cloneWithoutPaths(draft), ocrExecution };
}

export function applyAnnotationPreset(preset: AnnotationPreset, currentDraft: Draft): Draft {
  if (!isAnnotationPreset(preset) || !isDraft(currentDraft)) return currentDraft;
  return {
    ...preset.config,
    sourceRoot: currentDraft.sourceRoot,
    ...(preset.config.workMode === "full_copy" && hasOwn(currentDraft, "outputRoot") ? { outputRoot: currentDraft.outputRoot } : {}),
  } as Draft;
}

export function presetConfigContainsPaths(preset: AnnotationPreset): boolean {
  return isAnnotationPreset(preset) && (hasOwn(preset.config, "sourceRoot") || hasOwn(preset.config, "outputRoot"));
}
