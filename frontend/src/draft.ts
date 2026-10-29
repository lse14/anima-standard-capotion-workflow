import type { OcrDevice, OcrExecutionRequest, ResourceCatalogResponse, ResourceKind } from "./api";

export type WorkMode = "in_place" | "full_copy";
export type OverwriteMode = "incremental" | "rebuild";
export type CaptionThresholdMode = "model_default" | "uniform" | "per_category";
export type CaptionInputTxtMode = "tag" | "nl";
export type ReplaceIndexMode = "bundled" | "custom";
export type InvalidImageAction = "block" | "skip";
export type QualityDevice = "auto" | "cuda" | "cpu";
export type ExportFormat = "both" | "json" | "flat_txt";
export type FlatTxtLayout = "single_line" | "nl_newline";

export type ModuleBatchSize = {
  caption: number;
  classify: number;
  replace: number;
  ocr: number;
  nl: number;
  countReview: number;
  dropout: number;
  tokenBudget: number;
  export: number;
};

export type CaptionFormatDraft = {
  flatTxtLayout: FlatTxtLayout;
  replaceUnderscoresWithSpaces: boolean;
  preserveEscapes: boolean;
  triggersEnabled: boolean;
  triggerTerms: string[];
};

export type ImageDecodeDraft = {
  extensions: string[];
  rejectMultiFrame: boolean;
  applyExifTranspose: boolean;
  alphaBackground: string;
  invalidImageAction: InvalidImageAction;
};

export type CaptionDraft = {
  enabled: boolean;
  thresholdMode: CaptionThresholdMode;
  overwriteTxt: boolean;
  inputTxtMode: CaptionInputTxtMode;
  taggerFallbackOnMissingTxt: boolean;
  resourceId?: string;
  uniformThreshold?: number;
  categoryThresholds?: Record<string, number>;
};

export type ClassifyDraft = {
  enabled: boolean;
  indexMode: "bundled" | "custom";
  overwriteJson: boolean;
  overwriteCount: boolean;
  wikiDataSourceId?: string;
  resourceId?: string;
  customResourcePath?: string;
};

export type ReplaceDraft = {
  enabled: boolean;
  indexMode: ReplaceIndexMode;
  resourceId?: string;
  customIndexPath?: string;
};

export type OcrDraft = {
  enabled: boolean;
  device: OcrDevice;
  llmMinConfidence: number;
  forceReprocess: boolean;
  resourceId: "ocr-ppocrv5-server-paddle-v1";
};

export type NlApiPolicyDraft = {
  maxRequestsPerMinute: number | "unlimited";
  backupEnabled: boolean;
  maxHttpAttempts?: number;
};

export type NlDraft = {
  enabled: boolean;
  reuseOriginalNl: boolean;
  apiEnabled: boolean;
  useImage: boolean;
  useFullJson: boolean;
  systemPrompt: string;
  promptVersion: string;
  captionPreset: "general" | "style" | "character" | "moss-general" | "moss-style" | "moss-character";
  lengthDistribution: { short: number; medium: number; long: number };
  lengthSeed: string;
  apiProfileId: string;
  apiPolicy: NlApiPolicyDraft;
};

export type TokenBudgetDraft = {
  enabled: boolean;
  maxTokens: number;
  resourceId: string;
};

export type CountReviewDraft = {
  enabled: boolean;
  protocolVersion: string;
};

export type DropoutArtistDraft = {
  enabled: boolean;
  dropoutProbability: number;
  addAtPrefix?: boolean;
};

export type DropoutQualityDraft = {
  enabled: boolean;
  dropoutProbability: number;
  device: QualityDevice;
  batchSize: number;
  resourceId?: string;
};

export type AppearanceProbabilityDraft = {
  dropNl: number;
  dropAppearance: number;
};

export type AppearanceNlDraft = {
  enabled: boolean;
  solo: AppearanceProbabilityDraft;
  nonSolo: AppearanceProbabilityDraft;
  unknown: AppearanceProbabilityDraft;
};

export type DropoutDraft = {
  enabled: boolean;
  policyVersion: string;
  seed: string;
  artist: DropoutArtistDraft;
  quality: DropoutQualityDraft;
  appearanceNl: AppearanceNlDraft;
};

export type ExportDraft = {
  format: ExportFormat;
};

export type Draft = {
  schemaVersion: 10;
  moduleBatchSize: ModuleBatchSize;
  workMode: WorkMode;
  overwriteMode: OverwriteMode;
  sourceRoot: string;
  outputRoot?: string;
  annotationBackup: "required";
  recursive: boolean;
  captionFormat: CaptionFormatDraft;
  imageDecode: ImageDecodeDraft;
  caption: CaptionDraft;
  classify: ClassifyDraft;
  replace: ReplaceDraft;
  ocr: OcrDraft;
  nl: NlDraft;
  countReview: CountReviewDraft;
  dropout: DropoutDraft;
  tokenBudget: TokenBudgetDraft;
  export: ExportDraft;
};

export type DraftSectionKey =
  | "captionFormat"
  | "imageDecode"
  | "caption"
  | "classify"
  | "replace"
  | "ocr"
  | "nl"
  | "countReview"
  | "dropout"
  | "tokenBudget"
  | "export";

export function newDraft(): Draft {
  return {
    schemaVersion: 10,
    moduleBatchSize: { caption: 4, classify: 128, replace: 128, ocr: 4, nl: 3, countReview: 100, dropout: 4, tokenBudget: 128, export: 500 },
    workMode: "in_place", overwriteMode: "incremental", sourceRoot: "", annotationBackup: "required", recursive: false,
    captionFormat: { flatTxtLayout: "nl_newline", replaceUnderscoresWithSpaces: true, preserveEscapes: true, triggersEnabled: false, triggerTerms: [] },
    imageDecode: { extensions: [".jpg", ".jpeg", ".png", ".webp", ".bmp"], rejectMultiFrame: true, applyExifTranspose: true, alphaBackground: "#FFFFFF", invalidImageAction: "block" },
    caption: { enabled: true, thresholdMode: "model_default", overwriteTxt: false, inputTxtMode: "tag", taggerFallbackOnMissingTxt: true, resourceId: "caption-e621-eva02-large-full-v1" },
    classify: { enabled: true, indexMode: "bundled", overwriteJson: false, overwriteCount: false, resourceId: "classify-e621-20260724-v1" },
    replace: { enabled: true, indexMode: "bundled", resourceId: "replace-e621-20260726-v2" },
    ocr: { enabled: false, device: "auto", llmMinConfidence: 0.5, forceReprocess: false, resourceId: "ocr-ppocrv5-server-paddle-v1" },
    nl: { enabled: true, reuseOriginalNl: true, apiEnabled: true, useImage: true, useFullJson: false, systemPrompt: "", promptVersion: "nl-default-prompt-v4", captionPreset: "general", lengthDistribution: { short: 33, medium: 34, long: 33 }, lengthSeed: "anima-nl-length-v1", apiProfileId: "default", apiPolicy: { maxRequestsPerMinute: 60, backupEnabled: false } },
    countReview: { enabled: true, protocolVersion: "count-review-v1" },
    dropout: {
      enabled: false, policyVersion: "dataset-batch-policy-v1", seed: "anima-policy-default-v1",
      artist: { enabled: true, dropoutProbability: 0, addAtPrefix: true }, quality: { enabled: true, dropoutProbability: 0, device: "auto", batchSize: 4, resourceId: "lse14-scorer-5k-v1" },
      appearanceNl: { enabled: true, solo: { dropNl: 0.7, dropAppearance: 0.05 }, nonSolo: { dropNl: 0.05, dropAppearance: 0.7 }, unknown: { dropNl: 0.15, dropAppearance: 0.15 } },
    },
    tokenBudget: { enabled: true, maxTokens: 490, resourceId: "tokenizer-qwen3-0.6b-anima-v1" },
    export: { format: "both" },
  };
}

export function newOcrExecution(): OcrExecutionRequest {
  return {
    textDetLimitSideLen: { mode: "auto", value: null },
    textBatchSize: { mode: "auto", value: null },
  };
}

export function replaceDraftValue<K extends keyof Draft>(draft: Draft, key: K, value: Draft[K]): Draft {
  return { ...draft, [key]: value } as Draft;
}

export function patchDraftSection<K extends DraftSectionKey>(
  draft: Draft,
  key: K,
  patch: Partial<Draft[K]>,
): Draft {
  return { ...draft, [key]: { ...draft[key], ...patch } } as Draft;
}

export function applyResourceCatalogDefaults(draft: Draft, catalog: ResourceCatalogResponse): Draft {
  const defaults = catalog.defaults;
  const configured = (kind: ResourceKind, resourceId: string | undefined) => typeof resourceId === "string" && catalog.resources.some((item) =>
    item.kind === kind && item.resourceId === resourceId && item.available
    && !["incompatible", "unavailable"].includes(item.compatibility.status),
  );

  return {
    ...draft,
    caption: {
      ...draft.caption,
      resourceId: configured("tagging-model", draft.caption.resourceId) ? draft.caption.resourceId : defaults.taggingModel,
    },
    classify: {
      ...draft.classify,
      ...(draft.classify.indexMode === "custom"
        ? { indexMode: "custom" as const }
        : { indexMode: "bundled" as const, resourceId: configured("classification-index", draft.classify.resourceId) ? draft.classify.resourceId : defaults.classificationIndex }),
    },
    replace: draft.replace.indexMode === "bundled" ? {
        ...draft.replace,
        resourceId: configured("replacement-index", draft.replace.resourceId) ? draft.replace.resourceId : defaults.replacementIndex,
      } : draft.replace,
    dropout: {
      ...draft.dropout,
      quality: {
        ...draft.dropout.quality,
        resourceId: configured("dropout-model", draft.dropout.quality.resourceId) ? draft.dropout.quality.resourceId : defaults.dropoutModel,
      },
    },
  };
}
