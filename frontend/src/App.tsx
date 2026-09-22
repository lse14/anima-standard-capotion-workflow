import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  addNlBudget, cancelJob, cancelPreflight, confirmNlOutcomes, confirmWorkspace, deleteCompletedHistory, deleteNlSecret, deleteNlProfile, copyNlProfile, setNlProfileDefault, discardJob, getPreflightProgress, listNlProfiles, manualNlRetry, manualNlRetryBatch, manualNlWrite,
  pauseJob, recoverJob, restoreOriginalAnnotations, resumeJob, saveNlProfile, startPreflight, updateRunningNlApiConfig,
  repairJob, saveNlSecret, startPipeline, getDeviceRecommendations, type DeviceRecommendationResponse, type NlProfile, type OcrExecutionRequest, type PreflightSummary,
  type PipelineModuleId, type PreflightProgress, type ResourceCatalogResponse, type ResourceKind,
} from "./api";
import { CountReviewPanel } from "./CountReviewPanel";
import { IssuePanel } from "./components/IssuePanel";
import { makeFieldGuidanceCopy, ToggleField } from "./components/FormField";
import { ModuleBatchField } from "./components/ModuleBatchField";
import { resourceSelectable } from "./components/ResourcePicker";
import { TaskMonitor } from "./components/TaskMonitor";
import { RepairTasksPanel } from "./components/RepairTasksPanel";
import { WorkflowRail } from "./components/WorkflowRail";
import { CaptionStep } from "./components/steps/CaptionStep";
import { ClassifyStep } from "./components/steps/ClassifyStep";
import { ExportStep } from "./components/steps/ExportStep";
import { NlStep } from "./components/steps/NlStep";
import { OcrStep } from "./components/steps/OcrStep";
import { PolicyStep } from "./components/steps/PolicyStep";
import { PrecheckStep } from "./components/steps/PrecheckStep";
import { ReplaceStep } from "./components/steps/ReplaceStep";
import { SetupStep } from "./components/steps/SetupStep";
import { TokenBudgetStep } from "./components/steps/TokenBudgetStep";
import { applyResourceCatalogDefaults, newDraft, patchDraftSection, replaceDraftValue, type Draft, type DraftSectionKey } from "./draft";
import {
  applyAnnotationPreset, createAnnotationPreset, loadAnnotationPresets, loadDraftState, saveAnnotationPresets, saveDraftState,
  type AnnotationPreset,
} from "./draftPersistence";
import { text } from "./appCopy";
import { isActiveJobStatus, useJobMonitor } from "./hooks/useJobMonitor";
import { useResourceCatalog } from "./hooks/useResourceCatalog";
import { loadUiLanguage, moduleLabel, saveUiLanguage, statusLabel, translate, type UiLanguage } from "./i18n";
import { TokenBudgetReviewPanel } from "./TokenBudgetReviewPanel";
import "./styles.css";

type StepId = "setup" | PipelineModuleId | "precheck";
type WorkspaceView = "workflow" | "count_review" | "failures";
type PendingAction =
  | "preflight" | "preflight_cancel" | "confirm_workspace" | "start" | "repair" | "recover"
  | "terminate" | "discard" | "delete_history" | "restore" | "nl_manual_retry" | "nl_manual_retry_batch" | "nl_manual_write"
  | "nl_budget" | "nl_confirm_outcomes" | "pause" | "resume"
  | "profile_save" | "credential_delete"
  | "profile_default" | "profile_copy" | "profile_delete" | "runtime_api_update"
  | null;
type ActionName = Exclude<PendingAction, null>;

const draftModuleOrder: readonly PipelineModuleId[] = ["caption", "classify", "replace", "ocr", "nl", "count_review", "dropout", "token_budget", "export"];
const emptyProfile: NlProfile = { profileId: "default", name: "default", endpoint: "", model: "", backupModel: null, apiCredentialRef: "", credentialRef: "", systemPrompt: "", apiPolicy: { maxRequestsPerMinute: 60 }, policy: { maxRequestsPerMinute: 60 }, hasCredential: false };
const CREDENTIAL_REFERENCE = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;

function effectiveCredentialReference(profile: NlProfile): string {
  return CREDENTIAL_REFERENCE.test(profile.apiCredentialRef) ? profile.apiCredentialRef : `nl-profile-${profile.profileId}`;
}

export function App() {
  const [language, setLanguage] = useState<UiLanguage>(loadUiLanguage);
  const [profiles, setProfiles] = useState<NlProfile[]>([]);
  const [defaultProfileId, setDefaultProfileId] = useState<string | null>(null);
  const [initialPersistence] = useState(() => loadDraftState());
  const [draft, setDraft] = useState<Draft>(initialPersistence.draft);
  const draftDefaults = useMemo(newDraft, []);
  const [ocrExecution, setOcrExecution] = useState<OcrExecutionRequest>(initialPersistence.ocrExecution);
  const [presets, setPresets] = useState<AnnotationPreset[]>(loadAnnotationPresets);
  const [selectedPresetId, setSelectedPresetId] = useState("");
  const [presetName, setPresetName] = useState("");
  const [persistenceWarning, setPersistenceWarning] = useState<string | null>(null);
  const [preflight, setPreflight] = useState<PreflightSummary | null>(null);
  const [preflightStale, setPreflightStale] = useState(false);
  const [preflightProgress, setPreflightProgress] = useState<PreflightProgress | null>(null);
  const preflightGenerationRef = useRef(0);
  const [profile, setProfile] = useState<NlProfile>(emptyProfile);
  const [secret, setSecret] = useState("");
  const [diagnosticResetToken, setDiagnosticResetToken] = useState(0);
  const [budget, setBudget] = useState("100");
  const [triggerInput, setTriggerInput] = useState(() => initialPersistence.draft.captionFormat.triggerTerms.join(", "));
  const [attemptBudget, setAttemptBudget] = useState(() => {
    const value = initialPersistence.draft.nl.apiPolicy.maxHttpAttempts;
    return typeof value === "number" ? String(value) : "";
  });
  const [deviceRecommendation, setDeviceRecommendation] = useState<DeviceRecommendationResponse | null>(null);
  const [recommendationPending, setRecommendationPending] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [pendingActions, setPendingActions] = useState<ReadonlySet<ActionName>>(() => new Set<ActionName>());
  const pendingActionRef = useRef<Set<ActionName>>(new Set<ActionName>());
  const draftPersistenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const preflightAbortRef = useRef<AbortController | null>(null);
  const [activeStep, setActiveStep] = useState(0);
  const [openGuide, setOpenGuide] = useState(true);
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>("workflow");
  const t = (key: string, values?: Record<string, string | number>) => translate(language, key, values);
  const guidanceCopy = makeFieldGuidanceCopy(t);
  const copy = text[language];
  const isActionPending = (name: ActionName) => pendingActions.has(name);
  const applyCatalog = useCallback((catalog: ResourceCatalogResponse, invalidate: boolean) => {
    setDraft((current) => applyResourceCatalogDefaults(current, catalog));
    if (invalidate && preflight) setPreflightStale(true);
  }, [preflight]);
  const {
    catalog: resourceCatalog,
    loading: resourcesLoading,
    error: resourceError,
    refresh: refreshResources,
  } = useResourceCatalog({ failureMessage: copy.resourceLoadFailed, onCatalog: applyCatalog });
  const {
    jobId,
    jobs,
    jobsCursor,
    jobsLoading,
    jobsError,
    snapshot,
    snapshotLoading,
    snapshotError,
    issueCursor,
    selectJob,
    refreshJobs,
    loadMoreJobs,
    refreshSnapshot,
    firstIssuePage,
    nextIssuePage,
  } = useJobMonitor({
    jobListFailureMessage: t("jobListFailed"),
    jobRequestFailureMessage: t("jobRequestFailed"),
  });

  useEffect(() => setWorkspaceView("workflow"), [jobId]);
  useEffect(() => {
    if (snapshot?.job.jobId === jobId && snapshot.job.status === "reviewing" && snapshot.job.currentModuleId === "count_review") {
      setWorkspaceView("count_review");
    }
  }, [jobId, snapshot?.job.jobId, snapshot?.job.status, snapshot?.job.currentModuleId]);

  const moduleOrder = snapshot?.moduleOrder?.length ? snapshot.moduleOrder : draftModuleOrder;
  const steps = useMemo(() => {
    const guides: Record<PipelineModuleId, readonly string[]> = {
      caption: copy.captionGuide,
      classify: copy.classifyGuide,
      replace: copy.replaceGuide,
      ocr: copy.ocrGuide,
      nl: copy.nlGuide,
      count_review: copy.countReviewGuide,
      dropout: copy.policyGuide,
      token_budget: copy.tokenBudgetGuide,
      export: copy.exportGuide,
    };
    return [
      { id: "setup" as StepId, title: copy.setup, guide: copy.setupGuide },
      ...moduleOrder.map((id) => ({ id: id as StepId, title: moduleLabel(language, id), guide: guides[id] })),
      { id: "precheck" as StepId, title: t("precheckPage"), guide: copy.precheckGuide },
    ];
  }, [copy, language, moduleOrder, t]);

  const updateSection = <K extends DraftSectionKey>(name: K, patch: Partial<Draft[K]>) => {
    invalidatePreflight();
    setDraft((current) => patchDraftSection(current, name, patch));
  };
  const updateModuleBatchSize = (module: keyof Draft["moduleBatchSize"], value: number) => {
    invalidatePreflight();
    setDraft((current) => ({ ...current, moduleBatchSize: { ...current.moduleBatchSize, [module]: value } }));
  };
  const recommendModuleBatches = async () => {
    if (configurationLocked || recommendationPending) return;
    setRecommendationPending(true);
    setActionError(null);
    try {
      const rpm = draft.nl.apiPolicy.maxRequestsPerMinute === "unlimited" ? 100_000 : draft.nl.apiPolicy.maxRequestsPerMinute;
      const result = await getDeviceRecommendations(Number(rpm));
      setDeviceRecommendation(result);
      setDraft((current) => ({ ...current, moduleBatchSize: { ...current.moduleBatchSize, ...result.moduleBatchSize } }));
      invalidatePreflight();
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : t("recommendationUnavailable"));
    } finally {
      setRecommendationPending(false);
    }
  };
  const updateInputTxtMode = (inputTxtMode: Draft["caption"]["inputTxtMode"]) => {
    invalidatePreflight();
    setDraft((current) => patchDraftSection(current, "caption", {
      inputTxtMode,
      enabled: inputTxtMode === "nl" ? true : current.caption.enabled,
      overwriteTxt: false,
    }));
  };
  const selectedTagger = resourceCatalog?.resources.find((item) =>
    item.kind === "tagging-model" && item.resourceId === draft.caption.resourceId,
  );
  // F37: the three frozen threshold modes carry mutually exclusive fields, so the section is rebuilt.
  const updateThresholdMode = (mode: Draft["caption"]["thresholdMode"]) => {
    const defaults = selectedTagger?.defaultThresholds ?? {};
    invalidatePreflight();
    setDraft((current) => {
      const base: Draft["caption"] = {
        enabled: current.caption.enabled,
        thresholdMode: mode,
        overwriteTxt: current.caption.overwriteTxt,
        inputTxtMode: current.caption.inputTxtMode,
        taggerFallbackOnMissingTxt: current.caption.taggerFallbackOnMissingTxt,
        resourceId: current.caption.resourceId,
      };
      return replaceDraftValue(current, "caption",
        mode === "uniform" ? { ...base, uniformThreshold: defaults.general ?? Object.values(defaults)[0] ?? 0.5 }
        : mode === "per_category" ? { ...base, categoryThresholds: { ...defaults } }
        : base);
    });
  };
  const selectTagger = (resourceId: string) => {
    const resource = resourceCatalog?.resources.find((item) => item.resourceId === resourceId);
    if (!resourceSelectable(resource)) return;
    invalidatePreflight();
    setDraft((current) => replaceDraftValue(current, "caption", {
        enabled: current.caption.enabled,
        thresholdMode: "model_default",
        overwriteTxt: current.caption.overwriteTxt,
        inputTxtMode: current.caption.inputTxtMode,
        taggerFallbackOnMissingTxt: current.caption.taggerFallbackOnMissingTxt,
        resourceId,
      }));
  };
  const updateCategoryThreshold = (category: string, value: string) => {
    invalidatePreflight();
    setDraft((current) => patchDraftSection(current, "caption", {
      categoryThresholds: { ...current.caption.categoryThresholds, [category]: Number(value) },
    }));
  };
  const updateTriggerTerms = (value: string) => {
    setTriggerInput(value);
    updateSection("captionFormat", { triggerTerms: value.split(",").map((term) => term.trim()).filter(Boolean) });
  };
  const updateApiPolicy = (patch: Partial<Draft["nl"]["apiPolicy"]>) => {
    if (!runtimeEditable) invalidatePreflight();
    setDraft((current) => patchDraftSection(current, "nl", {
      apiPolicy: { ...current.nl.apiPolicy, ...patch },
    }));
  };
  // The runtime bounds (nl_runner.DEFAULT_POLICY) are enforced here so a cleared field cannot
  // fail the whole task later.
  const clamp = (value: string, low: number, high: number, fallback: number) =>
    Math.min(high, Math.max(low, Math.round(Number(value)) || fallback));
  // An empty budget must remove the key so the backend keeps deriving it from the dataset size.
  const updateAttemptBudget = (value: string) => {
    setAttemptBudget(value);
    if (!runtimeEditable) invalidatePreflight();
    setDraft((current) => {
      const { maxHttpAttempts: _frozen, ...rest } = current.nl.apiPolicy;
      return patchDraftSection(current, "nl", {
        apiPolicy: /^[1-9]\d*$/.test(value) ? { ...rest, maxHttpAttempts: Number(value) } : rest,
      });
    });
  };
  const updateDropoutNested = <K extends "artist" | "quality" | "appearanceNl">(
    section: K,
    patch: Partial<Draft["dropout"][K]>,
  ) => {
    invalidatePreflight();
    setDraft((current) => ({
      ...current,
      dropout: {
        ...current.dropout,
        [section]: { ...current.dropout[section], ...patch },
      },
    } as Draft));
  };
  const updateAppearanceProbability = (group: "solo" | "nonSolo" | "unknown", key: "dropNl" | "dropAppearance", value: string) => {
    invalidatePreflight();
    setDraft((current) => patchDraftSection(current, "dropout", {
      appearanceNl: {
        ...current.dropout.appearanceNl,
        [group]: { ...current.dropout.appearanceNl[group], [key]: Number(value) },
      },
    }));
  };
  const invalidatePreflight = () => {
    preflightGenerationRef.current += 1;
    preflightAbortRef.current?.abort();
    preflightAbortRef.current = null;
    setPreflightStale(Boolean(preflight));
    setPreflightProgress(null);
  };

  useEffect(() => {
    if (draftPersistenceTimerRef.current !== null) window.clearTimeout(draftPersistenceTimerRef.current);
    draftPersistenceTimerRef.current = window.setTimeout(() => {
      draftPersistenceTimerRef.current = null;
      if (!saveDraftState({ draft, ocrExecution })) setPersistenceWarning(translate(language, "draftStorageUnavailable"));
    }, 300);
    return () => {
      if (draftPersistenceTimerRef.current !== null) {
        window.clearTimeout(draftPersistenceTimerRef.current);
        draftPersistenceTimerRef.current = null;
      }
    };
  }, [draft, language, ocrExecution]);

  useEffect(() => {
    const flushDraft = () => { void saveDraftState({ draft, ocrExecution }); };
    window.addEventListener("beforeunload", flushDraft);
    return () => window.removeEventListener("beforeunload", flushDraft);
  }, [draft, ocrExecution]);

  const selectAnnotationPreset = (presetId: string) => {
    if (!presetId) {
      setSelectedPresetId("");
      setPresetName("");
      return;
    }
    const preset = presets.find((item) => item.presetId === presetId);
    if (!preset) return;
    setDraft((current) => applyAnnotationPreset(preset, current));
    setOcrExecution(preset.ocrExecution);
    setTriggerInput(preset.config.captionFormat.triggerTerms.join(", "));
    const maxHttpAttempts = preset.config.nl.apiPolicy.maxHttpAttempts;
    setAttemptBudget(typeof maxHttpAttempts === "number" ? String(maxHttpAttempts) : "");
    setSelectedPresetId(preset.presetId);
    setPresetName(preset.name);
    invalidatePreflight();
    setActionError(null);
  };

  const saveCurrentPreset = () => {
    const name = presetName.trim();
    if (!name) return;
    const existing = presets.find((item) => item.name.trim().toLocaleLowerCase() === name.toLocaleLowerCase());
    if (existing && !window.confirm(t("confirmOverwriteTaskPreset", { name }))) return;
    const nextPreset = createAnnotationPreset(name, draft, ocrExecution, undefined, existing?.presetId);
    if (!nextPreset) return;
    const nextPresets = existing
      ? presets.map((item) => item.presetId === existing.presetId ? nextPreset : item)
      : [...presets, nextPreset];
    if (!saveAnnotationPresets(nextPresets)) {
      setPersistenceWarning(t("presetStorageUnavailable"));
      return;
    }
    setPresets(nextPresets);
    setSelectedPresetId(nextPreset.presetId);
    setPresetName(nextPreset.name);
    setPersistenceWarning(null);
  };

  const deleteCurrentPreset = () => {
    const selected = presets.find((item) => item.presetId === selectedPresetId);
    if (!selected || !window.confirm(t("confirmDeleteTaskPreset", { name: selected.name }))) return;
    const nextPresets = presets.filter((item) => item.presetId !== selected.presetId);
    if (!saveAnnotationPresets(nextPresets)) {
      setPersistenceWarning(t("presetStorageUnavailable"));
      return;
    }
    setPresets(nextPresets);
    setSelectedPresetId("");
    setPresetName("");
    setPersistenceWarning(null);
  };

  const refreshProfiles = async (preferredProfileId = draft.nl.apiProfileId): Promise<NlProfile[] | undefined> => {
    try {
      const response = await listNlProfiles();
      const nextProfiles = response.profiles;
      setProfiles(nextProfiles);
      setDefaultProfileId(response.defaultProfileId);
      const selected = nextProfiles.find((item) => item.profileId === preferredProfileId)
        ?? nextProfiles.find((item) => item.profileId === response.defaultProfileId)
        ?? nextProfiles.find((item) => item.profileId === "default")
        ?? nextProfiles[0];
      if (selected) {
        setProfile(selected);
        setDraft((current) => ({ ...current, nl: { ...current.nl, apiProfileId: selected.profileId } }));
      }
      return nextProfiles;
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : t("profileRequestFailed"));
      return undefined;
    }
  };

  useEffect(() => { void refreshProfiles(); }, []);
  useEffect(() => { saveUiLanguage(language); document.documentElement.lang = language; }, [language]);
  useEffect(() => {
    if (activeStep >= steps.length) setActiveStep(Math.max(0, steps.length - 1));
  }, [activeStep, steps.length]);
  const runAction = async <T,>(name: ActionName, operation: () => Promise<T>, failureMessage = t("controlRequestFailed")): Promise<T | undefined> => {
    if (pendingActionRef.current.has(name)) return undefined;
    pendingActionRef.current.add(name);
    setPendingActions((current) => new Set(current).add(name));
    setActionError(null);
    try {
      return await operation();
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : failureMessage);
      return undefined;
    } finally {
      pendingActionRef.current.delete(name);
      setPendingActions((current) => {
        const next = new Set(current);
        next.delete(name);
        return next;
      });
    }
  };
  const control = <T,>(name: ActionName, operation: () => Promise<T>) => runAction(name, async () => {
    const result = await operation();
    await refreshSnapshot();
    return result;
  });
  const lifecycleControl = <T,>(name: ActionName, operation: () => Promise<T>) => runAction(name, async () => {
    const result = await operation();
    await Promise.all([refreshJobs(), refreshSnapshot(undefined, true)]);
    return result;
  });
  const discardSelectedTask = (targetJobId: string, parentJobId: string | null) => {
    if (!window.confirm(t("confirmDiscard"))) return;
    void runAction("discard", async () => {
      await discardJob(targetJobId);
      if (parentJobId) {
        selectJob(parentJobId);
        await Promise.all([refreshJobs(), refreshSnapshot(parentJobId, true)]);
      } else {
        selectJob("");
        await refreshJobs();
      }
    });
  };
  const deleteCompletedHistoryForTask = (targetJobId: string, parentJobId: string | null) => {
    if (!window.confirm(t("confirmDeleteCompletedHistory"))) return;
    void runAction("delete_history", async () => {
      await deleteCompletedHistory(targetJobId);
      if (parentJobId) {
        selectJob(parentJobId);
        await Promise.all([refreshJobs(), refreshSnapshot(parentJobId, true)]);
      } else {
        selectJob("");
        await refreshJobs();
      }
    });
  };
  const runPreflight = async () => {
    const distribution = draft.nl.lengthDistribution;
    const distributionValues = Object.values(distribution);
    if (!distributionValues.every((value) => Number.isInteger(value) && value >= 0 && value <= 100)
      || distributionValues.reduce((total, value) => total + value, 0) !== 100) {
      setActionError(t("lengthDistributionInvalid"));
      return;
    }
    if (draft.tokenBudget.enabled) {
      const tokenizer = resourceCatalog?.resources.find((item) => item.kind === "tokenizer" && item.resourceId === draft.tokenBudget.resourceId);
      if (!tokenizer?.available || !Number.isInteger(tokenizer.contextLimit)
        || tokenizer.contextLimit! < 1 || !Number.isInteger(draft.tokenBudget.maxTokens)
        || draft.tokenBudget.maxTokens < 1 || draft.tokenBudget.maxTokens > tokenizer.contextLimit!) {
        setActionError(t("tokenizerPreflightRequired"));
        return;
      }
    }
    invalidatePreflight();
    const generation = preflightGenerationRef.current;
    const abortController = new AbortController();
    preflightAbortRef.current?.abort();
    preflightAbortRef.current = abortController;
    const wait = (milliseconds: number) => new Promise<void>((resolve, reject) => {
      const timer = window.setTimeout(resolve, milliseconds);
      abortController.signal.addEventListener("abort", () => {
        window.clearTimeout(timer);
        reject(new DOMException("preflight cancelled", "AbortError"));
      }, { once: true });
    });
    const runAsyncPreflight = async (): Promise<PreflightSummary> => {
      const request = {
        config: draft,
        ...(draft.ocr.enabled ? { ocrExecution } : {}),
      };
      let current = await startPreflight(request);
      if (generation !== preflightGenerationRef.current) return Promise.reject(new DOMException("stale preflight", "AbortError"));
      setPreflightProgress(current);
      while (current.status !== "completed" && current.status !== "failed" && current.status !== "cancelled") {
        await wait(100);
        current = await getPreflightProgress(current.jobId, abortController.signal);
        if (generation !== preflightGenerationRef.current) return Promise.reject(new DOMException("stale preflight", "AbortError"));
        setPreflightProgress(current);
      }
      setPreflightProgress(current);
      if (current.status !== "completed") {
        throw new Error(current.error || (current.status === "cancelled" ? t("preflightCancelled") : t("preflightProgressFailed")));
      }
      if (!current.summary) throw new Error(t("preflightSummaryMissing"));
      return current.summary;
    };
    const result = await runAction("preflight", runAsyncPreflight, t("preflightFailed"));
    if (preflightAbortRef.current === abortController) preflightAbortRef.current = null;
    if (!result) return;
    setPreflight(result);
    setPreflightStale(false);
    selectJob(result.jobId);
    void refreshJobs();
  };
  const cancelRunningPreflight = async () => {
    const current = preflightProgress;
    if (!current || (current.status !== "queued" && current.status !== "preflighting")) return;
    await runAction("preflight_cancel", () => cancelPreflight(current.jobId), t("preflightCancelFailed"));
  };
  const startPipelineFromPrecheck = async () => {
    if (!preflight || preflightStale) return;
    const rebuild = draft.overwriteMode === "rebuild";
    if (!window.confirm(rebuild ? t("confirmRebuild") : t("confirmWorkspacePrompt"))) return;
    const targetJobId = preflight.jobId;
    const result = await runAction("start", async () => {
      await confirmWorkspace(targetJobId, rebuild);
      const started = await startPipeline(targetJobId);
      selectJob(targetJobId);
      await Promise.all([refreshJobs(), refreshSnapshot(targetJobId, true)]);
      return started;
    }, t("startPipelineFailed"));
    if (result) setPreflightStale(false);
  };
  const startRepair = async () => {
    const result = await runAction("repair", () => repairJob(jobId), t("repairStartFailed"));
    if (!result) return;
    selectJob(result.jobId);
    setPreflight(null);
    void refreshJobs();
  };
  const startManualNlRetry = (issue: { issueId: string; sampleId: number }) => {
    if (!window.confirm(t("confirmNlManualRetry"))) return;
    void control("nl_manual_retry", async () => {
      const result = await manualNlRetry(jobId, { issueId: issue.issueId });
      selectJob(result.jobId);
      await refreshJobs();
      return result;
    });
  };
  const startManualNlBatchRetry = (issues: Array<{ issueId: string; sampleId: number }>) => {
    if (!window.confirm(t("confirmNlManualBatchRetry", { count: issues.length }))) return;
    void control("nl_manual_retry_batch", async () => {
      const result = await manualNlRetryBatch(jobId, issues.map((issue) => issue.issueId));
      selectJob(result.jobId);
      await refreshJobs();
      return result;
    });
  };
  const startManualNlWrite = (issue: { issueId: string; sampleId: number }, nl: string) => {
    if (!window.confirm(t("confirmNlManualWrite"))) return;
    void control("nl_manual_write", async () => {
      const result = await manualNlWrite(jobId, { issueId: issue.issueId, nl });
      selectJob(result.jobId);
      await refreshJobs();
      return result;
    });
  };
  const submitProfile = async () => {
    // F25: the prompt the runtime sends is the task's, so the profile only mirrors it.
    const saved = await runAction("profile_save", async () => {
      const result = await saveNlProfile({ ...profile, apiCredentialRef: effectiveCredentialReference(profile), systemPrompt: draft.nl.systemPrompt });
      if (secret) await saveNlSecret(result.apiCredentialRef, secret);
      await refreshProfiles(result.profileId);
      return secret ? { ...result, hasCredential: true } : result;
    }, t("profileSaveFailed"));
    if (!saved) return;
    setProfile(saved);
    setDraft((current) => ({ ...current, nl: { ...current.nl, apiProfileId: saved.profileId } }));
    setDiagnosticResetToken((value) => value + 1);
    if (secret) setSecret("");
  };
  const setProfileDefault = () => {
    if (!profile.profileId) return;
    void runAction("profile_default", async () => {
      await setNlProfileDefault(profile.profileId, profile.revision);
      await refreshProfiles(profile.profileId);
    }, t("profileSaveFailed"));
  };
  const copyProfile = () => {
    const targetId = window.prompt(t("copyProfileId"), `${profile.profileId}-copy`);
    if (!targetId) return;
    void runAction("profile_copy", async () => {
      const copied = await copyNlProfile(profile.profileId, targetId);
      await refreshProfiles(copied.profileId);
    }, t("profileSaveFailed"));
  };
  const removeProfile = () => {
    if (!profile.profileId || !window.confirm(t("confirmDeleteProfile", { name: profile.name ?? profile.profileId }))) return;
    void runAction("profile_delete", async () => {
      await deleteNlProfile(profile.profileId);
      await refreshProfiles();
    }, t("profileDeleteFailed"));
  };

  const rebuild = draft.overwriteMode === "rebuild";
  const classifyEnabled = Boolean(draft.classify.enabled);
  const classificationResourceActive = classifyEnabled || Boolean(draft.caption.enabled);
  const enabledModules = draftModuleOrder
    .filter((id) => id === "export" || (id === "count_review" ? Boolean(draft.countReview.enabled)
      : id === "token_budget" ? draft.tokenBudget.enabled : Boolean(draft[id].enabled)))
    .map((id) => moduleLabel(language, id)).join(" / ");
  const workspaceReady = snapshot?.job.status === "preparing_workspace";
  // F28: the backend accepts the confirmation while paused or interrupted (scheduler.py:335).
  const pendingApiDecisions = snapshot?.nlPendingApiDecisions ?? 0;
  const rawE621Converted = snapshot?.captionDiagnostics.find((item) => item.code === "e621_raw_json_converted")?.count ?? 0;
  const captionGpuFallback = snapshot?.captionDiagnostics.find((item) => item.code === "caption_gpu_fallback")?.count ?? 0;
  const nlAwaitsDecision = snapshot?.job.currentModuleId === "nl" && ["paused", "interrupted"].includes(snapshot.job.status) && pendingApiDecisions > 0;
  const retriableCount = snapshot?.repairPreview?.eligibleTargetCount ?? 0;
  const failureCount = snapshot?.modules.reduce((total, module) => total + module.issue_count, 0) ?? 0;
  const latestEvent = snapshot?.events.length ? snapshot.events[snapshot.events.length - 1] : null;
  const ocrFailureMessage = snapshot?.job.status === "failed" && snapshot.job.currentModuleId === "ocr"
    ? [...snapshot.events].reverse().find((event) => event.module_id === "ocr" && event.status === "failed" && event.message)?.message ?? null
    : null;
  const currentStep = steps[activeStep] ?? steps[steps.length - 1];
  const visibleStepIndex = steps.indexOf(currentStep);
  const currentModuleNumber = currentStep.id === "setup" || currentStep.id === "precheck" ? null : moduleOrder.indexOf(currentStep.id) + 1;
  const workflowRailSteps = steps.map((step, index) => {
    const summary = step.id === "setup" || step.id === "precheck" ? null : snapshot?.modules.find((module) => module.module_id === step.id);
    const complete = step.id === "setup"
      ? true
      : step.id === "precheck"
        ? Boolean(preflight && !preflightStale)
      : Boolean(summary && ["completed", "completed_with_issues", "skipped", "skipped_not_available"].includes(summary.status));
    const label = index === visibleStepIndex ? copy.current : complete ? copy.completed : index < visibleStepIndex ? copy.visited : copy.pending;
    return {
      id: step.id,
      title: step.title,
      label,
      moduleNumber: step.id === "setup" || step.id === "precheck" ? null : moduleOrder.indexOf(step.id) + 1,
      isSetup: step.id === "setup" || step.id === "precheck",
    };
  });
  const stepReady = true;
  const taskLocked = Boolean(workspaceReady || (snapshot && (isActiveJobStatus(snapshot.job.status) || snapshot.job.status === "reviewing")));
  const configurationLocked = taskLocked || isActionPending("preflight");
  const runtimeEditable = Boolean(snapshot && ["preparing_workspace", "running", "paused", "reviewing"].includes(snapshot.job.status));
  const applyRuntimeApi = () => {
    if (!snapshot || !runtimeEditable) return;
    const credentialRef = effectiveCredentialReference(profile);
    void runAction("runtime_api_update", async () => {
      if (secret) await saveNlSecret(credentialRef, secret);
      const result = await updateRunningNlApiConfig(jobId, {
        expectedTaskRevision: snapshot.job.apiConfigRevision ?? 0,
        expectedProfileRevision: profile.revision ?? 1,
        profileId: profile.profileId,
        protocol: profile.protocol ?? "openai_chat_completions",
        providerLabel: profile.providerLabel ?? "gemini-relay",
        endpoint: profile.endpoint,
        model: profile.model,
        backupModel: profile.backupModel,
        credentialRef,
        jsonMode: profile.jsonMode ?? "prompt_only",
        policy: draft.nl.apiPolicy,
      });
      await refreshProfiles(profile.profileId);
      await refreshSnapshot(undefined, true);
      setSecret("");
      return result;
    }, t("runtimeApiUpdateFailed"));
  };
  const orderedModules = snapshot?.moduleOrder.map((id) => snapshot.modules.find((module) => module.module_id === id) ?? {
    module_id: id, status: "pending", completed: 0, failed: 0, skipped: 0, total: 0, issue_count: 0,
  }) ?? [];
  const numberedModuleLabel = (moduleId: string) => {
    const index = snapshot?.moduleOrder.indexOf(moduleId as PipelineModuleId) ?? -1;
    const label = moduleLabel(language, moduleId);
    return index >= 0 ? `${t("moduleNumber", { number: index + 1 })} ${label}` : label;
  };
  const resourcesOf = (kind: ResourceKind) => resourceCatalog?.resources.filter((item) =>
    item.kind === kind,
  ) ?? [];
  const resourceFor = (kind: ResourceKind, resourceId: unknown) => resourceCatalog?.resources.find((item) =>
    item.kind === kind && item.resourceId === resourceId,
  );
  const ocrResource = resourceFor("ocr-model", draft.ocr.resourceId);
  const tokenizerResource = resourceFor("tokenizer", draft.tokenBudget.resourceId);
  const requiredResources = [
    resourceFor("tagging-model", draft.caption.resourceId),
    ...(draft.classify.indexMode === "bundled" ? [resourceFor("classification-index", draft.classify.resourceId)] : []),
    ...(draft.dropout.enabled && draft.dropout.quality.enabled
      ? [resourceFor("dropout-model", draft.dropout.quality.resourceId)]
      : []),
    ...(draft.replace.indexMode === "bundled"
      ? [resourceFor("replacement-index", draft.replace.resourceId)]
      : []),
    ...(draft.tokenBudget.enabled ? [tokenizerResource] : []),
  ];
  const resourceProblems = requiredResources.filter((item) => !(
    item?.available && !["incompatible", "unavailable"].includes(item.compatibility.status)
  ));
  const resourceProblemNames = resourceProblems.map((item) => item?.displayName[language] || item?.officialModelId || item?.resourceId || copy.resourceUnavailable);
  const classificationPathReady = draft.classify.indexMode === "bundled" || Boolean(draft.classify.customResourcePath?.trim());
  const profileResourcesReady = Boolean(resourceCatalog) && resourceProblems.length === 0 && classificationPathReady;
  const resourcePickerCopy = {
    loading: copy.resourceLoading,
    unavailable: copy.resourceUnavailable,
    refresh: copy.refreshResources,
    retry: t("retryResources"),
    details: copy.resourceDetails,
    version: copy.resourceVersion,
    id: copy.resourceId,
    status: copy.resourceStatus,
    distribution: copy.resourceDistribution,
    ready: copy.resourceReady,
    notInstalled: copy.resourceNotInstalled,
    incompatible: copy.resourceIncompatible,
    localOnly: copy.resourceLocalOnly,
    bundled: copy.resourceBundled,
    source: copy.resourceSource,
    license: copy.resourceLicense,
    adjustable: copy.resourceAdjustable,
    excluded: copy.resourceExcluded,
    manualInstall: copy.resourceManualInstall,
    downloadUrl: copy.resourceDownloadUrl,
    installDirectory: copy.resourceInstallDirectory,
    requiredFiles: copy.resourceRequiredFiles,
    installHint: copy.resourceInstallHint,
    defaultSuffix: copy.defaultResource,
    invalid: copy.invalidResources,
  };
  const pathPickerCopy = {
    selectLabel: t("selectPath"),
    selectingLabel: t("selectingPath"),
    busyMessage: t("pathPickerBusy"),
    unavailableMessage: t("pathPickerUnavailable"),
    failedMessage: t("pathPickerFailed"),
  };
  const invalidResourceCount = resourceCatalog?.invalidResources.length ?? 0;
  const taskMonitorLabels = {
    taskOverview: copy.taskOverview,
    taskProgress: t("taskProgress"),
    currentModule: t("currentModule"),
    currentBatch: t("currentBatch"),
    waitingForHumanReview: t("waitingForHumanReview"),
    taskActions: copy.taskActions,
    activeModule: t("activeModule"),
    pauseTask: t("pauseTask"),
    resumeTask: t("resumeTask"),
    terminateTask: t("terminateTask"),
    recoverTask: t("recoverTask"),
    discardTask: t("discardTask"),
    deleteCompletedHistory: t("deleteCompletedHistory"),
    repairDeleteBlocked: t("repairDeleteBlocked"),
    repairTask: t("repairTask"),
    backToParentTask: t("backToParentTask"),
    additionalAttempts: t("additionalAttempts"),
    addBudget: copy.addBudget,
    pendingApiDecisions: t("pendingApiDecisions"),
    confirmUnknown: t("confirmUnknown"),
    issues: t("issues"),
    noTask: copy.noTask,
    loadingTask: t("loadingTask"),
    retryTask: t("retryTask"),
    ocrRuntime: copy.ocrRuntime,
    ocrAvailability: copy.ocrAvailability,
    ocrGpu: copy.ocrGpu,
    ocrRequestedDevice: copy.ocrRequestedDevice,
    ocrObservedDevice: copy.ocrObservedDevice,
    ocrRecommended: copy.ocrRecommended,
    ocrEffective: copy.ocrEffective,
    ocrStartupReason: copy.ocrStartupReason,
    apiSettings: t("apiSettings"), apiRevision: t("apiRevision"), apiProvider: t("apiProvider"), apiModel: t("apiModel"), apiJsonMode: t("apiJsonMode"), apiPending: t("apiRevision"),
  };
  const issuePanelLabels = {
    issues: t("issues"),
    shownRetriable: (shown: number, retriable: number) => t("shownRetriable", { shown, retriable }),
    attempt: (attempt: number) => t("attempt", { attempt }),
    retryFrom: (module: string) => t("retryFrom", { module }),
    notRetriable: t("notRetriable"),
    firstPage: t("firstPage"),
    nextPage: t("nextPage"),
    restoreOriginal: t("restoreOriginal"),
    reprocess: t("reprocess"),
    nlRetry: t("nlManualRetry"),
    nlBatchRetry: t("nlManualBatchRetry"),
    selectNlPage: t("selectNlPage"),
    selectedNlCount: (selected: number) => t("selectedNlCount", { selected }),
    nlWrite: t("nlManualWrite"),
    nlWritePlaceholder: t("nlManualWritePlaceholder"),
    workbenchTitle: t("failureWorkbench"),
    workbenchHelp: t("failureWorkbenchHelp"),
    moduleFilter: t("moduleFilter"),
    filterAll: t("filterAll"),
    noIssues: t("noFailures"),
    samplePath: t("samplePath"),
    retriable: t("retriableIssues"),
  };
  const taskMonitorModules = orderedModules.map((module) => ({
    moduleId: module.module_id,
    label: numberedModuleLabel(module.module_id),
    status: module.status,
    statusLabel: statusLabel(language, module.status),
    completed: module.completed,
    failed: module.failed,
    skipped: module.skipped,
    total: module.total,
    issueCount: module.issue_count,
    isCurrent: module.module_id === snapshot?.job.currentModuleId,
  }));

  const batchRecommendation = <section className="batch-recommendation top-batch-recommendation" aria-busy={recommendationPending}>
      <div className="batch-recommendation-heading">
        <strong>{t("recommendFromDevice")}</strong>
        <button className="secondary" type="button" disabled={configurationLocked || recommendationPending} onClick={() => void recommendModuleBatches()}>{recommendationPending ? t("recommendationLoading") : t("recommendFromDevice")}</button>
      </div>
      {deviceRecommendation && <>
        <p>{t("deviceRecommendationSummary", {
          cpu: deviceRecommendation.cpuPhysicalCores,
          gpu: deviceRecommendation.gpu.available ? (deviceRecommendation.gpu.name ?? "available") : "CPU-only",
          free: deviceRecommendation.gpu.freeVramBytes == null ? "-" : `${(deviceRecommendation.gpu.freeVramBytes / 1024 ** 3).toFixed(1)} GB`,
          total: deviceRecommendation.gpu.totalVramBytes == null ? "-" : `${(deviceRecommendation.gpu.totalVramBytes / 1024 ** 3).toFixed(1)} GB`,
          baseline: deviceRecommendation.baselineVersion,
        })}</p>
        <details><summary>{t("batchSize")}</summary><ul>{Object.entries(deviceRecommendation.moduleBatchSize).map(([module, value]) => <li key={module}><span>{moduleLabel(language, module === "countReview" ? "count_review" : module)}</span><strong>{value}</strong><small>{deviceRecommendation.reasons[module] ?? ""}</small></li>)}</ul></details>
        {deviceRecommendation.probeErrors.length > 0 && <small className="resource-warning">{deviceRecommendation.probeErrors.join(", ")}</small>}
      </>}
    </section>;

  const guide = <>
    <details className="module-guide" open={openGuide} onToggle={(event) => setOpenGuide((event.target as HTMLDetailsElement).open)}>
    <summary>{copy.details}</summary>
    <p>{currentStep.guide[0]}</p>
    <dl>
      <div><dt>{copy.input}</dt><dd>{currentStep.guide[1]}</dd></div>
      <div><dt>{copy.output}</dt><dd>{currentStep.guide[2]}</dd></div>
      <div><dt>{copy.effect}</dt><dd>{currentStep.guide[3]}</dd></div>
    </dl>
    </details>
  </>;

  const countReviewContent = <>
    <div className="option-stack count-review-config">
      <ToggleField id="count-review-enabled" label={t("enableCountReview")} checked={draft.countReview.enabled} disabled={configurationLocked} onChange={(enabled) => updateSection("countReview", { enabled })} copy={guidanceCopy} guidance={{ description: t("fieldHelp_countReviewEnabled"), defaultValue: draftDefaults.countReview.enabled ? t("fieldEnabled") : t("fieldDisabled") }} />
      <ModuleBatchField id="count-review-batch-size" label={t("batchSize")} value={draft.moduleBatchSize.countReview} defaultValue={draftDefaults.moduleBatchSize.countReview} recommended={deviceRecommendation?.moduleBatchSize.countReview} recommendationReason={deviceRecommendation?.reasons.countReview} minimum={1} maximum={500} disabled={configurationLocked || !draft.countReview.enabled} t={t} guidanceCopy={guidanceCopy} onChange={(value) => updateModuleBatchSize("countReview", value)} />
      <p className="hint">{t("countReviewConfigHelp")}</p>
    </div>
    {snapshot?.moduleOrder.includes("count_review")
      ? <div className="count-review-launch"><p>{t("countReviewWorkbenchHelp")}</p><button type="button" onClick={() => setWorkspaceView("count_review")}>{t("openCountReviewWorkbench")}</button></div>
      : <p className="review-empty">{t("countReviewUnavailableUntilTask")}</p>}
  </>;
  const tokenBudgetContent = <>
    <TokenBudgetStep
      draft={draft}
      defaults={draftDefaults}
      taskLocked={configurationLocked}
      language={language}
      resources={resourcesOf("tokenizer")}
      resourcesLoading={resourcesLoading}
      resourceError={resourceError}
      invalidResourceCount={invalidResourceCount}
      resourcePickerCopy={resourcePickerCopy}
      t={t}
      guidanceCopy={guidanceCopy}
      onTokenBudgetChange={(patch) => updateSection("tokenBudget", patch)}
      onRefreshResources={() => void refreshResources()}
      batchSize={draft.moduleBatchSize.tokenBudget}
      batchRecommended={deviceRecommendation?.moduleBatchSize.tokenBudget}
      batchReason={deviceRecommendation?.reasons.tokenBudget}
      onBatchSizeChange={(value) => updateModuleBatchSize("tokenBudget", value)}
    />
    {snapshot?.moduleOrder.includes("token_budget") && <TokenBudgetReviewPanel
      jobId={jobId}
      jobStatus={snapshot.job.status}
      currentModuleId={snapshot.job.currentModuleId}
      language={language}
      onApplied={refreshSnapshot}
    />}
  </>;
  const stepContent = currentStep.id === "setup" ? <SetupStep
    draft={draft}
    defaults={draftDefaults}
    taskLocked={configurationLocked}
    profileResourcesReady={profileResourcesReady}
    presets={presets}
    selectedPresetId={selectedPresetId}
    presetName={presetName}
    persistenceWarning={persistenceWarning}
    t={t}
    guidanceCopy={guidanceCopy}
    pathPickerCopy={pathPickerCopy}
    copy={{
      preflightHint: copy.preflightHint,
      profileReady: copy.profileReady,
      profileUnavailable: copy.profileUnavailable,
    }}
    onSourceRootChange={(value) => { setDraft((current) => replaceDraftValue(current, "sourceRoot", value)); invalidatePreflight(); }}
    onWorkModeChange={(workMode) => { setDraft((current) => ({ ...current, workMode, outputRoot: workMode === "in_place" ? undefined : current.outputRoot })); invalidatePreflight(); }}
    onOutputRootChange={(value) => { setDraft((current) => replaceDraftValue(current, "outputRoot", value)); invalidatePreflight(); }}
    onOverwriteModeChange={(overwriteMode) => { setDraft((current) => replaceDraftValue(current, "overwriteMode", overwriteMode)); invalidatePreflight(); }}
    onInvalidImageActionChange={(invalidImageAction) => updateSection("imageDecode", { invalidImageAction })}
    onRecursiveChange={(recursive) => { setDraft((current) => replaceDraftValue(current, "recursive", recursive)); invalidatePreflight(); }}
    onPresetSelect={selectAnnotationPreset}
    onPresetNameChange={setPresetName}
    onSavePreset={saveCurrentPreset}
    onDeletePreset={deleteCurrentPreset}
  /> : currentStep.id === "precheck" ? <PrecheckStep
    draft={draft}
    preflightPending={isActionPending("preflight")}
    preflightCancelPending={isActionPending("preflight_cancel")}
    startPending={isActionPending("start")}
    preflightProgress={preflightProgress}
    preflight={preflight}
    preflightStale={preflightStale}
    actionError={actionError}
    profileResourcesReady={profileResourcesReady}
    resourcesLoading={resourcesLoading}
    canStart={Boolean(preflight && !preflightStale && preflight.jobId && !(draft.imageDecode.invalidImageAction === "block" && preflight.imageIssueCount > 0))}
    enabledModules={enabledModules}
    t={t}
    onRunPreflight={() => void runPreflight()}
    onCancelPreflight={() => void cancelRunningPreflight()}
    onStartPipeline={() => void startPipelineFromPrecheck()}
  /> : currentStep.id === "caption" ? <CaptionStep
    draft={draft}
    defaults={draftDefaults}
    taskLocked={configurationLocked}
    rebuild={rebuild}
    language={language}
    resources={resourcesOf("tagging-model")}
    resourcesLoading={resourcesLoading}
    resourceError={resourceError}
    invalidResourceCount={invalidResourceCount}
    resourcePickerCopy={resourcePickerCopy}
    selectedTagger={selectedTagger}
    triggerInput={triggerInput}
    t={t}
    guidanceCopy={guidanceCopy}
    copy={{ captionModel: copy.captionModel, captionModelHelp: copy.captionModelHelp }}
    onCaptionChange={(patch) => updateSection("caption", patch)}
    onInputTxtModeChange={updateInputTxtMode}
    onCaptionFormatChange={(patch) => updateSection("captionFormat", patch)}
    onSelectTagger={selectTagger}
    onRefreshResources={() => void refreshResources()}
    onThresholdModeChange={updateThresholdMode}
    onCategoryThresholdChange={updateCategoryThreshold}
    onTriggerInputChange={updateTriggerTerms}
    batchSize={draft.moduleBatchSize.caption}
    batchRecommended={deviceRecommendation?.moduleBatchSize.caption}
    batchReason={deviceRecommendation?.reasons.caption}
    onBatchSizeChange={(value) => updateModuleBatchSize("caption", value)}
  /> : currentStep.id === "classify" ? <ClassifyStep
    draft={draft}
    taskLocked={configurationLocked}
    rebuild={rebuild}
    classifyEnabled={classifyEnabled}
    classificationResourceActive={classificationResourceActive}
    language={language}
    resources={resourcesOf("classification-index")}
    resourcesLoading={resourcesLoading}
    resourceError={resourceError}
    invalidResourceCount={invalidResourceCount}
    resourcePickerCopy={resourcePickerCopy}
    pathPickerCopy={pathPickerCopy}
    t={t}
    guidanceCopy={guidanceCopy}
    copy={{ classificationIndex: copy.classificationIndex, classificationIndexHelp: copy.classificationIndexHelp, bundledResource: copy.bundledResource, customResource: copy.customResource, classificationResourceJson: copy.classificationResourceJson, anthroReplacementNote: copy.anthroReplacementNote }}
    onClassifyChange={(patch) => updateSection("classify", patch)}
    onRefreshResources={() => void refreshResources()}
    batchSize={draft.moduleBatchSize.classify}
    batchRecommended={deviceRecommendation?.moduleBatchSize.classify}
    batchReason={deviceRecommendation?.reasons.classify}
    onBatchSizeChange={(value) => updateModuleBatchSize("classify", value)}
  /> : currentStep.id === "replace" ? <ReplaceStep
    draft={draft}
    taskLocked={configurationLocked}
    language={language}
    resources={resourcesOf("replacement-index")}
    resourcesLoading={resourcesLoading}
    resourceError={resourceError}
    invalidResourceCount={invalidResourceCount}
    resourcePickerCopy={resourcePickerCopy}
    pathPickerCopy={pathPickerCopy}
    replaceIndex={preflight?.replaceIndex ?? null}
    t={t}
    guidanceCopy={guidanceCopy}
    copy={{
      replaceMode: copy.replaceMode,
      bundledIndex: copy.bundledIndex,
      customIndex: copy.customIndex,
      customIndexPath: copy.customIndexPath,
      customIndexHelp: copy.customIndexHelp,
      replacementIndex: copy.replacementIndex,
      replacementIndexHelp: copy.replacementIndexHelp,
      indexRules: copy.indexRules,
    }}
    onReplaceChange={(patch) => updateSection("replace", patch)}
     onIndexModeChange={(indexMode) => {
       invalidatePreflight();
      setDraft((current) => replaceDraftValue(current, "replace", indexMode === "bundled"
        ? { enabled: current.replace.enabled, indexMode, resourceId: resourceCatalog?.defaults.replacementIndex ?? "replace-e621-20260726-v2" }
        : { enabled: current.replace.enabled, indexMode, customIndexPath: current.replace.customIndexPath ?? "" }));
    }}
    onRefreshResources={() => void refreshResources()}
    batchSize={draft.moduleBatchSize.replace}
    batchRecommended={deviceRecommendation?.moduleBatchSize.replace}
    batchReason={deviceRecommendation?.reasons.replace}
    onBatchSizeChange={(value) => updateModuleBatchSize("replace", value)}
  /> : currentStep.id === "ocr" ? <OcrStep
    draft={draft}
    defaults={draftDefaults}
     taskLocked={configurationLocked}
    ocrExecution={ocrExecution}
    runtime={snapshot?.ocrRuntime ?? null}
    resource={ocrResource}
    resourcesLoading={resourcesLoading}
    resourceError={resourceError}
    diagnostics={snapshot?.ocrDiagnostics ?? []}
    failureMessage={ocrFailureMessage}
    guidanceCopy={guidanceCopy}
    t={t}
    copy={copy}
    onOcrChange={(patch) => updateSection("ocr", patch)}
     onOcrExecutionChange={(next) => { invalidatePreflight(); setOcrExecution(next); }}
    onRefreshResources={() => void refreshResources()}
    batchSize={draft.moduleBatchSize.ocr}
    batchRecommended={deviceRecommendation?.moduleBatchSize.ocr}
    batchReason={deviceRecommendation?.reasons.ocr}
    onBatchSizeChange={(value) => updateModuleBatchSize("ocr", value)}
  /> : currentStep.id === "nl" ? <NlStep
    draft={draft}
    defaults={draftDefaults}
    taskLocked={configurationLocked}
    attemptBudget={attemptBudget}
    profile={profile}
    profiles={profiles}
    secret={secret}
    profileSavePending={isActionPending("profile_save")}
    credentialDeletePending={isActionPending("credential_delete")}
    runtimeEditable={runtimeEditable}
    runtimeTaskRevision={snapshot?.job.apiConfigRevision ?? 0}
    runtimePending={Boolean(snapshot?.job.apiRuntimePending)}
    diagnosticResetToken={diagnosticResetToken}
    t={t}
    guidanceCopy={guidanceCopy}
    profileHelp={copy.profileHelp}
    onNlChange={(patch) => updateSection("nl", patch)}
    onApiPolicyChange={updateApiPolicy}
    onRequestsPerMinuteChange={(value) => updateApiPolicy({ maxRequestsPerMinute: clamp(value, 1, 100_000, 60) })}
    onUnlimitedRpmChange={(enabled) => {
      if (!enabled) { updateApiPolicy({ maxRequestsPerMinute: 60 }); return; }
      if (window.confirm(t("confirmUnlimitedRpm"))) updateApiPolicy({ maxRequestsPerMinute: "unlimited" });
    }}
    onAttemptBudgetChange={updateAttemptBudget}
    onProfileSelect={(profileId) => {
      const selected = profiles.find((item) => item.profileId === profileId) ?? { ...emptyProfile, profileId };
      setProfile(selected);
      setSecret("");
      setDiagnosticResetToken((value) => value + 1);
      if (!runtimeEditable) updateSection("nl", { apiProfileId: selected.profileId });
    }}
    onProfileChange={(nextProfile) => setProfile((current) => ({ ...current, ...nextProfile }))}
    onSecretChange={setSecret}
    onSaveProfile={() => void submitProfile()}
    onApplyRuntime={applyRuntimeApi}
    defaultProfileId={defaultProfileId}
    onSetDefault={setProfileDefault}
    onCopyProfile={copyProfile}
    onDeleteProfile={removeProfile}
    onClearSecret={() => void control("credential_delete", async () => {
      const reference = effectiveCredentialReference(profile);
      await deleteNlSecret(reference);
      await refreshProfiles(profile.profileId);
      setProfile((current) => ({ ...current, apiCredentialRef: reference, hasCredential: false }));
      setSecret("");
      setDiagnosticResetToken((value) => value + 1);
    })}
    batchSize={draft.moduleBatchSize.nl}
    batchRecommended={deviceRecommendation?.moduleBatchSize.nl}
    batchReason={deviceRecommendation?.reasons.nl}
    onBatchSizeChange={(value) => updateModuleBatchSize("nl", value)}
  /> : currentStep.id === "count_review" ? countReviewContent : currentStep.id === "dropout" ? <PolicyStep
    draft={draft}
    defaults={draftDefaults}
    taskLocked={configurationLocked}
    language={language}
    resources={resourcesOf("dropout-model")}
    resourcesLoading={resourcesLoading}
    resourceError={resourceError}
    invalidResourceCount={invalidResourceCount}
    resourcePickerCopy={resourcePickerCopy}
    t={t}
    guidanceCopy={guidanceCopy}
    copy={{ qualityModel: copy.qualityModel, qualityModelHelp: copy.qualityModelHelp }}
    capabilities={resourceCatalog?.capabilities}
    onDropoutChange={(patch) => updateSection("dropout", patch)}
    onArtistChange={(patch) => updateDropoutNested("artist", patch)}
    onQualityChange={(patch) => updateDropoutNested("quality", patch)}
    onAppearanceNlChange={(patch) => updateDropoutNested("appearanceNl", patch)}
    onAppearanceProbabilityChange={updateAppearanceProbability}
    onRefreshResources={() => void refreshResources()}
    batchSize={draft.moduleBatchSize.dropout}
    batchRecommended={deviceRecommendation?.moduleBatchSize.dropout}
    batchReason={deviceRecommendation?.reasons.dropout}
    onBatchSizeChange={(value) => updateModuleBatchSize("dropout", value)}
  /> : currentStep.id === "token_budget" ? tokenBudgetContent : <ExportStep
    draft={draft}
    defaults={draftDefaults}
    taskLocked={configurationLocked}
    exportSummary={snapshot?.exportSummary ?? null}
    t={t}
    guidanceCopy={guidanceCopy}
    onExportChange={(patch) => updateSection("export", patch)}
    onCaptionFormatChange={(patch) => updateSection("captionFormat", patch)}
    batchSize={draft.moduleBatchSize.export}
    batchRecommended={deviceRecommendation?.moduleBatchSize.export}
    batchReason={deviceRecommendation?.reasons.export}
    onBatchSizeChange={(value) => updateModuleBatchSize("export", value)}
  />;

  const repairTasksPanel = snapshot && !snapshot.job.parentJobId && snapshot.repairChildren.length > 0 ? <RepairTasksPanel
    tasks={snapshot.repairChildren}
    pending={isActionPending("discard") || isActionPending("delete_history")}
    labels={{
      title: t("repairTasks"),
      open: t("openRepairTask"),
      delete: t("discardTask"),
      deleteCompletedHistory: t("deleteCompletedHistory"),
      targets: (sampleCount, targetCount) => t("repairTargets", { sampleCount, targetCount }),
      createdAt: (createdAt) => t("repairCreatedAt", { createdAt }),
    }}
    onOpen={selectJob}
    onDelete={(childJobId, status) => status === "succeeded"
      ? deleteCompletedHistoryForTask(childJobId, jobId)
      : discardSelectedTask(childJobId, jobId)}
  /> : null;
  const issuePanel = snapshot ? <IssuePanel
    issues={snapshot.issues.map((item) => ({
      issueId: item.issue_id,
      sampleId: item.sample_id,
      relativeImagePath: item.relative_image_path,
      moduleId: item.module_id,
      code: item.code,
      message: item.message,
      retriable: Boolean(item.retriable),
      attempt: item.attempt,
      repairStartModule: item.repair_start_module ?? null,
    }))}
    retriableCount={retriableCount}
    cursor={issueCursor}
    nextCursor={{ sampleId: snapshot.nextIssueAfterSampleId, issueId: snapshot.nextIssueAfterIssueId }}
    labels={issuePanelLabels}
    canRestore={snapshot.job.status === "succeeded"}
    canReprocess={retriableCount > 0 && ["reviewing", "failed"].includes(snapshot.job.status)}
    canManualNl={Boolean(snapshot.job.status === "reviewing" || snapshot.job.status === "failed" || snapshot.job.status === "succeeded")}
    pendingActions={pendingActions}
    onFirstPage={firstIssuePage}
    onNextPage={nextIssuePage}
    onRestore={() => { if (window.confirm(t("confirmRestore"))) void control("restore", () => restoreOriginalAnnotations(jobId)); }}
    onReprocess={() => void startRepair()}
    onManualNlRetry={startManualNlRetry}
    onManualNlBatchRetry={startManualNlBatchRetry}
    onManualNlWrite={startManualNlWrite}
  /> : <p className="review-empty">{copy.noTask}</p>;

  return <main className="app-shell">
    <header className="topbar"><div><p className="eyebrow">ANIMA</p><h1>Anima Dataset Tool</h1><p>{t("pipelineSubtitle")}</p></div><div className="topbar-controls"><div className="recent-task-control"><label>{t("recentTasks")}<select disabled={jobsLoading} value={jobs.some((item) => item.jobId === jobId) ? jobId : ""} onChange={(event) => { if (event.target.value === "__more__") { void loadMoreJobs(); return; } selectJob(event.target.value); }}><option value="">{jobs.length ? "-" : t("noRecentTasks")}</option>{jobs.map((item) => <option key={item.jobId} value={item.jobId}>{item.createdAt} {statusLabel(language, item.status)} {item.jobId.slice(0, 8)}</option>)}{jobsCursor && <option value="__more__">{t("loadMore")}</option>}</select></label><div className="recent-task-state" aria-live="polite" aria-atomic="true">{jobsLoading && <small role="status">{t("loadingTasks")}</small>}{jobsError && <><small role="alert">{jobsError}</small><button className="secondary" type="button" onClick={() => void refreshJobs()}>{t("retryTasks")}</button></>}</div></div><div className="language-switch" role="group" aria-label={t("language")}><button type="button" className={language === "zh-CN" ? "selected" : ""} onClick={() => setLanguage("zh-CN")}>中文</button><button type="button" className={language === "en" ? "selected" : ""} onClick={() => setLanguage("en")}>EN</button></div></div></header>
    {batchRecommendation}
    <nav className="workspace-tabs" aria-label={t("pipelineSubtitle")}>
      <button type="button" className={workspaceView === "workflow" ? "active" : ""} aria-current={workspaceView === "workflow" ? "page" : undefined} onClick={() => setWorkspaceView("workflow")}>{t("workspaceWorkflow")}</button>
      <button type="button" className={workspaceView === "count_review" ? "active" : ""} aria-current={workspaceView === "count_review" ? "page" : undefined} disabled={!snapshot?.moduleOrder.includes("count_review")} onClick={() => setWorkspaceView("count_review")}>{t("workspaceCountReview")}{snapshot?.job.status === "reviewing" && snapshot.job.currentModuleId === "count_review" && <span className="workspace-badge attention">!</span>}</button>
      <button type="button" className={workspaceView === "failures" ? "active" : ""} aria-current={workspaceView === "failures" ? "page" : undefined} onClick={() => setWorkspaceView("failures")}>{t("workspaceFailures")}{failureCount > 0 && <span className="workspace-badge">{failureCount}</span>}</button>
    </nav>
    {workspaceView === "workflow" && <div className="workflow-layout">
      <WorkflowRail flow={copy.flow} visibleStepIndex={visibleStepIndex} steps={workflowRailSteps} onSelect={(index) => { setActiveStep(index); setOpenGuide(true); if (steps[index]?.id === "count_review" && snapshot?.moduleOrder.includes("count_review")) setWorkspaceView("count_review"); }} />
      <section className="step-panel"><div className="step-heading"><div><p className="eyebrow">{copy.flow} / {currentModuleNumber === null ? copy.setup : t("moduleNumber", { number: currentModuleNumber })}</p><h2>{currentStep.title}</h2></div><span className="status">{currentStep.id === "setup" ? t("newTask") : copy.current}</span></div>{actionError && currentStep.id !== "precheck" && <div className="action-feedback" aria-live="assertive"><p role="alert">{actionError}</p></div>}{guide}<div className="step-content">{stepContent}</div><div className="wizard-controls"><button className="secondary" type="button" disabled={visibleStepIndex === 0} onClick={() => { setActiveStep(visibleStepIndex - 1); setOpenGuide(true); }}>{copy.back}</button><button type="button" disabled={visibleStepIndex === steps.length - 1 || !stepReady} onClick={() => { setActiveStep(visibleStepIndex + 1); setOpenGuide(true); }}>{copy.next}</button></div></section>
      <TaskMonitor
        snapshot={snapshot ? {
          status: snapshot.job.status,
          currentModuleId: snapshot.job.currentModuleId ?? null,
          apiRuntime: snapshot.job.apiRuntime ? { revision: snapshot.job.apiRuntime.revision, providerLabel: snapshot.job.apiRuntime.providerLabel, endpoint: snapshot.job.apiRuntime.endpoint, model: snapshot.job.apiRuntime.model, jsonMode: snapshot.job.apiRuntime.jsonMode } : null,
          apiRuntimePending: snapshot.job.apiRuntimePending ? { revision: snapshot.job.apiRuntimePending.revision, providerLabel: snapshot.job.apiRuntimePending.providerLabel, endpoint: snapshot.job.apiRuntimePending.endpoint, model: snapshot.job.apiRuntimePending.model, jsonMode: snapshot.job.apiRuntimePending.jsonMode } : null,
          ocrRuntime: snapshot.ocrRuntime,
        } : null}
        isRepairTask={Boolean(snapshot?.job.parentJobId)}
        hasRepairChildren={Boolean(snapshot?.repairChildren.length)}
        loading={snapshotLoading}
        error={snapshotError}
        statusLabel={statusLabel(language, snapshot?.job.status ?? "idle")}
        currentModuleLabel={snapshot?.job.currentModuleId ? numberedModuleLabel(snapshot.job.currentModuleId) : "-"}
        currentBatchLabel={latestEvent ? `${latestEvent.completed} / ${latestEvent.total}` : "-"}
        rawE621ConvertedMessage={rawE621Converted > 0 ? t("rawE621Converted", { count: rawE621Converted }) : null}
        captionGpuFallbackMessage={captionGpuFallback > 0 ? t("captionGpuFallback") : null}
        modules={taskMonitorModules}
        labels={taskMonitorLabels}
        canDiscard={Boolean(snapshot && snapshot.repairChildren.length === 0 && ["ready", "interrupted", "reviewing", "cancelled_recoverable", "failed"].includes(snapshot.job.status))}
        canDeleteHistory={Boolean(snapshot && snapshot.job.status === "succeeded" && snapshot.repairChildren.length === 0)}
        budget={budget}
        pendingApiDecisions={pendingApiDecisions}
        nlAwaitsDecision={nlAwaitsDecision}
        pendingActions={pendingActions}
        onPause={() => void lifecycleControl("pause", () => pauseJob(jobId))}
        onResume={() => void lifecycleControl("resume", () => resumeJob(jobId))}
        onTerminate={() => { if (window.confirm(t("confirmTerminate"))) void lifecycleControl("terminate", () => cancelJob(jobId)); }}
        onRecover={() => { if (window.confirm(t("confirmRecover"))) void lifecycleControl("recover", () => recoverJob(jobId)); }}
        onDiscard={() => discardSelectedTask(jobId, snapshot?.job.parentJobId ?? null)}
        onDeleteHistory={() => deleteCompletedHistoryForTask(jobId, snapshot?.job.parentJobId ?? null)}
        onBackToParent={() => { if (snapshot?.job.parentJobId) selectJob(snapshot.job.parentJobId); }}
        onBudgetChange={setBudget}
        onAddBudget={() => void control("nl_budget", () => addNlBudget(jobId, Number(budget)))}
        onConfirmUnknown={() => { if (window.confirm(t("confirmUnknownPrompt", { count: pendingApiDecisions }))) void control("nl_confirm_outcomes", () => confirmNlOutcomes(jobId)); }}
        onRetry={() => void refreshSnapshot()}
      />
    </div>}
    {workspaceView === "count_review" && <div className="workspace-page count-review-workspace">{actionError && <div className="action-feedback"><p role="alert">{actionError}</p></div>}{snapshot?.moduleOrder.includes("count_review") ? <CountReviewPanel jobId={jobId} jobStatus={snapshot.job.status} currentModuleId={snapshot.job.currentModuleId} language={language} onConfirmed={() => refreshSnapshot(undefined, true)} /> : <p className="review-empty">{t("countReviewUnavailableUntilTask")}</p>}</div>}
    {workspaceView === "failures" && <div className="workspace-page failure-workspace">{actionError && <div className="action-feedback"><p role="alert">{actionError}</p></div>}{repairTasksPanel}{issuePanel}</div>}
  </main>;
}
