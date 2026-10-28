import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  createNlPromptPreset,
  deleteNlPromptPreset,
  discoverNlModels,
  getNlPromptPreset,
  listNlPromptPresets,
  resetNlPromptPreset,
  testNlMessage,
  updateNlPromptPreset,
  type ModelDiscoveryResult,
  type NlPresetType,
  type NlPromptPresetDetail,
  type NlPromptPresetSummary,
  type TestMessageResult,
} from "../api";
import { FormField, type FieldGuidanceCopy } from "./FormField";

type Translate = (key: string, values?: Record<string, string | number>) => string;

const BUILTIN_PRESET_IDS = new Set([
  "builtin:nl-preset-v1-general",
  "builtin:nl-preset-v1-style",
  "builtin:nl-preset-v1-character",
]);

type Feedback = { kind: "discovery"; result: ModelDiscoveryResult } | { kind: "test"; result: TestMessageResult };

export type NlPromptPresetLibraryProps = {
  disabled: boolean;
  includeMoss?: boolean;
  promptText: string;
  presetType: NlPresetType;
  t: Translate;
  guidanceCopy: FieldGuidanceCopy;
  onPromptChange: (promptText: string) => void;
  onPresetTypeChange: (presetType: NlPresetType) => void;
  onDirtyChange?: (dirty: boolean) => void;
};

function promptTextOf(detail: NlPromptPresetDetail): string {
  return detail.promptText ?? detail.basePrompt ?? "";
}

export function NlPromptPresetLibrary({
  disabled, includeMoss = false, promptText, presetType, t, guidanceCopy, onPromptChange, onPresetTypeChange, onDirtyChange,
}: NlPromptPresetLibraryProps) {
  const [presets, setPresets] = useState<NlPromptPresetSummary[]>([]);
  const [details, setDetails] = useState<Record<string, NlPromptPresetDetail>>({});
  const [selectedId, setSelectedId] = useState("");
  const [selected, setSelected] = useState<NlPromptPresetDetail | null>(null);
  const [name, setName] = useState("");
  const [type, setType] = useState<NlPresetType | "">("");
  const [prompt, setPrompt] = useState("");
  const [savedName, setSavedName] = useState("");
  const [savedType, setSavedType] = useState<NlPresetType | "">("");
  const [savedPrompt, setSavedPrompt] = useState("");
  const [creating, setCreating] = useState(false);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const dirty = creating
    ? Boolean(name.trim() || type || prompt.trim())
    : name !== savedName || type !== savedType || prompt !== savedPrompt;

  useEffect(() => {
    onDirtyChange?.(dirty);
  }, [dirty, onDirtyChange]);

  const applyDetail = (detail: NlPromptPresetDetail) => {
    const text = promptTextOf(detail);
    setSelected(detail);
    setSelectedId(detail.presetId);
    setName(detail.name);
    setType(detail.type);
    setPrompt(text);
    setSavedName(detail.name);
    setSavedType(detail.type);
    setSavedPrompt(text);
    setCreating(false);
    onPromptChange(text);
    onPresetTypeChange(detail.type);
  };

  const loadAll = async (preferredId?: string) => {
    const summaries = await listNlPromptPresets(includeMoss);
    const visibleSummaries = includeMoss ? summaries.presets.filter((item) => item.type.startsWith("moss-")) : summaries.presets.filter((item) => !item.type.startsWith("moss-"));
    const details = await Promise.all(visibleSummaries.map((item) => getNlPromptPreset(item.presetId)));
    setPresets(visibleSummaries);
    setDetails(Object.fromEntries(details.map((item) => [item.presetId, item])));
    const preferred = preferredId && details.find((item) => item.presetId === preferredId);
    const matching = details.find((item) => promptTextOf(item) === promptText && item.type === presetType);
    const general = details.find((item) => item.type === (includeMoss ? "moss-general" : "general")) ?? details[0];
    const next = preferred ?? matching ?? general;
    if (next) applyDetail(next);
  };

  useEffect(() => {
    let active = true;
    void loadAll().catch((cause: unknown) => {
      if (active) setError(cause instanceof Error ? cause.message : t("presetLoadFailed"));
    }).finally(() => {
      if (active) setLoading(false);
    });
    return () => { active = false; };
  }, [includeMoss]);

  const selectPreset = async (presetId: string) => {
    if (pending || presetId === selectedId) return;
    if (dirty && !window.confirm(t("confirmDiscardPreset"))) return;
    setPending(true);
    setError(null);
    try {
      const detail = await getNlPromptPreset(presetId);
      setDetails((current) => ({ ...current, [detail.presetId]: detail }));
      applyDetail(detail);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : t("presetLoadFailed"));
    } finally {
      setPending(false);
    }
  };

  const startCreate = () => {
    if (dirty && !window.confirm(t("confirmDiscardPreset"))) return;
    setCreating(true);
    setSelected(null);
    setSelectedId("");
    setName("");
    setType("");
    setPrompt("");
    setSavedName("");
    setSavedType("");
    setSavedPrompt("");
    onDirtyChange?.(true);
  };

  const cancelCreate = () => {
    if (dirty && !window.confirm(t("confirmDiscardPreset"))) return;
    const general = presets.find((item) => item.type === (includeMoss ? "moss-general" : "general")) ?? presets[0];
    if (general) void selectPreset(general.presetId);
  };

  const save = async () => {
    if (!name.trim() || !type || !prompt.trim() || pending) return;
    setPending(true);
    setError(null);
    try {
      const body = { name, type, promptText: prompt } as const;
      const detail = creating
        ? await createNlPromptPreset(body)
        : await updateNlPromptPreset(selectedId, body);
      const nextSummaries = await listNlPromptPresets(includeMoss);
      setPresets(includeMoss ? nextSummaries.presets.filter((item) => item.type.startsWith("moss-")) : nextSummaries.presets.filter((item) => !item.type.startsWith("moss-")));
      setDetails((current) => ({ ...current, [detail.presetId]: detail }));
      applyDetail(detail);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : t("presetSaveFailed"));
    } finally {
      setPending(false);
    }
  };

  const reset = async () => {
    if (creating || !selected || !selected.builtIn || pending) return;
    if (dirty && !window.confirm(t("confirmResetPreset"))) return;
    setPending(true);
    setError(null);
    try {
      const detail = await resetNlPromptPreset(selected.presetId);
      setDetails((current) => ({ ...current, [detail.presetId]: detail }));
      applyDetail(detail);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : t("presetResetFailed"));
    } finally {
      setPending(false);
    }
  };

  const remove = async () => {
    if (creating || !selected || selected.builtIn || pending || !window.confirm(t("confirmDeletePreset"))) return;
    setPending(true);
    setError(null);
    try {
      await deleteNlPromptPreset(selected.presetId);
      await loadAll();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : t("presetDeleteFailed"));
    } finally {
      setPending(false);
    }
  };

  return <section className="nl-preset-library" aria-busy={loading || pending}>
    <div className="nl-preset-library-toolbar">
      <p className="hint">{t("nlPresetLibraryHelp")}</p>
      <button type="button" className="secondary" disabled={disabled || loading || pending} onClick={startCreate}>{t("newPreset")}</button>
    </div>
    <div className="nl-preset-cards" role="list" aria-label={t("nlPresetList")}>
        {presets.map((item) => <button
          key={item.presetId}
          data-nl-preset-card
          type="button"
          aria-label={item.name}
          className={item.presetId === selectedId ? "selected" : ""}
          disabled={disabled || loading || pending || creating}
          onClick={() => void selectPreset(item.presetId)}
        ><span className="nl-preset-card-name">{item.name}</span><span className="nl-preset-card-type">{t(`captionPreset_${item.type}`)}</span>{details[item.presetId] && <span className="nl-preset-preview">{promptTextOf(details[item.presetId])}</span>}</button>)}
    </div>
    <div className="nl-preset-editor">
        {creating && <p className="nl-preset-mode">{t("newPreset")}</p>}
        <FormField label={t("presetName")} id="nl-preset-name" copy={guidanceCopy} guidance={{ description: t("fieldHelp_presetFields") }}>
          <input id="nl-preset-name" disabled={disabled || loading || pending || (!creating && Boolean(selected?.builtIn))} value={name} onChange={(event) => setName(event.target.value)} />
        </FormField>
        <FormField label={t("presetType")} id="nl-preset-type" copy={guidanceCopy} guidance={{ description: t("fieldHelp_presetType") }}>
          <select id="nl-preset-type" disabled={disabled || loading || pending || (!creating && Boolean(selected?.builtIn))} value={type} onChange={(event) => setType(event.target.value as NlPresetType | "")}>
            <option value="">{t("selectPresetType")}</option>
            <option value="general">{t("captionPreset_general")}</option>
            <option value="style">{t("captionPreset_style")}</option>
            <option value="character">{t("captionPreset_character")}</option>
            {includeMoss && <><option value="moss-general">{t("captionPreset_moss_general")}</option><option value="moss-style">{t("captionPreset_moss_style")}</option><option value="moss-character">{t("captionPreset_moss_character")}</option></>}
          </select>
        </FormField>
        <FormField label={t("promptText")} id="nl-prompt-text" wide copy={guidanceCopy} guidance={{ description: t("fieldHelp_promptText") }}>
          <textarea id="nl-prompt-text" rows={12} disabled={disabled || loading || pending} value={prompt} onChange={(event) => { setPrompt(event.target.value); onPromptChange(event.target.value); }} />
        </FormField>
        <div className="step-actions">
          {creating
            ? <><button type="button" disabled={disabled || loading || pending || !name.trim() || !type || !prompt.trim()} onClick={() => void save()}>{t("createPreset")}</button><button className="secondary" type="button" disabled={pending} onClick={cancelCreate}>{t("cancel")}</button></>
            : <><button type="button" disabled={disabled || loading || pending || !selected || !name.trim() || !type || !prompt.trim() || !dirty} onClick={() => void save()}>{t("saveChanges")}</button>{selected?.builtIn && <button className="secondary" type="button" disabled={disabled || loading || pending} onClick={() => void reset()}>{t("resetToDefault")}</button>}{selected && !selected.builtIn && <button className="danger-action" type="button" disabled={disabled || loading || pending} onClick={() => void remove()}>{t("deletePreset")}</button>}</>}
        </div>
        {error && <p className="resource-error" role="alert">{error}</p>}
    </div>
  </section>;
}

export type NlApiToolsProps = {
  disabled: boolean;
  profileId: string;
  endpoint: string;
  model: string;
  backupModel: string | null;
  backupEnabled: boolean;
  apiCredentialRef: string;
  transientApiKey: string;
  promptText: string;
  promptDirty: boolean;
  resetToken?: number;
  onEndpointChange: (endpoint: string) => void;
  onModelChange: (model: string) => void;
  onBackupModelChange: (model: string | null) => void;
  t: Translate;
  guidanceCopy: FieldGuidanceCopy;
  children?: ReactNode;
};

function diagnosticCredentials(endpoint: string, apiCredentialRef: string, transientApiKey: string) {
  const credentials = { endpoint, apiCredentialRef };
  return transientApiKey.trim() ? { ...credentials, apiKey: transientApiKey } : credentials;
}

const MANUAL_MODEL_VALUE = "__manual__";

export function isRemoteHttpEndpoint(value: string): boolean {
  if (!value || /\s/.test(value)) return false;
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "http:" || !parsed.hostname || parsed.username || parsed.password || parsed.search || parsed.hash) return false;
    const hostname = parsed.hostname.toLowerCase();
    return !["localhost", "127.0.0.1", "::1", "[::1]"].includes(hostname);
  } catch {
    return false;
  }
}

type ModelFieldProps = {
  id: string;
  label: string;
  value: string | null;
  models: readonly string[];
  disabled: boolean;
  allowEmpty: boolean;
  t: Translate;
  guidanceCopy: FieldGuidanceCopy;
  guidanceKey: string;
  onChange: (value: string | null) => void;
};

function ModelField({ id, label, value, models, disabled, allowEmpty, t, guidanceCopy, guidanceKey, onChange }: ModelFieldProps) {
  const availableModels = useMemo(() => Array.from(new Set(models)).filter((item) => item !== MANUAL_MODEL_VALUE), [models]);
  const current = value ?? "";
  const [manualSelection, setManualSelection] = useState<boolean | null>(null);
  useEffect(() => {
    setManualSelection(current && availableModels.includes(current) ? false : allowEmpty && !current ? false : true);
  }, [allowEmpty, availableModels, current]);
  const selectedValue = manualSelection === true ? MANUAL_MODEL_VALUE : current && availableModels.includes(current) ? current : allowEmpty && !current ? "" : MANUAL_MODEL_VALUE;
  const manual = selectedValue === MANUAL_MODEL_VALUE;
  return <FormField label={label} id={id} copy={guidanceCopy} guidance={{ description: t(guidanceKey) }}>
    <select
      id={id}
      disabled={disabled}
      value={selectedValue}
      onChange={(event) => {
        const next = event.target.value;
        if (next === MANUAL_MODEL_VALUE) {
          setManualSelection(true);
          return;
        }
        setManualSelection(false);
        onChange(next === "" ? (allowEmpty ? null : "") : next);
      }}
    >
      <option value="">{allowEmpty ? t("noBackupModel") : t("selectPrimaryModel")}</option>
      {availableModels.map((item) => <option key={item} value={item}>{item}</option>)}
      <option value={MANUAL_MODEL_VALUE}>{t("manualModel")}</option>
    </select>
    {manual && <input
      aria-label={t(allowEmpty ? "manualBackupModel" : "manualPrimaryModel")}
      disabled={disabled}
      value={current}
      onChange={(event) => onChange(event.target.value)}
      placeholder={t("manualModel")}
      autoComplete="off"
    />}
  </FormField>;
}

export function NlApiTools({
  disabled, profileId, endpoint, model, backupModel, backupEnabled, apiCredentialRef, transientApiKey, promptText, promptDirty,
  resetToken = 0, onEndpointChange, onModelChange, onBackupModelChange, t, guidanceCopy, children,
}: NlApiToolsProps) {
  const [models, setModels] = useState<string[]>([]);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [pending, setPending] = useState<"discovery" | "test" | null>(null);
  const requestVersion = useRef(0);

  useEffect(() => {
    requestVersion.current += 1;
    setModels([]);
    setFeedback(null);
    setPending(null);
  }, [disabled, endpoint, profileId, resetToken]);

  const discover = async () => {
    if (!endpoint.trim() || pending) return;
    const version = ++requestVersion.current;
    setPending("discovery");
    try {
      const result = await discoverNlModels(diagnosticCredentials(endpoint, apiCredentialRef, transientApiKey));
      if (version !== requestVersion.current) return;
      setFeedback({ kind: "discovery", result });
      setModels(result.ok ? result.models : []);
    } catch (cause) {
      if (version !== requestVersion.current) return;
      setModels([]);
      setFeedback({ kind: "discovery", result: { ok: false, latencyMs: 0, models: [], errorCode: "request_failed", errorReason: cause instanceof Error ? cause.message : t("diagnosticFailed") } });
    } finally {
      if (version === requestVersion.current) setPending(null);
    }
  };

  const sendTest = async () => {
    if (!endpoint.trim() || !model.trim() || !promptText.trim() || promptDirty || pending) return;
    const version = ++requestVersion.current;
    setPending("test");
    try {
      const result = await testNlMessage({ ...diagnosticCredentials(endpoint, apiCredentialRef, transientApiKey), model, basePrompt: promptText });
      if (version !== requestVersion.current) return;
      setFeedback({ kind: "test", result });
    } catch (cause) {
      if (version !== requestVersion.current) return;
      setFeedback({ kind: "test", result: { ok: false, latencyMs: 0, actualModel: null, replyText: null, usage: null, errorCode: "request_failed", errorReason: cause instanceof Error ? cause.message : t("diagnosticFailed") } });
    } finally {
      if (version === requestVersion.current) setPending(null);
    }
  };

  return <div className="nl-diagnostics" aria-busy={pending !== null}>
    <div className="form-grid">
      <FormField label={t("endpoint")} id="nl-endpoint" copy={guidanceCopy} guidance={{ description: t("fieldHelp_endpoint") }}>
        <input id="nl-endpoint" disabled={disabled} value={endpoint} onChange={(event) => onEndpointChange(event.target.value)} autoComplete="url" />
      </FormField>
      <div className="step-actions wide">
        <button className="secondary" type="button" disabled={disabled || pending !== null || !endpoint.trim()} aria-busy={pending === "discovery"} onClick={() => void discover()}>{t("getModels")}</button>
      </div>
    </div>
    {isRemoteHttpEndpoint(endpoint) && <p className="nl-http-warning" role="status">{t("httpEndpointWarning")}</p>}
    <div className="form-grid nl-model-selection">
      <ModelField id="nl-model" label={t("primaryModel")} value={model} models={models} disabled={disabled || pending !== null} allowEmpty={false} t={t} guidanceCopy={guidanceCopy} guidanceKey="fieldHelp_model" onChange={(value) => onModelChange(value ?? "")} />
      <ModelField id="nl-backup-model" label={t("backupModel")} value={backupModel} models={models} disabled={disabled || pending !== null || !backupEnabled} allowEmpty t={t} guidanceCopy={guidanceCopy} guidanceKey="fieldHelp_backupModelProvider" onChange={onBackupModelChange} />
    </div>
    {children}
    <div className="step-actions wide">
      <button type="button" disabled={disabled || pending !== null || !endpoint.trim() || !model.trim() || !promptText.trim() || promptDirty} aria-busy={pending === "test"} onClick={() => void sendTest()}>{t("sendTestMessage")}</button>
      {promptDirty && <small className="field-help">{t("savePresetBeforeTest")}</small>}
    </div>
    {feedback?.kind === "discovery" && <div className="nl-api-feedback" aria-live="polite"><strong>{feedback.result.ok ? t("diagnosticSuccess") : t("diagnosticFailure")}</strong><span>{feedback.result.latencyMs} ms</span>{feedback.result.ok ? <span>{t("modelsFound", { count: feedback.result.models.length })}</span> : <span>{feedback.result.errorReason ?? t("diagnosticUnavailable")}</span>}</div>}
    {feedback?.kind === "test" && <div className="nl-api-feedback" aria-live="polite"><strong>{feedback.result.ok ? t("diagnosticSuccess") : t("diagnosticFailure")}</strong><span>{feedback.result.latencyMs} ms</span><span>{feedback.result.actualModel ?? t("diagnosticUnavailable")}</span><span>{feedback.result.replyText ?? ""}</span><span>{t("promptTokens")}: {feedback.result.usage?.promptTokens ?? t("diagnosticUnavailable")}</span><span>{t("completionTokens")}: {feedback.result.usage?.completionTokens ?? t("diagnosticUnavailable")}</span><span>{t("totalTokens")}: {feedback.result.usage?.totalTokens ?? t("diagnosticUnavailable")}</span>{!feedback.result.ok && <span>{feedback.result.errorReason ?? t("diagnosticUnavailable")}</span>}</div>}
  </div>;
}
