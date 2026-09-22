import { useEffect, useState } from "react";
import type { Draft } from "../../draft";
import { FormField, ToggleField, type FieldGuidanceCopy } from "../FormField";
import { ModuleBatchField } from "../ModuleBatchField";
import { NlApiTools, NlPromptPresetLibrary } from "../NlApiTools";

type Translate = (key: string, values?: Record<string, string | number>) => string;
type NlPresetType = "general" | "style" | "character";

type NlProfileForm = {
  profileId: string;
  name?: string;
  endpoint: string;
  model: string;
  backupModel: string | null;
  apiCredentialRef: string;
  hasCredential: boolean;
  revision?: number;
};

export type NlStepProps = {
  draft: Pick<Draft, "nl">;
  defaults: Pick<Draft, "nl">;
  taskLocked: boolean;
  attemptBudget: string;
  profile: NlProfileForm;
  profiles: readonly NlProfileForm[];
  defaultProfileId: string | null;
  secret: string;
  profileSavePending: boolean;
  credentialDeletePending: boolean;
  runtimeEditable: boolean;
  runtimeTaskRevision: number;
  runtimePending: boolean;
  diagnosticResetToken: number;
  t: Translate;
  guidanceCopy: FieldGuidanceCopy;
  profileHelp: string;
  onNlChange: (patch: Partial<Draft["nl"]>) => void;
  onApiPolicyChange: (patch: Partial<Draft["nl"]["apiPolicy"]>) => void;
  onRequestsPerMinuteChange: (value: string) => void;
  onUnlimitedRpmChange: (enabled: boolean) => void;
  onAttemptBudgetChange: (value: string) => void;
  onProfileSelect: (profileId: string) => void;
  onProfileChange: (profile: NlProfileForm) => void;
  onSecretChange: (value: string) => void;
  onSaveProfile: () => void;
  onApplyRuntime: () => void;
  onSetDefault: () => void;
  onCopyProfile: () => void;
  onDeleteProfile: () => void;
  onClearSecret: () => void;
  batchSize: number;
  batchRecommended?: number;
  batchReason?: string;
  onBatchSizeChange: (value: number) => void;
};

function distributionError(distribution: Draft["nl"]["lengthDistribution"]): boolean {
  const values = Object.values(distribution);
  return !values.every((value) => Number.isInteger(value) && value >= 0 && value <= 100)
    || values.reduce((total, value) => total + value, 0) !== 100;
}

export function NlStep({
  draft, defaults, taskLocked, attemptBudget, profile, profiles, defaultProfileId, secret, profileSavePending, credentialDeletePending, runtimeEditable, runtimeTaskRevision, runtimePending, t,
  onNlChange, onApiPolicyChange, onRequestsPerMinuteChange,
  onUnlimitedRpmChange, onAttemptBudgetChange, onProfileSelect, onProfileChange, onSecretChange, onSaveProfile, onApplyRuntime, onSetDefault, onCopyProfile, onDeleteProfile, onClearSecret, guidanceCopy, profileHelp, diagnosticResetToken,
  batchSize, batchRecommended, batchReason, onBatchSizeChange,
}: NlStepProps) {
  const apiDisabled = !draft.nl.enabled || !draft.nl.apiEnabled;
  const apiControlsDisabled = apiDisabled || (taskLocked && !runtimeEditable);
  const apiCollapsed = !draft.nl.enabled || !draft.nl.apiEnabled;
  const unlimitedRpm = draft.nl.apiPolicy.maxRequestsPerMinute === "unlimited";
  const invalidDistribution = distributionError(draft.nl.lengthDistribution);
  const [promptDirty, setPromptDirty] = useState(false);
  const [keyVisible, setKeyVisible] = useState(false);
  useEffect(() => {
    if (!secret || taskLocked || apiDisabled) setKeyVisible(false);
  }, [apiDisabled, secret, taskLocked]);
  useEffect(() => {
    setKeyVisible(false);
  }, [diagnosticResetToken, profile.endpoint, profile.profileId]);
  const profileOptions = profiles.filter((item) => item.profileId !== profile.profileId);

  return <div className="nl-step settings-section" data-config-surface="nl">
    <section className="settings-section">
      <h3>{t("nlGenerationSettings")}</h3>
      <div className="form-grid">
        <ToggleField id="nl-enabled" label={t("enableNl")} checked={draft.nl.enabled} disabled={taskLocked} guidance={{ description: t("fieldHelp_enableNl") }} copy={guidanceCopy} onChange={(enabled) => onNlChange({ enabled })} />
        <ToggleField id="nl-reuse-original" label={t("reuseOriginalNl")} checked={draft.nl.reuseOriginalNl} disabled={taskLocked || !draft.nl.enabled} guidance={{ description: t("fieldHelp_reuseOriginalNl") }} copy={guidanceCopy} onChange={(reuseOriginalNl) => onNlChange({ reuseOriginalNl })} />
        <ToggleField id="nl-api-enabled" label={t("generateNl")} checked={draft.nl.apiEnabled} disabled={taskLocked || !draft.nl.enabled} guidance={{ description: t("fieldHelp_generateNl") }} copy={guidanceCopy} onChange={(apiEnabled) => onNlChange({ apiEnabled })} />
        <ToggleField id="nl-use-image" label={t("useImage")} checked={draft.nl.useImage} disabled={taskLocked || !draft.nl.enabled || !draft.nl.apiEnabled} guidance={{ description: t("fieldHelp_useImage") }} copy={guidanceCopy} onChange={(useImage) => onNlChange({ useImage })} />
        <ToggleField id="nl-use-json" label={t("useJsonContext")} checked={draft.nl.useFullJson} disabled={taskLocked || !draft.nl.enabled || !draft.nl.apiEnabled} guidance={{ description: t("fieldHelp_useJsonContext") }} copy={guidanceCopy} onChange={(useFullJson) => onNlChange({ useFullJson })} />
      </div>
      <ModuleBatchField id="nl-batch-size" label={t("batchSize")} value={batchSize} defaultValue={3} recommended={batchRecommended} recommendationReason={batchReason} minimum={1} maximum={16} disabled={taskLocked || !draft.nl.enabled} t={t} guidanceCopy={guidanceCopy} onChange={onBatchSizeChange} />
    </section>

    <section className="settings-section nl-prompt-section">
      <h3>{t("nlPromptPresets")}</h3>
      <NlPromptPresetLibrary
        disabled={taskLocked || !draft.nl.enabled}
        promptText={draft.nl.systemPrompt}
        presetType={draft.nl.captionPreset as NlPresetType}
        t={t}
        guidanceCopy={guidanceCopy}
        onPromptChange={(systemPrompt) => onNlChange({ systemPrompt, promptVersion: "nl-default-prompt-v4" })}
        onPresetTypeChange={(captionPreset) => onNlChange({ captionPreset, promptVersion: "nl-default-prompt-v4" })}
        onDirtyChange={setPromptDirty}
      />
      <div className="form-grid nl-length-settings">
        {(["short", "medium", "long"] as const).map((tier) => <FormField key={tier} id={`nl-length-${tier}`} label={t(`lengthTier_${tier}`)} copy={guidanceCopy} guidance={{ description: t("fieldHelp_lengthTiers"), defaultValue: String(defaults.nl.lengthDistribution[tier]), range: "0-100 integer; total 100" }}>
          <input id={`nl-length-${tier}`} disabled={taskLocked} type="number" min="0" max="100" step="1" value={draft.nl.lengthDistribution[tier]} onChange={(event) => onNlChange({ lengthDistribution: { ...draft.nl.lengthDistribution, [tier]: Number(event.target.value) } })} />
        </FormField>)}
      </div>
      {invalidDistribution && <small className="resource-error" role="alert">{t("lengthDistributionInvalid")}</small>}
    </section>

    <section className={`settings-section nl-api-section${apiCollapsed ? " is-collapsed" : ""}`}>
      <details open={!apiCollapsed} onClick={(event) => { if (apiCollapsed) event.preventDefault(); }}>
        <summary><span>{t("nlApiProfile")}</span><small>{apiCollapsed ? t("apiControlsCollapsed") : t("apiControlsExpanded")}</small></summary>
        <div className="nl-api-content">
          <p className="hint">{profileHelp}</p>
          <div className="form-grid">
            <FormField label={t("profileId")} id="nl-profile-select" copy={guidanceCopy} guidance={{ description: t("fieldHelp_profileId") }}>
              <select id="nl-profile-select" disabled={apiControlsDisabled || profileSavePending} value={profile.profileId} onChange={(event) => onProfileSelect(event.target.value)}>
                <option value={profile.profileId}>{profile.profileId}</option>
                {profileOptions.map((item) => <option key={item.profileId} value={item.profileId}>{item.profileId}</option>)}
              </select>
            </FormField>
          </div>
          <div className="nl-profile-meta" aria-live="polite">
            <span>{t("profileRevision")}: {profile.revision ?? 1}</span>
            <span>{profile.profileId === defaultProfileId ? t("defaultProfile") : t("notDefaultProfile")}</span>
          </div>
            <NlApiTools
            disabled={apiControlsDisabled}
            profileId={profile.profileId}
            endpoint={profile.endpoint}
            model={profile.model}
            backupModel={profile.backupModel}
            backupEnabled={draft.nl.apiPolicy.backupEnabled}
            apiCredentialRef={profile.apiCredentialRef}
            transientApiKey={secret}
            promptText={draft.nl.systemPrompt}
            promptDirty={promptDirty}
            resetToken={diagnosticResetToken}
            onEndpointChange={(endpoint) => onProfileChange({ ...profile, endpoint })}
            onModelChange={(model) => onProfileChange({ ...profile, model })}
            onBackupModelChange={(backupModel) => onProfileChange({ ...profile, backupModel })}
            t={t}
            guidanceCopy={guidanceCopy}
          >
            <div className="form-grid nl-api-secret-grid">
              <FormField label={t("apiKey")} id="nl-api-key" wide copy={guidanceCopy} guidance={{ description: t("fieldHelp_apiKey") }}>
                <div className="secret-input-control">
                <input id="nl-api-key" disabled={apiControlsDisabled} type={keyVisible ? "text" : "password"} value={secret} autoComplete="off" onChange={(event) => onSecretChange(event.target.value)} placeholder={profile.hasCredential ? t("savedKey") : t("dpapiKey")} />
                  <button className="icon-button secondary" type="button" disabled={taskLocked || apiDisabled || !secret} aria-label={t(keyVisible ? "hideApiKey" : "showApiKey")} title={t(keyVisible ? "hideApiKey" : "showApiKey")} onClick={() => setKeyVisible((visible) => !visible)}><span aria-hidden="true">&#128065;</span></button>
                </div>
              </FormField>
            </div>
            <div className="step-actions"><button disabled={apiControlsDisabled || profileSavePending} aria-busy={profileSavePending} type="button" onClick={runtimeEditable ? onApplyRuntime : onSaveProfile}>{t(runtimeEditable ? "applyRuntimeApi" : "saveProfile")}</button><button className="secondary" disabled={taskLocked || profileSavePending || profile.profileId === defaultProfileId} type="button" onClick={onSetDefault}>{t("setDefaultProfile")}</button><button className="secondary" disabled={taskLocked || profileSavePending} type="button" onClick={onCopyProfile}>{t("copyProfile")}</button><button className="danger-action" disabled={taskLocked || profileSavePending || profile.profileId === defaultProfileId || profiles.length <= 1} type="button" onClick={onDeleteProfile}>{t("deleteProfile")}</button><button className="secondary" disabled={taskLocked || credentialDeletePending || (!profile.hasCredential && !profile.apiCredentialRef)} aria-busy={credentialDeletePending} type="button" onClick={onClearSecret}>{t("clearKey")}</button></div>
            {runtimeEditable && <p className="hint" role="status">{runtimePending ? t("runtimeApiPending", { revision: runtimeTaskRevision + 1 }) : t("runtimeApiHint", { revision: runtimeTaskRevision })}</p>}

            <div className="form-grid nl-request-limits">
              <FormField label={t("requestsPerMinute")} id="nl-rpm" copy={guidanceCopy} guidance={{ description: t("fieldHelp_requestsPerMinute"), defaultValue: String(defaults.nl.apiPolicy.maxRequestsPerMinute === "unlimited" ? 60 : defaults.nl.apiPolicy.maxRequestsPerMinute), range: "1-100000" }}>
                <input id="nl-rpm" disabled={apiControlsDisabled || unlimitedRpm} type="number" min="1" max="100000" step="1" value={unlimitedRpm ? 60 : draft.nl.apiPolicy.maxRequestsPerMinute} onChange={(event) => onRequestsPerMinuteChange(event.target.value)} />
              </FormField>
              <ToggleField id="nl-unlimited-rpm" label={t("unlimitedRpm")} checked={unlimitedRpm} disabled={apiControlsDisabled} guidance={{ description: t("fieldHelp_unlimitedRpm"), defaultValue: t("fieldDisabled") }} copy={guidanceCopy} onChange={onUnlimitedRpmChange} />
              <FormField label={t("attemptBudget")} id="nl-attempt-budget" copy={guidanceCopy} guidance={{ description: t("fieldHelp_attemptBudget") }}>
                <input id="nl-attempt-budget" disabled={apiControlsDisabled} inputMode="numeric" value={attemptBudget} onChange={(event) => onAttemptBudgetChange(event.target.value)} /><small className="field-help">{t("attemptBudgetHelp")}</small>
              </FormField>
              <ToggleField id="nl-backup-enabled" label={t("enableBackupModel")} checked={draft.nl.apiPolicy.backupEnabled} disabled={apiControlsDisabled} guidance={{ description: t("fieldHelp_backupModel"), defaultValue: t("fieldDisabled") }} copy={guidanceCopy} onChange={(backupEnabled) => onApiPolicyChange({ backupEnabled })} />
            </div>
          </NlApiTools>
        </div>
      </details>
    </section>
  </div>;
}
