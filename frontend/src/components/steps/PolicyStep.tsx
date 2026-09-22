import { ResourcePicker, type ResourcePickerProps } from "../ResourcePicker";
import { FormField, ToggleField, type FieldGuidanceCopy } from "../FormField";
import { ModuleBatchField } from "../ModuleBatchField";
import type { Draft } from "../../draft";
import type { UiLanguage } from "../../i18n";
import type { ResourceCatalogResponse } from "../../api";

type Translate = (key: string, values?: Record<string, string | number>) => string;
type AppearanceGroup = "solo" | "nonSolo" | "unknown";

export type PolicyStepProps = {
  draft: Pick<Draft, "dropout">;
  defaults: Pick<Draft, "dropout">;
  taskLocked: boolean;
  language: UiLanguage;
  resources: ResourcePickerProps["resources"];
  resourcesLoading: boolean;
  resourceError: string | null;
  invalidResourceCount: number;
  resourcePickerCopy: ResourcePickerProps["copy"];
  t: Translate;
  guidanceCopy: FieldGuidanceCopy;
  copy: { qualityModel: string; qualityModelHelp: string };
  capabilities?: ResourceCatalogResponse["capabilities"];
  onDropoutChange: (patch: Partial<Draft["dropout"]>) => void;
  onArtistChange: (patch: Partial<Draft["dropout"]["artist"]>) => void;
  onQualityChange: (patch: Partial<Draft["dropout"]["quality"]>) => void;
  onAppearanceNlChange: (patch: Partial<Draft["dropout"]["appearanceNl"]>) => void;
  onAppearanceProbabilityChange: (group: AppearanceGroup, key: "dropNl" | "dropAppearance", value: string) => void;
  onRefreshResources: () => void;
  batchSize: number;
  batchRecommended?: number;
  batchReason?: string;
  onBatchSizeChange: (value: number) => void;
};

export function PolicyStep({
  draft, defaults, taskLocked, language, resources, resourcesLoading, resourceError, invalidResourceCount,
  resourcePickerCopy, t, guidanceCopy, copy, onDropoutChange, onArtistChange, onQualityChange, onAppearanceNlChange,
  onAppearanceProbabilityChange, onRefreshResources,
  batchSize, batchRecommended, batchReason, onBatchSizeChange, capabilities,
}: PolicyStepProps) {
  const appearanceDisabled = taskLocked || !draft.dropout.enabled || !draft.dropout.appearanceNl.enabled;
  const groupLabels = { solo: t("dropoutSolo"), nonSolo: t("dropoutNonSolo"), unknown: t("dropoutUnknown") };
  const retainBoth = (group: AppearanceGroup) => Math.max(0, 1 - Number(draft.dropout.appearanceNl[group].dropNl) - Number(draft.dropout.appearanceNl[group].dropAppearance));
  const capabilityLabel = (status: "available" | "missing" | "invalid" | undefined, needed: boolean) => {
    if (status === "available") return t("capabilityInstalled");
    if (status === "invalid") return t("capabilityInvalid");
    return needed ? t("capabilityRequiredMissing") : t("capabilityNotNeeded");
  };
  const capabilityRows = [
    { id: "policy-runtime", label: t("policyRuntime"), status: capabilities?.policy.status, needed: draft.dropout.enabled },
    { id: "quality-stack", label: t("qualityStack"), status: capabilities?.qualityStack.status, needed: draft.dropout.enabled && draft.dropout.quality.enabled },
  ] as const;

  return <div className="option-stack" data-config-surface="policy">
    <section className="capability-status" aria-label={t("policyRuntime")}>
      {capabilityRows.map((row) => <div className="capability-status-row" key={row.id}>
        <strong>{row.label}</strong>
        <span>{capabilityLabel(row.status, row.needed)}</span>
        {(row.status === "missing" || row.status === "invalid") && row.needed && <small>{t("capabilityInstallHint")}</small>}
      </div>)}
    </section>
    <ToggleField id="policy-enabled" label={t("enablePolicy")} checked={draft.dropout.enabled} disabled={taskLocked} onChange={(enabled) => onDropoutChange({ enabled })} copy={guidanceCopy} guidance={{ description: t("fieldHelp_policyEnabled"), defaultValue: defaults.dropout.enabled ? t("fieldEnabled") : t("fieldDisabled") }} />
    <ModuleBatchField id="dropout-batch-size" label={t("batchSize")} value={batchSize} defaultValue={4} recommended={batchRecommended} recommendationReason={batchReason} minimum={1} maximum={16} disabled={taskLocked || !draft.dropout.enabled} t={t} guidanceCopy={guidanceCopy} onChange={onBatchSizeChange} />
    <div className="form-grid">
      <FormField id="policy-seed" label={t("seed")} copy={guidanceCopy} guidance={{ description: t("fieldHelp_policySeed"), defaultValue: defaults.dropout.seed }}>
        <input id="policy-seed" disabled={taskLocked || !draft.dropout.enabled} value={draft.dropout.seed} onChange={(event) => onDropoutChange({ seed: event.target.value })} />
      </FormField>
      <ToggleField id="policy-artist-enabled" label={t("appendFolderArtist")} checked={draft.dropout.artist.enabled} disabled={taskLocked || !draft.dropout.enabled} onChange={(enabled) => onArtistChange({ enabled })} copy={guidanceCopy} guidance={{ description: t("fieldHelp_appendFolderArtist"), defaultValue: defaults.dropout.artist.enabled ? t("fieldEnabled") : t("fieldDisabled") }} />
      <FormField id="policy-artist-dropout" label={t("artistDropout")} copy={guidanceCopy} guidance={{ description: t("fieldHelp_policyProbability"), defaultValue: String(defaults.dropout.artist.dropoutProbability), range: "0-1, step 0.01" }}>
        <input id="policy-artist-dropout" disabled={taskLocked || !draft.dropout.enabled || !draft.dropout.artist.enabled} type="number" min="0" max="1" step="0.01" value={draft.dropout.artist.dropoutProbability} onChange={(event) => onArtistChange({ dropoutProbability: Number(event.target.value) })} />
      </FormField>
      <ToggleField id="policy-quality-enabled" label={t("qualityScore")} checked={draft.dropout.quality.enabled} disabled={taskLocked || !draft.dropout.enabled} onChange={(enabled) => onQualityChange({ enabled })} copy={guidanceCopy} guidance={{ description: t("fieldHelp_policyBranch"), defaultValue: defaults.dropout.quality.enabled ? t("fieldEnabled") : t("fieldDisabled") }} />
    </div>
    {draft.dropout.enabled && draft.dropout.quality.enabled && <ResourcePicker
      id="policy-quality-model"
      label={copy.qualityModel} language={language} selectedId={draft.dropout.quality.resourceId ?? ""} resources={resources}
      loading={resourcesLoading} error={resourceError} invalidCount={invalidResourceCount}
      guidance={{ description: t("fieldHelp_qualityModel") }} guidanceCopy={guidanceCopy}
      selectionDisabled={taskLocked} refreshDisabled={taskLocked} note={copy.qualityModelHelp} copy={resourcePickerCopy}
      onChange={(resourceId) => onQualityChange({ resourceId })} onRefresh={onRefreshResources}
    />}
    <div className="form-grid">
      <FormField id="policy-quality-dropout" label={t("qualityDropout")} copy={guidanceCopy} guidance={{ description: t("fieldHelp_policyProbability"), defaultValue: String(defaults.dropout.quality.dropoutProbability), range: "0-1, step 0.01" }}>
        <input id="policy-quality-dropout" disabled={taskLocked || !draft.dropout.enabled || !draft.dropout.quality.enabled} type="number" min="0" max="1" step="0.01" value={draft.dropout.quality.dropoutProbability} onChange={(event) => onQualityChange({ dropoutProbability: Number(event.target.value) })} />
      </FormField>
      <FormField id="policy-quality-device" label={t("qualityDevice")} copy={guidanceCopy} guidance={{ description: t("fieldHelp_qualityDevice"), defaultValue: t("automatic"), recommendation: t("automatic") }}>
        <select id="policy-quality-device" disabled={taskLocked || !draft.dropout.enabled || !draft.dropout.quality.enabled} value={draft.dropout.quality.device} onChange={(event) => onQualityChange({ device: event.target.value as Draft["dropout"]["quality"]["device"] })}><option value="auto">{t("automatic")}</option><option value="cuda">CUDA</option><option value="cpu">CPU</option></select>
      </FormField>
      <FormField id="policy-quality-batch" label={t("qualityBatch")} copy={guidanceCopy} guidance={{ description: t("fieldHelp_qualityBatch"), defaultValue: String(defaults.dropout.quality.batchSize), range: "1-16, step 1" }}>
        <input id="policy-quality-batch" disabled={taskLocked || !draft.dropout.enabled || !draft.dropout.quality.enabled} type="number" min="1" max="16" step="1" value={draft.dropout.quality.batchSize} onChange={(event) => onQualityChange({ batchSize: Number(event.target.value) })} />
      </FormField>
    </div>
    <section className={`dropout-strategy ${appearanceDisabled ? "is-disabled" : ""}`}>
      <div className="dropout-strategy-heading">
        <ToggleField id="policy-appearance-enabled" label={t("appearanceNl")} checked={draft.dropout.appearanceNl.enabled} disabled={taskLocked || !draft.dropout.enabled} onChange={(enabled) => onAppearanceNlChange({ enabled })} copy={guidanceCopy} guidance={{ description: t("appearanceNlHelp"), defaultValue: defaults.dropout.appearanceNl.enabled ? t("fieldEnabled") : t("fieldDisabled") }} />
      </div>
      <div className="dropout-strategy-body">
        {(["solo", "nonSolo", "unknown"] as const).map((group) => <div className="probability-row" key={group}><strong>{groupLabels[group]}</strong><FormField id={`policy-${group}-drop-nl`} label={t("dropNl")} copy={guidanceCopy} guidance={{ description: t("dropNlHelp"), defaultValue: String(defaults.dropout.appearanceNl[group].dropNl), range: "0-1, step 0.01" }}><input id={`policy-${group}-drop-nl`} disabled={appearanceDisabled} type="number" min="0" max="1" step="0.01" value={draft.dropout.appearanceNl[group].dropNl} onChange={(event) => onAppearanceProbabilityChange(group, "dropNl", event.target.value)} /></FormField><FormField id={`policy-${group}-drop-appearance`} label={t("dropAppearance")} copy={guidanceCopy} guidance={{ description: t("dropAppearanceHelp"), defaultValue: String(defaults.dropout.appearanceNl[group].dropAppearance), range: "0-1, step 0.01" }}><input id={`policy-${group}-drop-appearance`} disabled={appearanceDisabled} type="number" min="0" max="1" step="0.01" value={draft.dropout.appearanceNl[group].dropAppearance} onChange={(event) => onAppearanceProbabilityChange(group, "dropAppearance", event.target.value)} /></FormField><small>{t("keepBoth", { value: (retainBoth(group) * 100).toFixed(0) })}</small></div>)}
      </div>
    </section>
  </div>;
}
