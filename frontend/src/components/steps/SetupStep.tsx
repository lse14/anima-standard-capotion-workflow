import type { AnnotationPreset } from "../../draftPersistence";
import type { Draft } from "../../draft";
import { FormField, ToggleField, type FieldGuidanceCopy } from "../FormField";
import { PathPicker, type PathPickerCopy } from "../PathPicker";

type Translate = (key: string, values?: Record<string, string | number>) => string;

export type SetupStepProps = {
  draft: Pick<Draft, "sourceRoot" | "workMode" | "outputRoot" | "overwriteMode" | "imageDecode" | "recursive">;
  defaults: Pick<Draft, "workMode" | "overwriteMode" | "imageDecode" | "recursive">;
  taskLocked: boolean;
  profileResourcesReady: boolean;
  presets: readonly AnnotationPreset[];
  selectedPresetId: string;
  presetName: string;
  persistenceWarning: string | null;
  t: Translate;
  guidanceCopy: FieldGuidanceCopy;
  pathPickerCopy: PathPickerCopy;
  copy: { preflightHint: string; profileReady: string; profileUnavailable: string };
  onSourceRootChange: (value: string) => void;
  onWorkModeChange: (value: Draft["workMode"]) => void;
  onOutputRootChange: (value: string) => void;
  onOverwriteModeChange: (value: Draft["overwriteMode"]) => void;
  onInvalidImageActionChange: (value: Draft["imageDecode"]["invalidImageAction"]) => void;
  onRecursiveChange: (value: boolean) => void;
  onPresetSelect: (presetId: string) => void;
  onPresetNameChange: (value: string) => void;
  onSavePreset: () => void;
  onDeletePreset: () => void;
};

export function SetupStep({
  draft, defaults, taskLocked, profileResourcesReady, presets, selectedPresetId, presetName, persistenceWarning, t, guidanceCopy, pathPickerCopy, copy,
  onSourceRootChange, onWorkModeChange, onOutputRootChange, onOverwriteModeChange, onInvalidImageActionChange, onRecursiveChange,
  onPresetSelect, onPresetNameChange, onSavePreset, onDeletePreset,
}: SetupStepProps) {
  return <>
    <div className="step-intro"><p>{copy.preflightHint}</p><span className={`status ${profileResourcesReady ? "available" : "failed"}`}>{profileResourcesReady ? copy.profileReady : copy.profileUnavailable}</span></div>
    <section className="annotation-presets" aria-labelledby="annotation-presets-heading">
      <div className="subsection-heading"><h3 id="annotation-presets-heading">{t("taskPresets")}</h3></div>
      <div className="form-grid">
        <FormField id="annotation-preset-select" label={t("taskPreset")} copy={guidanceCopy} guidance={{ description: t("fieldHelp_taskPreset") }}>
          <select id="annotation-preset-select" disabled={taskLocked} value={selectedPresetId} onChange={(event) => onPresetSelect(event.target.value)}>
            <option value="">{t("noTaskPreset")}</option>
            {presets.map((preset) => <option key={preset.presetId} value={preset.presetId}>{preset.name}</option>)}
          </select>
        </FormField>
        <FormField id="annotation-preset-name" label={t("presetName")} copy={guidanceCopy} guidance={{ description: t("fieldHelp_taskPreset") }}>
          <input id="annotation-preset-name" disabled={taskLocked} value={presetName} onChange={(event) => onPresetNameChange(event.target.value)} placeholder={t("taskPresetNamePlaceholder")} />
        </FormField>
      </div>
      <div className="step-actions">
        <button type="button" disabled={taskLocked || !presetName.trim()} onClick={onSavePreset}>{t("saveTaskPreset")}</button>
        <button className="secondary" type="button" disabled={taskLocked || !selectedPresetId} onClick={onDeletePreset}>{t("deleteTaskPreset")}</button>
      </div>
      <p className="hint">{t("taskPresetHelp")}</p>
      {persistenceWarning && <p className="resource-warning" role="status" data-persistence-warning>{persistenceWarning}</p>}
    </section>
    <div data-config-surface="setup" className="form-grid">
      <FormField id="setup-source-dataset" label={t("sourceDataset")} copy={guidanceCopy} guidance={{ description: t("fieldHelp_sourceDataset") }}>
        <PathPicker id="setup-source-dataset" purpose="source_dataset" disabled={taskLocked} value={draft.sourceRoot} onChange={onSourceRootChange} placeholder={t("absoluteWindowsPath")} copy={pathPickerCopy} />
      </FormField>
      <FormField id="setup-work-mode" label={t("workMode")} copy={guidanceCopy} guidance={{ description: t("fieldHelp_workMode"), defaultValue: defaults.workMode === "in_place" ? t("inPlace") : t("fullCopy") }}>
        <select id="setup-work-mode" disabled={taskLocked} value={draft.workMode} onChange={(event) => onWorkModeChange(event.target.value as Draft["workMode"])}><option value="in_place">{t("inPlace")}</option><option value="full_copy">{t("fullCopy")}</option></select>
      </FormField>
      {draft.workMode === "full_copy" && <FormField id="setup-output-dataset" label={t("outputDataset")} copy={guidanceCopy} guidance={{ description: t("fieldHelp_outputDataset") }} wide>
        <PathPicker id="setup-output-dataset" purpose="output_dataset" disabled={taskLocked} value={draft.outputRoot ?? ""} onChange={onOutputRootChange} placeholder={t("emptyOrAbsentDirectory")} copy={pathPickerCopy} />
      </FormField>}
      <FormField id="setup-overwrite-mode" label={t("overwriteMode")} copy={guidanceCopy} guidance={{ description: t("fieldHelp_overwriteMode"), defaultValue: defaults.overwriteMode === "incremental" ? t("incremental") : t("rebuild") }}>
        <select id="setup-overwrite-mode" disabled={taskLocked} value={draft.overwriteMode} onChange={(event) => onOverwriteModeChange(event.target.value as Draft["overwriteMode"])}><option value="incremental">{t("incremental")}</option><option value="rebuild">{t("rebuild")}</option></select>
      </FormField>
      <FormField id="setup-invalid-image-action" label={t("invalidImageAction")} copy={guidanceCopy} guidance={{ description: t("fieldHelp_invalidImageAction"), defaultValue: defaults.imageDecode.invalidImageAction === "block" ? t("blockInvalidImages") : t("skipInvalidImages") }}>
        <select id="setup-invalid-image-action" disabled={taskLocked} value={draft.imageDecode.invalidImageAction} onChange={(event) => onInvalidImageActionChange(event.target.value as Draft["imageDecode"]["invalidImageAction"])}><option value="block">{t("blockInvalidImages")}</option><option value="skip">{t("skipInvalidImages")}</option></select>
      </FormField>
      <ToggleField id="setup-recursive" label={t("searchSubfolders")} checked={draft.recursive} disabled={taskLocked} onChange={onRecursiveChange} copy={guidanceCopy} guidance={{ description: t("fieldHelp_recursive"), defaultValue: defaults.recursive ? t("fieldEnabled") : t("fieldDisabled") }} />
    </div>
    <details className="input-format-help" data-input-format-help>
      <summary>{t("inputFormatHelpTitle")}</summary>
      <ul>
        <li>{t("inputFormatHelpPairing")}</li>
        <li>{t("inputFormatHelpStandardJson")}</li>
        <li>{t("inputFormatHelpRawJson")}</li>
        <li>{t("inputFormatHelpTxt")}</li>
        <li>{t("inputFormatHelpValidation")}</li>
      </ul>
    </details>
    <p className="hint">{copy.preflightHint}</p>
  </>;
}
