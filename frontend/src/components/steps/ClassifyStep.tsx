import { ResourcePicker, type ResourcePickerProps } from "../ResourcePicker";
import { FormField, ToggleField, type FieldGuidanceCopy } from "../FormField";
import { ModuleBatchField } from "../ModuleBatchField";
import { PathPicker, type PathPickerCopy } from "../PathPicker";
import type { Draft } from "../../draft";
import type { UiLanguage } from "../../i18n";

type Translate = (key: string, values?: Record<string, string | number>) => string;

export type ClassifyStepProps = {
  draft: Pick<Draft, "classify">;
  taskLocked: boolean;
  rebuild: boolean;
  classifyEnabled: boolean;
  classificationResourceActive: boolean;
  language: UiLanguage;
  resources: ResourcePickerProps["resources"];
  resourcesLoading: boolean;
  resourceError: string | null;
  invalidResourceCount: number;
  resourcePickerCopy: ResourcePickerProps["copy"];
  pathPickerCopy: PathPickerCopy;
  t: Translate;
  guidanceCopy: FieldGuidanceCopy;
  copy: { classificationIndex: string; classificationIndexHelp: string; bundledResource: string; customResource: string; classificationResourceJson: string; anthroReplacementNote: string };
  onClassifyChange: (patch: Partial<Draft["classify"]>) => void;
  onRefreshResources: () => void;
  batchSize: number;
  batchRecommended?: number;
  batchReason?: string;
  onBatchSizeChange: (value: number) => void;
};

export function ClassifyStep({
  draft, taskLocked, rebuild, classifyEnabled, classificationResourceActive, language, resources,
  resourcesLoading, resourceError, invalidResourceCount, resourcePickerCopy, pathPickerCopy, t, guidanceCopy, copy, onClassifyChange, onRefreshResources,
  batchSize, batchRecommended, batchReason, onBatchSizeChange,
}: ClassifyStepProps) {
  return <div className="option-stack" data-config-surface="classify">
    <ToggleField id="classify-enabled" label={t("enableClassify")} checked={classifyEnabled} disabled={taskLocked} onChange={(enabled) => onClassifyChange({ enabled })} copy={guidanceCopy} guidance={{ description: t("fieldHelp_enableClassify"), defaultValue: classifyEnabled ? t("fieldEnabled") : t("fieldDisabled") }} />
    <ModuleBatchField id="classify-batch-size" label={t("batchSize")} value={batchSize} defaultValue={128} recommended={batchRecommended} recommendationReason={batchReason} minimum={1} maximum={500} disabled={taskLocked || !classifyEnabled} t={t} guidanceCopy={guidanceCopy} onChange={onBatchSizeChange} />
    <div className="segmented-control" role="group" aria-label={copy.classificationIndex}>
      <button type="button" className={draft.classify.indexMode === "bundled" ? "selected" : ""} disabled={taskLocked || !classificationResourceActive} onClick={() => onClassifyChange({ indexMode: "bundled", resourceId: draft.classify.resourceId, customResourcePath: undefined })}>{copy.bundledResource}</button>
      <button type="button" className={draft.classify.indexMode === "custom" ? "selected" : ""} disabled={taskLocked || !classificationResourceActive} onClick={() => onClassifyChange({ indexMode: "custom", resourceId: undefined, customResourcePath: draft.classify.customResourcePath ?? "" })}>{copy.customResource}</button>
    </div>
    {draft.classify.indexMode === "bundled" ? <ResourcePicker
      id="classify-resource"
      label={copy.classificationIndex} language={language} selectedId={draft.classify.resourceId ?? ""} resources={resources}
      loading={resourcesLoading} error={resourceError} invalidCount={invalidResourceCount}
      guidance={{ description: t("fieldHelp_classificationIndex") }} guidanceCopy={guidanceCopy}
      selectionDisabled={taskLocked || !classificationResourceActive} refreshDisabled={taskLocked} note={copy.classificationIndexHelp} copy={resourcePickerCopy}
      onChange={(resourceId) => onClassifyChange({ resourceId })} onRefresh={onRefreshResources}
    /> : <FormField id="classify-custom-resource" label={copy.classificationResourceJson} copy={guidanceCopy} guidance={{ description: copy.classificationIndexHelp }}>
      <PathPicker id="classify-custom-resource" purpose="classification_resource_json" disabled={taskLocked || !classificationResourceActive} value={draft.classify.customResourcePath ?? ""} onChange={(customResourcePath) => onClassifyChange({ customResourcePath })} placeholder={t("absoluteWindowsPath")} copy={pathPickerCopy} />
    </FormField>}
    <small>{copy.anthroReplacementNote}</small>
    {rebuild ? <>
      <div className="readonly-setting" data-classify-overwrite-mode="rebuild">
        <ToggleField id="classify-overwrite-json" label={t("overwriteJson")} checked disabled onChange={() => undefined} copy={guidanceCopy} guidance={{ description: t("rebuildOverwriteHelp"), defaultValue: t("fieldEnabled") }} />
        <small>{t("rebuildOverwriteHelp")}</small>
      </div>
      <div className="readonly-setting">
        <ToggleField id="classify-overwrite-count" label={t("overwriteCount")} checked disabled onChange={() => undefined} copy={guidanceCopy} guidance={{ description: t("rebuildOverwriteHelp"), defaultValue: t("fieldEnabled") }} />
        <small>{t("rebuildOverwriteHelp")}</small>
      </div>
    </> : <>
      <ToggleField id="classify-overwrite-json" label={t("overwriteJson")} checked={draft.classify.overwriteJson} disabled={taskLocked || !classifyEnabled} onChange={(overwriteJson) => onClassifyChange({ overwriteJson })} copy={guidanceCopy} guidance={{ description: t("fieldHelp_overwriteJson"), defaultValue: draft.classify.overwriteJson ? t("fieldEnabled") : t("fieldDisabled") }} />
      <ToggleField id="classify-overwrite-count" label={t("overwriteCount")} checked={draft.classify.overwriteCount} disabled={taskLocked || !classifyEnabled} onChange={(overwriteCount) => onClassifyChange({ overwriteCount })} copy={guidanceCopy} guidance={{ description: t("fieldHelp_overwriteCount"), defaultValue: draft.classify.overwriteCount ? t("fieldEnabled") : t("fieldDisabled") }} />
    </>}
  </div>;
}
