import type { Draft } from "../../draft";
import { FormField, type FieldGuidanceCopy } from "../FormField";
import { ModuleBatchField } from "../ModuleBatchField";

type Translate = (key: string, values?: Record<string, string | number>) => string;

type ExportSummary = {
  valid: number;
  invalid: number;
  exported: number;
  issueCount: number;
  convertedSamples: number;
  unlabeled: number;
  unlabeledFolder: string | null;
} | null;

export type ExportStepProps = {
  draft: Pick<Draft, "export" | "captionFormat">;
  defaults: Pick<Draft, "export" | "captionFormat">;
  taskLocked: boolean;
  exportSummary: ExportSummary;
  t: Translate;
  guidanceCopy: FieldGuidanceCopy;
  onExportChange: (patch: Partial<Draft["export"]>) => void;
  onCaptionFormatChange: (patch: Partial<Draft["captionFormat"]>) => void;
  batchSize: number;
  batchRecommended?: number;
  batchReason?: string;
  onBatchSizeChange: (value: number) => void;
};

export function ExportStep({
  draft, defaults, taskLocked, exportSummary, t, guidanceCopy, onExportChange, onCaptionFormatChange,
  batchSize, batchRecommended, batchReason, onBatchSizeChange,
}: ExportStepProps) {
  return <>
    <div className="form-grid" data-config-surface="export">
      <ModuleBatchField id="export-batch-size" label={t("batchSize")} value={batchSize} defaultValue={500} recommended={batchRecommended} recommendationReason={batchReason} minimum={1} maximum={500} disabled={taskLocked} t={t} guidanceCopy={guidanceCopy} onChange={onBatchSizeChange} />
      <FormField id="export-format" label={t("format")} copy={guidanceCopy} guidance={{ description: t("fieldHelp_exportFormat"), defaultValue: t("both") }}><select id="export-format" disabled={taskLocked} value={draft.export.format} onChange={(event) => onExportChange({ format: event.target.value as Draft["export"]["format"] })}><option value="both">{t("both")}</option><option value="json">{t("jsonOnly")}</option><option value="flat_txt">{t("flatTxtOnly")}</option></select></FormField>
      <FormField id="flat-txt-layout" label={t("flatTxtLayout")} copy={guidanceCopy} guidance={{ description: t("fieldHelp_flatTxtLayout"), defaultValue: defaults.captionFormat.flatTxtLayout === "single_line" ? t("singleLine") : t("nlNewline") }}><select id="flat-txt-layout" disabled={taskLocked} value={draft.captionFormat.flatTxtLayout} onChange={(event) => onCaptionFormatChange({ flatTxtLayout: event.target.value as Draft["captionFormat"]["flatTxtLayout"] })}><option value="nl_newline">{t("nlNewline")}</option><option value="single_line">{t("singleLine")}</option></select></FormField>
    </div>
    {exportSummary && <dl className="facts"><div><dt>{t("validInvalid")}</dt><dd>{exportSummary.valid} / {exportSummary.invalid}</dd></div><div><dt>{t("committed")}</dt><dd>{exportSummary.exported}</dd></div><div><dt>{t("issues")}</dt><dd>{exportSummary.issueCount}</dd></div><div><dt>{t("converted")}</dt><dd>{exportSummary.convertedSamples}</dd></div><div><dt>{t("unlabeledImages")}</dt><dd>{exportSummary.unlabeled}</dd></div>{exportSummary.unlabeledFolder && <div className="wide"><dt>{t("unlabeledFolder")}</dt><dd><code>{exportSummary.unlabeledFolder}</code></dd></div>}</dl>}
  </>;
}
