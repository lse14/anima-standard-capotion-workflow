import type { PreflightProgress, PreflightSummary } from "../../api";
import type { Draft } from "../../draft";

type Translate = (key: string, values?: Record<string, string | number>) => string;

function formatBytes(value: number): string {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let size = Math.max(0, value), unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
  return `${size < 10 && unit ? size.toFixed(1) : Math.round(size)} ${units[unit]}`;
}

export type PrecheckStepProps = {
  draft: Pick<Draft, "sourceRoot" | "overwriteMode" | "imageDecode" | "export">;
  preflightPending: boolean;
  preflightCancelPending: boolean;
  startPending: boolean;
  preflightProgress: PreflightProgress | null;
  preflight: PreflightSummary | null;
  preflightStale: boolean;
  actionError: string | null;
  profileResourcesReady: boolean;
  resourcesLoading: boolean;
  canStart: boolean;
  enabledModules: string;
  t: Translate;
  onRunPreflight: () => void;
  onCancelPreflight: () => void;
  onStartPipeline: () => void;
};

export function PrecheckStep({
  draft, preflightPending, preflightCancelPending, startPending, preflightProgress, preflight, preflightStale,
  actionError, profileResourcesReady, resourcesLoading, canStart, enabledModules, t, onRunPreflight,
  onCancelPreflight, onStartPipeline,
}: PrecheckStepProps) {
  const running = preflightProgress?.status === "queued" || preflightProgress?.status === "preflighting";
  const blockingImageIssues = Boolean(preflight && draft.imageDecode.invalidImageAction === "block" && preflight.imageIssueCount > 0);
  const preflightButtonDisabled = preflightPending || resourcesLoading || !profileResourcesReady || !draft.sourceRoot.trim();
  const startButtonDisabled = startPending || preflightPending || !canStart || blockingImageIssues;
  const format = preflight?.projection.format ?? draft.export.format;
  const formatText = format === "both" ? t("both") : format === "json" ? t("jsonOnly") : t("flatTxtOnly");

  return <>
    <div className="step-intro">
      <p>{t("precheckPageHelp")}</p>
      <span className={`status ${preflightStale ? "failed" : preflight ? "available" : "pending"}`}>
        {preflightStale ? t("preflightStale") : preflight ? t("preflightReady") : t("preflightRequired")}
      </span>
    </div>
    <div className="step-actions precheck-actions" aria-busy={preflightPending || startPending}>
      <button type="button" disabled={preflightButtonDisabled} aria-busy={preflightPending} onClick={onRunPreflight}>
        {t("preflight")}
      </button>
      {running && <button className="secondary" type="button" disabled={preflightCancelPending} aria-busy={preflightCancelPending} onClick={onCancelPreflight}>
        {t("cancelPreflight")}
      </button>}
      <button type="button" disabled={startButtonDisabled} aria-busy={startPending} onClick={onStartPipeline}>
        {t("startPipeline")}
      </button>
    </div>
    {actionError && <div className="action-feedback" aria-live="assertive"><p role="alert">{actionError}</p></div>}
    {preflightProgress && <section className="preflight-progress" aria-live="polite" data-preflight-progress>
      <div className="preflight-progress-heading"><strong>{t("preflightImageProgress")}</strong><span className={`status status-${preflightProgress.status}`}>{t(`preflightStatus_${preflightProgress.status}`)}</span></div>
      <progress aria-label={t("preflightImageProgress")} max={Math.max(1, preflightProgress.totalFiles)} value={Math.min(preflightProgress.checkedFiles, preflightProgress.totalFiles)} />
      <div className="preflight-progress-count">{preflightProgress.checkedFiles} / {preflightProgress.totalFiles}</div>
      {preflightProgress.error && <p className="resource-warning" role="alert">{preflightProgress.error}</p>}
    </section>}
    {preflight && <>
      <section className="preflight-diagnostics" aria-labelledby="preflight-diagnostics-heading" data-preflight-diagnostics>
        <div className="subsection-heading"><h3 id="preflight-diagnostics-heading">{t("preflightDiagnostics")}</h3></div>
        {preflightStale && <p className="resource-warning" role="status">{t("preflightStaleHelp")}</p>}
        {blockingImageIssues && <p className="action-feedback" role="alert">{t("preflightBlockingImages", { count: preflight.imageIssueCount })}</p>}
        {!blockingImageIssues && preflight.imageIssueCount > 0 && <p className="resource-warning" role="status">{t("preflightSkippedImages", { count: preflight.imageIssueCount })}</p>}
        {preflight.annotationKeyCollisionCount > 0 && <p className="action-feedback" role="alert">{t("preflightKeyCollisions", { count: preflight.annotationKeyCollisionCount })}</p>}
        {!preflightStale && preflight.imageIssueCount === 0 && preflight.annotationKeyCollisionCount === 0 && <p className="hint">{t("preflightNoIssues")}</p>}
      </section>
      <dl className="facts preflight">
        <div><dt>{t("samples")}</dt><dd>{preflight.sampleCount}</dd></div>
        <div><dt>{t("inScope")}</dt><dd>{preflight.inScopeCount}</dd></div>
        <div><dt>{t("existingTxtJson")}</dt><dd>{preflight.nonblankTxtCount} / {preflight.nonblankJsonCount}</dd></div>
        <div><dt>{t("blankTxtJson")}</dt><dd>{preflight.blankTxtCount} / {preflight.blankJsonCount}</dd></div>
        <div><dt>{t("plannedJson")}</dt><dd>{preflight.projection.jsonCreate ?? 0} / {preflight.projection.jsonOverwrite ?? 0} / {preflight.projection.jsonDelete ?? 0}</dd></div>
        <div><dt>{t("plannedTxt")}</dt><dd>{preflight.projection.txtCreate ?? 0} / {preflight.projection.txtOverwrite ?? 0} / {preflight.projection.txtDelete ?? 0}</dd></div>
        <div><dt>{t("retainedSamples")}</dt><dd>{preflight.projection.retainedSamples ?? 0}</dd></div>
        <div><dt>{t("keyCollisions")}</dt><dd>{preflight.annotationKeyCollisionCount}</dd></div>
        <div><dt>{t("imageIssues")}</dt><dd>{preflight.imageIssueCount}</dd></div>
        <div><dt>{t("spaceEstimate")}</dt><dd>{formatBytes(preflight.estimate.incrementalWriteBytes ?? 0)} / {formatBytes(preflight.estimate.backupUpperBoundBytes ?? 0)}</dd></div>
        <div><dt>{t("apiRequests")}</dt><dd>{preflight.api.minRequests ?? 0} / {preflight.api.maxWithBackupRequests ?? 0}</dd></div>
        <div><dt>{t("httpBudget")}</dt><dd>{preflight.api.httpAttemptBudget ?? 0}</dd></div>
        <div><dt>{t("uploadEstimate")}</dt><dd>{formatBytes(preflight.api.estimatedUploadBytes ?? 0)}</dd></div>
        <div><dt>{t("format")}</dt><dd>{formatText}</dd></div>
        <div><dt>{t("enabledModules")}</dt><dd>{enabledModules}</dd></div>
      </dl>
    </>}
  </>;
}
