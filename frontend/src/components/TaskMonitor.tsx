export type TaskMonitorSnapshot = {
  status: string;
  currentModuleId: string | null;
  apiRuntime?: { revision: number; providerLabel: string; endpoint: string; model: string; jsonMode: string } | null;
  apiRuntimePending?: { revision: number; providerLabel: string; endpoint: string; model: string; jsonMode: string } | null;
  ocrRuntime: {
    availability: "pending" | "available" | "unavailable";
    runtimeId: "ocr-paddle" | "ocr-paddle-gpu" | null;
    gpuName: string | null;
    totalVramBytes: number | null;
    requestedDevice: "auto" | "cuda" | "cpu" | null;
    observedDevice: "cpu" | "cuda" | null;
    recommended: { textDetLimitSideLen: number; textBatchSize: number } | null;
    effective: { textDetLimitSideLen: number; textBatchSize: number } | null;
    startupReason: "gpu_runtime_unavailable" | "binding_invalid" | null;
  } | null;
};

export type TaskMonitorModule = {
  moduleId: string;
  label: string;
  status: string;
  statusLabel: string;
  completed: number;
  failed: number;
  skipped: number;
  total: number;
  issueCount: number;
  isCurrent: boolean;
};

export type TaskMonitorProps = {
  snapshot: TaskMonitorSnapshot | null;
  isRepairTask: boolean;
  hasRepairChildren: boolean;
  loading: boolean;
  error: string | null;
  statusLabel: string;
  currentModuleLabel: string;
  currentBatchLabel: string;
  rawE621ConvertedMessage: string | null;
  captionGpuFallbackMessage: string | null;
  modules: TaskMonitorModule[];
  labels: {
    taskOverview: string; taskProgress: string; currentModule: string; currentBatch: string;
    waitingForHumanReview: string;
    taskActions: string; activeModule: string; pauseTask: string; resumeTask: string; terminateTask: string; recoverTask: string;
    discardTask: string; deleteCompletedHistory: string; repairDeleteBlocked: string; repairTask: string; backToParentTask: string;
    additionalAttempts: string; addBudget: string; pendingApiDecisions: string; confirmUnknown: string;
    issues: string; noTask: string; loadingTask: string; retryTask: string;
    ocrRuntime: string; ocrAvailability: string; ocrGpu: string; ocrRequestedDevice: string; ocrObservedDevice: string;
    ocrRecommended: string; ocrEffective: string; ocrStartupReason: string;
    apiSettings: string; apiRevision: string; apiProvider: string; apiModel: string; apiJsonMode: string; apiPending: string;
  };
  canDiscard: boolean;
  canDeleteHistory: boolean;
  budget: string;
  pendingApiDecisions: number;
  nlAwaitsDecision: boolean;
  pendingActions: ReadonlySet<string>;
  onPause: () => void;
  onResume: () => void;
  onTerminate: () => void;
  onRecover: () => void;
  onDiscard: () => void;
  onDeleteHistory: () => void;
  onBackToParent: () => void;
  onBudgetChange: (value: string) => void;
  onAddBudget: () => void;
  onConfirmUnknown: () => void;
  onRetry: () => void;
};

function formatGiB(bytes: number): string {
  return `${(bytes / (1024 ** 3)).toFixed(bytes % (1024 ** 3) ? 1 : 0)} GiB`;
}

export function TaskMonitor({
  snapshot, isRepairTask, hasRepairChildren, loading, error, statusLabel, currentModuleLabel, currentBatchLabel, rawE621ConvertedMessage, captionGpuFallbackMessage, modules, labels,
  canDiscard, canDeleteHistory, budget, pendingApiDecisions, nlAwaitsDecision, pendingActions, onPause, onResume, onTerminate, onRecover, onDiscard, onDeleteHistory, onBackToParent,
  onBudgetChange, onAddBudget, onConfirmUnknown, onRetry,
}: TaskMonitorProps) {
  const isPending = (action: string) => pendingActions.has(action);
  const canRecover = snapshot?.status === "interrupted" || snapshot?.status === "cancelled_recoverable";
  const canTerminate = Boolean(snapshot && ["preparing_workspace", "running", "paused", "reviewing", "exporting"].includes(snapshot.status));

  return <aside className="task-monitor" aria-busy={loading || pendingActions.size > 0}>
    <div className="monitor-heading">
      <div><p className="eyebrow">{labels.taskOverview}</p><h2>{isRepairTask ? labels.repairTask : labels.taskProgress}</h2></div>
      <span className={`status ${snapshot?.status ?? "idle"}`} aria-live="polite" aria-atomic="true">{statusLabel}</span>
      {isRepairTask && <button className="secondary" type="button" onClick={onBackToParent}>{labels.backToParentTask}</button>}
    </div>
    {loading && <p className="monitor-state" role="status">{labels.loadingTask}</p>}
    {error && <div className="monitor-error"><p role="alert">{error}</p><button className="secondary" type="button" onClick={onRetry}>{labels.retryTask}</button></div>}
    {snapshot ? <>
      <dl className="facts">
        <div><dt>{labels.currentModule}</dt><dd>{currentModuleLabel}</dd></div>
        <div><dt>{labels.currentBatch}</dt><dd>{currentBatchLabel}</dd></div>
      </dl>
      {snapshot.apiRuntime ? <section className="ocr-runtime" aria-label={labels.apiSettings}>
        <h3>{labels.apiSettings}</h3>
        <dl className="facts">
          <div><dt>{labels.apiProvider}</dt><dd>{snapshot.apiRuntime.providerLabel}</dd></div>
          <div><dt>{labels.apiModel}</dt><dd>{snapshot.apiRuntime.model}</dd></div>
          <div><dt>{labels.apiRevision}</dt><dd>{snapshot.apiRuntime.revision}</dd></div>
          <div><dt>{labels.apiJsonMode}</dt><dd>{snapshot.apiRuntime.jsonMode}</dd></div>
        </dl>
        {snapshot.apiRuntimePending ? <p className="hint" role="status">{labels.apiPending}: rev {snapshot.apiRuntimePending.revision}</p> : null}
      </section> : null}
      {snapshot.ocrRuntime ? <section className="ocr-runtime" aria-label={labels.ocrRuntime}>
        <h3>{labels.ocrRuntime}</h3>
        <dl className="facts">
          <div><dt>{labels.ocrAvailability}</dt><dd>{snapshot.ocrRuntime.availability}{snapshot.ocrRuntime.runtimeId ? ` / ${snapshot.ocrRuntime.runtimeId}` : ""}</dd></div>
          {snapshot.ocrRuntime.gpuName ? <div><dt>{labels.ocrGpu}</dt><dd>{snapshot.ocrRuntime.gpuName}{snapshot.ocrRuntime.totalVramBytes === null ? "" : ` ${formatGiB(snapshot.ocrRuntime.totalVramBytes)}`}</dd></div> : null}
          <div><dt>{labels.ocrRequestedDevice}</dt><dd>{snapshot.ocrRuntime.requestedDevice ?? "-"}</dd></div>
          <div><dt>{labels.ocrObservedDevice}</dt><dd>{snapshot.ocrRuntime.observedDevice ?? "-"}</dd></div>
          {snapshot.ocrRuntime.recommended ? <div><dt>{labels.ocrRecommended}</dt><dd>{snapshot.ocrRuntime.recommended.textDetLimitSideLen} / {snapshot.ocrRuntime.recommended.textBatchSize}</dd></div> : null}
          {snapshot.ocrRuntime.effective ? <div><dt>{labels.ocrEffective}</dt><dd>{snapshot.ocrRuntime.effective.textDetLimitSideLen} / {snapshot.ocrRuntime.effective.textBatchSize}</dd></div> : null}
          {snapshot.ocrRuntime.startupReason ? <div><dt>{labels.ocrStartupReason}</dt><dd>{snapshot.ocrRuntime.startupReason}</dd></div> : null}
        </dl>
      </section> : null}
      {captionGpuFallbackMessage ? <p className="hint">{captionGpuFallbackMessage}</p> : null}
      {rawE621ConvertedMessage ? <p className="hint">{rawE621ConvertedMessage}</p> : null}
      <div className="module-progress">{modules.map((module) => {
        const settled = module.completed + module.failed + module.skipped;
        const percentage = module.total ? Math.min(100, Math.round(settled / module.total * 100)) : 0;
        const awaitingHumanReview = module.isCurrent && snapshot.status === "reviewing";
        return <div className={`module-row ${module.isCurrent ? "current" : ""}`} key={module.moduleId}>
          <div className="module-row-heading"><div><strong>{module.label}</strong><span className={`status ${awaitingHumanReview ? "reviewing" : module.status}`}>{awaitingHumanReview ? labels.waitingForHumanReview : module.statusLabel}</span></div></div>
          <progress value={settled} max={Math.max(module.total, 1)} />
          <small>{awaitingHumanReview ? labels.waitingForHumanReview : `${settled} / ${module.total} (${percentage}%)${module.issueCount ? ` - ${module.issueCount} ${labels.issues}` : ""}`}</small>
        </div>;
      })}</div>
      <section className="task-actions">
        <h3>{labels.taskActions}</h3>
        <p className="task-action-context">{labels.activeModule}: {currentModuleLabel}</p>
        <div className="action-grid">
          {snapshot.status === "paused"
            ? <button className="secondary" type="button" disabled={isPending("resume")} aria-busy={isPending("resume")} onClick={onResume}>{labels.resumeTask}</button>
            : <button className="secondary" type="button" disabled={!(["running", "exporting"].includes(snapshot.status)) || isPending("pause")} aria-busy={isPending("pause")} onClick={onPause}>{labels.pauseTask}</button>}
          <button className="secondary" type="button" disabled={!canTerminate || isPending("terminate")} aria-busy={isPending("terminate")} onClick={onTerminate}>{labels.terminateTask}</button>
          <button className="secondary" type="button" disabled={!canRecover || isPending("recover")} aria-busy={isPending("recover")} onClick={onRecover}>{labels.recoverTask}</button>
        </div>
        <div className="action-grid">
          {snapshot.status === "succeeded"
            ? <button className="danger-action" type="button" disabled={!canDeleteHistory || isPending("delete_history")} aria-busy={isPending("delete_history")} onClick={onDeleteHistory}>{labels.deleteCompletedHistory}</button>
            : <button className="danger-action" type="button" disabled={!canDiscard || isPending("discard")} aria-busy={isPending("discard")} onClick={onDiscard}>{labels.discardTask}</button>}
        </div>
      </section>
      {hasRepairChildren && <p className="hint">{labels.repairDeleteBlocked}</p>}
      {pendingApiDecisions > 0 && <section className="task-actions">
        <div className="inline-control">
          <input inputMode="numeric" value={budget} onChange={(event) => onBudgetChange(event.target.value)} aria-label={labels.additionalAttempts} />
          <button type="button" disabled={!/^\d+$/.test(budget) || Number(budget) < 1 || isPending("nl_budget")} aria-busy={isPending("nl_budget")} onClick={onAddBudget}>{labels.addBudget}</button>
        </div>
        <div className="inline-control">
          <span>{labels.pendingApiDecisions}: {pendingApiDecisions}</span>
          <button className="warning-action" type="button" disabled={!nlAwaitsDecision || isPending("nl_confirm_outcomes")} aria-busy={isPending("nl_confirm_outcomes")} onClick={onConfirmUnknown}>{labels.confirmUnknown}</button>
        </div>
      </section>}
    </> : !loading && !error && <p className="empty-state">{labels.noTask}</p>}
  </aside>;
}
