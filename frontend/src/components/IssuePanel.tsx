import { useEffect, useState } from "react";

export type IssuePanelIssue = {
  issueId: string;
  sampleId: number;
  relativeImagePath?: string;
  moduleId: string;
  code: string;
  message: string;
  retriable: boolean;
  attempt: number;
  repairStartModule: string | null;
};

export type IssuePanelProps = {
  issues: IssuePanelIssue[];
  retriableCount: number;
  cursor: { sampleId: number; issueId: string | null };
  nextCursor: { sampleId: number; issueId: string | null };
  labels: {
    issues: string;
    shownRetriable: (shown: number, retriable: number) => string;
    attempt: (attempt: number) => string;
    retryFrom: (module: string) => string;
    notRetriable: string;
    firstPage: string;
    nextPage: string;
    restoreOriginal: string;
    reprocess: string;
    nlRetry: string;
    nlBatchRetry: string;
    selectNlPage: string;
    selectedNlCount: (selected: number) => string;
    nlWrite: string;
    nlWritePlaceholder: string;
    workbenchTitle: string;
    workbenchHelp: string;
    moduleFilter: string;
    filterAll: string;
    noIssues: string;
    samplePath: string;
    retriable: string;
  };
  canRestore: boolean;
  canReprocess: boolean;
  pendingActions: ReadonlySet<string>;
  onFirstPage: () => void;
  onNextPage: () => void;
  onRestore: () => void;
  onReprocess: () => void;
  canManualNl: boolean;
  onManualNlRetry: (issue: IssuePanelIssue) => void;
  onManualNlBatchRetry: (issues: IssuePanelIssue[]) => void;
  onManualNlWrite: (issue: IssuePanelIssue, nl: string) => void;
};

export function IssuePanel({
  issues, retriableCount, cursor, nextCursor, labels, canRestore, canReprocess, pendingActions, onFirstPage, onNextPage, onRestore, onReprocess,
  canManualNl, onManualNlRetry, onManualNlBatchRetry, onManualNlWrite,
}: IssuePanelProps) {
  const atFirstPage = cursor.sampleId === 0 && cursor.issueId === null;
  const atLastPage = issues.length < 1 || (nextCursor.sampleId === cursor.sampleId && nextCursor.issueId === cursor.issueId);
  const restoring = pendingActions.has("restore");
  const repairing = pendingActions.has("repair");
  const manualNlRetrying = pendingActions.has("nl_manual_retry");
  const manualNlBatchRetrying = pendingActions.has("nl_manual_retry_batch");
  const manualNlWriting = pendingActions.has("nl_manual_write");
  const [nlDrafts, setNlDrafts] = useState<Record<string, string>>({});
  const [selectedNlIssueIds, setSelectedNlIssueIds] = useState<Set<string>>(new Set());
  const [moduleFilter, setModuleFilter] = useState("");
  const modules = Array.from(new Set(issues.map((item) => item.moduleId))).sort();
  const visibleIssues = moduleFilter ? issues.filter((item) => item.moduleId === moduleFilter) : issues;
  const selectableNlIssues = canManualNl ? visibleIssues.filter((item) => item.moduleId === "nl" && !item.retriable) : [];
  const selectedNlIssues = selectableNlIssues.filter((item) => selectedNlIssueIds.has(item.issueId));
  const allNlPageSelected = selectableNlIssues.length > 0 && selectedNlIssues.length === selectableNlIssues.length;

  useEffect(() => {
    setSelectedNlIssueIds(new Set());
  }, [cursor.sampleId, cursor.issueId]);

  const toggleNlPage = () => {
    setSelectedNlIssueIds(allNlPageSelected ? new Set() : new Set(selectableNlIssues.map((item) => item.issueId)));
  };

  return <section className="issues-panel failure-issues" aria-busy={restoring || repairing || manualNlRetrying || manualNlBatchRetrying || manualNlWriting}>
    <div className="panel-heading">
      <div><p className="eyebrow">{labels.issues}</p><h2>{labels.workbenchTitle}</h2><p className="panel-help">{labels.workbenchHelp}</p></div>
      <span>{labels.shownRetriable(issues.length, retriableCount)}</span>
    </div>
    <div className="failure-toolbar">
      <label>{labels.moduleFilter}<select value={moduleFilter} onChange={(event) => setModuleFilter(event.target.value)}><option value="">{labels.filterAll}</option>{modules.map((module) => <option key={module} value={module}>{module}</option>)}</select></label>
      <dl className="failure-facts"><div><dt>{labels.issues}</dt><dd>{visibleIssues.length}</dd></div><div><dt>{labels.retriable}</dt><dd>{retriableCount}</dd></div></dl>
    </div>
    {selectableNlIssues.length > 0 && <div className="issue-batch-bar">
      <label className="checkbox"><input type="checkbox" checked={allNlPageSelected} disabled={manualNlRetrying || manualNlBatchRetrying || manualNlWriting} onChange={toggleNlPage} />{labels.selectNlPage}</label>
      <span>{labels.selectedNlCount(selectedNlIssues.length)}</span>
      <button className="secondary" type="button" disabled={selectedNlIssues.length === 0 || manualNlRetrying || manualNlBatchRetrying || manualNlWriting} aria-busy={manualNlBatchRetrying} onClick={() => onManualNlBatchRetry(selectedNlIssues)}>{labels.nlBatchRetry}</button>
    </div>}
    <div className="step-actions issue-pagination">
      <button className="secondary" type="button" disabled={atFirstPage} onClick={onFirstPage}>{labels.firstPage}</button>
      <button className="secondary" type="button" disabled={atLastPage} onClick={onNextPage}>{labels.nextPage}</button>
    </div>
    {visibleIssues.length === 0 && <p className="review-empty">{labels.noIssues}</p>}
    <ul className="issue-list">{visibleIssues.map((item) => {
      const isNlManual = canManualNl && item.moduleId === "nl" && !item.retriable;
      const draft = nlDrafts[item.issueId] ?? "";
      return <li key={item.issueId}>
        <div className="issue-row-selection">
          {isNlManual && <label className="checkbox"><input type="checkbox" checked={selectedNlIssueIds.has(item.issueId)} disabled={manualNlRetrying || manualNlBatchRetrying || manualNlWriting} onChange={(event) => setSelectedNlIssueIds((current) => { const next = new Set(current); if (event.target.checked) next.add(item.issueId); else next.delete(item.issueId); return next; })} /><span>#{item.sampleId}</span></label>}
          <div className="issue-primary"><code>{item.moduleId}:{item.code}</code>{item.relativeImagePath && <span className="issue-path"><strong>{labels.samplePath}</strong> {item.relativeImagePath}</span>}<span>{item.message}</span></div>
        </div>
        <small>{labels.attempt(item.attempt)}; {item.retriable ? labels.retryFrom(item.repairStartModule ?? "-") : labels.notRetriable}</small>
        {isNlManual && <div className="issue-manual-nl">
          <button className="secondary" type="button" disabled={manualNlRetrying || manualNlBatchRetrying || manualNlWriting} onClick={() => onManualNlRetry(item)}>{labels.nlRetry}</button>
          <input value={draft} placeholder={labels.nlWritePlaceholder} disabled={manualNlRetrying || manualNlBatchRetrying || manualNlWriting} onChange={(event) => setNlDrafts((current) => ({ ...current, [item.issueId]: event.target.value }))} />
          <button className="secondary" type="button" disabled={!draft.trim() || manualNlRetrying || manualNlBatchRetrying || manualNlWriting} onClick={() => onManualNlWrite(item, draft)}>{labels.nlWrite}</button>
        </div>}
      </li>;
    })}</ul>
    <div className="step-actions">
      <button className="warning-action" type="button" disabled={!canRestore || restoring} aria-busy={restoring} onClick={onRestore}>{labels.restoreOriginal}</button>
      <button className="secondary" type="button" disabled={!canReprocess || repairing} aria-busy={repairing} onClick={onReprocess}>{labels.reprocess}</button>
    </div>
  </section>;
}
