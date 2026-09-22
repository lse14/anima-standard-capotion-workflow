import { useEffect, useMemo, useState } from "react";

import {
  ApiError,
  confirmCountReview,
  countReviewImageUrl,
  listCountReview,
  updateCountReviewBatch,
  updateCountReviewDecision,
  type CountReviewDecision,
  type CountReviewDecisionSource,
  type CountReviewFilters,
  type CountReviewItem,
  type CountReviewPage,
  type CountValue,
} from "./api";
import { translate, type UiLanguage } from "./i18n";

const PAGE_SIZE_OPTIONS = [20, 50, 100] as const;
const DEFAULT_PAGE_SIZE = 50;
const countValues: readonly CountValue[] = ["solo", "duo", "trio", "group"];
const reasonCodes = [
  "count_source_conflict",
  "original_count_invalid",
  "count_character_lower_bound",
  "count_relationship_lower_bound",
  "count_conflict",
  "wiki_missing",
  "count_observation_mismatch",
  "count_observation_invalid",
  "count_observation_unknown",
] as const;

type SaveState = "saving" | "saved" | "failed" | "conflict";

type CountReviewPanelProps = {
  jobId: string;
  jobStatus: string;
  currentModuleId?: string;
  language: UiLanguage;
  onConfirmed: () => Promise<void>;
};

function isFinalCount(value: string | null): value is CountValue {
  return value !== null && (countValues as readonly string[]).includes(value);
}

function ReviewImage({ src, alt, failedLabel }: { src: string; alt: string; failedLabel: string }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);
  return <figure className={`review-preview ${failed ? "failed" : ""}`}>
    {!failed && <img src={src} alt={alt} loading="lazy" onError={() => setFailed(true)} />}
    {failed && <figcaption>{failedLabel}</figcaption>}
  </figure>;
}

function replaceDecisions(
  page: CountReviewPage,
  updates: ReadonlyArray<{ sampleId: number; decision: CountReviewDecision }>,
): CountReviewPage {
  const bySample = new Map(updates.map((item) => [item.sampleId, item.decision]));
  let pendingCount = page.pendingCount;
  const items = page.items.map((item) => {
    const decision = bySample.get(item.sampleId);
    if (!decision) return item;
    if (item.decision.status === "pending" && decision.status !== "pending") pendingCount -= 1;
    if (item.decision.status !== "pending" && decision.status === "pending") pendingCount += 1;
    return { ...item, decision };
  });
  return { ...page, items, pendingCount: Math.max(0, pendingCount) };
}

export function CountReviewPanel({ jobId, jobStatus, currentModuleId, language, onConfirmed }: CountReviewPanelProps) {
  const t = (key: string, values?: Record<string, string | number>) => translate(language, key, values);
  const [filters, setFilters] = useState<CountReviewFilters>({ status: "pending" });
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [pageCursors, setPageCursors] = useState([0]);
  const [pageIndex, setPageIndex] = useState(0);
  const [page, setPage] = useState<CountReviewPage | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [saveStates, setSaveStates] = useState<Record<number, SaveState>>({});
  const [batchManualCount, setBatchManualCount] = useState<CountValue>("solo");
  const [loading, setLoading] = useState(true);
  const [batchSaving, setBatchSaving] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [checking, setChecking] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [reloadRevision, setReloadRevision] = useState(0);
  const afterSampleId = pageCursors[pageIndex] ?? 0;
  const editable = jobStatus === "reviewing" && currentModuleId === "count_review";
  const itemSaving = Object.values(saveStates).some((state) => state === "saving");
  const confirmBlocker = !editable
    ? t("countReviewStateBlocker")
    : loading || !page
      ? t("countReviewLoading")
      : page.pendingCount > 0
        ? t("countReviewPendingBlocker", { count: page.pendingCount })
        : t("countReviewReady");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setLoadError(null);
    setSelected(new Set());
    void listCountReview(jobId, filters, afterSampleId, pageSize)
      .then((result) => {
        if (cancelled) return;
        setPage(result);
        setSaveStates({});
      })
      .catch((cause) => {
        if (cancelled) return;
        setPage(null);
        setLoadError(cause instanceof Error ? cause.message : t("countReviewLoadFailed"));
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [
    jobId,
    jobStatus,
    currentModuleId,
    afterSampleId,
    filters.status,
    filters.reason,
    filters.classifyCount,
    filters.vlmCount,
    filters.mismatchOnly,
    pageSize,
    reloadRevision,
    language,
  ]);

  const selectedItems = useMemo(
    () => page?.items.filter((item) => selected.has(item.sampleId)) ?? [],
    [page, selected],
  );
  const allPageSelected = Boolean(page?.items.length) && page!.items.every((item) => selected.has(item.sampleId));
  const canBatchClassify = selectedItems.length > 0 && selectedItems.every((item) => item.classify.count !== null);
  const canBatchVlm = selectedItems.length > 0 && selectedItems.every((item) => isFinalCount(item.vlm.count));

  const resetPaging = () => {
    setPageCursors([0]);
    setPageIndex(0);
    setSelected(new Set());
  };
  const patchFilters = (patch: Partial<CountReviewFilters>) => {
    setFilters((current) => ({ ...current, ...patch }));
    setNotice(null);
    resetPaging();
  };
  const markSaveState = (sampleIds: number[], state: SaveState) => {
    setSaveStates((current) => {
      const next = { ...current };
      for (const sampleId of sampleIds) next[sampleId] = state;
      return next;
    });
  };
  const handleConflict = (sampleIds: number[]) => {
    markSaveState(sampleIds, "conflict");
    setNotice(t("countReviewConflict"));
    setReloadRevision((value) => value + 1);
  };

  const saveOne = async (item: CountReviewItem, source: CountReviewDecisionSource, count?: CountValue) => {
    if (!editable || batchSaving) return;
    markSaveState([item.sampleId], "saving");
    setNotice(null);
    try {
      const result = await updateCountReviewDecision(jobId, {
        sampleId: item.sampleId,
        expectedVersion: item.decision.version,
        source,
        ...(count ? { count } : {}),
      });
      setPage((current) => current ? replaceDecisions(current, [result]) : current);
      markSaveState([item.sampleId], "saved");
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 409) {
        handleConflict([item.sampleId]);
        return;
      }
      markSaveState([item.sampleId], "failed");
      setNotice(cause instanceof Error ? cause.message : t("countReviewSaveFailed"));
    }
  };

  const saveBatch = async (source: CountReviewDecisionSource, count?: CountValue) => {
    if (!editable || selectedItems.length === 0 || itemSaving) return;
    const sampleIds = selectedItems.map((item) => item.sampleId);
    setBatchSaving(true);
    markSaveState(sampleIds, "saving");
    setNotice(null);
    try {
      const result = await updateCountReviewBatch(jobId, selectedItems.map((item) => ({
        sampleId: item.sampleId,
        expectedVersion: item.decision.version,
        source,
        ...(count ? { count } : {}),
      })));
      setPage((current) => current ? replaceDecisions(current, result.items) : current);
      markSaveState(sampleIds, "saved");
      setSelected(new Set());
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 409) {
        handleConflict(sampleIds);
        return;
      }
      markSaveState(sampleIds, "failed");
      setNotice(cause instanceof Error ? cause.message : t("countReviewBatchFailed"));
    } finally {
      setBatchSaving(false);
    }
  };

  const confirmAndContinue = async () => {
    if (!editable || !page || page.pendingCount !== 0 || confirming) return;
    setChecking(true);
    setNotice(null);
    try {
      const authority = await listCountReview(jobId, {}, 0, 1);
      if (authority.pendingCount !== 0) {
        setPage((current) => current ? { ...current, pendingCount: authority.pendingCount } : current);
        setFilters({ status: "pending" });
        resetPaging();
        setNotice(t("countReviewPendingBlocker", { count: authority.pendingCount }));
        return;
      }
      if (!window.confirm(t("countReviewConfirmPrompt"))) return;
      setConfirming(true);
      await confirmCountReview(jobId);
      setNotice(t("countReviewContinuing"));
      await onConfirmed();
    } catch (cause) {
      setNotice(cause instanceof Error ? cause.message : t("countReviewConfirmFailed"));
    } finally {
      setConfirming(false);
      setChecking(false);
    }
  };

  const countLabel = (value: string | null) => t(`countValue_${value ?? "unavailable"}`);
  const codeLabel = (code: string) => t(`countReason_${code}`);
  const toggleAll = () => {
    if (!page) return;
    setSelected(allPageSelected ? new Set() : new Set(page.items.map((item) => item.sampleId)));
  };

  return <section className="count-review-panel" aria-busy={loading || batchSaving || confirming || checking}>
    <div className="review-workbench-heading">
      <div><p className="eyebrow">{t("workspaceCountReview")}</p><h2>{t("workspaceCountReview")}</h2><p>{t("countReviewWorkbenchHelp")}</p></div>
      <span className={`status ${editable ? "reviewing" : "idle"}`}>{t(editable ? "countReviewEditable" : "countReviewReadOnly")}</span>
    </div>
    <div className="review-summary">
      <dl>
        <div><dt>{t("countReviewTargets")}</dt><dd>{page?.targetCount ?? 0}</dd></div>
        <div><dt>{t("countReviewPending")}</dt><dd>{page?.pendingCount ?? 0}</dd></div>
        <div><dt>{t("countReviewSelected")}</dt><dd>{selectedItems.length}</dd></div>
      </dl>
    </div>

    <div className={`review-command ${page?.pendingCount === 0 && editable ? "ready" : "blocked"}`}>
      <div><strong>{confirmBlocker}</strong><small>{checking ? t("countReviewChecking") : t("countReviewConfirmPrompt")}</small></div>
      <button type="button" className="confirm-review" disabled={!editable || loading || checking || confirming || batchSaving || itemSaving || !page || page.pendingCount !== 0} onClick={() => void confirmAndContinue()}>{confirming || checking ? t(confirming ? "countReviewConfirming" : "countReviewChecking") : t("countReviewConfirmContinue")}</button>
    </div>

    <div className="review-filters" aria-label={t("countReviewFilters")}>
      <label>{t("countReviewStatus")}
        <select value={filters.status ?? ""} onChange={(event) => patchFilters({ status: (event.target.value || undefined) as CountReviewFilters["status"] })}>
          <option value="">{t("filterAll")}</option>
          <option value="pending">{t("countDecision_pending")}</option>
          <option value="auto_resolved">{t("countDecision_auto_resolved")}</option>
          <option value="manual_resolved">{t("countDecision_manual_resolved")}</option>
        </select>
      </label>
      <label>{t("countReviewReason")}
        <select value={filters.reason ?? ""} onChange={(event) => patchFilters({ reason: event.target.value || undefined })}>
          <option value="">{t("filterAll")}</option>
          {reasonCodes.map((code) => <option key={code} value={code}>{codeLabel(code)}</option>)}
        </select>
      </label>
      <label>{t("countReviewClassify")}
        <select value={filters.classifyCount ?? ""} onChange={(event) => patchFilters({ classifyCount: (event.target.value || undefined) as CountReviewFilters["classifyCount"] })}>
          <option value="">{t("filterAll")}</option>
          {countValues.map((value) => <option key={value} value={value}>{countLabel(value)}</option>)}
          <option value="unavailable">{countLabel(null)}</option>
        </select>
      </label>
      <label>{t("countReviewVlm")}
        <select value={filters.vlmCount ?? ""} onChange={(event) => patchFilters({ vlmCount: (event.target.value || undefined) as CountReviewFilters["vlmCount"] })}>
          <option value="">{t("filterAll")}</option>
          {countValues.map((value) => <option key={value} value={value}>{countLabel(value)}</option>)}
          <option value="unknown">{countLabel("unknown")}</option>
          <option value="unavailable">{countLabel(null)}</option>
        </select>
      </label>
      <label className="checkbox review-mismatch"><input type="checkbox" checked={filters.mismatchOnly ?? false} onChange={(event) => patchFilters({ mismatchOnly: event.target.checked || undefined })} />{t("countReviewMismatchOnly")}</label>
      <label>{t("countReviewPageSize")}
        <select value={pageSize} onChange={(event) => {
          setPageSize(Number(event.target.value));
          resetPaging();
        }}>
          {PAGE_SIZE_OPTIONS.map((size) => <option key={size} value={size}>{size}</option>)}
        </select>
      </label>
    </div>

    {notice && <p className="review-notice" role="status">{notice}</p>}
    {loadError && <div className="review-error"><p role="alert">{loadError}</p><button className="secondary" type="button" onClick={() => setReloadRevision((value) => value + 1)}>{t("retryCountReview")}</button></div>}

    <div className="review-batch-bar">
      <label className="checkbox"><input type="checkbox" disabled={!editable || loading || !page?.items.length} checked={allPageSelected} onChange={toggleAll} />{t("countReviewSelectPage")}</label>
      <div className="review-batch-actions">
        <button type="button" className="secondary" disabled={!editable || batchSaving || itemSaving || !canBatchClassify} onClick={() => void saveBatch("classify")}>{t("countReviewUseClassify")}</button>
        <button type="button" className="secondary" disabled={!editable || batchSaving || itemSaving || !canBatchVlm} onClick={() => void saveBatch("vlm")}>{t("countReviewUseVlm")}</button>
        <div className="count-segments" role="group" aria-label={t("countReviewManualCount")}>
          {countValues.map((value) => <button key={value} type="button" className={batchManualCount === value ? "selected" : ""} disabled={!editable || batchSaving || itemSaving} onClick={() => setBatchManualCount(value)}>{countLabel(value)}</button>)}
        </div>
        <button type="button" disabled={!editable || batchSaving || itemSaving || selectedItems.length === 0} onClick={() => void saveBatch("manual", batchManualCount)}>{t("countReviewApplyManual")}</button>
      </div>
    </div>

    {loading && <p className="review-empty">{t("countReviewLoading")}</p>}
    {!loading && !loadError && page?.items.length === 0 && <p className="review-empty">{t("countReviewEmpty")}</p>}
    {!loading && page && <div className="review-list">
      {page.items.map((item) => {
        const state = saveStates[item.sampleId];
        const selectedSource = item.decision.selectedSource;
        return <article className="review-item" key={item.sampleId}>
          <ReviewImage
            src={countReviewImageUrl(jobId, item.sampleId)}
            alt={t("countReviewImageAlt", { path: item.relativeImagePath })}
            failedLabel={t("countReviewPreviewFailed")}
          />
          <div className="review-item-body">
            <header>
              <label className="checkbox"><input type="checkbox" disabled={!editable || batchSaving || state === "saving"} checked={selected.has(item.sampleId)} onChange={(event) => setSelected((current) => { const next = new Set(current); if (event.target.checked) next.add(item.sampleId); else next.delete(item.sampleId); return next; })} /><span>#{item.sampleId}</span></label>
              <code title={item.relativeImagePath}>{item.relativeImagePath}</code>
              <span className={`status ${item.decision.status}`}>{t(`countDecision_${item.decision.status}`)}</span>
            </header>
            <div className="review-evidence">
              <section>
                <span>{t("countReviewClassify")}</span>
                <strong>{countLabel(item.classify.count)}</strong>
                <small>{item.classify.warningCodes.length ? item.classify.warningCodes.map(codeLabel).join(" / ") : t("countReviewNoWarnings")}</small>
              </section>
              <section>
                <span>{t("countReviewVlm")}</span>
                <strong>{countLabel(item.vlm.count)}</strong>
                <small>{t(`countLayout_${item.vlm.layout ?? "unavailable"}`)}; {t("countReviewRepeated")}: {item.vlm.sameCharacterRepeated === null ? t("notAvailable") : item.vlm.sameCharacterRepeated ? t("yes") : t("no")}</small>
                {item.vlm.notRequestedReason && <small>{item.vlm.notRequestedReason}</small>}
              </section>
              <section>
                <span>{t("countReviewFinal")}</span>
                <strong>{countLabel(item.decision.finalCount)}</strong>
                <small>{selectedSource ? t(`countSource_${selectedSource}`) : t("notAvailable")}; v{item.decision.version}</small>
              </section>
            </div>
            {item.decision.reviewReasons.length > 0 && <div className="review-reasons" aria-label={t("countReviewReason")}>{item.decision.reviewReasons.map((code) => <span key={code}>{codeLabel(code)}</span>)}</div>}
            <div className="review-actions">
              <button type="button" className={`secondary ${selectedSource === "classify" ? "selected-action" : ""}`} disabled={!editable || batchSaving || state === "saving" || item.classify.count === null} onClick={() => void saveOne(item, "classify")}>{t("countReviewUseClassify")}</button>
              <button type="button" className={`secondary ${selectedSource === "vlm" ? "selected-action" : ""}`} disabled={!editable || batchSaving || state === "saving" || !isFinalCount(item.vlm.count)} onClick={() => void saveOne(item, "vlm")}>{t("countReviewUseVlm")}</button>
              <div className="count-segments" role="group" aria-label={t("countReviewManualCount")}>
                {countValues.map((value) => <button key={value} type="button" className={selectedSource === "manual" && item.decision.finalCount === value ? "selected" : ""} disabled={!editable || batchSaving || state === "saving"} onClick={() => void saveOne(item, "manual", value)}>{countLabel(value)}</button>)}
              </div>
              <small className={`save-state ${state ?? ""}`}>{state ? t(`countSave_${state}`) : item.decision.resolvedAt ? t("countSave_saved") : ""}</small>
            </div>
          </div>
        </article>;
      })}
    </div>}

    <div className="review-footer">
      <div className="review-pagination">
        <button type="button" className="secondary" disabled={loading || pageIndex === 0} onClick={() => setPageIndex(0)}>{t("firstPage")}</button>
        <button type="button" className="secondary" disabled={loading || pageIndex === 0} onClick={() => setPageIndex((value) => Math.max(0, value - 1))}>{t("previousPage")}</button>
        <span>{t("pageNumber", { page: pageIndex + 1 })}</span>
        <button type="button" className="secondary" disabled={loading || page?.nextAfterSampleId === null || page?.nextAfterSampleId === undefined} onClick={() => {
          if (page?.nextAfterSampleId === null || page?.nextAfterSampleId === undefined) return;
          const nextCursor = page.nextAfterSampleId;
          setPageCursors((current) => [...current.slice(0, pageIndex + 1), nextCursor]);
          setPageIndex((value) => value + 1);
        }}>{t("nextPage")}</button>
      </div>
    </div>
  </section>;
}
