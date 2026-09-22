import {
  DEFAULT_JOB_ID,
  expect,
  failRoute,
  holdRoute,
  makeSnapshot,
  mutationsFor,
  openApp,
  openPrecheck,
  openSetup,
  setJobSnapshot,
  test,
} from "./mockApi";

async function openTrackedJob(page: Parameters<typeof openApp>[0], status: string) {
  await openApp(page, { language: "en" });
  await page.getByLabel("Recent tasks").selectOption(DEFAULT_JOB_ID);
  await expect(page.locator(".task-monitor > .monitor-heading > .status")).toHaveText(status);
}

async function selectTask(page: Parameters<typeof openApp>[0], jobId: string): Promise<void> {
  await page.getByLabel("Recent tasks").selectOption(jobId);
}

type SnapshotFetchProbe = { active: number; maxActive: number; started: number };

async function installSnapshotFetchProbe(page: Parameters<typeof openApp>[0], jobId: string): Promise<void> {
  await page.addInitScript((path) => {
    const probe: SnapshotFetchProbe = { active: 0, maxActive: 0, started: 0 };
    const scope = window as typeof window & { __snapshotFetchProbe?: SnapshotFetchProbe };
    const originalFetch = window.fetch.bind(window);
    scope.__snapshotFetchProbe = probe;
    window.fetch = async (...args) => {
      const input = args[0];
      const href = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (new URL(href, window.location.href).pathname !== path) return originalFetch(...args);
      probe.active += 1;
      probe.started += 1;
      probe.maxActive = Math.max(probe.maxActive, probe.active);
      try {
        return await originalFetch(...args);
      } finally {
        probe.active -= 1;
      }
    };
  }, `/api/jobs/${encodeURIComponent(jobId)}`);
}

async function snapshotFetchProbe(page: Parameters<typeof openApp>[0]): Promise<SnapshotFetchProbe> {
  return page.evaluate(() => {
    const probe = (window as typeof window & { __snapshotFetchProbe?: SnapshotFetchProbe }).__snapshotFetchProbe;
    if (!probe) throw new Error("snapshot fetch probe is not installed");
    return { ...probe };
  });
}

const lifecycleButtonNames = ["Terminate task", "Recover task"] as const;

const lifecycleStateMatrix = [
  { status: "running", statusLabel: "running", enabled: ["Terminate task"] },
  { status: "paused", statusLabel: "paused", enabled: ["Terminate task"] },
  { status: "preparing_workspace", statusLabel: "preparing workspace", enabled: ["Terminate task"] },
  { status: "reviewing", statusLabel: "reviewing", enabled: ["Terminate task"] },
  { status: "exporting", statusLabel: "exporting", enabled: ["Terminate task"] },
  { status: "interrupted", statusLabel: "interrupted", enabled: ["Recover task"] },
  { status: "cancelled_recoverable", statusLabel: "cancelled, recoverable", enabled: ["Recover task"] },
  { status: "committing", statusLabel: "committing", enabled: [] },
  { status: "cancelling", statusLabel: "cancelling", enabled: [] },
  { status: "succeeded", statusLabel: "succeeded", enabled: [] },
  { status: "discarded", statusLabel: "status_discarded", enabled: [] },
] as const;

test.describe("task status and issue characterization", () => {
  test("rejects obsolete JobConfig snapshot fixtures", () => {
    expect(() => makeSnapshot({ schemaVersion: 8 })).toThrow("only supports JobConfig schema v10");
  });

  test("keeps the task monitor idle when no recent task exists", async ({ page, api }) => {
    const staleJobId = "stale-task-from-previous-session";
    failRoute(api, `GET /api/jobs/${staleJobId}`, "job does not exist", 404);
    await installSnapshotFetchProbe(page, staleJobId);
    await openApp(page, { jobId: staleJobId, language: "en" });

    await expect(page.getByLabel("Task ID")).toHaveCount(0);
    await expect(page.getByLabel("Recent tasks")).toHaveValue("");
    await expect(page.locator(".empty-state")).toBeVisible();
    await expect(page.locator(".monitor-error")).toHaveCount(0);
    expect((await snapshotFetchProbe(page)).started).toBe(0);
  });

  test("loads task switches without overlapping a repeated snapshot target", async ({ page, api }) => {
    const nextJobId = "job-selected-while-previous-pending";
    setJobSnapshot(api, makeSnapshot({ jobId: DEFAULT_JOB_ID, status: "running", currentModuleId: "nl" }));
    setJobSnapshot(api, makeSnapshot({ jobId: nextJobId, status: "failed", currentModuleId: "ocr" }));
    await installSnapshotFetchProbe(page, DEFAULT_JOB_ID);
    const releasePrevious = holdRoute(api, `GET /api/jobs/${DEFAULT_JOB_ID}`);
    try {
      await openApp(page, { jobId: DEFAULT_JOB_ID, language: "en" });
      await expect.poll(async () => (await snapshotFetchProbe(page)).started).toBe(1);
      await selectTask(page, nextJobId);
      await expect(page.locator(".task-monitor > .monitor-heading > .status")).toHaveText("failed", { timeout: 2000 });
      await selectTask(page, DEFAULT_JOB_ID);
      await expect.poll(async () => (await snapshotFetchProbe(page)).started).toBe(2);
      expect((await snapshotFetchProbe(page)).maxActive).toBe(1);
    } finally {
      releasePrevious();
    }
    await expect(page.locator(".task-monitor > .monitor-heading > .status")).toHaveText("running");
  });

  test("switches issue pages without overlapping snapshot requests", async ({ page, api }) => {
    const snapshot = makeSnapshot({ jobId: DEFAULT_JOB_ID, status: "running", currentModuleId: "nl" });
    snapshot.issues = [{
      issue_id: "issue-page-one", sample_id: 1, module_id: "nl", code: "review", severity: "warning",
      message: "Review this sample", retriable: 1, attempt: 1, repair_start_module: "nl",
    }];
    snapshot.nextIssueAfterSampleId = 1;
    snapshot.nextIssueAfterIssueId = "issue-page-one";
    setJobSnapshot(api, snapshot);
    await installSnapshotFetchProbe(page, DEFAULT_JOB_ID);
    await openApp(page, { jobId: DEFAULT_JOB_ID, language: "en" });
    await page.getByRole("button", { name: /Failures/ }).click();
    await expect(page.getByText("Review this sample")).toBeVisible();
    const issuePanel = page.locator(".issues-panel");
    await expect(issuePanel.locator(".issue-pagination")).toBeVisible();
    expect(await issuePanel.evaluate((root) => {
      const pagination = root.querySelector(".issue-pagination");
      const list = root.querySelector(".issue-list");
      return Boolean(pagination && list && (pagination.compareDocumentPosition(list) & Node.DOCUMENT_POSITION_FOLLOWING));
    })).toBe(true);

    const releasePrevious = holdRoute(api, `GET /api/jobs/${DEFAULT_JOB_ID}`);
    try {
      await expect.poll(async () => (await snapshotFetchProbe(page)).active, { timeout: 2000 }).toBe(1);
      await page.getByRole("button", { name: "Next page", exact: true }).click();
      await expect.poll(async () => (await snapshotFetchProbe(page)).started).toBeGreaterThanOrEqual(3);
      expect((await snapshotFetchProbe(page)).maxActive).toBe(1);
    } finally {
      releasePrevious();
    }
  });

  test("submits task-only OCR execution tuning and shows the compact runtime recommendation", async ({ page, api }) => {
    await openApp(page, { language: "en" });

    await page.locator(".workflow-rail").getByRole("button", { name: /OCR/ }).click();
    await page.getByRole("checkbox", { name: "Enable OCR" }).check();
    await expect(page.getByLabel("OCR device", { exact: true })).toBeVisible({ timeout: 2000 });
    await page.getByLabel("OCR device", { exact: true }).selectOption("cuda");
    await page.getByRole("radio", { name: "Manual detection limit" }).check();
    await page.getByLabel("OCR detection limit", { exact: true }).fill("2560");
    await page.getByRole("radio", { name: "Manual text batch" }).check();
    await page.getByLabel("OCR text batch", { exact: true }).fill("4");

    await openSetup(page);
    await page.getByRole("textbox", { name: "Source dataset" }).fill("E:\\datasets\\ocr-ui");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/preflight").length).toBe(1);
    expect(mutationsFor(api, "POST", "/api/jobs/preflight")[0]?.body).toMatchObject({
      config: { schemaVersion: 10, ocr: { enabled: true, device: "cuda" } },
      ocrExecution: {
        textDetLimitSideLen: { mode: "manual", value: 2560 },
        textBatchSize: { mode: "manual", value: 4 },
      },
    });
    await expect(page.getByLabel("OCR runtime")).toContainText("GPU 24 GiB");
    await expect(page.getByLabel("OCR runtime")).toContainText("2560 / 4");
    await page.locator(".workflow-rail").getByRole("button", { name: /OCR/ }).click();
    await expect(page.getByLabel("OCR device", { exact: true })).toBeEnabled();
  });

  test("shows the backend CUDA failure guidance in the OCR step", async ({ page, api }) => {
    const snapshot = makeSnapshot({ status: "failed", currentModuleId: "ocr", schemaVersion: 10 });
    snapshot.events = [{
      event_id: 8,
      module_id: "ocr",
      status: "failed",
      completed: 0,
      total: 3,
      attempt: 0,
      message: "The OCR CUDA runtime is unavailable or incompatible with this GPU. Choose Auto or CPU.",
    }];
    setJobSnapshot(api, snapshot);
    await openTrackedJob(page, "failed");
    await page.locator(".workflow-rail").getByRole("button", { name: /OCR/ }).click();
    await expect(page.locator(".ocr-failure")).toContainText("Choose Auto or CPU");
  });

  test("shows the Caption GPU fallback guidance", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({
      status: "running",
      currentModuleId: "caption",
      schemaVersion: 10,
      captionDiagnostics: [{ code: "caption_gpu_fallback", severity: "warning", count: 1 }],
    }));
    await openTrackedJob(page, "running");
    await expect(page.getByText("Caption GPU initialization failed; automatically using CPU.", { exact: true })).toBeVisible();
  });

  test("task lifecycle exposes only global pause and resume controls", async ({ page, api }) => {
    const snapshot = makeSnapshot({ status: "running", currentModuleId: "caption", schemaVersion: 10 });
    setJobSnapshot(api, snapshot);
    await openTrackedJob(page, "running");

    await expect(page.locator(".module-progress .module-row")).toHaveCount(snapshot.moduleOrder.length);
    await expect(page.locator(".module-controls")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Terminate task" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Pause task" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Resume task" })).toHaveCount(0);
    await page.getByRole("button", { name: "Pause task" }).click();
    await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${DEFAULT_JOB_ID}/pause`).length).toBe(1);

    await expect(page.getByRole("button", { name: "Resume task" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Pause task" })).toHaveCount(0);
    await page.getByRole("button", { name: "Resume task" }).click();
    await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${DEFAULT_JOB_ID}/resume`).length).toBe(1);

    await page.setViewportSize({ width: 390, height: 844 });
    await expect(page.getByRole("button", { name: "Pause task" })).toBeVisible();
  });

  test("hides NL API controls without pending API decisions", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({ status: "running", currentModuleId: "nl" }));
    await openTrackedJob(page, "running");

    await expect(page.getByLabel("Additional HTTP attempts")).toHaveCount(0);
    await expect(page.getByText("Pending API decisions: 0", { exact: true })).toHaveCount(0);
  });

  test("修复子任务独立展示、切换并可删除任务", async ({ page, api }) => {
    const parent = makeSnapshot({
      jobId: DEFAULT_JOB_ID,
      status: "failed",
      currentModuleId: "export",
    });
    const childJobId = `${DEFAULT_JOB_ID}-repair`;
    const child = makeSnapshot({
      jobId: childJobId,
      status: "failed",
      currentModuleId: "caption",
      parentJobId: DEFAULT_JOB_ID,
    });

    parent.repairChildren = [{
      jobId: childJobId,
      status: "failed",
      currentModuleId: "caption",
      sampleCount: 2,
      targetCount: 2,
      createdAt: child.job.createdAt,
      finishedAt: child.job.finishedAt,
    }];

    setJobSnapshot(api, parent);
    api.snapshots.set(childJobId, child);

    await openApp(page, { jobId: DEFAULT_JOB_ID, language: "zh-CN" });
    await page.getByRole("button", { name: /失败处理/ }).click();

    await expect(page.getByRole("heading", { name: "修复子任务", exact: true })).toBeVisible();
    await expect(page.getByText(`创建于 ${child.job.createdAt}`, { exact: true })).toBeVisible();
    await expect(page.getByLabel("最近任务").locator(`option[value="${childJobId}"]`)).toHaveCount(0);
    await page.getByRole("button", { name: "任务", exact: true }).click();
    await expect(page.getByRole("button", { name: "固定任务" })).toHaveCount(0);
    await expect(page.locator(".task-monitor .task-actions").getByRole("button", { name: "删除任务并释放训练集占用" })).toBeDisabled();
    await expect(page.getByText("请先删除修复子任务")).toBeVisible();
    await page.getByRole("button", { name: /失败处理/ }).click();

    await page.setViewportSize({ width: 390, height: 844 });
    await expect.poll(() => page.locator(".repair-tasks-panel").evaluate((element) => (
      element.getBoundingClientRect().right <= window.innerWidth
    ))).toBe(true);

    await page.getByRole("button", { name: "打开修复任务" }).click();
    await expect(page.getByText("修复任务", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "返回父任务" })).toBeVisible();
    await expect(page.locator(".task-monitor .task-actions").getByRole("button", { name: "删除任务并释放训练集占用" })).toBeEnabled();

    await page.getByRole("button", { name: "返回父任务" }).click();
    await expect(page.getByLabel("最近任务")).toHaveValue(DEFAULT_JOB_ID);
    await page.getByRole("button", { name: /失败处理/ }).click();
    await expect(page.getByRole("heading", { name: "修复子任务", exact: true })).toBeVisible();
    await page.getByRole("button", { name: "打开修复任务" }).click();

    page.once("dialog", (dialog) => {
      expect(dialog.message()).toBe("确认删除该任务并释放训练集占用吗？未提交进度和任务记录将被删除，且无法恢复。");
      dialog.accept();
    });
    await page.locator(".task-monitor .task-actions").getByRole("button", { name: "删除任务并释放训练集占用" }).click();
    await expect.poll(() =>
      mutationsFor(api, "POST", `/api/jobs/${childJobId}/discard`).length,
    ).toBe(1);
    await expect(page.getByLabel("最近任务")).toHaveValue(DEFAULT_JOB_ID);
    await expect(page.getByRole("heading", { name: "修复子任务", exact: true })).toHaveCount(0);
  });

  test("deletes only the control record for a succeeded task", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({ status: "succeeded", currentModuleId: "export" }));
    await openTrackedJob(page, "succeeded");

    const deleteHistory = page.getByRole("button", { name: "Delete completed history", exact: true });
    await expect(deleteHistory).toBeEnabled();
    page.once("dialog", (dialog) => {
      expect(dialog.message()).toBe(
        "Delete only this completed task record? Dataset annotations, exports, and the original backup will be preserved.",
      );
      dialog.accept();
    });
    await deleteHistory.click();

    await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${DEFAULT_JOB_ID}/delete-history`).length).toBe(1);
    await expect(page.getByLabel("Recent tasks")).toHaveValue("");
    await expect(page.locator(".empty-state")).toBeVisible();
  });

  test("uses the completed-history action for a succeeded repair child", async ({ page, api }) => {
    const parent = makeSnapshot({ status: "failed", currentModuleId: "export" });
    const childJobId = `${DEFAULT_JOB_ID}-repair-succeeded`;
    const child = makeSnapshot({ jobId: childJobId, status: "succeeded", currentModuleId: "export", parentJobId: DEFAULT_JOB_ID });
    parent.repairChildren = [{
      jobId: childJobId,
      status: "succeeded",
      currentModuleId: "export",
      sampleCount: 2,
      targetCount: 2,
      createdAt: child.job.createdAt,
      finishedAt: child.job.finishedAt,
    }];
    setJobSnapshot(api, parent);
    api.snapshots.set(childJobId, child);
    await openApp(page, { jobId: DEFAULT_JOB_ID, language: "en" });
    await page.getByRole("button", { name: /Failures/ }).click();

    const deleteHistory = page.locator(".repair-tasks-panel").getByRole("button", { name: "Delete completed history", exact: true });
    await expect(deleteHistory).toBeEnabled();
    page.once("dialog", (dialog) => {
      expect(dialog.message()).toBe(
        "Delete only this completed task record? Dataset annotations, exports, and the original backup will be preserved.",
      );
      dialog.accept();
    });
    await deleteHistory.click();

    await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${childJobId}/delete-history`).length).toBe(1);
    await expect(page.locator(".repair-tasks-panel")).toHaveCount(0);
  });

  test("retries the selected current-page NL issues as one child after Export succeeds", async ({ page, api }) => {
    const snapshot = makeSnapshot({ status: "succeeded", currentModuleId: "export", schemaVersion: 10 });
    snapshot.issues = [
      { issue_id: "nl-failed-1", sample_id: 1, module_id: "nl", code: "nl_api_unavailable", severity: "error", message: "provider unavailable", retriable: 0, attempt: 1 },
      { issue_id: "nl-failed-2", sample_id: 2, module_id: "nl", code: "nl_response_invalid", severity: "error", message: "invalid provider response", retriable: 0, attempt: 1 },
    ];
    setJobSnapshot(api, snapshot);
    await openApp(page, { jobId: DEFAULT_JOB_ID, language: "en" });
    await page.getByRole("button", { name: /Failures/ }).click();

    const selectCurrentPage = page.getByRole("checkbox", { name: "Select NL issues on this page" });
    await expect(selectCurrentPage).toBeVisible();
    await selectCurrentPage.check();
    const retrySelected = page.getByRole("button", { name: "Retry selected NL" });
    await expect(retrySelected).toBeEnabled();
    page.once("dialog", (dialog) => dialog.accept());
    await retrySelected.click();

    await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${DEFAULT_JOB_ID}/nl/manual-retry-batch`).length).toBe(1);
    expect(mutationsFor(api, "POST", `/api/jobs/${DEFAULT_JOB_ID}/nl/manual-retry-batch`)[0]?.body).toEqual({
      issueIds: ["nl-failed-1", "nl-failed-2"], confirmed: true,
    });
    await expect(page.getByRole("heading", { name: "Task progress", exact: true })).toBeVisible();
  });

  test("disables individual NL actions while a selected batch retry is pending", async ({ page, api }) => {
    const snapshot = makeSnapshot({ status: "succeeded", currentModuleId: "export", schemaVersion: 10 });
    snapshot.issues = [
      { issue_id: "nl-failed-1", sample_id: 1, module_id: "nl", code: "nl_api_unavailable", severity: "error", message: "provider unavailable", retriable: 0, attempt: 1 },
      { issue_id: "nl-failed-2", sample_id: 2, module_id: "nl", code: "nl_response_invalid", severity: "error", message: "invalid provider response", retriable: 0, attempt: 1 },
    ];
    setJobSnapshot(api, snapshot);
    const releaseBatchRetry = holdRoute(api, `POST /api/jobs/${DEFAULT_JOB_ID}/nl/manual-retry-batch`);
    try {
      await openApp(page, { jobId: DEFAULT_JOB_ID, language: "en" });
      await page.getByRole("button", { name: /Failures/ }).click();
      await page.getByRole("checkbox", { name: "Select NL issues on this page" }).check();
      page.once("dialog", (dialog) => dialog.accept());
      await page.getByRole("button", { name: "Retry selected NL" }).click();
      await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${DEFAULT_JOB_ID}/nl/manual-retry-batch`).length).toBe(1);
      await expect(page.getByRole("button", { name: "Retry NL API" }).first()).toBeDisabled();
      await expect(page.getByPlaceholder("Manual NL text").first()).toBeDisabled();
    } finally {
      releaseBatchRetry();
    }
  });

  test("refreshes the selected snapshot after pausing while a poll is in flight", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({ status: "running", currentModuleId: "caption" }));
    await installSnapshotFetchProbe(page, DEFAULT_JOB_ID);
    await openTrackedJob(page, "running");
    const initialRequests = (await snapshotFetchProbe(page)).started;

    const releasePoll = holdRoute(api, `GET /api/jobs/${DEFAULT_JOB_ID}`);
    try {
      await expect.poll(async () => (await snapshotFetchProbe(page)).started).toBeGreaterThan(initialRequests);
      const inFlightRequests = (await snapshotFetchProbe(page)).started;
      await page.getByRole("button", { name: "Pause task" }).click();
      await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${DEFAULT_JOB_ID}/pause`).length).toBe(1);
      await expect.poll(async () => (await snapshotFetchProbe(page)).started, { timeout: 1_500 }).toBeGreaterThan(inFlightRequests);
    } finally {
      releasePoll();
    }
  });

  test("recovers a selected cancelled recoverable task", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({ status: "cancelled_recoverable", currentModuleId: "caption" }));
    await openTrackedJob(page, "cancelled, recoverable");

    page.once("dialog", (dialog) => dialog.accept());
    const recoverTask = page.getByRole("button", { name: "Recover task" });
    await expect(recoverTask).toBeEnabled({ timeout: 1_500 });
    await recoverTask.click();
    await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${DEFAULT_JOB_ID}/recover`).length).toBe(1);
    await expect(page.locator(".task-monitor > .monitor-heading > .status")).toHaveText("running");
  });

  test("terminates a selected task through cancel after confirmation", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({ status: "running", currentModuleId: "caption" }));
    await openTrackedJob(page, "running");

    page.once("dialog", (dialog) => {
      expect(dialog.message()).toContain("safely drains active work");
      dialog.accept();
    });
    const terminateTask = page.getByRole("button", { name: "Terminate task" });
    await expect(terminateTask).toBeVisible({ timeout: 1_500 });
    await terminateTask.click();
    await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${DEFAULT_JOB_ID}/cancel`).length).toBe(1);
    await expect(page.locator(".task-monitor > .monitor-heading > .status")).toHaveText("cancelling");
  });

  test("binds a held lifecycle action to its click-time task", async ({ page, api }) => {
    const taskA = DEFAULT_JOB_ID;
    const taskB = "job-selected-after-pause";
    setJobSnapshot(api, makeSnapshot({ jobId: taskA, status: "running", currentModuleId: "caption" }));
    setJobSnapshot(api, makeSnapshot({ jobId: taskB, status: "succeeded", currentModuleId: "export" }));
    await installSnapshotFetchProbe(page, taskB);
    await openApp(page, { jobId: taskA, language: "en" });
    await expect(page.locator(".task-monitor > .monitor-heading > .status")).toHaveText("running");

    const releasePause = holdRoute(api, `POST /api/jobs/${taskA}/pause`);
    try {
      await page.getByRole("button", { name: "Pause task" }).click();
      await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${taskA}/pause`).length).toBe(1);

      await selectTask(page, taskB);
      await expect(page.locator(".task-monitor > .monitor-heading > .status")).toHaveText("succeeded");
      const selectedTaskRequests = (await snapshotFetchProbe(page)).started;
      expect(selectedTaskRequests).toBeGreaterThan(0);
      releasePause();
      await expect.poll(async () => (await snapshotFetchProbe(page)).started).toBeGreaterThan(selectedTaskRequests);
    } finally {
      releasePause();
    }

    await expect(page.locator(".task-monitor > .monitor-heading > .status")).toHaveText("succeeded");
    await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${taskA}/pause`).length).toBe(1);
    expect(mutationsFor(api, "POST", `/api/jobs/${taskB}/pause`)).toHaveLength(0);
  });

  test("routes recovery to the task selected at click time", async ({ page, api }) => {
    const taskA = DEFAULT_JOB_ID;
    const taskB = "job-recover-selected-task";
    setJobSnapshot(api, makeSnapshot({ jobId: taskA, status: "running", currentModuleId: "caption" }));
    setJobSnapshot(api, makeSnapshot({ jobId: taskB, status: "cancelled_recoverable", currentModuleId: "caption" }));
    await openApp(page, { jobId: taskA, language: "en" });
    await expect(page.locator(".task-monitor > .monitor-heading > .status")).toHaveText("running");

    await selectTask(page, taskB);
    await expect(page.locator(".task-monitor > .monitor-heading > .status")).toHaveText("cancelled, recoverable");
    page.once("dialog", (dialog) => dialog.accept());
    await page.getByRole("button", { name: "Recover task" }).click();

    await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${taskB}/recover`).length).toBe(1);
    expect(mutationsFor(api, "POST", `/api/jobs/${taskA}/recover`)).toHaveLength(0);
    await expect(page.locator(".task-monitor > .monitor-heading > .status")).toHaveText("running");
  });

  test("gates lifecycle buttons by task status", async ({ page, api }) => {
    const jobs = lifecycleStateMatrix.map((entry) => ({
      ...entry,
      jobId: `job-lifecycle-gate-${entry.status}`,
    }));
    for (const entry of jobs) {
      setJobSnapshot(api, makeSnapshot({ jobId: entry.jobId, status: entry.status, currentModuleId: "caption" }));
    }

    await openApp(page, { jobId: jobs[0].jobId, language: "en" });
    const status = page.locator(".task-monitor > .monitor-heading > .status");
    await expect(status).toHaveText(jobs[0].statusLabel);
    for (const entry of jobs) {
      await selectTask(page, entry.jobId);
      await expect(status).toHaveText(entry.statusLabel);
      for (const buttonName of lifecycleButtonNames) {
        const button = page.getByRole("button", { name: buttonName });
        if (entry.enabled.includes(buttonName)) await expect(button, `${entry.status} ${buttonName}`).toBeEnabled();
        else await expect(button, `${entry.status} ${buttonName}`).toBeDisabled();
      }
    }
  });

  test("shows OCR diagnostics and starts OCR repair through the existing endpoint", async ({ page, api }) => {
    const snapshot = makeSnapshot({ status: "failed", currentModuleId: "ocr", schemaVersion: 10 });
    snapshot.ocrDiagnostics = [
      { code: "ocr_total", severity: "info", count: 3 },
      { code: "ocr_new", severity: "info", count: 1 },
      { code: "ocr_reused", severity: "info", count: 2 },
      { code: "ocr_success", severity: "info", count: 1 },
      { code: "ocr_no_text", severity: "info", count: 1 },
      { code: "ocr_failed", severity: "info", count: 1 },
      { code: "ocr_text_items", severity: "info", count: 7 },
      { code: "ocr_included_for_llm", severity: "info", count: 5 },
      { code: "nl_ocr_context_omitted_too_large", severity: "warning", count: 1 },
    ];
    snapshot.issues = [{
      issue_id: "issue-1",
      sample_id: 1,
      module_id: "ocr",
      code: "ocr_inference_failed",
      severity: "error",
      message: "A test issue is available for repair.",
      retriable: 1,
      attempt: 2,
      repair_start_module: "ocr",
    }];
    setJobSnapshot(api, snapshot);
    await openTrackedJob(page, "failed");

    await page.locator(".workflow-rail").getByRole("button", { name: /OCR/ }).click();
    await expect(page.getByText("OCR summary")).toBeVisible();
    await expect(page.getByLabel("OCR summary").getByText("3", { exact: true })).toBeVisible();
    await page.getByRole("button", { name: /Failures/ }).click();
    await expect(page.getByText("A test issue is available for repair.")).toBeVisible();
    await page.getByRole("button", { name: "Reprocess retriable samples", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${DEFAULT_JOB_ID}/repair`).length).toBe(1);
  });
});
