import {
  DEFAULT_JOB_ID,
  clearRouteFailure,
  failRoute,
  expect,
  makeSnapshot,
  mutationsFor,
  openApp,
  setJobSnapshot,
  test,
} from "./mockApi";

async function openCountReview(page: Parameters<typeof openApp>[0]) {
  await openApp(page, { jobId: DEFAULT_JOB_ID, language: "en" });
  await expect(page.getByRole("button", { name: /Count Review/ }).filter({ has: page.locator(".workspace-badge") })).toBeVisible();
  await expect(page.locator(".count-review-workspace")).toBeVisible();
  await expect(page.getByText("sample-a.png")).toBeVisible();
}

test.describe("count review characterization", () => {
  test("shows the current module as waiting for human review", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({ status: "reviewing", currentModuleId: "count_review" }));
    await openCountReview(page);

    await page.getByRole("button", { name: "Task", exact: true }).click();
    const countModule = page.locator(".module-progress .module-row").filter({ hasText: "Count Review" });
    await expect(countModule.locator(".status")).toHaveText("Waiting for human review");
    await expect(countModule.locator("small")).toHaveText("Waiting for human review");
  });

  test("selects a review page size without overlapping preview rows", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({ status: "reviewing", currentModuleId: "count_review" }));
    await openCountReview(page);

    const panel = page.locator(".count-review-panel");
    const pageSize = panel.getByLabel("Items per page");
    await expect(pageSize).toHaveValue("50");
    await expect(pageSize.locator("option")).toHaveText(["20", "50", "100"]);
    await expect.poll(() => api.countReviewRequests.includes(50)).toBe(true);
    await pageSize.selectOption("100");
    await expect(pageSize).toHaveValue("100");
    await expect.poll(() => api.countReviewRequests.includes(100)).toBe(true);

    const previewRows = await panel.locator(".review-item").evaluateAll((items) => items.map((item) => {
      const box = item.getBoundingClientRect();
      return { top: box.top, bottom: box.bottom };
    }));
    expect(previewRows).toHaveLength(2);
    expect(previewRows[1].top).toBeGreaterThanOrEqual(previewRows[0].bottom);
  });

  test("shows a Count Review load error with a scoped retry", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({ status: "reviewing", currentModuleId: "count_review" }));
    failRoute(api, `GET /api/jobs/${DEFAULT_JOB_ID}/count-review`, "count review unavailable");
    await openApp(page, { jobId: DEFAULT_JOB_ID, language: "en" });

    const panel = page.locator(".count-review-panel");
    await expect(panel.getByRole("alert")).toHaveText("count review unavailable");
    await expect(panel.getByRole("button", { name: "Retry count review" })).toBeVisible();

    clearRouteFailure(api, `GET /api/jobs/${DEFAULT_JOB_ID}/count-review`);
    await panel.getByRole("button", { name: "Retry count review" }).click();
    await expect(panel.getByText("sample-a.png")).toBeVisible();
  });

  test("saves one review decision through the existing item endpoint", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({ status: "reviewing", currentModuleId: "count_review" }));
    await openCountReview(page);

    const firstItem = page.locator(".review-item").filter({ hasText: "sample-a.png" });
    await firstItem.getByRole("button", { name: "Use Classify" }).click();
    await expect.poll(() => mutationsFor(api, "PUT", `/api/jobs/${DEFAULT_JOB_ID}/count-review/1`).length).toBe(1);
    expect(mutationsFor(api, "PUT", `/api/jobs/${DEFAULT_JOB_ID}/count-review/1`)[0].body).toEqual({
      expectedVersion: 1,
      source: "classify",
    });
    await expect(firstItem.getByText("Saved")).toBeVisible();
  });

  test("saves a selected page in batch and confirms the review", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({ status: "reviewing", currentModuleId: "count_review" }));
    await openCountReview(page);

    await page.getByLabel("Select this page").check();
    const batchActions = page.locator(".review-batch-actions");
    await batchActions.getByRole("button", { name: "Use VLM" }).click();
    await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${DEFAULT_JOB_ID}/count-review/batch`).length).toBe(1);
    expect(mutationsFor(api, "POST", `/api/jobs/${DEFAULT_JOB_ID}/count-review/batch`)[0].body).toEqual({
      updates: [
        { sampleId: 1, expectedVersion: 1, source: "vlm" },
        { sampleId: 2, expectedVersion: 1, source: "vlm" },
      ],
    });

    page.once("dialog", (dialog) => dialog.accept());
    await page.getByRole("button", { name: "Confirm and continue" }).click();
    await expect.poll(() => mutationsFor(api, "POST", `/api/jobs/${DEFAULT_JOB_ID}/count-review/confirm`).length).toBe(1);
    await page.getByRole("button", { name: "Task", exact: true }).click();
    await expect(page.locator(".task-monitor > .monitor-heading > .status")).toHaveText("running");
  });

  test("keeps the continue blocker visible while decisions remain", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({ status: "reviewing", currentModuleId: "count_review" }));
    await openCountReview(page);

    await expect(page.getByText("Resolve 2 pending decisions before continuing.")).toBeVisible();
    await expect(page.getByRole("button", { name: "Confirm and continue" })).toBeDisabled();
  });
});
