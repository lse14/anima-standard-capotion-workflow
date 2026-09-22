import { clearRouteFailure, expect, failRoute, holdRoute, makeSnapshot, mutationsFor, openApp, openPrecheck, openSetup, setJobSnapshot, test } from "./mockApi";

test.describe("v10 Token Budget configuration", () => {
  test("starts a v10 draft with the default tokenizer budget and prompt route", async ({ page, api }) => {
    await openApp(page, { language: "en" });

    await page.locator(".workflow-rail").getByRole("button", { name: /Token Budget/ }).click();
    await expect(page.getByRole("checkbox", { name: "Enable Token Budget" })).toBeChecked();
    await expect(page.getByLabel("Maximum training tokens", { exact: true })).toHaveValue("512");
    expect(api.promptRequests).toEqual([]);
  });

  test("sends the selected NL preset and exact length distribution in the v10 preflight body", async ({ page, api }) => {
    await openApp(page, { language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /NL/ }).click();
    const presets = page.locator("[data-nl-preset-card]");
    await expect(presets).toHaveCount(3);
    await expect(presets.nth(0)).toContainText("General task preset prompt.");
    await expect(presets.nth(1)).toContainText("Style task preset prompt.");
    await expect(presets.nth(2)).toContainText("Character task preset prompt.");
    await page.getByRole("button", { name: "Style", exact: true }).click();
    await page.getByLabel("Short (%)", { exact: true }).fill("20");
    await page.getByLabel("Medium (%)", { exact: true }).fill("30");
    await page.getByLabel("Long (%)", { exact: true }).fill("50");
    await openSetup(page);
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\v10-budget");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();

    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/preflight").length).toBe(1);
    const config = (mutationsFor(api, "POST", "/api/jobs/preflight")[0].body as { config: Record<string, unknown> }).config;
    expect(config.schemaVersion).toBe(10);
    expect(config.nl).toMatchObject({ promptVersion: "nl-default-prompt-v4", captionPreset: "style", lengthDistribution: { short: 20, medium: 30, long: 50 }, lengthSeed: "anima-nl-length-v1" });
    expect(config.tokenBudget).toEqual({ enabled: true, maxTokens: 512, resourceId: "tokenizer-qwen3-0.6b-anima-v1" });
    expect(config).not.toHaveProperty("resourceManifestRelativePath");
    expect(config).not.toHaveProperty("resourceFingerprint");
    expect(config).not.toHaveProperty("contextLimit");
  });

  test("allows disabled Token Budget preflight even when its tokenizer is missing", async ({ page, api }) => {
    const anima = api.resources.resources.find((item) => item.resourceId === "tokenizer-qwen3-0.6b-anima-v1");
    if (!anima) throw new Error("Anima tokenizer fixture is missing");
    anima.available = false;
    await openApp(page, { language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /Token Budget/ }).click();
    await page.getByRole("checkbox", { name: "Enable Token Budget" }).uncheck();
    await expect(page.getByLabel("Maximum training tokens", { exact: true })).toBeDisabled();
    await openSetup(page);
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\disabled-budget");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/preflight").length).toBe(1);
    expect((mutationsFor(api, "POST", "/api/jobs/preflight")[0].body as { config: { tokenBudget: unknown } }).config.tokenBudget).toEqual({ enabled: false, maxTokens: 512, resourceId: "tokenizer-qwen3-0.6b-anima-v1" });
  });

  test("uses the selected Krea resource context limit without a browser constant", async ({ page, api }) => {
    await openApp(page, { language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /Token Budget/ }).click();
    const tokenizer = page.getByLabel("Tokenizer resource", { exact: true });
    await tokenizer.selectOption({ label: "Qwen/Qwen3-VL-4B-Instruct" });
    const maximum = page.getByLabel("Maximum training tokens", { exact: true });
    await expect(maximum).toHaveAttribute("max", "262144");
    await maximum.fill("50000");
    await openSetup(page);
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\krea-budget");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/preflight").length).toBe(1);
    expect((mutationsFor(api, "POST", "/api/jobs/preflight")[0].body as { config: { tokenBudget: unknown } }).config.tokenBudget).toEqual({ enabled: true, maxTokens: 50000, resourceId: "tokenizer-qwen3-vl-4b-krea2-v1" });
  });

  test("edits with a debounced recount and applies only the saved proposal", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({ schemaVersion: 10, status: "reviewing", currentModuleId: "token_budget" }));
    await openApp(page, { jobId: "job-e621-characterization", language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /Token Budget/ }).click();
    const first = page.locator(".token-budget-review-item").first();
    await expect(first).toBeVisible();
    await first.getByLabel("NL").fill("Edited caption.");
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/job-e621-characterization/token-budget/recount").length).toBe(1);
    expect(mutationsFor(api, "POST", "/api/jobs/job-e621-characterization/token-budget/recount")[0].body).toEqual({
      sampleId: 1,
      expectedVersion: 1,
      annotation: { quality: ["high quality"], count: "solo", character: "", series: "", artist: "", appearance: ["red jacket"], tags: ["tag-1"], environment: ["street"], nl: "Edited caption." },
    });
    await first.getByRole("button", { name: "Apply", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/job-e621-characterization/token-budget/apply").length).toBe(1);
    expect(mutationsFor(api, "POST", "/api/jobs/job-e621-characterization/token-budget/apply")[0].body).toEqual({ sampleId: 1, expectedVersion: 2 });
    await expect(page.getByText("Applied durably; Export has not started yet.")).toBeVisible();
  });

  test("runs one explicit short rewrite for the selected samples", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({ schemaVersion: 10, status: "reviewing", currentModuleId: "token_budget" }));
    await openApp(page, { jobId: "job-e621-characterization", language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /Token Budget/ }).click();
    await page.getByRole("checkbox", { name: "#1" }).check();
    await page.getByRole("button", { name: "Rewrite short", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/job-e621-characterization/token-budget/rewrite-short").length).toBe(1);
    expect(mutationsFor(api, "POST", "/api/jobs/job-e621-characterization/token-budget/rewrite-short")[0].body).toEqual({ sampleIds: [1], expectedVersions: { "1": 1 } });
    await expect(page.locator(".token-budget-review-item").first().getByLabel("NL")).toHaveValue("Short rewrite 1.");
  });

  test("pages overflow reviews through the bounded sample keyset", async ({ page, api }) => {
    const template = api.tokenBudgetReviews.items[0];
    api.tokenBudgetReviews.items = Array.from({ length: 51 }, (_, index) => ({
      ...template,
      sampleId: index + 1,
      relativeImagePath: `review/sample-${index + 1}.png`,
      review: { ...template.review, version: 1 },
      annotation: { ...template.annotation, tags: [`tag-${index + 1}`], nl: `Original caption ${index + 1}.` },
      proposal: null,
      rewriteProposal: null,
    }));
    api.tokenBudgetReviews.targetCount = 51;
    setJobSnapshot(api, makeSnapshot({ schemaVersion: 10, status: "reviewing", currentModuleId: "token_budget" }));
    await openApp(page, { jobId: "job-e621-characterization", language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /Token Budget/ }).click();
    await expect(page.getByRole("checkbox", { name: "#1", exact: true })).toBeVisible();
    const panel = page.locator(".token-budget-review-panel");
    await panel.getByRole("button", { name: "Next page", exact: true }).click();
    await expect(page.getByRole("checkbox", { name: "#51" })).toBeVisible();
    await panel.getByRole("button", { name: "Previous", exact: true }).click();
    await expect(page.getByRole("checkbox", { name: "#1", exact: true })).toBeVisible();
  });

  test("cancels an expired recount and applies only the newer edit", async ({ page, api }) => {
    const recountPath = "/api/jobs/job-e621-characterization/token-budget/recount";
    const aborted: string[] = [];
    page.on("requestfailed", (request) => {
      if (new URL(request.url()).pathname === recountPath) aborted.push(request.failure()?.errorText ?? "aborted");
    });
    setJobSnapshot(api, makeSnapshot({ schemaVersion: 10, status: "reviewing", currentModuleId: "token_budget" }));
    const release = holdRoute(api, `POST ${recountPath}`);
    await openApp(page, { jobId: "job-e621-characterization", language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /Token Budget/ }).click();
    const nl = page.locator(".token-budget-review-item").first().getByLabel("NL");
    await nl.fill("First caption.");
    await expect.poll(() => mutationsFor(api, "POST", recountPath).length).toBe(1);
    await nl.fill("Second caption.");
    await expect.poll(() => aborted.length).toBe(1);
    await expect.poll(() => mutationsFor(api, "POST", recountPath).length).toBe(2);
    release();
    await expect(nl).toHaveValue("Second caption.");
  });

  test("refreshes on a recount version conflict and does not repeat a held rewrite", async ({ page, api }) => {
    setJobSnapshot(api, makeSnapshot({ schemaVersion: 10, status: "reviewing", currentModuleId: "token_budget" }));
    failRoute(api, "POST /api/jobs/job-e621-characterization/token-budget/recount", "Token Budget review version conflict", 409);
    await openApp(page, { jobId: "job-e621-characterization", language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /Token Budget/ }).click();
    await page.locator(".token-budget-review-item").first().getByLabel("NL").fill("Conflict caption.");
    await expect(page.getByText("This review changed on the server. The current page was refreshed.")).toBeVisible();

    clearRouteFailure(api, "POST /api/jobs/job-e621-characterization/token-budget/recount");
    const release = holdRoute(api, "POST /api/jobs/job-e621-characterization/token-budget/rewrite-short");
    await page.getByRole("checkbox", { name: "#1" }).check();
    const rewrite = page.locator(".review-batch-bar").getByRole("button");
    await rewrite.click();
    await expect(rewrite).toBeDisabled();
    await rewrite.dispatchEvent("click");
    release();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/job-e621-characterization/token-budget/rewrite-short").length).toBe(1);
  });
});
