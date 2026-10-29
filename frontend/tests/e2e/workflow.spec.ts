import {
  expect,
  failRoute,
  holdRoute,
  mutationsFor,
  openApp,
  openPrecheck,
  openSetup,
  test,
} from "./mockApi";

function expectedE621Draft(sourceRoot: string, systemPrompt: string): Record<string, unknown> {
  return {
    schemaVersion: 10,
    moduleBatchSize: { caption: 4, classify: 128, replace: 128, ocr: 4, nl: 3, countReview: 100, dropout: 4, tokenBudget: 128, export: 500 },
    workMode: "in_place",
    overwriteMode: "incremental",
    sourceRoot,
    annotationBackup: "required",
    recursive: false,
    captionFormat: { flatTxtLayout: "nl_newline", replaceUnderscoresWithSpaces: true, preserveEscapes: true, triggersEnabled: false, triggerTerms: [] },
    imageDecode: {
      extensions: [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
      rejectMultiFrame: true,
      applyExifTranspose: true,
      alphaBackground: "#FFFFFF",
      invalidImageAction: "block",
    },
    caption: {
      enabled: true,
      thresholdMode: "model_default",
      overwriteTxt: false,
      inputTxtMode: "tag",
      taggerFallbackOnMissingTxt: true,
      resourceId: "caption-e621-eva02-large-full-v1",
    },
    classify: {
      enabled: true,
      indexMode: "bundled",
      overwriteJson: false,
      overwriteCount: false,
      resourceId: "classify-e621-20260724-v1",
    },
    replace: { enabled: true, indexMode: "bundled", resourceId: "replace-e621-20260726-v2" },
    ocr: {
      enabled: false,
      device: "auto",
      llmMinConfidence: 0.5,
      forceReprocess: false,
      resourceId: "ocr-ppocrv5-server-paddle-v1",
    },
    nl: {
      enabled: true,
      reuseOriginalNl: true,
      apiEnabled: true,
      useImage: true,
      useFullJson: false,
      systemPrompt,
      promptVersion: "nl-default-prompt-v4",
      captionPreset: "general",
      lengthDistribution: { short: 33, medium: 34, long: 33 },
      lengthSeed: "anima-nl-length-v1",
      apiProfileId: "default",
      apiPolicy: { maxRequestsPerMinute: 60, backupEnabled: false },
    },
    countReview: { enabled: true, protocolVersion: "count-review-v1" },
    dropout: {
      enabled: false,
      policyVersion: "dataset-batch-policy-v1",
      seed: "anima-policy-default-v1",
      artist: { enabled: true, dropoutProbability: 0 },
      quality: {
        enabled: true,
        dropoutProbability: 0,
        device: "auto",
        batchSize: 4,
        resourceId: "lse14-scorer-5k-v1",
      },
      appearanceNl: {
        enabled: true,
        solo: { dropNl: 0.7, dropAppearance: 0.05 },
        nonSolo: { dropNl: 0.05, dropAppearance: 0.7 },
        unknown: { dropNl: 0.15, dropAppearance: 0.15 },
      },
    },
    tokenBudget: { enabled: true, maxTokens: 490, resourceId: "tokenizer-qwen3-0.6b-anima-v1" },
    export: { format: "both" },
  };
}

test.describe("workflow characterization", () => {
  test("exposes v10 batch controls and the default NL-newline layout", async ({ page }) => {
    await openApp(page, { language: "en" });
    await expect(page.getByRole("button", { name: "Recommend from device", exact: true })).toBeVisible();
    const moduleControls: Array<[RegExp, string]> = [
      [/Caption/, "caption-batch-size"], [/Classify/, "classify-batch-size"], [/Replace/, "replace-batch-size"],
      [/OCR/, "ocr-batch-size"], [/NL/, "nl-batch-size"], [/Count Review/, "count-review-batch-size"],
      [/Dropout/, "dropout-batch-size"], [/Token Budget/, "token-budget-batch-size"], [/Export/, "export-batch-size"],
    ];
    for (const [stepName, controlId] of moduleControls) {
      await page.locator(".workflow-rail").getByRole("button", { name: stepName }).click();
      await expect(page.locator(`#${controlId}`)).toBeVisible();
    }
    const layout = page.getByLabel("Flat TXT layout", { exact: true });
    await expect(layout).toHaveValue("nl_newline");
    await layout.selectOption("single_line");
    await expect(layout).toHaveValue("single_line");
  });

  test("restores the complete draft after refresh without persisting the API secret", async ({ page }) => {
    await openApp(page, { language: "en" });
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\draft-source");
    await page.getByLabel("Work mode", { exact: true }).selectOption("full_copy");
    await page.getByRole("textbox", { name: "Output dataset", exact: true }).fill("E:\\datasets\\draft-output");
    await page.getByLabel("Overwrite mode", { exact: true }).selectOption("incremental");
    await page.getByRole("checkbox", { name: "Search subfolders", exact: true }).check();

    await page.locator(".workflow-rail").getByRole("button", { name: /Caption/ }).click();
    await page.getByLabel("TXT input mode", { exact: true }).selectOption("nl");
    await page.getByRole("checkbox", { name: "Add trigger terms", exact: true }).check();
    await page.getByRole("textbox", { name: "Trigger terms (comma separated)", exact: true }).fill("wolf, fox");

    await page.locator(".workflow-rail").getByRole("button", { name: /Classify/ }).click();
    await page.getByRole("checkbox", { name: "Overwrite JSON", exact: true }).check();
    await page.getByRole("checkbox", { name: "Overwrite count", exact: true }).check();

    await page.locator(".workflow-rail").getByRole("button", { name: /OCR/ }).click();
    await page.getByRole("checkbox", { name: "Enable OCR", exact: true }).check();
    await page.getByLabel("OCR device", { exact: true }).selectOption("cpu");
    const tuning = page.locator("fieldset.ocr-tuning");
    await tuning.nth(0).getByRole("radio", { name: "Manual detection limit", exact: true }).check();
    await tuning.nth(0).getByRole("spinbutton", { name: "OCR detection limit", exact: true }).fill("2560");
    await tuning.nth(1).getByRole("radio", { name: "Manual text batch", exact: true }).check();
    await tuning.nth(1).getByRole("spinbutton", { name: "OCR text batch", exact: true }).fill("3");

    await page.locator(".workflow-rail").getByRole("button", { name: /NL/ }).click();
    await page.getByRole("textbox", { name: "HTTP attempt budget", exact: true }).fill("17");
    await page.getByRole("textbox", { name: "API key", exact: true }).fill("do-not-persist-this-secret");

    await page.reload();
    await expect(page.getByRole("heading", { name: "Anima Dataset Tool" })).toBeVisible();
    await expect(page.getByRole("textbox", { name: "Source dataset", exact: true })).toHaveValue("E:\\datasets\\draft-source");
    await expect(page.getByLabel("Work mode", { exact: true })).toHaveValue("full_copy");
    await expect(page.getByRole("textbox", { name: "Output dataset", exact: true })).toHaveValue("E:\\datasets\\draft-output");
    await expect(page.getByLabel("Overwrite mode", { exact: true })).toHaveValue("incremental");
    await expect(page.getByRole("checkbox", { name: "Search subfolders", exact: true })).toBeChecked();

    await page.locator(".workflow-rail").getByRole("button", { name: /Caption/ }).click();
    await expect(page.getByLabel("TXT input mode", { exact: true })).toHaveValue("nl");
    await expect(page.getByRole("checkbox", { name: "Add trigger terms", exact: true })).toBeChecked();
    await expect(page.getByRole("textbox", { name: "Trigger terms (comma separated)", exact: true })).toHaveValue("wolf, fox");
    await page.locator(".workflow-rail").getByRole("button", { name: /Classify/ }).click();
    await expect(page.getByRole("checkbox", { name: "Overwrite JSON", exact: true })).toBeChecked();
    await expect(page.getByRole("checkbox", { name: "Overwrite count", exact: true })).toBeChecked();
    await page.locator(".workflow-rail").getByRole("button", { name: /OCR/ }).click();
    await expect(page.getByRole("checkbox", { name: "Enable OCR", exact: true })).toBeChecked();
    await expect(page.getByLabel("OCR device", { exact: true })).toHaveValue("cpu");
    await expect(page.locator("fieldset.ocr-tuning").nth(0).getByRole("spinbutton", { name: "OCR detection limit", exact: true })).toHaveValue("2560");
    await expect(page.locator("fieldset.ocr-tuning").nth(1).getByRole("spinbutton", { name: "OCR text batch", exact: true })).toHaveValue("3");
    await page.locator(".workflow-rail").getByRole("button", { name: /NL/ }).click();
    await expect(page.getByRole("textbox", { name: "HTTP attempt budget", exact: true })).toHaveValue("17");

    const persisted = await page.evaluate(() => Object.values(window.localStorage).join("\n"));
    expect(persisted).not.toContain("do-not-persist-this-secret");
  });

  test("saves, loads, and deletes a complete local preset while retaining current paths", async ({ page }) => {
    await openApp(page, { language: "en" });
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\preset-current-source");
    await page.getByLabel("Work mode", { exact: true }).selectOption("full_copy");
    await page.getByRole("textbox", { name: "Output dataset", exact: true }).fill("E:\\datasets\\preset-current-output");
    await page.locator(".workflow-rail").getByRole("button", { name: /Export/ }).click();
    await page.getByLabel("Format", { exact: true }).selectOption("json");
    await page.getByRole("button", { name: /Dataset and configuration/ }).click();

    await page.getByRole("textbox", { name: "Preset name", exact: true }).fill("Local review");
    await page.getByRole("button", { name: "Save task preset", exact: true }).click();
    await expect(page.getByRole("option", { name: "Local review", exact: true })).toHaveCount(1);
    const presetStorage = await page.evaluate(() => JSON.parse(window.localStorage.getItem("anima.ui.annotation-presets.v1") ?? "{}"));
    expect(presetStorage.presets[0].config).not.toHaveProperty("sourceRoot");
    expect(presetStorage.presets[0].config).not.toHaveProperty("outputRoot");

    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\changed-source");
    await page.getByRole("textbox", { name: "Output dataset", exact: true }).fill("E:\\datasets\\changed-output");
    await page.locator(".workflow-rail").getByRole("button", { name: /Export/ }).click();
    await page.getByLabel("Format", { exact: true }).selectOption("flat_txt");
    await page.getByRole("button", { name: /Dataset and configuration/ }).click();
    await page.getByRole("combobox", { name: "Task preset", exact: true }).selectOption({ label: "Local review" });

    await expect(page.getByRole("textbox", { name: "Source dataset", exact: true })).toHaveValue("E:\\datasets\\changed-source");
    await expect(page.getByRole("textbox", { name: "Output dataset", exact: true })).toHaveValue("E:\\datasets\\changed-output");
    await page.locator(".workflow-rail").getByRole("button", { name: /Export/ }).click();
    await expect(page.getByLabel("Format", { exact: true })).toHaveValue("json");
    await page.getByRole("button", { name: /Dataset and configuration/ }).click();

    page.once("dialog", (dialog) => dialog.accept());
    await page.getByRole("button", { name: "Delete task preset", exact: true }).click();
    await expect(page.getByRole("option", { name: "Local review", exact: true })).toHaveCount(0);
    await page.reload();
    await expect(page.getByRole("option", { name: "Local review", exact: true })).toHaveCount(0);
  });

  test("falls back to defaults when local draft storage is malformed or from an unknown version", async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem("anima.ui.draft.v1", JSON.stringify({ schemaVersion: 999, draft: {}, ocrExecution: {} }));
      window.localStorage.setItem("anima.ui.annotation-presets.v1", "not-json");
    });
    await openApp(page, { language: "en" });
    await expect(page.getByRole("textbox", { name: "Source dataset", exact: true })).toHaveValue("");
    await expect(page.getByLabel("Work mode", { exact: true })).toHaveValue("in_place");
    await expect(page.getByRole("option", { name: "No task preset", exact: true })).toHaveCount(1);
  });

  test("keeps Classify JSON and count overwrite independent in incremental mode and explains rebuild", async ({ page, api }) => {
    await openApp(page, { language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /Classify/ }).click();
    const overwriteJson = page.getByRole("checkbox", { name: "Overwrite JSON", exact: true });
    const overwriteCount = page.getByRole("checkbox", { name: "Overwrite count", exact: true });
    await expect(overwriteJson).toBeEnabled();
    await expect(overwriteCount).toBeEnabled();
    await overwriteJson.check();
    await expect(overwriteJson).toBeChecked();
    await expect(overwriteCount).not.toBeChecked();

    await openSetup(page);
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\classify-overwrite");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/preflight").length).toBe(1);
    expect((mutationsFor(api, "POST", "/api/jobs/preflight")[0].body as { config: { classify: Record<string, unknown> } }).config.classify).toMatchObject({ overwriteJson: true, overwriteCount: false });

    await page.reload();
    await page.locator(".workflow-rail").getByRole("button", { name: /Dataset and configuration/ }).click();
    await page.getByLabel("Overwrite mode", { exact: true }).selectOption("rebuild");
    await page.locator(".workflow-rail").getByRole("button", { name: /Classify/ }).click();
    await expect(page.getByRole("checkbox", { name: "Overwrite JSON", exact: true })).toBeDisabled();
    await expect(page.getByRole("checkbox", { name: "Overwrite count", exact: true })).toBeDisabled();
    await expect(page.locator("[data-classify-overwrite-mode='rebuild']").getByText("Rebuild always overwrites JSON and count.", { exact: true })).toHaveCount(1);
    await expect(page.locator(".readonly-setting").getByText("Rebuild always overwrites JSON and count.", { exact: true })).toHaveCount(2);
  });

  test("offers an accessible cancellation action while asynchronous preflight is running", async ({ page, api }) => {
    await openApp(page, { language: "en" });
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\preflight-cancel");
    await openPrecheck(page);
    const releaseProgress = holdRoute(api, "GET /api/jobs/preflight/job-e621-characterization/progress");
    await page.getByRole("button", { name: "Preflight", exact: true }).click();
    await expect(page.locator("[data-preflight-progress]")).toBeVisible();
    await expect(page.getByRole("button", { name: "Cancel preflight", exact: true })).toBeVisible();
    await page.getByRole("button", { name: "Cancel preflight", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/preflight/job-e621-characterization/cancel").length).toBe(1);
    releaseProgress();
    await expect(page.locator("[data-preflight-progress] .status")).toHaveText("Cancelled");
  });

  test("fills module recommendations once, keeps manual overrides, and protects values on probe failure", async ({ page, api }) => {
    await openApp(page, { language: "en" });
    const recommend = page.getByRole("button", { name: "Recommend from device", exact: true });
    await recommend.click();
    await expect.poll(() => api.recommendationRequests).toBe(1);
    await page.locator(".workflow-rail").getByRole("button", { name: /Caption/ }).click();
    await page.locator("#caption-batch-size").fill("10");
    await page.locator("#caption-batch-size").press("Tab");
    await expect(page.locator("#caption-batch-size")).toHaveValue("10");
    await expect(page.getByRole("alert").filter({ hasText: "above the device recommendation" })).toBeVisible();

    failRoute(api, "GET /api/application/device-recommendations", "device probe unavailable", 503);
    await recommend.click();
    await expect(page.getByRole("alert").filter({ hasText: "device probe unavailable" })).toBeVisible();
    await expect(page.locator("#caption-batch-size")).toHaveValue("10");
  });

  test("keeps closed module batch controls disabled while retaining their draft values", async ({ page }) => {
    await openApp(page, { language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /OCR/ }).click();
    const batch = page.locator("#ocr-batch-size");
    await expect(batch).toHaveValue("4");
    await expect(batch).toBeDisabled();
    await page.locator(".workflow-rail").getByRole("button", { name: /Dropout/ }).click();
    await expect(page.locator("#dropout-batch-size")).toHaveValue("4");
    await expect(page.locator("#dropout-batch-size")).toBeDisabled();
  });

  test("locks every module batch control after workspace confirmation", async ({ page }) => {
    await openApp(page, { language: "en" });
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\batch-lock");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();
    await expect(page.getByRole("button", { name: "Start pipeline", exact: true })).toBeEnabled();
    page.once("dialog", (dialog) => dialog.accept());
    await page.getByRole("button", { name: "Start pipeline", exact: true }).click();
    await page.locator(".workflow-rail").getByRole("button", { name: /Caption/ }).click();
    await expect(page.locator("#caption-batch-size")).toBeDisabled();
    await page.locator(".workflow-rail").getByRole("button", { name: /Export/ }).click();
    await expect(page.locator("#export-batch-size")).toBeDisabled();
    await expect(page.getByRole("button", { name: "Recommend from device", exact: true })).toBeDisabled();
  });

  test("keeps the E621 Draft shape through preflight, workspace confirmation, and start", async ({ page, api }) => {
    const sourceRoot = "E:\\datasets\\e621-characterization";
    await openApp(page, { language: "en" });

    await page.locator(".workflow-rail").getByRole("button", { name: /NL/ }).click();
    await expect(page.getByLabel("User supplement", { exact: true })).toHaveCount(0);
    await expect(page.locator("[data-nl-preset-card]")).toHaveCount(3);
    expect(api.promptRequests).toEqual([]);

    await openSetup(page);
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill(sourceRoot);
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();

    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/preflight").length).toBe(1);
    expect((mutationsFor(api, "POST", "/api/jobs/preflight")[0].body as { config: { classify: Record<string, unknown> } }).config.classify)
      .not.toHaveProperty("wikiDataSourceId");
    expect(mutationsFor(api, "POST", "/api/jobs/preflight")[0].body).toEqual({
      config: expectedE621Draft(sourceRoot, "General task preset prompt."),
    });
    await expect(page.getByRole("button", { name: "Start pipeline", exact: true })).toBeEnabled();

    page.once("dialog", (dialog) => dialog.accept());
    await page.getByRole("button", { name: "Start pipeline", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/job-e621-characterization/confirm-workspace").length).toBe(1);
    expect(mutationsFor(api, "POST", "/api/jobs/job-e621-characterization/confirm-workspace")[0].body).toEqual({
      confirmed: true,
      confirmedRebuild: false,
    });
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/job-e621-characterization/start").length).toBe(1);
    await expect(page.locator(".task-monitor > .monitor-heading > .status")).toHaveText("running");
  });

  test("lets Caption choose v10 TXT input handling and sends the selected mode", async ({ page, api }) => {
    await openApp(page, { language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /Caption/ }).click();

    const inputMode = page.getByLabel("TXT input mode", { exact: true });
    const tagger = page.getByRole("checkbox", { name: "Enable caption" });
    const fallback = page.getByRole("checkbox", { name: "Use Tagger for missing or empty TXT" });
    await expect(inputMode).toHaveValue("tag");
    await expect(fallback).toBeChecked();
    await page.getByRole("button", { name: "TXT input mode information" }).click();
    await expect(page.getByRole("tooltip").filter({ hasText: "JSON nl" })).toBeVisible();
    await page.getByRole("button", { name: "Use Tagger for missing or empty TXT information" }).click();
    await expect(page.getByRole("tooltip").filter({ hasText: "new task" })).toBeVisible();

    await tagger.uncheck();
    await inputMode.selectOption("nl");
    await expect(tagger).toBeChecked();
    await expect(tagger).toBeDisabled();
    await expect(fallback).toHaveCount(0);
    await expect(page.getByRole("checkbox", { name: "Overwrite TXT" })).toBeDisabled();

    await openSetup(page);
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\txt-input-nl");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/preflight").length).toBe(1);
    const config = (mutationsFor(api, "POST", "/api/jobs/preflight")[0].body as { config: Record<string, unknown> }).config;
    expect(config.caption).toEqual({
      enabled: true,
      thresholdMode: "model_default",
      overwriteTxt: false,
      inputTxtMode: "nl",
      taggerFallbackOnMissingTxt: true,
      resourceId: "caption-e621-eva02-large-full-v1",
    });
  });

  test("shows the agreed Chinese TXT input guidance", async ({ page }) => {
    await openApp(page, { language: "zh-CN" });
    await page.locator(".workflow-rail").getByRole("button", { name: /Caption/ }).click();
    await page.getByRole("button", { name: "TXT 输入模式说明" }).click();
    await expect(page.getByRole("tooltip").filter({ hasText: "写入 JSON nl" })).toBeVisible();
    await page.getByRole("button", { name: "缺失或空 TXT 时启用 Tagger 补全说明" }).click();
    await expect(page.getByRole("tooltip").filter({ hasText: "重新创建任务运行" })).toBeVisible();
  });

  test("explains JSON and TXT input formats in the Setup step", async ({ page }) => {
    await openApp(page, { language: "en" });
    const help = page.locator("[data-input-format-help]");
    await help.locator("summary").click();
    await expect(help).toContainText("same-stem siblings");
    await expect(help).toContainText("quality, count");
    await expect(help).toContainText("artist, character, contributor, copyright");
    await expect(help).toContainText("Tag TXT");
    await expect(help).toContainText("NL TXT");

    await page.getByRole("button", { name: "中文", exact: true }).click();
    const chineseHelp = page.locator("[data-input-format-help]");
    await expect(chineseHelp.locator("summary")).toHaveText("JSON / TXT 输入格式");
    await chineseHelp.locator("summary").click();
    await expect(chineseHelp).toContainText("同 stem 兄弟文件");
    await expect(chineseHelp).toContainText("原始 E621 JSON");
    await expect(chineseHelp).toContainText("Tag TXT");
    await expect(chineseHelp).toContainText("NL TXT");
  });

  test("labels appearance dropout controls as NL and Tag rates", async ({ page }) => {
    await openApp(page, { language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /Dropout/ }).click();

    const nlRates = page.getByLabel("NL dropout rate", { exact: true });
    const tagRates = page.getByLabel("Tag dropout rate", { exact: true });
    await expect(nlRates).toHaveCount(3);
    await expect(tagRates).toHaveCount(3);
    for (const index of [0, 1, 2]) {
      await expect(nlRates.nth(index)).toBeVisible();
      await expect(tagRates.nth(index)).toBeVisible();
    }
    await expect(page.getByLabel("Drop description", { exact: true })).toHaveCount(0);
    await expect(page.getByLabel("Drop appearance tags", { exact: true })).toHaveCount(0);

    await expect(page.getByRole("button", { name: "NL dropout rate information", exact: true })).toHaveCount(3);
    await expect(page.getByRole("button", { name: "Tag dropout rate information", exact: true })).toHaveCount(3);
    await page.getByRole("button", { name: "NL dropout rate information", exact: true }).first().click();
    await expect(page.getByRole("tooltip").filter({ hasText: "JSON nl" })).toBeVisible();
    await page.getByRole("button", { name: "Tag dropout rate information", exact: true }).first().click();
    await expect(page.getByRole("tooltip").filter({ hasText: "JSON tags" })).toBeVisible();
  });

  test("keeps OCR and NL independent while sending the exact v10 OCR request object", async ({ page, api }) => {
    await openApp(page, { language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /OCR/ }).click();
    const enableOcr = page.getByRole("checkbox", { name: "Enable OCR" });
    const confidence = page.getByLabel("Minimum LLM confidence", { exact: true });
    await expect(enableOcr).not.toBeChecked();
    await expect(confidence).toHaveValue("0.5");
    await expect(confidence).toBeDisabled();
    await enableOcr.check();
    const device = page.getByLabel("OCR device", { exact: true });
    await expect(device).toBeEnabled();
    await expect(device).toHaveValue("auto");
    await expect(page.getByLabel("Automatic detection limit")).toBeChecked();
    await expect(page.getByLabel("Automatic text batch")).toBeChecked();
    await expect(page.getByLabel("OCR detection limit", { exact: true })).toBeDisabled();
    await expect(page.getByLabel("OCR text batch", { exact: true })).toBeDisabled();
    await confidence.fill("1.4");
    await confidence.press("Tab");
    await expect(confidence).toHaveValue("1");
    await page.locator(".workflow-rail").getByRole("button", { name: /NL/ }).click();
    await expect(page.getByRole("checkbox", { name: "Enable NL" })).toBeChecked();
    await openSetup(page);
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\ocr-independent");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/preflight").length).toBe(1);
    const config = (mutationsFor(api, "POST", "/api/jobs/preflight")[0].body as { config: Record<string, unknown> }).config;
    expect(mutationsFor(api, "POST", "/api/jobs/preflight")[0].body).toMatchObject({
      config: { schemaVersion: 10, ocr: { enabled: true, device: "auto", llmMinConfidence: 1, forceReprocess: false, resourceId: "ocr-ppocrv5-server-paddle-v1" } },
      ocrExecution: {
        textDetLimitSideLen: { mode: "auto", value: null },
        textBatchSize: { mode: "auto", value: null },
      },
    });
  });

  test("allows OCR-disabled preflight and disables unavailable OCR while showing the install command", async ({ page, api }) => {
    const resource = api.resources.resources.find((item) => item.resourceId === "ocr-ppocrv5-server-paddle-v1");
    if (!resource) throw new Error("OCR fixture resource is missing");
    resource.available = false;
    await openApp(page, { language: "en" });
    await expect(page.getByText("Resources ready", { exact: true })).toBeVisible();
    await page.locator(".workflow-rail").getByRole("button", { name: /OCR/ }).click();
    await expect(page.getByText("Import-OcrResource.bat -Apply", { exact: true })).toBeVisible();
    await openSetup(page);
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\ocr-disabled");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/preflight").length).toBe(1);
    await page.locator(".workflow-rail").getByRole("button", { name: /OCR/ }).click();
    await expect(page.getByRole("checkbox", { name: "Enable OCR" })).toBeDisabled();
  });

  test("creates v10 tasks with independent E621 defaults", async ({ page, api }) => {
    await openApp(page, { language: "en" });
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\e621-v10");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/preflight").length).toBe(1);
    const config = (mutationsFor(api, "POST", "/api/jobs/preflight")[0].body as { config: Record<string, unknown> }).config;
    expect(config.schemaVersion).toBe(10);
    expect(config.profile).toBeUndefined();
    expect(config.classify).toMatchObject({ indexMode: "bundled", resourceId: "classify-e621-20260724-v1" });
    expect(config.ocr).toEqual({ enabled: false, device: "auto", llmMinConfidence: 0.5, forceReprocess: false, resourceId: "ocr-ppocrv5-server-paddle-v1" });
  });

  test("selects a custom classification resource.json independently", async ({ page, api }) => {
    await openApp(page, { language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /Classify/ }).click();
    await page.getByRole("button", { name: "Custom resource", exact: true }).click();
    await page.getByRole("button", { name: "Select path", exact: true }).click();
    await expect(page.getByLabel("Classification resource.json", { exact: true })).toHaveValue("E:\\picked\\resource.json");

    await openSetup(page);
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\custom-classify");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/preflight").length).toBe(1);
    const config = (mutationsFor(api, "POST", "/api/jobs/preflight")[0].body as { config: Record<string, unknown> }).config;
    expect(config).toMatchObject({
      classify: { indexMode: "custom", customResourcePath: "E:\\picked\\resource.json" },
    });
    expect((config.classify as Record<string, unknown>).resourceId).toBeUndefined();
  });

  test("a pending preflight disables only its duplicate trigger", async ({ page, api }) => {
    const releasePreflight = holdRoute(api, "POST /api/jobs/preflight");
    await openApp(page, { language: "en" });
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\pending-preflight");
    await openPrecheck(page);

    const preflight = page.getByRole("button", { name: "Preflight", exact: true });
    await preflight.click();
    await expect(preflight).toBeDisabled();
    releasePreflight();
  });

  test("a pending preflight freezes Caption, Classify, and Replace configuration", async ({ page, api }) => {
    const releasePreflight = holdRoute(api, "POST /api/jobs/preflight");
    await openApp(page, { language: "en" });
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\pending-preflight-controls");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();
    await expect(page.getByRole("button", { name: "Preflight", exact: true })).toBeDisabled();

    await page.locator(".workflow-rail").getByRole("button", { name: /Caption/ }).click();
    await expect(page.locator("#caption-tagging-model")).toBeDisabled();

    await page.locator(".workflow-rail").getByRole("button", { name: /Classify/ }).click();
    await expect(page.getByRole("button", { name: "Custom resource", exact: true })).toBeDisabled();
    await expect(page.locator("#classify-resource")).toBeDisabled();

    await page.locator(".workflow-rail").getByRole("button", { name: /Replace/ }).click();
    await expect(page.getByRole("checkbox", { name: "Enable E621 replacement" })).toBeDisabled();
    await expect(page.locator("#replace-mode")).toBeDisabled();
    releasePreflight();
  });

  test("a pending preflight cannot submit the same mutation twice", async ({ page, api }) => {
    const releasePreflight = holdRoute(api, "POST /api/jobs/preflight");
    await openApp(page, { language: "en" });
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\pending-preflight");
    await openPrecheck(page);

    const preflight = page.getByRole("button", { name: "Preflight", exact: true });
    await preflight.click();
    await expect(preflight).toBeDisabled();
    await preflight.dispatchEvent("click");
    releasePreflight();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/preflight").length).toBe(1);
  });

  test("the pending preflight region advertises scoped busy state", async ({ page, api }) => {
    const releasePreflight = holdRoute(api, "POST /api/jobs/preflight");
    await openApp(page, { language: "en" });
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\pending-preflight");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();

    await expect(page.locator(".step-actions[aria-busy]").first()).toHaveAttribute("aria-busy", "true");
    releasePreflight();
  });

  test("renders a failed preflight in the action-feedback region", async ({ page, api }) => {
    failRoute(api, "POST /api/jobs/preflight", "preflight unavailable");
    await openApp(page, { language: "en" });
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\failed-preflight");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();

    await expect(page.locator('.action-feedback p[role="alert"]')).toHaveText("preflight unavailable");
  });

  test("shows determinate image progress while asynchronous preflight is running", async ({ page, api }) => {
    await openApp(page, { language: "en" });
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\progress");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();
    await expect.poll(() => mutationsFor(api, "POST", "/api/jobs/preflight/async").length).toBe(1);
    const releasePreflight = holdRoute(api, "GET /api/jobs/preflight/job-e621-characterization/progress");
    await expect(page.getByRole("progressbar", { name: "Preflight image progress", exact: true })).toBeVisible();
    await expect(page.getByText("0 / 3", { exact: true })).toBeVisible();
    releasePreflight();
    await expect(page.getByRole("progressbar", { name: "Preflight image progress", exact: true })).toHaveAttribute("value", "3");
    await expect(page.getByText("3 / 3", { exact: true })).toBeVisible();
    await expect(page.getByText("Samples", { exact: true })).toBeVisible();
  });

  test("renders actionable dataset claim guidance after workspace confirmation conflicts", async ({ page, api }) => {
    const detail = "Dataset is claimed by task 3bc585. Select it under Recent tasks: Recover keeps its progress and continues to hold the dataset; Discard deletes its overlay and releases the dataset.";
    failRoute(api, "POST /api/jobs/job-e621-characterization/confirm-workspace", detail, 409);
    await openApp(page, { language: "en" });
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\claimed");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();

    page.once("dialog", (dialog) => dialog.accept());
    await page.getByRole("button", { name: "Start pipeline", exact: true }).click();

    const alert = page.locator('.action-feedback p[role="alert"]');
    await expect(alert).toHaveText(detail);
    await expect(alert).toContainText("task 3bc585");
    await expect(alert).toContainText("Recent tasks");
    await expect(alert).toContainText("Recover keeps its progress and continues to hold the dataset");
    await expect(alert).toContainText("Discard deletes its overlay and releases the dataset");
    await expect(alert).not.toContainText("request failed: 409");
    await expect(alert).not.toContainText("request failed: 500");
  });
});
