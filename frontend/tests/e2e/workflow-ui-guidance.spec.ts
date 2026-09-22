import { expect, openApp, openPrecheck, openSetup, test } from "./mockApi";

async function expectNoHorizontalOverflow(page: Parameters<typeof openApp>[0]) {
  const size = await page.evaluate(() => ({ scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth }));
  expect(size.scrollWidth).toBeLessThanOrEqual(size.clientWidth);
}

test.describe("workflow field guidance foundation", () => {
  test("shows an evidence-backed Setup default without changing the control", async ({ page }) => {
    await openApp(page, { language: "en" });
    const workMode = page.getByLabel("Work mode", { exact: true });
    await expect(workMode).toHaveValue("in_place");
    const info = page.getByRole("button", { name: "Work mode information" });
    await info.click();
    const tooltip = page.getByRole("tooltip");
    await expect(tooltip).toContainText("Controls whether annotations are committed to the source dataset or a copied dataset.");
    await expect(tooltip).toContainText("Default: In place");
    await expect(workMode).toHaveValue("in_place");
  });

  test("opens guidance by keyboard and closes it with Escape", async ({ page }) => {
    await openApp(page, { language: "en" });
    const info = page.getByRole("button", { name: "Source dataset information" });
    await info.focus();
    await info.press("Enter");
    await expect(page.getByRole("tooltip")).toBeVisible();
    await info.press("Escape");
    await expect(page.getByRole("tooltip")).toBeHidden();
  });

  test("opens guidance only by activation and toggles it closed on a second click", async ({ page }) => {
    await openApp(page, { language: "en" });
    const info = page.getByRole("button", { name: "Work mode information" });
    const tooltip = page.getByRole("tooltip");
    await info.hover();
    await expect(tooltip).toBeHidden();
    await info.click();
    await expect(tooltip).toBeVisible();
    await info.click();
    await expect(tooltip).toBeHidden();
  });

  test("documents the JSON and TXT input formats in Setup", async ({ page }) => {
    for (const scenario of [
      { language: "en" as const, title: "JSON / TXT input formats", pairing: "same-stem siblings", raw: "Raw E621 JSON", txt: "NL TXT supplies JSON nl" },
      { language: "zh-CN" as const, title: "JSON / TXT 输入格式", pairing: "同 stem 兄弟文件", raw: "原始 E621 JSON", txt: "NL TXT 写入 JSON 的 nl" },
    ]) {
      await openApp(page, { language: scenario.language });
      const help = page.locator('[data-input-format-help]');
      await help.locator("summary").click();
      await expect(help).toContainText(scenario.title);
      await expect(help).toContainText(scenario.pairing);
      await expect(help).toContainText(scenario.raw);
      await expect(help).toContainText(scenario.txt);
    }
  });
});

test.describe("workflow guidance for resource-backed steps", () => {
  test("uses independent resources without a task annotation profile", async ({ page }) => {
    await openApp(page, { language: "en" });
    const setup = page.locator('[data-config-surface="setup"]');
    await expect(setup.getByText("Annotation profile", { exact: true })).toHaveCount(0);
    await expect(setup.getByRole("button", { name: "E621", exact: true })).toHaveCount(0);
    await expect(setup.getByRole("button", { name: "Danbooru", exact: true })).toHaveCount(0);

    await page.locator(".workflow-rail").getByRole("button", { name: /Classify/ }).click();
    await expect(page.getByRole("button", { name: "Bundled resource", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Custom resource", exact: true })).toBeVisible();
    await page.getByRole("button", { name: "Custom resource", exact: true }).click();
    await expect(page.getByLabel("Classification resource.json", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Select path", exact: true })).toBeVisible();

    await page.locator(".workflow-rail").getByRole("button", { name: /Replace/ }).click();
    await expect(page.getByRole("checkbox", { name: "Enable E621 replacement", exact: true })).toBeChecked();
  });

  test("shows the selected resource explanation and profile recommendation", async ({ page }) => {
    await openApp(page, { language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /Caption/ }).click();
    const info = page.getByRole("button", { name: "Tagging model information" });
    await info.click();
    await expect(page.getByRole("tooltip")).toContainText("Recommended: E621 tagger");

    await page.locator(".workflow-rail").getByRole("button", { name: /Classify/ }).click();
    await page.getByRole("button", { name: "Classification and Count index information" }).click();
    await expect(page.getByRole("tooltip")).toContainText("Recommended: E621 classify index");
  });

  test("wraps every visible Caption, Classify, and Replace control in a setting field", async ({ page }) => {
    await openApp(page, { language: "en" });
    for (const stepName of [/Caption/, /Classify/, /Replace/]) {
      await page.locator(".workflow-rail").getByRole("button", { name: stepName }).click();
      const controls = page.locator(".step-content input:visible, .step-content select:visible, .step-content textarea:visible");
      const count = await controls.count();
      expect(count).toBeGreaterThan(0);
      for (let index = 0; index < count; index += 1) {
        await expect(controls.nth(index).locator("xpath=ancestor::*[@data-setting-field][1]")).toHaveCount(1);
      }
    }
  });
});

test.describe("workflow guidance for remaining modules", () => {
  test("shows OCR defaults and runtime recommendations without enabling disabled controls", async ({ page }) => {
    await openApp(page, { language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /OCR/ }).click();
    await expect(page.getByLabel("OCR device", { exact: true })).toBeDisabled();
    const deviceInfo = page.getByRole("button", { name: "OCR device information" });
    await deviceInfo.click();
    await expect(page.getByRole("tooltip")).toContainText("Default: Auto");
    await expect(page.getByRole("tooltip")).toContainText("Recommended: Auto");

    await page.getByLabel("Enable OCR", { exact: true }).check();
    await openSetup(page);
    await page.getByRole("textbox", { name: "Source dataset", exact: true }).fill("E:\\datasets\\guidance");
    await openPrecheck(page);
    await page.getByRole("button", { name: "Preflight", exact: true }).click();
    await expect.poll(() => page.getByRole("button", { name: "Preflight", exact: true }).getAttribute("aria-busy")).toBe("false");
    await page.locator(".workflow-rail").getByRole("button", { name: /OCR/ }).click();
    const detectionInfo = page.getByRole("button", { name: "OCR detection limit information" });
    await detectionInfo.click();
    await expect(page.getByRole("tooltip")).toContainText("Recommended: 2560");
    const ocrTooltip = page.locator(".ocr-tuning .field-tooltip:visible");
    await expect(ocrTooltip).toBeVisible();
    const ocrBounds = await ocrTooltip.evaluate((node) => {
      const rect = node.getBoundingClientRect();
      return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom, width: window.innerWidth, height: window.innerHeight };
    });
    expect(ocrBounds.left).toBeGreaterThanOrEqual(0);
    expect(ocrBounds.right).toBeLessThanOrEqual(ocrBounds.width);
    expect(ocrBounds.top).toBeGreaterThanOrEqual(0);
    expect(ocrBounds.bottom).toBeLessThanOrEqual(ocrBounds.height);
    await page.getByRole("button", { name: "OCR text batch information" }).click();
    await expect(page.getByRole("tooltip")).toContainText("Recommended: 4");
  });

  test("separates Token Budget defaults from the tokenizer range", async ({ page, api }) => {
    const tokenizer = api.resources.resources.find((item) => item.resourceId === "tokenizer-qwen3-0.6b-anima-v1");
    if (!tokenizer) throw new Error("tokenizer fixture missing");
    tokenizer.default = true;
    await openApp(page, { language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /Token Budget/ }).click();
    await page.getByRole("button", { name: "Maximum training tokens information" }).click();
    await expect(page.getByRole("tooltip")).toContainText("Default: 512");
    await expect(page.getByRole("tooltip")).toContainText("Range: 1-40960");
    const tokenizerInfo = page.getByRole("button", { name: "Tokenizer resource information" });
    await tokenizerInfo.click();
    await expect(page.getByRole("tooltip")).toContainText("Recommended: Qwen/Qwen3-0.6B");
    await expect(page.getByRole("tooltip")).not.toContainText("Recommended: 40960");

  });

  test("wraps Count Review, Policy, and Export controls and documents policy probabilities", async ({ page }) => {
    await openApp(page, { language: "en" });
    for (const stepName of [/Count Review/, /Dropout/, /Export/]) {
      await page.locator(".workflow-rail").getByRole("button", { name: stepName }).click();
      const controls = page.locator(".step-content input:visible, .step-content select:visible, .step-content textarea:visible");
      const count = await controls.count();
      expect(count).toBeGreaterThan(0);
      for (let index = 0; index < count; index += 1) {
        await expect(controls.nth(index).locator("xpath=ancestor::*[@data-setting-field][1]")).toHaveCount(1);
      }
    }
    await page.locator(".workflow-rail").getByRole("button", { name: /Dropout/ }).click();
    const nlDropoutInfo = page.getByRole("button", { name: "NL dropout rate information" }).first();
    await expect(nlDropoutInfo).toBeEnabled();
    await nlDropoutInfo.click();
    await expect(page.getByRole("tooltip")).toContainText("Default: 0.7");
    await expect(page.getByRole("tooltip")).toContainText("Range: 0-1, step 0.01");
    await expect(page.getByRole("tooltip")).toContainText("JSON nl");
    const tagDropoutInfo = page.getByRole("button", { name: "Tag dropout rate information" }).first();
    await tagDropoutInfo.click();
    await expect(page.getByRole("tooltip")).toContainText("JSON tags");
  });

  test("explains the folder artist mapping independently from the Character preset", async ({ page }) => {
    for (const scenario of [
      { language: "en" as const, label: "Append folder to JSON artist", firstLine: "Extract the name from the image's first-level folder (format: n_name), or from a flat dataset root named n_name", character: "independent of the Character preset", preflight: "Import preflight validates the folder or root name", folder: "10_uiokv" },
      { language: "zh-CN" as const, label: "将目录名追加到 JSON artist", firstLine: "从图片的一级目录（格式：n_名称）或符合命名规则的平铺数据集根目录提取名称", character: "独立于 Character 预设", preflight: "导入预检会校验目录或根目录名称", folder: "10_uiokv" },
    ]) {
      await openApp(page, { language: scenario.language });
      await page.locator(".workflow-rail").getByRole("button", { name: /Dropout/ }).click();
      const field = page.locator("#policy-artist-enabled").locator("xpath=ancestor::*[@data-setting-field][1]");
      await expect(field.getByText(scenario.label, { exact: true })).toBeVisible();
      await expect(field.locator("input")).toBeDisabled();
      await field.getByRole("button").click();
      const tooltip = page.getByRole("tooltip");
      await expect(tooltip).toContainText(scenario.firstLine);
      await expect(tooltip).toContainText("@" + (scenario.language === "en" ? "name" : "名称"));
      await expect(tooltip).toContainText(scenario.preflight);
      await expect(tooltip).toContainText(scenario.folder);
      await expect(tooltip).toContainText(scenario.character);
    }
  });
});

test.describe("workflow guidance for NL", () => {
  test("wraps configurable controls and distinguishes NL defaults from provider settings", async ({ page }) => {
    await openApp(page, { language: "en" });
    await page.locator(".workflow-rail").getByRole("button", { name: /NL/ }).click();
    const controls = page.locator(".step-content input:visible, .step-content select:visible, .step-content textarea:visible");
    const count = await controls.count();
    expect(count).toBeGreaterThan(0);
    for (let index = 0; index < count; index += 1) {
      await expect(controls.nth(index).locator("xpath=ancestor::*[@data-setting-field][1]")).toHaveCount(1);
    }

    await page.getByRole("button", { name: "Batch size information" }).click();
    await expect(page.getByRole("tooltip")).toContainText("Default: 3");
    await expect(page.getByRole("tooltip")).toContainText("Range: 1-16");
    await page.getByRole("button", { name: "Endpoint URL information" }).click();
    await expect(page.getByRole("tooltip")).not.toContainText("Default:");
    await expect(page.getByRole("tooltip")).not.toContainText("Recommended:");
  });
});

test.describe("workflow guidance responsive contracts", () => {
  test("keeps every visible workflow control in a setting field without desktop overlap", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await openApp(page, { language: "en" });
    const navigation = page.locator(".workflow-rail .step-nav");
    const stepCount = await navigation.count();
    expect(stepCount).toBeGreaterThan(0);
    for (let stepIndex = 0; stepIndex < stepCount; stepIndex += 1) {
      await navigation.nth(stepIndex).click();
      const controls = page.locator(".step-content input:visible, .step-content select:visible, .step-content textarea:visible");
      for (let controlIndex = 0; controlIndex < await controls.count(); controlIndex += 1) {
        await expect(controls.nth(controlIndex).locator("xpath=ancestor::*[@data-setting-field][1]")).toHaveCount(1);
      }
      const fields = await page.locator(".step-content .form-field:visible").evaluateAll((nodes) => nodes.map((node) => {
        const rect = node.getBoundingClientRect();
        return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom };
      }));
      for (let firstIndex = 0; firstIndex < fields.length; firstIndex += 1) {
        for (let secondIndex = firstIndex + 1; secondIndex < fields.length; secondIndex += 1) {
          const first = fields[firstIndex];
          const second = fields[secondIndex];
          const overlaps = first.left < second.right && first.right > second.left && first.top < second.bottom && first.bottom > second.top;
          expect(overlaps, `overlapping fields at step ${stepIndex}`).toBe(false);
        }
      }
    }
    await page.locator(".workflow-rail").getByRole("button", { name: /NL/ }).click();
    await expect(page.locator('.step-content [data-config-surface="nl"]')).toHaveCount(1);
  });

  test("keeps a clicked mobile tooltip inside the Chinese viewport and closes it outside", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await openApp(page, { language: "zh" });
    await page.locator(".field-info-button:visible").first().click();
    const tooltip = page.locator(".field-tooltip:visible");
    await expect(tooltip).toBeVisible();
    const bounds = await tooltip.evaluate((node) => {
      const rect = node.getBoundingClientRect();
      return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom, width: window.innerWidth, height: window.innerHeight };
    });
    expect(bounds.left).toBeGreaterThanOrEqual(0);
    expect(bounds.right).toBeLessThanOrEqual(bounds.width);
    expect(bounds.top).toBeGreaterThanOrEqual(0);
    expect(bounds.bottom).toBeLessThanOrEqual(bounds.height);
    await page.mouse.click(2, 2);
    await expect(tooltip).toBeHidden();

    const navigation = page.locator(".workflow-rail .step-nav");
    for (let stepIndex = 0; stepIndex < await navigation.count(); stepIndex += 1) {
      await navigation.nth(stepIndex).click();
      await expectNoHorizontalOverflow(page);
    }
  });

  test("wraps English labels and keeps every control inside the 320px viewport", async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 844 });
    await openApp(page, { language: "en" });
    const navigation = page.locator(".workflow-rail .step-nav");
    for (let stepIndex = 0; stepIndex < await navigation.count(); stepIndex += 1) {
      await navigation.nth(stepIndex).click();
      await expectNoHorizontalOverflow(page);
      const viewportWidth = await page.evaluate(() => document.documentElement.clientWidth);
      const elements = page.locator(".step-content .form-field:visible, .step-content button:visible, .step-content .field-label:visible");
      for (let elementIndex = 0; elementIndex < await elements.count(); elementIndex += 1) {
        const geometry = await elements.nth(elementIndex).evaluate((node) => {
          const rect = node.getBoundingClientRect();
          return { left: rect.left, right: rect.right, scrollWidth: node.scrollWidth, clientWidth: node.clientWidth };
        });
        expect(geometry.left, `element extends left at step ${stepIndex}`).toBeGreaterThanOrEqual(0);
        expect(geometry.right, `element extends right at step ${stepIndex}`).toBeLessThanOrEqual(viewportWidth);
        expect(geometry.scrollWidth, `element clips text at step ${stepIndex}`).toBeLessThanOrEqual(geometry.clientWidth + 1);
      }
    }
  });
});
