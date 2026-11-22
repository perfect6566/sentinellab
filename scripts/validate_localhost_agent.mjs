import { mkdir, readFile, writeFile } from "node:fs/promises";

const origin = process.env.LOCAL_AGENT_ORIGIN || "http://127.0.0.1:5000";
const output = process.argv[2] || "reports/localhost-agent-evaluation-2026-11-03.json";
const browserVersion = await fetch("http://127.0.0.1:9333/json/version").then((response) => response.json());
const browserWs = new WebSocket(browserVersion.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  browserWs.addEventListener("open", resolve, { once: true });
  browserWs.addEventListener("error", reject, { once: true });
});
let browserNextId = 0;
const browserPending = new Map();
browserWs.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id && browserPending.has(message.id)) {
    browserPending.get(message.id)(message);
    browserPending.delete(message.id);
  }
});
const browserCdp = (method, params = {}) => new Promise((resolve) => {
  const id = ++browserNextId;
  browserPending.set(id, resolve);
  browserWs.send(JSON.stringify({ id, method, params }));
});
const context = await browserCdp("Target.createBrowserContext", { disposeOnDetach: true });
const target = await browserCdp("Target.createTarget", {
  url: `${origin}/`,
  browserContextId: context.result.browserContextId,
});
await new Promise((resolve) => setTimeout(resolve, 1000));
const cdpPages = await fetch("http://127.0.0.1:9333/json/list").then((response) => response.json());
const page = cdpPages.find((item) => item.id === target.result.targetId);
if (!page) throw new Error(`Could not open an isolated browser context at ${origin}`);

const localFiles = {};
for (const [key, path] of Object.entries({
  single: "samples/single-record.json",
  batch: "samples/batch-sample.csv",
  labeled: "samples/labeled-sample.csv",
  flipped: "samples/evaluation/labels-flipped.csv",
  invalid: "samples/evaluation/invalid-feature.csv",
  missing: "samples/evaluation/missing-label.csv",
  singleClass: "samples/evaluation/single-class.csv",
})) {
  localFiles[key] = (await readFile(path)).toString("base64");
}

const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  ws.addEventListener("open", resolve, { once: true });
  ws.addEventListener("error", reject, { once: true });
});
let nextId = 0;
const pending = new Map();
ws.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    pending.get(message.id)(message);
    pending.delete(message.id);
  }
});
const cdp = (method, params = {}) => new Promise((resolve) => {
  const id = ++nextId;
  pending.set(id, resolve);
  ws.send(JSON.stringify({ id, method, params }));
});

await cdp("Runtime.enable");
const expression = `
(async () => {
  const files = ${JSON.stringify(localFiles)};
  const decode = (value) => Uint8Array.from(atob(value), (character) => character.charCodeAt(0));
  const readJson = (encoded) => JSON.parse(new TextDecoder().decode(decode(encoded)));
  const health = await fetch('/health').then((response) => response.json());
  if (!health.ready) throw new Error('Localhost health is not ready');
  const cases = [];
  const upload = async (conversationId, name, content) => {
    const form = new FormData();
    form.append('file', new File([content], name, {type: 'text/csv'}));
    const response = await fetch('/api/upload', {
      method: 'POST', headers: {'X-Conversation-ID': conversationId}, body: form
    });
    const data = await response.json();
    if (!response.ok) throw new Error('Upload failed: ' + JSON.stringify(data));
    return data.file.reference;
  };
  const chat = async (conversationId, prompt, options = {}) => {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Conversation-ID': conversationId},
      body: JSON.stringify({
        message: prompt,
        ...(options.condition ? {condition: options.condition} : {}),
        ...(options.contextResultId ? {context_result_id: options.contextResultId} : {})
      })
    });
    return {status: response.status, data: await response.json()};
  };
  const resultDetails = async (conversationId, response) => Promise.all(
    (response.data.result_ids || []).map(async (resultId) => {
      const detail = await fetch('/api/results/' + encodeURIComponent(resultId), {
        headers: {'X-Conversation-ID': conversationId}
      });
      return detail.ok ? (await detail.json()).result : null;
    })
  );
  const addCase = async ({name, id, prompt, expectedTools, expectedOk = true, options, extraCheck}) => {
    const response = await chat(id, prompt, options);
    const results = (await resultDetails(id, response)).filter(Boolean);
    const actualTools = (response.data.activities || []).map((activity) => activity.tool);
    const checks = {
      tools: JSON.stringify(actualTools) === JSON.stringify(expectedTools),
      success: response.data.ok === expectedOk,
      ...(extraCheck ? extraCheck(response, results) : {})
    };
    cases.push({
      name, conversation_id: id, prompt, http_status: response.status,
      expected_tools: expectedTools, actual_tools: actualTools,
      expected_success: expectedOk, ok: response.data.ok,
      behavior_ok: Object.values(checks).every(Boolean), checks,
      error_code: response.data.error_code, final_answer: response.data.answer,
      activities: response.data.activities || [], structured_results: results,
      turns: response.data.turns, tool_calls: response.data.tool_calls
    });
    return response;
  };
  const single = readJson(files.single);
  const singleId = crypto.randomUUID();
  await addCase({
    name: 'AE-01 single prediction', id: singleId,
    prompt: 'Use predict_single for this complete feature record: ' + JSON.stringify(single),
    expectedTools: ['predict_single'],
    extraCheck: (response, results) => ({hasPrediction: results.some((item) => item.tool === 'predict_single')})
  });

  const batchId = crypto.randomUUID();
  const batchRef = await upload(batchId, 'batch-sample.csv', decode(files.batch));
  await addCase({
    name: 'AE-02 batch prediction', id: batchId,
    prompt: 'Run batch prediction for file reference ' + batchRef + '.',
    expectedTools: ['predict_batch'],
    extraCheck: (response, results) => ({fiveRows: results.some((item) => item.data.summary.input_rows === 5)})
  });

  const metId = crypto.randomUUID();
  const labeledRef = await upload(metId, 'labeled-sample.csv', decode(files.labeled));
  const metPrompt = 'Evaluate labeled file ' + labeledRef + '. If accuracy is at least 98%, predict this record: ' + JSON.stringify(single);
  const metResponse = await addCase({
    name: 'AE-03 threshold met', id: metId, prompt: metPrompt,
    expectedTools: ['evaluate_labeled', 'predict_single'],
    options: {condition: {minimum_accuracy: 98, evaluation_file_reference: labeledRef}},
    extraCheck: (response, results) => ({accuracyOne: results.some((item) => item.tool === 'evaluate_labeled' && item.data.accuracy === 1)})
  });
  const evaluationResultId = metResponse.data.result_ids?.[0];
  if (!evaluationResultId) throw new Error('AE-03 did not store the evaluation result');

  const notMetId = crypto.randomUUID();
  const flippedRef = await upload(notMetId, 'labels-flipped.csv', decode(files.flipped));
  await addCase({
    name: 'AE-04 threshold not met', id: notMetId,
    prompt: 'Evaluate this labeled file and predict the supplied record only if accuracy is at least 98%. File reference: ' + flippedRef + '. Record: ' + JSON.stringify(single),
    expectedTools: ['evaluate_labeled'],
    options: {condition: {minimum_accuracy: 98, evaluation_file_reference: flippedRef}},
    extraCheck: (response, results) => ({accuracyZero: results.some((item) => item.tool === 'evaluate_labeled' && item.data.accuracy === 0), noPrediction: !(response.data.activities || []).some((activity) => activity.tool === 'predict_single')})
  });

  await addCase({
    name: 'AE-05 false-negative follow-up', id: metId,
    prompt: 'How many false negatives were there in that evaluation?',
    expectedTools: [], options: {contextResultId: evaluationResultId},
    extraCheck: (response) => ({mentionsZero: /\\b0\\b/.test(response.data.answer || '')})
  });

  const invalidId = crypto.randomUUID();
  const invalidRef = await upload(invalidId, 'invalid-feature.csv', decode(files.invalid));
  await addCase({
    name: 'AE-06 invalid feature input', id: invalidId,
    prompt: 'Use predict_batch for file reference ' + invalidRef,
    expectedTools: ['predict_batch'], expectedOk: false,
    extraCheck: (response) => ({failureSurfaced: /invalid|valid row|failed/i.test(response.data.answer || '')})
  });

  const missingId = crypto.randomUUID();
  const missingRef = await upload(missingId, 'missing-label.csv', decode(files.missing));
  await addCase({
    name: 'AE-07 missing labels', id: missingId,
    prompt: 'Evaluate labeled file reference ' + missingRef,
    expectedTools: ['evaluate_labeled'], expectedOk: false,
    extraCheck: (response) => ({missingLabelSurfaced: /label/i.test(response.data.answer || '')})
  });

  const oneClassId = crypto.randomUUID();
  const oneClassRef = await upload(oneClassId, 'single-class.csv', decode(files.singleClass));
  await addCase({
    name: 'AE-08 single-class evaluation', id: oneClassId,
    prompt: 'Evaluate this one-class labeled CSV file reference ' + oneClassRef,
    expectedTools: ['evaluate_labeled'],
    extraCheck: (_response, results) => ({aucUndefined: results.some((item) => item.tool === 'evaluate_labeled' && item.data.auc === null)})
  });

  const failureId = crypto.randomUUID();
  const absentRef = crypto.randomUUID();
  await addCase({
    name: 'AE-09 unavailable file tool failure', id: failureId,
    prompt: 'Evaluate labeled CSV file reference ' + absentRef,
    expectedTools: ['evaluate_labeled'], expectedOk: false,
    extraCheck: (response) => ({toolFailureSurfaced: /failed|unavailable|invalid/i.test(response.data.answer || '')})
  });

  const metadataId = crypto.randomUUID();
  await addCase({
    name: 'LOCAL-META dataset size question', id: metadataId,
    prompt: 'Hello, could you describe your dataset size for training and testing?',
    expectedTools: [],
    extraCheck: (response) => ({hasTrainSize: /38,122/.test(response.data.answer || ''), hasTestSize: /9,531/.test(response.data.answer || '')})
  });
  const required = cases.filter((item) => item.name.startsWith('AE-'));
  const report = {
    created_at_utc: new Date().toISOString(),
    scope: 'localhost Flask application treated as final deployment; actual browser-origin API and live configured Agent provider',
    origin: location.origin,
    provider: health.agent,
    production_model: health.model.model_version,
    required_scenario_count: required.length,
    required_scenarios_passed: required.filter((item) => item.behavior_ok).length,
    additional_scenarios: cases.filter((item) => !item.name.startsWith('AE-')).length,
    failed: cases.filter((item) => !item.behavior_ok).map((item) => item.name),
    scenarios: cases
  };
  return JSON.stringify(report);
})()`;
const evaluation = await cdp("Runtime.evaluate", {
  expression,
  awaitPromise: true,
  returnByValue: true,
});
const serialized = evaluation.result?.result?.value;
if (typeof serialized !== "string") {
  throw new Error(JSON.stringify(evaluation.result?.exceptionDetails || evaluation));
}
const report = JSON.parse(serialized);
const summary = {
  required_scenario_count: report.required_scenario_count,
  required_scenarios_passed: report.required_scenarios_passed,
  failed: report.failed,
};
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
const screenshotDirectory = "reports/screenshots/localhost-evaluation-2026-11-03";
await mkdir(screenshotDirectory, { recursive: true });
for (const item of report.scenarios) {
  const render = `(() => {
    document.querySelector('.conversation-section')?.setAttribute('hidden', '');
    document.querySelector('#uploads')?.replaceChildren();
    const messages = document.querySelector('#messages');
    messages.replaceChildren();
    state.conversationId = ${JSON.stringify(item.conversation_id)};
    renderMessage('user', ${JSON.stringify(item.prompt)});
    renderMessage('assistant', ${JSON.stringify(item.final_answer)}, ${JSON.stringify(item.activities)});
    ${JSON.stringify(item.structured_results)}.forEach(addResultCard);
    messages.scrollTop = messages.scrollHeight;
    return true;
  })()`;
  await cdp("Runtime.evaluate", { expression: render, returnByValue: true });
  const metrics = await cdp("Page.getLayoutMetrics");
  const size = metrics.result.cssContentSize;
  const screenshot = await cdp("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: true,
    fromSurface: true,
    clip: { x: 0, y: 0, width: size.width, height: size.height, scale: 1 },
  });
  const suffix = item.behavior_ok ? "pass" : "fail";
  const slug = item.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  await writeFile(`${screenshotDirectory}/${slug}-${suffix}.png`, Buffer.from(screenshot.result.data, "base64"));
}
console.log(JSON.stringify({
  output,
  screenshotDirectory,
  provider: report.provider,
  production_model: report.production_model,
  required_scenario_count: summary.required_scenario_count,
  required_scenarios_passed: summary.required_scenarios_passed,
  failed: summary.failed,
  cases: report.scenarios.map((item) => ({name: item.name, behavior_ok: item.behavior_ok, actual_tools: item.actual_tools, error_code: item.error_code})),
}));
ws.close();
await browserCdp("Target.closeTarget", { targetId: target.result.targetId });
await browserCdp("Target.disposeBrowserContext", {
  browserContextId: context.result.browserContextId,
});
browserWs.close();
