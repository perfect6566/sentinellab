const $ = (selector) => document.querySelector(selector);
const STORAGE_KEY = "sentinel.conversations.v1";
const MAX_CONVERSATIONS = 12;
const WELCOME_MARKUP = $(".welcome")?.outerHTML || "";
const state = {
  conversations: [],
  conversationId: null,
  files: [],
  latestResultId: null,
  pendingCondition: null,
};

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.add("show");
  window.setTimeout(() => el.classList.remove("show"), 3200);
}

function activeConversation() {
  return state.conversations.find((item) => item.id === state.conversationId);
}

function persistConversations() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      conversations: state.conversations,
      activeId: state.conversationId,
    }));
  } catch {
    toast("Could not save conversation history in this browser.");
  }
}

function saveActiveConversation({ updateList = true } = {}) {
  const conversation = activeConversation();
  if (!conversation) return;
  conversation.files = state.files;
  conversation.latestResultId = state.latestResultId;
  conversation.pendingCondition = state.pendingCondition;
  conversation.updatedAt = Date.now();
  persistConversations();
  if (updateList) renderConversationList();
}

function newConversationRecord() {
  return {
    id: crypto.randomUUID(),
    title: "New analysis",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
    files: [],
    draft: "",
    latestResultId: null,
    pendingCondition: null,
  };
}

function renderConversationList() {
  const list = $("#conversation-list");
  if (!list) return;
  list.replaceChildren();
  [...state.conversations]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .forEach((conversation) => {
      const row = document.createElement("div");
      row.className = "conversation-item";
      if (conversation.id === state.conversationId) row.classList.add("active");
      const open = document.createElement("button");
      open.className = "conversation-open";
      open.type = "button";
      open.textContent = conversation.title || "New analysis";
      open.title = open.textContent;
      open.addEventListener("click", () => switchConversation(conversation.id));
      const remove = document.createElement("button");
      remove.className = "conversation-delete";
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Delete ${conversation.title || "analysis"}`);
      remove.addEventListener("click", () => deleteConversation(conversation.id));
      row.append(open, remove);
      list.append(row);
    });
}

function renderConversation(conversation) {
  state.conversationId = conversation.id;
  state.files = [...(conversation.files || [])];
  state.latestResultId = conversation.latestResultId || null;
  state.pendingCondition = conversation.pendingCondition || null;
  $("#message-input").value = conversation.draft || "";
  $("#messages").replaceChildren();
  if (!conversation.messages?.length) {
    const template = document.createElement("template");
    template.innerHTML = WELCOME_MARKUP;
    $("#messages").append(template.content.cloneNode(true));
  } else {
    conversation.messages.forEach((message) => {
      renderMessage(message.role, message.text, message.activities || []);
      (message.results || []).forEach(addResultCard);
    });
  }
  renderFiles();
  renderConversationList();
  $("#messages").scrollTop = $("#messages").scrollHeight;
}

function switchConversation(conversationId) {
  const previous = activeConversation();
  if (previous) {
    previous.draft = $("#message-input").value;
    saveActiveConversation({ updateList: false });
  }
  const conversation = state.conversations.find((item) => item.id === conversationId);
  if (!conversation) return;
  renderConversation(conversation);
  persistConversations();
}

function createConversation({ initial = false } = {}) {
  if (state.conversations.length >= MAX_CONVERSATIONS) {
    toast("Delete an analysis before creating another. The limit is 12.");
    return;
  }
  const conversation = newConversationRecord();
  state.conversations.unshift(conversation);
  renderConversation(conversation);
  persistConversations();
  if (!initial) $("#message-input").focus();
}

async function deleteConversation(conversationId) {
  const conversation = state.conversations.find((item) => item.id === conversationId);
  if (!conversation) return;
  if (!window.confirm(`Delete “${conversation.title}” from this browser?`)) return;
  try {
    await fetch(`/api/sessions/${encodeURIComponent(conversationId)}`, {
      method: "DELETE",
      headers: { "X-Conversation-ID": conversationId },
    });
  } catch {
    toast("Could not clear this conversation's server session.");
    return;
  }
  state.conversations = state.conversations.filter((item) => item.id !== conversationId);
  if (state.conversationId === conversationId) {
    state.conversationId = null;
    state.files = [];
    state.latestResultId = null;
    state.pendingCondition = null;
    if (!state.conversations.length) createConversation();
    else renderConversation(state.conversations[0]);
  } else {
    renderConversationList();
  }
  persistConversations();
}

function initializeConversations() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (Array.isArray(saved?.conversations)) {
      state.conversations = saved.conversations
        .filter((item) => typeof item.id === "string" && Array.isArray(item.messages))
        .slice(0, MAX_CONVERSATIONS);
      state.conversationId = saved.activeId;
    }
  } catch {
    state.conversations = [];
  }
  const active = state.conversations.find((item) => item.id === state.conversationId);
  if (active) renderConversation(active);
  else if (state.conversations.length) renderConversation(state.conversations[0]);
  else createConversation({ initial: true });
  persistConversations();
}

function renderMessage(role, text, activities = []) {
  if (role === "user") $(".welcome")?.remove();
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Y" : "S";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const content = document.createElement("p");
  content.textContent = text;
  bubble.append(content);
  if (activities.length) {
    const activity = document.createElement("details");
    activity.className = "activity";
    const summary = document.createElement("summary");
    summary.textContent = `${activities.length} tool ${activities.length === 1 ? "call" : "calls"}`;
    const list = document.createElement("ul");
    activities.forEach((item) => {
      const row = document.createElement("li");
      const head = document.createElement("div");
      head.textContent = `${item.tool}: ${item.status}`;
      row.append(head);
      const extras = [];
      if (item.arguments && Object.keys(item.arguments).length) {
        extras.push(`arguments: ${JSON.stringify(item.arguments)}`);
      }
      if (item.minimum_accuracy !== undefined && item.minimum_accuracy !== null) {
        extras.push(`minimum_accuracy: ${item.minimum_accuracy}`);
      }
      if (item.result_id) {
        extras.push(`result_id: ${item.result_id}`);
      }
      if (extras.length) {
        const detail = document.createElement("div");
        detail.textContent = extras.join(" · ");
        row.append(detail);
      }
      list.append(row);
    });
    activity.append(summary, list);
    bubble.append(activity);
  }
  article.append(avatar, bubble);
  $("#messages").append(article);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return article;
}

function addMessage(role, text, activities = [], results = []) {
  const article = renderMessage(role, text, activities);
  const conversation = activeConversation();
  if (conversation) {
    if (role === "user" && conversation.title === "New analysis") {
      conversation.title = text.replace(/\s+/g, " ").trim().slice(0, 38) || "New analysis";
    }
    conversation.messages.push({ role, text, activities, results });
    if (role === "user") conversation.draft = "";
    saveActiveConversation();
  }
  results.forEach(addResultCard);
  return article;
}

function addTypingIndicator() {
  const article = document.createElement("article");
  article.className = "message assistant typing-message";
  article.setAttribute("role", "status");
  article.innerHTML = `
    <div class="avatar">S</div>
    <div class="bubble typing-bubble">
      <span>Agent is typing</span>
      <span class="typing-dots" aria-hidden="true"><i></i><i></i><i></i></span>
    </div>`;
  $("#messages").append(article);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return article;
}

function metric(label, value) {
  const node = document.createElement("div");
  node.className = "metric";
  const small = document.createElement("small");
  small.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value ?? "—";
  node.append(small, strong);
  return node;
}

// Show the score exactly as the model reports it. Rounding it here would make
// the card disagree with the tool result the answer cites.
function formatScore(value) {
  if (value == null) return "—";
  const number = Number(value);
  return Number.isFinite(number) ? String(value) : "—";
}

function addResultCard(result) {
  const card = document.createElement("section");
  card.className = "result-card";
  const title = document.createElement("div");
  title.className = "result-title";
  const heading = document.createElement("h3");
  heading.textContent = result.tool.replaceAll("_", " ");
  const badge = document.createElement("span");
  badge.textContent = "MODEL RESULT";
  title.append(heading, badge);
  card.append(title);
  const data = result.data;
  const metrics = document.createElement("div");
  metrics.className = "metric-grid";
  if (result.tool === "predict_single") {
    metrics.append(
      metric("Classification", data.class_name),
      metric("Malware score", formatScore(data.malware_score)),
      metric("Threshold", data.threshold),
      metric("Model", data.model_version),
    );
  } else if (result.tool === "predict_batch") {
    const summary = data.summary;
    metrics.append(
      metric("Input rows", summary.input_rows), metric("Valid", summary.valid_rows),
      metric("Invalid", summary.invalid_rows), metric("Malware", summary.malware_count),
    );
    const rows = document.createElement("table");
    rows.className = "result-table";
    rows.innerHTML = "<thead><tr><th>Row</th><th>Status</th><th>Class</th><th>Score</th></tr></thead>";
    const body = document.createElement("tbody");
    (data.rows || []).slice(0, 8).forEach((item) => {
      const row = document.createElement("tr");
      [item.row_id, item.status, item.class_name || "—",
        item.malware_score == null ? "—" : formatScore(item.malware_score)]
        .forEach((value) => { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); });
      body.append(row);
    });
    rows.append(body);
    card.append(metrics, rows);
    if (data.result_file_reference) {
      const link = document.createElement("a");
      link.className = "download-link";
      link.href = `/api/download/${encodeURIComponent(data.result_file_reference)}?conversation_id=${encodeURIComponent(state.conversationId)}`;
      link.textContent = "Download complete predictions →";
      card.append(link);
    }
  } else if (result.tool === "evaluate_labeled") {
    const matrix = data.confusion_matrix;
    metrics.append(
      metric("Samples", data.sample_count), metric("Accuracy", `${(data.accuracy * 100).toFixed(2)}%`),
      metric("AUC", data.auc == null ? "Undefined" : data.auc.toFixed(4)), metric("Threshold", data.threshold),
    );
    const grid = document.createElement("div");
    grid.className = "matrix";
    [["TN", matrix.tn], ["FP", matrix.fp], ["FN", matrix.fn], ["TP", matrix.tp]].forEach(([label, value]) => {
      const cell = document.createElement("div"); cell.textContent = `${label} ${value}`; grid.append(cell);
    });
    card.append(metrics, grid);
    if (data.interpretation_note) {
      const note = document.createElement("p");
      note.className = "result-note";
      note.textContent = data.interpretation_note;
      card.append(note);
    }
  }
  if (!card.contains(metrics)) card.append(metrics);
  $("#messages").append(card);
  $("#messages").scrollTop = $("#messages").scrollHeight;
}

function displayAnswer(answer, results) {
  const batch = results.find((result) => result.tool === "predict_batch");
  const summary = batch?.data?.summary;
  if (!summary || !Number.isFinite(summary.input_rows)) return answer;
  const goodware = Number(summary.goodware_count) || 0;
  const malware = Number(summary.malware_count) || 0;
  return `Batch prediction complete: ${summary.input_rows} rows, ${summary.valid_rows} valid, ` +
    `${summary.invalid_rows} invalid. ${goodware} goodware and ${malware} malware predictions. ` +
    "Row-level scores and the complete download are below.";
}

// The response guard rejects a final answer whose wording contains numbers that
// no tool result supports. The tool calls themselves already succeeded, so the
// result cards below are correct. When that happens, restate the outcome from
// the tool results instead of showing the guard's internal message.
function groundedSummary(results) {
  const evaluation = results.find((result) => result.tool === "evaluate_labeled");
  const prediction = results.find((result) => result.tool === "predict_single");
  const batch = results.find((result) => result.tool === "predict_batch");
  if (!evaluation && !prediction && !batch) return null;

  const lines = [];
  if (evaluation) {
    const data = evaluation.data;
    const matrix = data.confusion_matrix || {};
    lines.push(
      `Labeled-file evaluation: ${data.sample_count} samples, accuracy ${data.accuracy}, ` +
      `AUC ${data.auc == null ? "undefined" : data.auc}, threshold ${data.threshold}. ` +
      `Confusion matrix: TN ${matrix.tn}, FP ${matrix.fp}, FN ${matrix.fn}, TP ${matrix.tp}.`
    );
  }
  if (prediction) {
    const data = prediction.data;
    lines.push(
      `Prediction: ${data.class_name}, malware_score ${data.malware_score}, ` +
      `threshold ${data.threshold}, model ${data.model_version}.`
    );
  }
  if (batch) {
    const data = batch.data.summary;
    lines.push(
      `Batch prediction: ${data.input_rows} rows, ${data.valid_rows} valid, ` +
      `${data.invalid_rows} invalid, ${data.goodware_count} goodware, ` +
      `${data.malware_count} malware.`
    );
  }
  return lines.join("\n\n");
}

function displayFallback(data, results) {
  const summary = groundedSummary(results);
  if (!summary) return data.answer || data.error || "No response";
  return (
    "The analysis completed and the model results below are authoritative. " +
    "The generated wording was rejected by the response guard because it stated a " +
    "figure that no tool result supports, so the outcome is reported directly from " +
    "the tool results.\n\n" + summary
  );
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    const health = $("#health");
    health.classList.add(data.ready ? "ready" : "error");
    health.querySelector("span:last-child").textContent = data.ready ? "Runtime ready" : "Not ready";
    $("#model-version").textContent = data.model.model_version || "Unavailable";
    $("#model-meta").textContent = data.ready
      ? `${data.model.feature_count} features · threshold ${data.model.threshold}`
      : "Check health details";
  } catch {
    $("#health").classList.add("error");
    $("#health span:last-child").textContent = "Health unavailable";
  }
}

async function upload(file) {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch("/api/upload", { method: "POST", body, headers: { "X-Conversation-ID": state.conversationId } });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Upload failed");
  state.files.push(data.file);
  saveActiveConversation();
  renderFiles();
  toast(`${data.file.name} uploaded`);
}

function renderFiles() {
  const attachments = $("#attachments");
  const uploads = $("#uploads");
  if (!attachments || !uploads) return;
  attachments.replaceChildren();
  uploads.replaceChildren();
  attachments.hidden = !state.files.length;
  state.files.forEach((file) => {
    const card = document.createElement("div");
    card.className = "attachment-card";
    card.innerHTML = `<span class="attachment-icon" aria-hidden="true">▤</span><span class="attachment-copy"><strong></strong><small>CSV · ready</small></span><button type="button" aria-label="Remove uploaded file">×</button>`;
    card.querySelector("strong").textContent = file.name;
    card.querySelector("button").addEventListener("click", () => {
      state.files = state.files.filter((item) => item.reference !== file.reference);
      saveActiveConversation();
      renderFiles();
    });
    attachments.append(card);
    const chip = document.createElement("div");
    chip.className = "file-chip";
    chip.innerHTML = `<span></span><small>ready</small>`;
    chip.querySelector("span").textContent = file.name;
    uploads.append(chip);
  });
}

function conversationHeaders(extra = {}) {
  return { ...extra, "X-Conversation-ID": state.conversationId };
}

function isExplicitNewTask(message) {
  return /^(?:use\s+)?predict_single\b|^run\s+batch\s+prediction\b|^evaluate\s+(?:labeled|labelled)\b/i
    .test(message.trim());
}

function latestFile() { return state.files.at(-1); }

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-template]");
  if (button) void (async () => {
    const kind = button.dataset.template;
    const file = latestFile();
    const templates = {
      single: "Use predict_single for this complete feature record: ",
      batch: file ? `Run batch prediction for file reference ${file.reference}.` : "Upload a batch CSV first.",
      evaluate: file ? `Evaluate labeled file reference ${file.reference}.` : "Upload a labeled CSV first.",
      conditional: file ? `Evaluate labeled file ${file.reference}.` : "Upload a labeled CSV first.",
    };
    if (kind === "single") {
      try {
        const sample = await fetch("/samples/single").then((response) => response.json());
        $("#message-input").value = templates.single + JSON.stringify(sample);
      } catch { $("#message-input").value = templates.single; }
    } else if (kind === "conditional" && file) {
      try {
        const sample = await fetch("/samples/single").then((response) => response.json());
        state.pendingCondition = {
          minimum_accuracy: 98,
          evaluation_file_reference: file.reference,
        };
        $("#message-input").value = `${templates.conditional} If accuracy is at least 98%, `
          + `predict this record: ${JSON.stringify(sample)}`;
      } catch { $("#message-input").value = templates.conditional; }
    } else {
      state.pendingCondition = null;
      $("#message-input").value = templates[kind];
    }
    const conversation = activeConversation();
    if (conversation) conversation.draft = $("#message-input").value;
    saveActiveConversation();
    $("#message-input").focus();
  })();
});

$("#file-input").addEventListener("change", async (event) => {
  if (!event.target.files.length) return;
  try { await upload(event.target.files[0]); } catch (error) { toast(error.message); }
  event.target.value = "";
});

$("#composer-file-input").addEventListener("change", async (event) => {
  if (!event.target.files.length) return;
  try { await upload(event.target.files[0]); } catch (error) { toast(error.message); }
  event.target.value = "";
});

const dropzone = $("#dropzone");
["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault(); dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault(); dropzone.classList.remove("dragging");
}));
dropzone.addEventListener("drop", async (event) => {
  if (!event.dataTransfer.files.length) return;
  try { await upload(event.dataTransfer.files[0]); } catch (error) { toast(error.message); }
});

$("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("#message-input");
  const message = input.value.trim();
  if (!message) return;
  addMessage("user", message);
  input.value = "";
  const sendButton = $("#send-button");
  const originalButtonText = sendButton.textContent;
  sendButton.disabled = true;
  sendButton.textContent = "Working…";
  const typingIndicator = addTypingIndicator();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: conversationHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        message,
        context_result_id: isExplicitNewTask(message) ? null : state.latestResultId,
        condition: state.pendingCondition,
      }),
    });
    const data = await response.json();
    typingIndicator.remove();
    if (data.error_code === "missing_context") {
      state.latestResultId = null;
      saveActiveConversation();
    }
    const resultIds = Array.isArray(data.result_ids) ? data.result_ids : [];
    const results = await Promise.all(resultIds.map(async (resultId) => {
      const resultResponse = await fetch(`/api/results/${encodeURIComponent(resultId)}`, { headers: conversationHeaders() });
      if (!resultResponse.ok) return null;
      const resultData = await resultResponse.json();
      return resultData.result || null;
    }));
    const availableResults = results.filter(Boolean);
	// Follow-up context: prefer latest evaluate_labeled (has confusion matrix); fallback to last result.
    if (resultIds.length) {
      // A follow-up question needs the result that actually carries the numbers
      // being asked about. In the conditional task the evaluation runs before
      // the prediction, so the last result is the prediction and holds no
      // confusion matrix. Prefer the latest labeled evaluation.
      let evaluationIndex = -1;
      for (let index = availableResults.length - 1; index >= 0; index -= 1) {
        if (availableResults[index].tool === "evaluate_labeled") {
          evaluationIndex = index;
          break;
        }
      }
      if (evaluationIndex !== -1) {
        const evaluationId = availableResults[evaluationIndex].result_id;
        const position = resultIds.indexOf(evaluationId);
        state.latestResultId = position === -1 ? resultIds.at(-1) : resultIds[position];
      } else {
        state.latestResultId = resultIds.at(-1);
      }
    }
    addMessage(
      "assistant",
      displayAnswer(
        data.error_code === "ungrounded_response"
          ? displayFallback(data, availableResults)
          : (data.answer || data.error || "No response"),
        availableResults,
      ),
      data.activities || [],
      availableResults,
    );
  } catch (error) {
    typingIndicator.remove();
    addMessage("assistant", `Request failed: ${error.message}`);
  }
  finally {
    typingIndicator.remove();
    state.pendingCondition = null;
    saveActiveConversation();
    sendButton.disabled = false;
    sendButton.textContent = originalButtonText;
    input.focus();
  }
});

$("#message-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!$("#send-button").disabled) $("#chat-form").requestSubmit();
  }
});

$("#message-input").addEventListener("input", (event) => {
  event.target.style.height = "auto";
  event.target.style.height = `${Math.min(event.target.scrollHeight, 180)}px`;
  const conversation = activeConversation();
  if (conversation) { conversation.draft = event.target.value; persistConversations(); }
});

$("#clear-chat").addEventListener("click", () => createConversation());

initializeConversations();
checkHealth();
