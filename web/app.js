const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chatForm");
const inputEl = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const newChatButton = document.querySelector("#newChat");
const temperatureEl = document.querySelector("#temperature");
const temperatureValueEl = document.querySelector("#temperatureValue");
const maxTokensEl = document.querySelector("#maxTokens");
const sessionLabelEl = document.querySelector("#sessionLabel");

const storageKey = "nanonona.sessionId";
let sessionId = sessionStorage.getItem(storageKey) || crypto.randomUUID();
sessionStorage.setItem(storageKey, sessionId);

function shortSession(id) {
  return id.slice(0, 8);
}

function setBusy(busy) {
  sendButton.disabled = busy;
  inputEl.disabled = busy;
}

function renderMessages(messages) {
  messagesEl.innerHTML = "";
  if (!messages.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "开始新的对话";
    messagesEl.appendChild(empty);
    return;
  }
  for (const message of messages) {
    appendMessage(message.role, message.content);
  }
}

function appendMessage(role, content, extraClass = "") {
  const item = document.createElement("div");
  item.className = `message ${role} ${extraClass}`.trim();
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  item.appendChild(bubble);
  messagesEl.appendChild(item);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return item;
}

async function api(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "请求失败");
  }
  return data;
}

async function loadHistory() {
  sessionLabelEl.textContent = `会话 ${shortSession(sessionId)}`;
  const response = await fetch(`/api/history?session_id=${encodeURIComponent(sessionId)}`);
  const data = await response.json();
  renderMessages(data.messages || []);
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = "";
  inputEl.style.height = "auto";
  appendMessage("user", text);
  const loading = appendMessage("assistant", "思考中...", "loading");
  setBusy(true);

  try {
    const data = await api("/api/chat", {
      session_id: sessionId,
      message: text,
      temperature: Number(temperatureEl.value),
      max_tokens: Number(maxTokensEl.value),
    });
    loading.remove();
    renderMessages(data.messages || []);
  } catch (error) {
    loading.querySelector(".bubble").textContent = `请求失败：${error.message}`;
  } finally {
    setBusy(false);
    inputEl.focus();
  }
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
});

temperatureEl.addEventListener("input", () => {
  temperatureValueEl.textContent = temperatureEl.value;
});

newChatButton.addEventListener("click", async () => {
  sessionId = crypto.randomUUID();
  sessionStorage.setItem(storageKey, sessionId);
  await loadHistory();
  inputEl.focus();
});

loadHistory();
