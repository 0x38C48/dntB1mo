const appShell = document.querySelector("#appShell");
const statusEl = document.querySelector("#status");
const ragBadge = document.querySelector("#ragBadge");
const axesEl = document.querySelector("#personaAxes");
const behaviorEl = document.querySelector("#behaviorBox");
const messagesEl = document.querySelector("#messages");
const memoriesEl = document.querySelector("#memories");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const clearBtn = document.querySelector("#clearBtn");
const collapseLeft = document.querySelector("#collapseLeft");
const collapseRight = document.querySelector("#collapseRight");
const expandLeft = document.querySelector("#expandLeft");
const expandRight = document.querySelector("#expandRight");

const t = {
  noMemory: "\u6682\u65e0\u76f8\u5173\u5386\u53f2\u7247\u6bb5",
  thinking: "\u751f\u6210\u4e2d...",
  error: "\u51fa\u9519\u4e86\uff1a",
  failed: "\u8bf7\u6c42\u5931\u8d25\uff1a",
  hello: "\u55ef",
  axes: ["\u600e\u4e48\u8bf4\u8bdd", "\u600e\u4e48\u60f3", "\u600e\u4e48\u5224\u65ad", "\u4ec0\u4e48\u4e0d\u505a", "\u77e5\u9053\u5c40\u9650"],
};

const profiles = {
  bot: { name: "backup", avatar: "/static/assets/backup.jpg" },
  user: { name: "NonForgetter", avatar: "/static/assets/nonforgetter.jpg" },
};

let chatHistory = [];

function hashString(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function pickClient(items, seed) {
  return items[hashString(seed) % items.length];
}

function polishClientReply(reply, userText) {
  const cleaned = String(reply || "").trim();
  if (/^[?？]{1,8}$/.test(cleaned) && hashString(userText) % 100 >= 20) {
    return pickClient(["\u600e\u4e48\u8bf4", "\u4f60\u8bf4", "\u554a", "\u4ec0\u4e48"], userText);
  }
  return cleaned;
}

function addMessage(role, text) {
  if (role === "meta") {
    const node = document.createElement("div");
    node.className = "msg-row meta";
    node.textContent = text;
    messagesEl.appendChild(node);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return node;
  }

  const profile = profiles[role] || profiles.bot;
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;
  row.innerHTML = `
    <img class="avatar" src="${profile.avatar}" alt="${profile.name}">
    <div class="msg-stack">
      <div class="nickname">${profile.name}</div>
      <div class="bubble"></div>
    </div>
  `;
  wireAvatar(row.querySelector(".avatar"), profile.name);
  row.querySelector(".bubble").textContent = text;
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return row;
}

function wireAvatar(image, name) {
  if (!image) return;
  image.addEventListener("error", () => {
    const fallback = document.createElement("div");
    fallback.className = `${image.className} avatar-fallback`;
    fallback.textContent = String(name || "?").slice(0, 1).toUpperCase();
    image.replaceWith(fallback);
  }, { once: true });
}

function addTypingIndicator() {
  const node = addMessage("meta", "");
  node.classList.add("typing-row");
  node.innerHTML = `<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>`;
  return node;
}

function addBotReply(text) {
  const parts = String(text || "")
    .split(/\n+/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length === 0) {
    addMessage("bot", "\uff1f");
    return;
  }
  for (const part of parts) {
    addMessage("bot", part);
  }
}

function renderMemories(memories) {
  memoriesEl.innerHTML = "";
  const rows = (memories || []).slice(0, 10);
  if (rows.length === 0) {
    memoriesEl.innerHTML = `<div class="memory"><p>${t.noMemory}</p></div>`;
    return;
  }
  rows.forEach((memory, index) => {
    const node = document.createElement("details");
    node.className = "memory";
    if (index < 2) node.open = true;
    node.innerHTML = `
      <summary>
        <span>${index + 1}. ${memory.chunk_id || "chunk"}</span>
        <b>${Number(memory.score || 0).toFixed(2)}</b>
      </summary>
      <div class="memory-time">${memory.start_time || ""}</div>
      <p></p>
    `;
    node.querySelector("p").textContent = memory.text || "";
    memoriesEl.appendChild(node);
  });
}

function renderBehavior(behavior) {
  if (!behavior || !behavior.topic_initiation) return;
  const topic = behavior.topic_initiation;
  const response = behavior.response_to_other_side || {};
  const categories = response.no_reply_categories || {};
  const topCats = Object.entries(categories)
    .slice(0, 5)
    .map(([key, value]) => `<span class="tag">${key}: ${value}</span>`)
    .join("");

  behaviorEl.innerHTML = `
    <section class="axis">
      <h2>\u65f6\u95f4\u884c\u4e3a</h2>
      <p>backup \u4e3b\u52a8\u5f00\u8bdd\u9898\u7684\u4e2d\u4f4d\u95f4\u9694\uff1a${topic.median_gap_minutes ?? "-"} min</p>
      <p>\u5bf9\u53e6\u4e00\u4fa7\u6d88\u606f\u6162\u56de/\u4e0d\u56de\u6bd4\u4f8b\uff1a${response.slow_or_no_reply_ratio ?? "-"}</p>
      <div class="tags">${topCats}</div>
    </section>
  `;
}

async function loadStatus() {
  const status = await fetch("/api/status").then((r) => r.json());
  statusEl.textContent = `${status.mode} · ${status.runtime_version || ""}`;
  if (ragBadge && status.rag) {
    ragBadge.textContent = "\u68d7/\u8c10\u97f3 RAG";
  }
}

async function loadPersona() {
  const persona = await fetch("/api/persona").then((r) => r.json());
  axesEl.innerHTML = "";
  const axes = persona.five_axes || {};
  for (const key of t.axes) {
    const axis = axes[key];
    if (!axis) continue;
    const node = document.createElement("section");
    node.className = "axis";
    const phrases = (axis.top_short_phrases || []).slice(0, 8);
    node.innerHTML = `<h2>${key}</h2><p>${axis.summary || ""}</p><div class="tags"></div>`;
    const tags = node.querySelector(".tags");
    for (const phrase of phrases) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = phrase;
      tags.appendChild(tag);
    }
    axesEl.appendChild(node);
  }
}

async function loadBehavior() {
  try {
    const response = await fetch("/api/behavior");
    if (response.ok) {
      renderBehavior(await response.json());
      return;
    }
  } catch (_) {
    // Older running backends do not expose /api/behavior yet.
  }
  try {
    const staticResponse = await fetch("/static/behavior_analysis.json");
    if (staticResponse.ok) {
      renderBehavior(await staticResponse.json());
    }
  } catch (_) {
    behaviorEl.innerHTML = "";
  }
}

async function submitMessage() {
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  resizeInput();
  addMessage("user", text);
  chatHistory.push({ role: "user", content: text });
  const meta = addTypingIndicator();
  try {
    const result = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ message: text, history: chatHistory.slice(-12) }),
    }).then((r) => r.json());
    meta.remove();
    if (result.error) {
      addMessage("bot", `${t.error}${result.error}`);
      return;
    }
    const polished = polishClientReply(result.reply, text);
    addBotReply(polished);
    chatHistory.push({ role: "assistant", content: polished });
    renderMemories(result.memories || []);
  } catch (error) {
    meta.remove();
    addMessage("bot", `${t.failed}${error}`);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitMessage();
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitMessage();
  }
});

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
}

input.addEventListener("input", resizeInput);

clearBtn.addEventListener("click", () => {
  chatHistory = [];
  messagesEl.innerHTML = "";
  renderMemories([]);
});

collapseLeft.addEventListener("click", () => appShell.classList.add("left-collapsed"));
collapseRight.addEventListener("click", () => appShell.classList.add("right-collapsed"));
expandLeft.addEventListener("click", () => appShell.classList.remove("left-collapsed"));
expandRight.addEventListener("click", () => appShell.classList.remove("right-collapsed"));

loadStatus();
loadPersona();
loadBehavior();
renderMemories([]);
addMessage("bot", t.hello);
document.querySelectorAll("img").forEach((img) => wireAvatar(img, img.alt || "?"));
resizeInput();
