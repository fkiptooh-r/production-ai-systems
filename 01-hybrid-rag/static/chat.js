// Chat page behaviour. The server renders the agency into <body data-tenant data-agency>,
// so this script never chooses a tenant: it only asks the one the URL already names.
const $ = (id) => document.getElementById(id);
const { tenant, agency } = document.body.dataset;
const log = $("log"), form = $("form"), input = $("question"), send = $("send");

for (const chip of document.querySelectorAll(".chip")) {
  chip.addEventListener("click", () => { input.value = chip.dataset.question; form.requestSubmit(); });
}

function bubble(cls, who) {
  $("empty")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  if (who) { const w = document.createElement("div"); w.className = "who"; w.textContent = who; el.append(w); }
  const body = document.createElement("div");
  el.append(body);
  log.append(el);
  el.scrollIntoView({ block: "end", behavior: "smooth" });
  return { el, body };
}

// Render text with [1]-style citations as badges, using text nodes only (never innerHTML).
function renderAnswer(target, text) {
  target.replaceChildren();
  for (const part of text.split(/(\[\d+\])/)) {
    const m = part.match(/^\[(\d+)\]$/);
    if (m) { const s = document.createElement("span"); s.className = "cite"; s.textContent = m[1]; target.append(s); }
    else if (part) target.append(document.createTextNode(part));
  }
}

function renderSources(el, sources, cited) {
  const box = document.createElement("div");
  box.className = "sources";
  for (const s of sources) {
    const d = document.createElement("details");
    d.className = cited.includes(s.n) ? "cited" : "uncited";
    const sum = document.createElement("summary");
    sum.textContent = `[${s.n}] ${s.source} › ${s.section}${cited.includes(s.n) ? "  (cited)" : ""}`;
    const p = document.createElement("p");
    p.textContent = s.text;
    d.append(sum, p);
    box.append(d);
  }
  el.append(box);
}

// POST + Server-Sent Events: EventSource only does GET, so read the stream by hand.
async function* sse(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, end); buffer = buffer.slice(end + 2);
      const event = raw.match(/^event: (.*)$/m)?.[1];
      const data = raw.match(/^data: (.*)$/m)?.[1];
      if (event && data) yield { event, data: JSON.parse(data) };
    }
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = ""; send.disabled = true;
  bubble("user").body.textContent = question;
  const bot = bubble("bot", agency);
  bot.body.textContent = "Searching the documents…";

  let text = "", sources = [];
  try {
    const response = await fetch(`/t/${encodeURIComponent(tenant)}/ask`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      const detail = (await response.json()).detail;
      throw new Error(Array.isArray(detail) ? detail[0].msg : detail ?? response.statusText);
    }
    for await (const { event, data } of sse(response)) {
      if (event === "sources") { sources = data.sources; bot.body.textContent = "Writing the answer…"; }
      else if (event === "token") { text += data.text; renderAnswer(bot.body, text); }
      else if (event === "error") throw new Error(data.message);
      else if (event === "done") {
        renderAnswer(bot.body, text.trim());
        if (data.abstained) bot.el.classList.add("idk");
        else if (!data.cited.length) {
          const warn = document.createElement("div"); warn.className = "meta error";
          warn.textContent = "No sources cited: treat this answer with suspicion.";
          bot.el.append(warn);
        }
        if (!data.abstained) renderSources(bot.el, sources, data.cited);
        const meta = document.createElement("div"); meta.className = "meta";
        meta.textContent = `${data.seconds}s · searched ${agency} only`;
        bot.el.append(meta);
      }
      bot.el.scrollIntoView({ block: "end" });
    }
  } catch (err) {
    bot.body.textContent = "";
    const p = document.createElement("span"); p.className = "error";
    p.textContent = `Something went wrong: ${err.message}`;
    bot.body.append(p);
  } finally {
    send.disabled = false; input.focus();
  }
});
