import {
  createPoller,
  initialScreen,
  pollIdentity,
  selectionComplete,
} from "./session.mjs";
("use strict");
const app = document.querySelector("#app");
const resetButton = document.querySelector("#reset");
const SESSION_KEY = "fan-session";
const DRAFT_KEY = "fan-selection";
let cfg,
  session = null,
  screen = "welcome",
  selection = [], // [{card_id, aspects_off: []}] — the visitor's working copy
  topic = null,
  ownSelection = false, // the visitor picked these cards (a sample's refs are not theirs)
  pending = false,
  lastActive = Date.now(),
  lastTouch = 0,
  renderedRun = "";

const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const two = (n) => String(n).padStart(2, "0");
const cardOf = (id) => cfg.cards.find((c) => c.id === id) || null;
const topicOf = (id) => cfg.topics.find((t) => t.id === id) || null;
const topicLabel = (id) => topicOf(id)?.label || id;
const aspectKeys = () => Object.keys(cfg.aspects || {});
const profileLabel = (cardId) =>
  cardOf(cardId)?.profile_label || cardOf(cardId)?.label || cardId;
/** Two cards can share a profile, so name a reference by profile and subject. */
const refName = (cardId) => {
  const card = cardOf(cardId);
  if (!card) return cardId;
  const profile = card.profile_label || card.label;
  return card.subject_label ? `${profile}（${card.subject_label}）` : profile;
};
/** The English reference text a card contributes, minus the aspects turned off. */
const refText = (entry) => {
  const card = cardOf(entry.card_id);
  if (!card) return "";
  const off = entry.aspects_off || [];
  return aspectKeys()
    .filter((k) => !off.includes(k) && card.aspects?.[k])
    .map((k) => card.aspects[k])
    .join(", ");
};
const num = (value) =>
  Number.isFinite(Number(value)) ? String(Number(Number(value).toFixed(2))) : String(value ?? "");
const MODES = {
  live: "この場で生成",
  "exact-cache": "同一条件のキャッシュ",
  sample: "事前生成サンプル",
};

function error(text) {
  const box = document.querySelector("#error");
  box.textContent = text;
  box.hidden = false;
  clearTimeout(error.timer);
  error.timer = setTimeout(() => (box.hidden = true), 6000);
}
async function api(path, method = "GET", body) {
  const r = await fetch("/api" + path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let message;
    try {
      message = (await r.json()).detail;
    } catch {
      message = `HTTP ${r.status}`;
    }
    throw Object.assign(
      new Error(
        typeof message === "string" ? message : "入力を確認してください。",
      ),
      { status: r.status },
    );
  }
  return r.json();
}
/** Single-flight guard: one visitor action at a time, errors surface as a toast. */
async function act(fn) {
  if (pending) return;
  pending = true;
  try {
    await fn();
  } catch (e) {
    error(e.message);
  } finally {
    pending = false;
  }
}
function bind(id, fn) {
  document.getElementById(id)?.addEventListener("click", () => act(fn));
}
function on(attr, fn) {
  app
    .querySelectorAll(`[data-${attr}]`)
    .forEach((el) =>
      el.addEventListener("click", () => act(() => fn(el.dataset))),
    );
}
const STEPS = ["好きな画像を選ぶ", "お題を選ぶ", "見比べる"];
function steps(n) {
  return `<ol class="stepbar">${STEPS.map(
    (label, i) =>
      `<li class="${i + 1 === n ? "active" : ""}"${i + 1 === n ? ' aria-current="step"' : ""}><b>${two(i + 1)}</b>${esc(label)}</li>`,
  ).join("")}</ol>`;
}
function render() {
  resetButton.hidden = !session;
  (
    {
      welcome,
      cards: cardsScreen,
      topics: topicsScreen,
      compare: compareScreen,
    }[screen] || welcome
  )();
}

/* ---------------------------------------------------------------- welcome */
function sampleControl() {
  return cfg.samples?.length
    ? `<div class="sample-picker"><span class="sample-lead">まずは結果だけ見る</span><select id="sample-id" aria-label="事前生成サンプル">${cfg.samples
        .map(
          (s) =>
            `<option value="${esc(s.id)}">${esc(s.label || `${s.id} · ${topicLabel(s.topic_id)}`)}</option>`,
        )
        .join(
          "",
        )}</select><button class="secondary" id="sample">サンプルを見る</button></div>`
    : "";
}
function bindSample() {
  bind("sample", async () => {
    if (!session) await start(false);
    session = await api(`/sessions/${session.id}/sample`, "POST", {
      sample_id: document.querySelector("#sample-id").value,
    });
    screen = "compare";
    render();
    window.scrollTo(0, 0);
  });
}
/** Same prompt, same seed: the plain picture beside a sample's personalized one. */
function heroArt() {
  const sample = cfg.samples?.find((x) => x.preview_url && topicOf(x.topic_id));
  const plain = sample ? topicOf(sample.topic_id) : cfg.topics?.[0];
  if (!plain?.preview_url) return "";
  return `<div class="hero-art ${sample ? "pair" : ""}"><figure class="plain"><img src="${esc(plain.preview_url)}" alt="${esc(plain.label)}のパーソナライズなしの生成画像"><figcaption>パーソナライズなし</figcaption></figure>${
    sample
      ? `<figure class="mine"><img src="${esc(sample.preview_url)}" alt="${esc(plain.label)}の、好みを反映した生成サンプル"><figcaption>好みを反映</figcaption></figure>`
      : ""
  }</div>`;
}
function welcome() {
  app.innerHTML = `<section class="hero"><div class="hero-copy"><span class="pill">好みを反映する画像生成 · 体験展示</span><h1>同じ一文から、<br><em>あなたの一枚</em>を。</h1><p class="intro">好きな絵を数枚えらぶだけ。入力文も生成モデルも変えずに、絵がどこまで「あなた好み」に寄るのかを、その場で見比べられます。</p><button class="primary" id="start" ${cfg.ready ? "" : "disabled"}>体験をはじめる <span>→</span></button><ul class="facts"><li>約3分</li><li>登録不要</li><li>この端末の中だけで動作</li></ul>${
    cfg.ready
      ? ""
      : '<p class="note ready-note">画像を準備中です。準備が終わると体験できます。</p>'
  }${sampleControl()}</div>${heroArt()}</section><ol class="journey"><li><b>01</b><h3>好きな画像を選ぶ</h3><p>16枚から${cfg.selection.min}〜${cfg.selection.max}枚</p></li><li><b>02</b><h3>お題を選ぶ</h3><p>一文は誰でも同じ</p></li><li><b>03</b><h3>見比べる</h3><p>パーソナライズなし・あり</p></li></ol>`;
  bind("start", () => start(true));
  bindSample();
}
async function start(show) {
  poller.stop();
  session = await api("/sessions", "POST");
  sessionStorage.setItem(SESSION_KEY, session.id);
  selection = [];
  topic = null;
  ownSelection = false;
  saveDraft();
  lastActive = Date.now();
  if (show) {
    screen = "cards";
    render();
    window.scrollTo(0, 0);
  }
}
function saveDraft() {
  try {
    sessionStorage.setItem(
      DRAFT_KEY,
      JSON.stringify({ sid: session?.id || null, selection }),
    );
  } catch {}
}

/* ------------------------------------------------ 01 好きな画像を選ぶ */
const picked = (id) => selection.findIndex((s) => s.card_id === id);
function cardsScreen() {
  const { min, max } = cfg.selection;
  const ok = selectionComplete(selection, cfg.selection);
  const order = session.card_order?.length
    ? session.card_order
    : cfg.cards.map((c) => c.id);
  app.innerHTML = `${steps(1)}<div class="topline"><div><h2>好きな画像を選んでください。</h2><p>同じ人物が4つの描き方で並んでいます。人物ではなく、描き方の好みで選んでください。</p></div><div class="counter" aria-label="選んだ枚数"><b>${selection.length}</b><small>/ ${min}〜${max}枚</small></div></div><div class="card-grid">${order
    .map((id) => {
      const card = cardOf(id);
      if (!card) return "";
      const n = picked(id);
      return `<button class="card ${n >= 0 ? "selected" : ""}" data-card="${esc(id)}" aria-pressed="${n >= 0}" aria-label="${esc(card.label)}を${n >= 0 ? "選択解除" : "選ぶ"}"><img src="${esc(card.url)}" alt="${esc(card.label)}" loading="lazy">${n >= 0 ? `<span class="order">${n + 1}</span>` : ""}<span class="card-foot">${esc(card.subject_label || card.label)}</span></button>`;
    })
    .join("")}</div>${
    selection.length
      ? `<div class="picked-head"><h3>選んだ${selection.length}枚を、参照に使います。</h3><span>この画像のどこが好き？ 外した側面は参照から消えます。</span></div><div class="picked-list">${selection
          .map((entry, i) => {
            const card = cardOf(entry.card_id);
            if (!card) return "";
            const off = entry.aspects_off || [];
            return `<div class="ref-edit"><img src="${esc(card.url)}" alt="${esc(card.label)}"><div class="ref-body"><div class="ref-head"><b>${two(i + 1)}</b><span class="ref-profile">${esc(card.profile_label || card.label)}</span><span class="ref-subject">${esc(card.subject_label || "")}</span></div><div class="chips" role="group" aria-label="${esc(card.label)}のどこが好きか">${aspectKeys()
              .map(
                (k) =>
                  `<button class="chip ${off.includes(k) ? "" : "on"}" data-aspect="${k}" data-target="${esc(entry.card_id)}" aria-pressed="${!off.includes(k)}"><b>${esc(cfg.aspects[k])}</b><small>${esc(card.aspects_ja?.[k] || card.aspects?.[k] || "")}</small></button>`,
              )
              .join(
                "",
              )}</div><code class="ref-en">${esc(refText(entry) || "（側面がすべて外れています）")}</code></div><button class="quiet" data-drop="${esc(entry.card_id)}">外す ✕</button></div>`;
          })
          .join("")}</div>`
      : `<p class="note">まだ選ばれていません。気になる描き方の画像を${min}枚以上えらんでください。</p>`
  }<div class="action-row"><span class="note">${ok ? `${selection.length}枚を参照にします。` : `あと${Math.max(0, min - selection.length)}枚えらぶと次へ進めます（最大${max}枚）。`}</span><button id="to-topics" class="primary" ${ok ? "" : "disabled"}>次へ <span>→</span></button></div><p class="note">選ばなかった画像を「嫌い」とは扱いません。並び順は端末ごとにシャッフルしています。</p>`;
  on("card", ({ card }) => {
    const n = picked(card);
    if (n >= 0) selection.splice(n, 1);
    else if (selection.length >= cfg.selection.max)
      return error(`選べるのは${cfg.selection.max}枚までです。`);
    else selection.push({ card_id: card, aspects_off: [] });
    saveDraft();
    cardsScreen();
  });
  on("aspect", ({ aspect, target }) => {
    const entry = selection.find((s) => s.card_id === target);
    if (!entry) return;
    const off = new Set(entry.aspects_off || []);
    if (off.has(aspect)) off.delete(aspect);
    else if (off.size + 1 >= aspectKeys().length)
      return error("この画像のどこが好きかを、少なくとも1つ残してください。");
    else off.add(aspect);
    entry.aspects_off = aspectKeys().filter((k) => off.has(k));
    saveDraft();
    cardsScreen();
  });
  on("drop", ({ drop }) => {
    const n = picked(drop);
    if (n >= 0) selection.splice(n, 1);
    saveDraft();
    cardsScreen();
  });
  bind("to-topics", async () => {
    session = await api(`/sessions/${session.id}/selection`, "PUT", {
      cards: selection.map((s) => ({
        card_id: s.card_id,
        aspects_off: s.aspects_off || [],
      })),
    });
    if (session.selection?.length) selection = structuredClone(session.selection);
    ownSelection = true;
    saveDraft();
    screen = "topics";
    render();
    window.scrollTo(0, 0);
  });
}

/* ------------------------------------------------------ 02 お題を選ぶ */
function refStrip() {
  return `<div class="refbar">${selection
    .map((entry, i) => {
      const card = cardOf(entry.card_id);
      if (!card) return "";
      return `<div class="refchip"><img src="${esc(card.url)}" alt="${esc(card.label)}"><div><b>${two(i + 1)} ${esc(refName(entry.card_id))}</b><code>${esc(refText(entry))}</code></div></div>`;
    })
    .join("")}</div>`;
}
function topicsScreen() {
  if (!topic) topic = session.run?.topic_id || cfg.topics[0]?.id;
  app.innerHTML = `${steps(2)}<h2>お題を選んでください。</h2><p>お題の一文は誰でも同じです。渡すのは、下の英語の説明文だけです。</p>${refStrip()}<div class="topics">${cfg.topics
    .map(
      (t) =>
        `<button class="topic ${t.id === topic ? "selected" : ""}" data-topic="${esc(t.id)}" aria-pressed="${t.id === topic}"><img src="${esc(t.preview_url)}" alt="${esc(t.label)}の参照なし生成サンプル" loading="lazy"><span>${esc(t.label)}</span></button>`,
    )
    .join(
      "",
    )}</div><div class="action-row"><button id="back" class="quiet">← ${session.run ? "比較に戻る" : "好きな画像を選び直す"}</button><button id="generate" class="primary">この好みで描く <span>→</span></button></div><p class="note">パーソナライズなしの4枚と、同じ入力文・同じseedで描いたあなた向けの4枚を並べます。</p>`;
  on("topic", ({ topic: id }) => {
    topic = id;
    topicsScreen();
  });
  bind("back", () => {
    screen = session.run ? "compare" : "cards";
    render();
    window.scrollTo(0, 0);
  });
  bind("generate", async () => {
    const button = document.querySelector("#generate");
    if (button) button.disabled = true;
    try {
      // The finished comparison for this topic is already on hand.
      const shown = session.run;
      const kept =
        shown?.topic_id === topic &&
        shown.mode !== "sample" &&
        shown.status === "done" &&
        !shown.error;
      if (!kept)
        session = await api(`/sessions/${session.id}/runs`, "POST", {
          topic_id: topic,
          request_id: crypto.randomUUID(),
        });
      screen = "compare";
      renderedRun = JSON.stringify(session.run);
      render();
      window.scrollTo(0, 0);
      if (session.run?.status !== "done") watch();
    } finally {
      const b = document.querySelector("#generate");
      if (b) b.disabled = false;
    }
  });
}

/* ------------------------------------------------------ 03 見比べる */
function statusbox(run) {
  const working = run.status !== "done";
  return `<div class="statusbox" role="status">${working ? '<span class="spinner"></span>' : "<span>✓</span>"}<p>${esc(run.message || (working ? "描いています…" : "できました。"))}${run.elapsed_seconds ? ` · ${esc(run.elapsed_seconds)}秒` : ""}</p></div>`;
}
/** Past the card grid the server owns the selection (a sample run sets its own). */
function adoptSelection() {
  const theirs = session?.selection;
  if (theirs?.length && JSON.stringify(theirs) !== JSON.stringify(selection))
    selection = structuredClone(theirs);
}
function shots(images, working, count = 4) {
  return `<div class="image-grid">${Array.from({ length: count }, (_, i) => {
    const x = images?.[i];
    return x
      ? `<figure class="shot"><img src="${esc(x.url)}" alt="seed ${esc(x.seed)} の生成画像" loading="lazy"><figcaption><span>${two(i + 1)}</span><span>seed ${esc(x.seed)}</span></figcaption></figure>`
      : `<div class="placeholder">${working ? "描いています…" : "—"}</div>`;
  }).join("")}</div>`;
}
/**
 * One row per distinct aspect phrase, grouped by aspect. A phrase that several
 * selected cards share carries a larger weight, and its thumbnails show why.
 */
function refRows(refs) {
  const keys = aspectKeys();
  const groups = keys.map((key) => [key, refs.filter((r) => r.aspect === key)]);
  const rest = refs.filter((r) => !keys.includes(r.aspect));
  if (rest.length) groups.push([null, rest]);
  return (
    groups
      .filter(([, list]) => list.length)
      .map(
        ([key, list]) =>
          `<div class="ref-aspect"><span class="ref-aspect-name">${esc(key ? cfg.aspects[key] : "その他")}</span><div class="ref-phrases">${[
            ...list,
          ]
            .sort((a, b) => (b.weight || 0) - (a.weight || 0))
            .map(
              (ref) =>
                `<div class="ref-phrase"><code>${esc(ref.text)}</code><span class="ref-weight">重み ${esc(num(ref.weight))}</span><span class="ref-thumbs">${(
                  ref.card_ids || []
                )
                  .map(
                    (id) =>
                      `<img src="${esc(cardOf(id)?.url || "")}" alt="${esc(refName(id))}" title="${esc(refName(id))}">`,
                  )
                  .join("")}</span></div>`,
            )
            .join("")}</div></div>`,
      )
      .join("") || "<p>—</p>"
  );
}
function inputsPanel(run) {
  const p = run.personalization;
  return `<details class="prompt-details"><summary>パーソナライズに渡したもの</summary><p class="kv"><b>お題の一文（全員に共通・変えていません）</b></p><code class="block">${esc(run.prompt || "—")}</code><p class="kv"><b>参照した好み（選んだ画像に付けた確認済みの説明文）</b>　同じ説明文を複数の画像が持つと、その分だけ重みが大きくなります。</p><div class="ref-used">${refRows(
    p?.refs || [],
  )}</div><p class="kv"><b>反映の強さ alpha</b>　${esc(p?.alpha ?? cfg.alpha ?? "—")}</p><p>生成モデル・seed・負のプロンプト・生成設定は、パーソナライズなしの4枚とまったく同じです。</p></details>`;
}
function compareScreen() {
  const run = session.run;
  if (!run) {
    screen = selectionComplete(selection, cfg.selection) ? "topics" : "cards";
    return render();
  }
  adoptSelection();
  const sample = run.mode === "sample";
  const working = run.status !== "done";
  const canRetopic = ownSelection && selectionComplete(selection, cfg.selection);
  app.innerHTML = `${steps(3)}<h2>パーソナライズなし・ありを見比べてください。</h2><div class="callout"><b>${esc(topicLabel(run.topic_id))}</b><span>入力文も生成モデルも seed も同じです。違うのは、あなたの好みを参照したかどうかだけです。</span></div>${
    working ? statusbox(run) : ""
  }<section class="row"><div class="comparison-label"><h3>パーソナライズなし</h3><span>好みを使わずに描いた4枚</span></div>${shots(
    run.plain,
    working,
  )}</section><section class="row mine"><div class="comparison-label"><h3>パーソナライズあり</h3><span>選んだ画像の好みを参照して描いた4枚</span><span class="badge ${sample ? "sample" : ""}">${esc(MODES[run.mode] || run.mode)}</span>${
    run.error ? `<span class="badge sample">${esc(run.error)}</span>` : ""
  }</div>${shots(run.personal, working)}${inputsPanel(run)}</section>${
    sample
      ? `<p class="error-message">事前生成のサンプルです。あなたの選択を反映した結果ではありません。${canRetopic ? "" : "ご自分の好みで試すには、最初から始めてください。"}</p>`
      : ""
  }<div class="action-row">${
    canRetopic
      ? `<button id="another" class="secondary" ${working ? "disabled" : ""}>別のお題で描く</button>`
      : '<button id="restart" class="secondary">最初から始める</button>'
  }${working ? '<button id="cancel" class="secondary">描くのを中止する</button>' : ""}<button id="finish" class="primary">体験を終了</button></div><p class="note">終了すると、この体験の選択・参照・生成画像は削除されます。${cfg.idle_seconds}秒の無操作でも終了します。${working ? "処理中は無操作リセットを止めています。" : ""}</p>`;
  bind("another", () => {
    topic = null;
    screen = "topics";
    render();
    window.scrollTo(0, 0);
  });
  bind("restart", async () => {
    await finish();
    await start(true);
  });
  bind("cancel", async () => {
    poller.stop();
    session = await api(`/sessions/${session.id}/cancel`, "POST");
    // Cancelling drops the unfinished run; the selection survives.
    adoptSelection();
    if (!session.run) screen = "topics";
    renderedRun = JSON.stringify(session.run);
    render();
    window.scrollTo(0, 0);
  });
  bind("finish", finish);
}
function keepScroll(fn) {
  const y = window.scrollY;
  fn();
  window.scrollTo(0, y);
}

/* ------------------------------------------------------------- lifecycle */
async function finish() {
  poller.stop();
  if (session) {
    try {
      await api(`/sessions/${session.id}`, "DELETE");
    } catch (e) {
      if (e.status !== 404) throw e;
    }
  }
  forget();
  render();
  window.scrollTo(0, 0);
}
function forget() {
  session = null;
  selection = [];
  topic = null;
  ownSelection = false;
  renderedRun = "";
  sessionStorage.removeItem(SESSION_KEY);
  sessionStorage.removeItem(DRAFT_KEY);
  screen = "welcome";
}
const poller = createPoller({
  identity: () => pollIdentity(session),
  fetchSnapshot: (sid) => api(`/sessions/${sid}`),
  onSnapshot: (current) => {
    session = current;
    const signature = JSON.stringify(session.run);
    if (signature !== renderedRun && screen === "compare") {
      renderedRun = signature;
      keepScroll(render);
    }
  },
  onExpired: () => {
    forget();
    render();
  },
  onError: (e) => error(e.message),
});
function watch() {
  poller.start();
}
resetButton.addEventListener("click", () => act(finish));
for (const name of ["pointerdown", "keydown", "change", "scroll"])
  document.addEventListener(
    name,
    () => {
      lastActive = Date.now();
      if (session && Date.now() - lastTouch > 10000) {
        lastTouch = Date.now();
        api(`/sessions/${session.id}/touch`, "POST").catch(() => {});
      }
    },
    { passive: true },
  );
setInterval(() => {
  const working = session?.run && session.run.status !== "done";
  if (working) lastActive = Date.now();
  if (
    session &&
    !pending &&
    !working &&
    Date.now() - lastActive > cfg.idle_seconds * 1000
  )
    act(finish);
}, 1000);
(async () => {
  try {
    cfg = await api("/config");
    const saved = sessionStorage.getItem(SESSION_KEY);
    if (saved) {
      try {
        session = await api(`/sessions/${saved}`);
        if (session.selection?.length)
          selection = structuredClone(session.selection);
        else {
          const draft = JSON.parse(sessionStorage.getItem(DRAFT_KEY) || "null");
          if (draft?.sid === session.id && Array.isArray(draft.selection))
            selection = draft.selection;
        }
        topic = session.run?.topic_id || null;
        ownSelection =
          Boolean(session.selection?.length) && session.run?.mode !== "sample";
        screen = initialScreen(session, cfg.selection);
        if (session.run && session.run.status !== "done") watch();
      } catch {
        sessionStorage.removeItem(SESSION_KEY);
        sessionStorage.removeItem(DRAFT_KEY);
        session = null;
      }
    }
    render();
  } catch (e) {
    app.innerHTML =
      "<h2>接続できませんでした。</h2><p>ローカルサーバーを確認し、ページを再読み込みしてください。</p>";
    error(e.message);
  }
})();
