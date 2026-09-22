import {
  canRequestRound,
  commitBlocker,
  createPoller,
  createSelectionWriter,
  createTicket,
  draftFromSnapshot,
  draftMatchesSnapshot,
  emptyDraft,
  gainIndex,
  GAIN_LABELS,
  GAIN_STEPS,
  initialScreen,
  likeAll,
  pollIdentity,
  removeCard,
  schemaSupported,
  setGain,
  setStrength,
  toggleAspect,
  toggleCard,
  withLimits,
} from "./session.mjs";
("use strict");
const app = document.querySelector("#app");
const resetButton = document.querySelector("#reset");
const SESSION_KEY = "fan-session";
let cfg,
  limits,
  session = null, // the server snapshot: the only source of truth
  draft = emptyDraft(), // the working copy the screen edits
  screen = "welcome",
  topic = null,
  pending = false,
  notice = "", // one line explaining what the server just did
  lastActive = Date.now(),
  lastTouch = 0,
  saveTimer = null,
  renderedRun = "";
const roundTicket = createTicket(() => crypto.randomUUID());
const runTicket = createTicket(() => crypto.randomUUID());

const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const two = (n) => String(n).padStart(2, "0");
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
const cardOf = (id) => cfg.cards.find((c) => c.id === id) || null;
const topicOf = (id) => cfg.topics.find((t) => t.id === id) || null;
const topicLabel = (id) => topicOf(id)?.label || id;
const aspectKeys = () => Object.keys(cfg.aspects || {});
const picked = (id) => draft.selection.findIndex((s) => s.card_id === id);
/** Two cards can share a profile, so name a reference by profile and subject. */
const refName = (cardId) => {
  const card = cardOf(cardId);
  if (!card) return cardId;
  const profile = card.profile_label || card.label;
  return card.subject_label ? `${profile}（${card.subject_label}）` : profile;
};
/** The English reference text a card contributes through its chosen aspects. */
const refText = (entry) => {
  const card = cardOf(entry.card_id);
  if (!card) return "";
  return aspectKeys()
    .filter((k) => entry.aspects.includes(k) && card.aspects?.[k])
    .map((k) => card.aspects[k])
    .join(", ");
};
const num = (value) =>
  Number.isFinite(Number(value))
    ? String(Number(Number(value).toFixed(2)))
    : String(value ?? "");
const MODES = {
  live: "この場で生成",
  "exact-cache": "同じ内容の描き直し",
  sample: "サンプル",
};

function error(text) {
  const box = document.querySelector("#error");
  box.textContent = text;
  box.hidden = false;
  clearTimeout(error.timer);
  error.timer = setTimeout(() => (box.hidden = true), 6000);
}
function say(text) {
  notice = text || "";
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
/** Single-flight guard for network actions; local edits never wait on it. */
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
/** Local edits: instant, never blocked by an in-flight save. */
function on(attr, fn) {
  app.querySelectorAll(`[data-${attr}]`).forEach((el) =>
    el.addEventListener("click", () => {
      try {
        fn(el.dataset);
      } catch (e) {
        error(e.message);
      }
    }),
  );
}

/* -------------------------------------------------- server-owned snapshot */
function adoptSession(snapshot) {
  session = snapshot;
}
/** Adopt the server's answer, draft included: the screen mirrors the snapshot. */
function adoptAll(snapshot) {
  session = snapshot;
  draft = draftFromSnapshot(snapshot, aspectKeys());
}
/**
 * Serialised full-replace writes. A 409 (stale revision, or a worker that still
 * owns the GPU right after cancel/done) adopts the server snapshot and retries.
 */
const writer = createSelectionWriter({
  revision: () => session.revision,
  put: (payload) => api(`/sessions/${session.id}/selection`, "PUT", payload),
  refetch: () => api(`/sessions/${session.id}`),
  adopt: adoptSession,
  wait: sleep,
});

/** Persist the working copy. `commit` marks it ready to generate from. */
async function save(commit) {
  clearTimeout(saveTimer);
  saveTimer = null;
  if (!session) return null;
  const value = { ...draft, committed: Boolean(commit) };
  if (draftMatchesSnapshot(value, session, aspectKeys())) {
    draft = value;
    return session;
  }
  try {
    const snapshot = await writer.push({ draft: structuredClone(value), commit });
    if (writer.busy) adoptSession(snapshot);
    else adoptAll(snapshot);
    // A content change detaches the finished comparison: nothing left to poll.
    if (!session.run) poller.stop();
    return session;
  } catch (e) {
    adoptAll(await api(`/sessions/${session.id}`));
    say("画面を最新の状態に合わせました。もう一度お試しください。");
    render();
    throw e;
  }
}
/**
 * Local edit: draw at once, save shortly after. On the result screen the edits
 * stay local until 「同じお題で描き直す」, so the comparison on screen survives.
 */
function edited() {
  if (screen === "compare") return redrawAdjust();
  render();
  clearTimeout(saveTimer);
  saveTimer = null;
  if (!session) return;
  // Deliberately outside `act`: an autosave must never swallow the next click.
  saveTimer = setTimeout(() => save(false).catch((e) => error(e.message)), 350);
}

const STEPS = ["好きな画像を選ぶ", "お題を選ぶ", "見比べる"];
function steps(n) {
  return `<ol class="stepbar">${STEPS.map(
    (label, i) =>
      `<li class="${i + 1 === n ? "active" : ""}"${i + 1 === n ? ' aria-current="step"' : ""}><b>${two(i + 1)}</b>${esc(label)}</li>`,
  ).join("")}</ol>`;
}
function noticeLine() {
  return notice ? `<p class="notice" role="status">${esc(notice)}</p>` : "";
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
function goto(next) {
  screen = next;
  render();
  window.scrollTo(0, 0);
}

/* ---------------------------------------------------------------- welcome */
/** Same prompt, same seed: the plain picture beside a sample's personalized one. */
function heroArt() {
  // The server names the pair; without it, fall back to the first sample's first seed.
  const sample = cfg.samples?.find((x) => x.preview_url && topicOf(x.topic_id));
  const topicId = cfg.hero?.topic_id || sample?.topic_id;
  const plain = topicOf(topicId) || cfg.topics?.[0];
  const plainUrl = cfg.hero?.plain_url || plain?.preview_url;
  const mineUrl = cfg.hero?.personal_url || sample?.preview_url;
  if (!plainUrl) return "";
  return `<div class="hero-art ${mineUrl ? "pair" : ""}"><figure class="plain"><img src="${esc(plainUrl)}" alt="${esc(plain.label)}のパーソナライズなしの生成画像"><figcaption>パーソナライズなし</figcaption></figure>${
    mineUrl
      ? `<figure class="mine"><img src="${esc(mineUrl)}" alt="${esc(plain.label)}の、パーソナライズありの生成サンプル"><figcaption>パーソナライズあり</figcaption></figure>`
      : ""
  }</div>`;
}
/** Where to follow up after the visit: the author's X account and the repository. */
const LINKS = [
  { name: "X", label: "@tmp_friends", url: "https://x.com/tmp_friends", qr: "qr-x.svg" },
  {
    name: "GitHub",
    label: "tmp-friends/personalized-t2i",
    url: "https://github.com/tmp-friends/personalized-t2i",
    qr: "qr-github.svg",
  },
];
function linksBand() {
  return `<section class="links-band"><div class="links-lead"><h3>もっと知りたい方へ</h3><p>作者の X と、この展示のソースコードです。スマートフォンで読み取れます。</p></div>${LINKS.map(
    (l) =>
      `<a class="qr-link" href="${l.url}" target="_blank" rel="noopener"><img src="/static/${l.qr}" alt="${l.name} ${l.label} の QR コード" width="104" height="104"><span><b>${l.name}</b>${l.label}</span></a>`,
  ).join("")}</section>`;
}
function welcome() {
  app.innerHTML = `<section class="hero"><div class="hero-copy"><span class="pill">パーソナライズ画像生成 · 体験展示</span><h1>好きな絵を選ぶだけ。<br>あなた好みに描く、<em>パーソナライズ画像生成</em>。</h1><p class="intro">好きな絵を数枚選ぶだけで、あなたの好みを反映した画像をその場で生成します。入力文や生成モデルは変えず、絵がどこまで「あなた好み」に寄るのかを見比べられます。</p><ol class="journey"><li><b>01</b><h3>好きな画像を選ぶ</h3><p>候補の一覧から、合計${limits.min}〜${limits.max}枚</p></li><li><b>02</b><h3>お題を選ぶ</h3><p>描いてほしい場面を選ぶ</p></li><li><b>03</b><h3>見比べる</h3><p>あなた向けの結果を見る</p></li></ol><button class="primary" id="start" ${cfg.ready ? "" : "disabled"}>体験をはじめる <span>→</span></button><ul class="facts"><li>約3分</li></ul>${
    cfg.ready
      ? ""
      : '<p class="note ready-note">画像を準備中です。準備が終わると体験できます。</p>'
  }</div>${heroArt()}</section>${linksBand()}`;
  bind("start", start);
}
async function start() {
  poller.stop();
  roundTicket.done();
  runTicket.done();
  adoptAll(await api("/sessions", "POST"));
  sessionStorage.setItem(SESSION_KEY, session.id);
  topic = null;
  say("");
  lastActive = Date.now();
  await loadAllRounds();
  goto("cards");
}

/* ------------------------------------------------ 01 好きな画像を選ぶ */
/** Strength, aspects and 「全部好き」 for one chosen card. */
function pickRow(entry, index) {
  const card = cardOf(entry.card_id);
  if (!card) return "";
  const unanswered = !entry.aspects.length;
  const strengths = [1, 2].map((value) => {
    const label = value === 1 ? "好き" : "とても好き";
    return `<button class="step ${entry.strength === value ? "on" : ""}" data-strength="${value}" data-target="${esc(entry.card_id)}" aria-pressed="${entry.strength === value}">${label}</button>`;
  });
  const chips = aspectKeys().map((k) => {
    const on = entry.aspects.includes(k);
    return `<button class="chip ${on ? "on" : ""}" data-aspect="${k}" data-target="${esc(entry.card_id)}" aria-pressed="${on}"><b>${esc(cfg.aspects[k])}</b><small>${esc(card.aspects_ja?.[k] || card.aspects?.[k] || "")}</small></button>`;
  });
  return `<div class="ref-edit ${unanswered ? "needs" : ""}" id="pick-${esc(entry.card_id)}"><img src="${esc(card.url)}" alt="${esc(card.label)}"><div class="ref-body"><div class="ref-head"><b>${two(index + 1)}</b><span class="ref-profile">${esc(card.profile_label || card.label)}</span><span class="ref-subject">${esc(card.subject_label || "")}</span></div><div class="seg small" role="group" aria-label="${esc(refName(entry.card_id))}の好きの強さ">${strengths.join("")}</div><div class="chips" role="group" aria-label="${esc(refName(entry.card_id))}のどこが好きか">${chips.join("")}<button class="chip all" data-likeall="${esc(entry.card_id)}">全部好き</button></div>${
    unanswered
      ? '<p class="hint">この画像のどこが好きかを、1つ以上えらんでください。</p>'
      : `<code class="ref-en">${esc(refText(entry))}</code>`
  }</div><button class="quiet" data-drop="${esc(entry.card_id)}">外す ✕</button></div>`;
}
function trayBlock(title, lead) {
  if (!draft.selection.length)
    return `<p class="note">まだ選ばれていません。気になる描き方の画像を${limits.min}枚以上えらんでください。</p>`;
  return `<div class="picked-head"><h3>${esc(title)}</h3><span>${esc(lead)}</span></div><div class="picked-list">${draft.selection
    .map((entry, i) => pickRow(entry, i))
    .join("")}</div>`;
}
const BLOCK_TEXT = (blocker) => {
  if (!blocker) return null;
  if (blocker.code === "too_few")
    return `あと${blocker.missing}枚えらぶと次へ進めます（最大${limits.max}枚）。`;
  if (blocker.code === "too_many")
    return `選べるのは${limits.max}枚までです。`;
  return `「${refName(blocker.card_id)}」のどこが好きかを、1つ以上えらんでください。`;
};
function bindPickRows() {
  on("strength", ({ strength, target }) => {
    draft = setStrength(draft, target, Number(strength));
    edited();
  });
  on("aspect", ({ aspect, target }) => {
    draft = toggleAspect(draft, target, aspect, aspectKeys());
    edited();
  });
  on("likeall", ({ likeall }) => {
    draft = likeAll(draft, likeall, aspectKeys());
    edited();
  });
  on("drop", ({ drop }) => {
    draft = removeCard(draft, drop);
    edited();
  });
}
/** Every served round, in order: the visitor sees all candidates in one list. */
const shownIds = () => [
  ...new Set((session?.rounds || []).flatMap((round) => round.card_ids || [])),
];
function cardsScreen() {
  const ids = shownIds();
  const blocker = commitBlocker(draft, limits);
  app.innerHTML = `${steps(1)}<div class="topline"><div><h2>好きな画像を選んでください。</h2><p>人物ではなく、色・光・描き方・雰囲気の好みで選びます。選んだ画像は最初「全部好き」になっているので、好きなところだけに絞り込めます。</p></div><div class="counter" aria-label="選んだ枚数"><b>${draft.selection.length}</b><small>/ ${limits.min}〜${limits.max}枚</small></div></div>${noticeLine()}<div class="roundline"><span>候補 ${ids.length}枚</span></div><div class="card-grid">${ids
    .map((id) => {
      const card = cardOf(id);
      if (!card) return "";
      const n = picked(id);
      return `<button class="card ${n >= 0 ? "selected" : ""}" data-card="${esc(id)}" aria-pressed="${n >= 0}" aria-label="${esc(card.label)}を${n >= 0 ? "選択解除" : "選ぶ"}"><img src="${esc(card.url)}" alt="${esc(card.label)}" loading="lazy">${n >= 0 ? `<span class="order">${n + 1}</span>` : ""}<span class="card-foot">${esc(card.subject_label || card.label)}</span></button>`;
    })
    .join("")}</div><div class="round-actions"><span class="note">選ばなかった画像を「嫌い」とは扱いません。</span></div>${trayBlock(
    "選んだ画像",
    "それぞれ、どこが好きかを教えてください。",
  )}<div class="action-row"><span class="note">${esc(
    BLOCK_TEXT(blocker) || `${draft.selection.length}枚をあなたの好みとして使います。`,
  )}</span><button id="commit" class="primary" ${blocker ? "disabled" : ""}>これで決定 <span>→</span></button></div>`;
  on("card", ({ card }) => {
    const result = toggleCard(draft, card, limits, aspectKeys());
    draft = result.draft;
    if (result.error) return error(result.error);
    edited();
  });
  bindPickRows();
  bind("commit", async () => {
    await save(true);
    const blocked = commitBlocker(draft, limits);
    if (blocked) return error(BLOCK_TEXT(blocked));
    say("");
    goto("topics");
  });
}
/**
 * Serve every remaining round up front, one request at a time, so the candidates
 * appear as one list. A retry reuses its request id: never an extra round.
 */
async function loadAllRounds() {
  while (canRequestRound(session, limits)) {
    const request_id = roundTicket.take();
    if (!request_id) return;
    try {
      for (let attempt = 0; ; attempt++) {
        try {
          // A round never changes the selection, so local edits are kept.
          adoptSession(
            await api(`/sessions/${session.id}/rounds`, "POST", {
              request_id,
              expected_revision: session.revision,
            }),
          );
          break;
        } catch (e) {
          if (e.status !== 409) throw e;
          adoptAll(await api(`/sessions/${session.id}`));
          if (!canRequestRound(session, limits) || attempt >= 1) throw e;
          await sleep(250);
        }
      }
      roundTicket.done();
    } catch (e) {
      if (!canRequestRound(session, limits)) {
        roundTicket.done();
        return;
      }
      roundTicket.fail();
      throw e;
    }
  }
}

/* ------------------------------------------------------ 02 お題を選ぶ */
function refStrip() {
  return `<div class="refbar">${draft.selection
    .map((entry, i) => {
      const card = cardOf(entry.card_id);
      if (!card) return "";
      return `<div class="refchip"><img src="${esc(card.url)}" alt="${esc(card.label)}"><div><b>${two(i + 1)} ${esc(refName(entry.card_id))}</b><code>${esc(refText(entry))}</code></div></div>`;
    })
    .join("")}</div>`;
}
function topicsScreen() {
  if (!topic) topic = session.run?.topic_id || cfg.topics[0]?.id;
  app.innerHTML = `${steps(2)}<h2>お題を選んでください。</h2><p>お題を選ぶと、あなたの好みを反映して描きます。反映に使うのは、選んだ画像の説明文です。</p>${noticeLine()}${refStrip()}<div class="topics">${cfg.topics
    .map(
      (t) =>
        `<button class="topic ${t.id === topic ? "selected" : ""}" data-topic="${esc(t.id)}" aria-pressed="${t.id === topic}"><img src="${esc(t.preview_url)}" alt="${esc(t.label)}のパーソナライズなし生成サンプル" loading="lazy"><span>${esc(t.label)}</span></button>`,
    )
    .join(
      "",
    )}</div><div class="action-row"><button id="back" class="quiet">← ${session.run ? "比較に戻る" : "好きな画像を選び直す"}</button><button id="generate" class="primary">この好みで描く <span>→</span></button></div><p class="note">あなたの好みを反映した4枚を、比較用に同じ入力文・同じseedで描いた通常の4枚と並べます。</p>`;
  on("topic", ({ topic: id }) => {
    topic = id;
    topicsScreen();
  });
  bind("back", () => {
    say("");
    goto(session.run ? "compare" : "cards");
  });
  bind("generate", () => generate(topic));
}
/** Commit the draft, then start one run for it. Seeds are fixed server-side. */
async function generate(topicId) {
  await save(true);
  const blocked = commitBlocker(draft, limits);
  if (blocked) {
    say(BLOCK_TEXT(blocked));
    return goto("cards");
  }
  const request_id = runTicket.take();
  if (!request_id) return;
  const button = document.querySelector("#generate") || document.querySelector("#redraw");
  if (button) button.disabled = true;
  try {
    adoptAll(
      await api(`/sessions/${session.id}/runs`, "POST", {
        topic_id: topicId,
        request_id,
        expected_revision: session.revision,
      }),
    );
    runTicket.done();
    topic = topicId;
    say("");
    renderedRun = JSON.stringify(session.run);
    goto("compare");
    if (session.run?.status !== "done") poller.start();
  } catch (e) {
    runTicket.fail();
    if (e.status === 409) {
      adoptAll(await api(`/sessions/${session.id}`));
      say("直前の処理が終わっていません。少し待ってからもう一度お試しください。");
      render();
    }
    throw e;
  } finally {
    if (button) button.disabled = false;
  }
}

/* ------------------------------------------------------ 03 見比べる */
function statusbox(run) {
  const working = run.status !== "done";
  return `<div class="statusbox ${working ? "" : "quiet-box"}" role="status">${working ? '<span class="spinner"></span>' : "<span>✓</span>"}<p>${esc(run.message || (working ? "描いています…" : "できました。"))}${run.elapsed_seconds ? ` · ${esc(run.elapsed_seconds)}秒` : ""}</p></div>`;
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
 * One row per distinct aspect phrase. A phrase that several selected cards share
 * carries a larger weight, and its thumbnails show why.
 */
function refRows(refs) {
  const keys = aspectKeys();
  const groups = keys.map((key) => [
    key,
    refs.filter((r) => (r.aspects || []).includes(key)),
  ]);
  const rest = refs.filter((r) => !(r.aspects || []).some((a) => keys.includes(a)));
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
/** Everything technical lives here, folded away from the main copy. */
function inputsPanel(run) {
  const p = run.personalization || {};
  const policy = p.effective_policy || {};
  return `<details class="prompt-details"><summary>パーソナライズに渡したもの</summary><p class="kv"><b>お題の一文（全員に共通・変えていません）</b></p><code class="block">${esc(run.prompt || "—")}</code><p class="kv"><b>参照した好み（選んだ画像に付けた確認済みの説明文）</b>　同じ説明文を複数の画像が持つと、その分だけ重みが大きくなります。</p><div class="ref-used">${refRows(
    p.refs || [],
  )}</div><p class="kv"><b>設定</b>　alpha ${esc(p.alpha ?? cfg.policy?.alpha ?? "—")} / policy ${esc(run.policy_id || "—")} / pooled ${esc(policy.pooled_mode || "—")} / Reference Prompt 単位 ${esc(policy.reference_unit || "—")}</p><p class="kv"><b>内容ハッシュ</b>　<code>${esc(run.personalization_hash || "—")}</code></p><p>生成モデル・seed・Negative Prompt・生成設定は、パーソナライズなしの4枚とまったく同じです。</p></details>`;
}
function gainBlock() {
  return `<div class="gains" role="group" aria-label="側面ごとの強さ">${aspectKeys()
    .map((key) => {
      const index = gainIndex(draft.aspect_gains[key]);
      return `<div class="gain"><span class="gain-name">${esc(cfg.aspects[key])}</span><div class="seg small" role="group" aria-label="${esc(cfg.aspects[key])}の強さ">${GAIN_STEPS.map(
        (value, i) =>
          `<button class="step ${i === index ? "on" : ""}" data-gain="${key}" data-level="${value}" aria-pressed="${i === index}">${GAIN_LABELS[i]}</button>`,
      ).join("")}</div></div>`;
    })
    .join("")}</div>`;
}
function adjustBlock(run) {
  if (run.mode === "sample") return "";
  const blocker = commitBlocker(draft, limits);
  const changed = !draftMatchesSnapshot(
    { ...draft, committed: true },
    session,
    aspectKeys(),
  );
  return `<section class="adjust"><div class="picked-head"><h3>好みを調整する</h3><span>変更は「同じお題で描き直す」を押したときに反映されます。お題と seed は同じままです。</span></div>${gainBlock()}${trayBlock(
    "選んだ画像",
    "好きな側面と強さを直せます。",
  )}<div class="action-row"><span class="note">${esc(
    BLOCK_TEXT(blocker) ||
      (changed
        ? "変更があります。描き直すと反映されます。"
        : "変更していません。描き直すと同じ結果になります。"),
  )}</span><button id="redraw" class="primary" ${blocker || run.status !== "done" ? "disabled" : ""}>同じお題で描き直す <span>→</span></button></div></section>`;
}
/** Redraw only the adjust panel, so the eight images on screen are not reloaded. */
function redrawAdjust() {
  const host = app.querySelector(".adjust");
  if (!host || !session?.run) return render();
  host.outerHTML = adjustBlock(session.run);
  bindAdjust(session.run);
}
function bindAdjust(run) {
  bindPickRows();
  on("gain", ({ gain, level }) => {
    draft = setGain(draft, gain, Number(level));
    edited();
  });
  bind("redraw", () => generate(run.topic_id));
}
function compareScreen() {
  const run = session.run;
  if (!run) {
    return goto(commitBlocker(draft, limits) ? "cards" : "topics");
  }
  const sample = run.mode === "sample";
  const working = run.status !== "done";
  app.innerHTML = `${steps(3)}<h2>あなたの好みで描いた結果です。</h2><div class="callout"><b>${esc(topicLabel(run.topic_id))}</b><span>違いは、あなたの好みを参照したかどうかだけです。</span></div>${noticeLine()}${statusbox(
    run,
  )}<section class="row"><div class="comparison-label"><h3>通常の生成</h3><span>好みを使わずに描いた4枚</span></div>${shots(
    run.plain,
    working,
  )}</section><section class="row mine"><div class="comparison-label"><h3>あなた向けにパーソナライズ</h3><span>選んだ画像の好みを参照して描いた4枚</span><span class="badge ${sample ? "sample" : ""}">${esc(MODES[run.mode] || run.mode)}</span>${
    run.error ? `<span class="badge sample">${esc(run.error)}</span>` : ""
  }</div>${shots(run.personal, working)}${inputsPanel(run)}</section>${
    sample
      ? '<p class="error-message">事前生成のサンプルです。あなたの選択を反映した結果ではありません。あなたの好みで試すには、下の「好きな画像を選ぶ」へ進んでください。</p>'
      : ""
  }${adjustBlock(run)}<div class="action-row"><button id="reselect" class="secondary">好きな画像を選ぶ</button><button id="another" class="secondary" ${working ? "disabled" : ""}>別のお題で描く</button>${
    working ? '<button id="cancel" class="secondary">描くのを中止する</button>' : ""
  }<button id="finish" class="primary">体験を終了</button></div><p class="note">終了すると、この体験の選択・ Reference Prompt ・生成画像は削除されます。${cfg.idle_seconds}秒の無操作でも終了します。${working ? "処理中は無操作リセットを止めています。" : ""}</p>`;
  bindAdjust(run);
  bind("reselect", async () => {
    say("");
    await loadAllRounds(); // fill in rounds a reload interrupted
    goto("cards");
  });
  bind("another", () => {
    topic = null;
    say("");
    goto("topics");
  });
  bind("cancel", async () => {
    poller.stop();
    adoptSession(await api(`/sessions/${session.id}/cancel`, "POST"));
    say("生成を中止しました。");
    renderedRun = JSON.stringify(session.run);
    goto(session.run ? "compare" : "topics");
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
  draft = emptyDraft(cfg ? aspectKeys() : undefined);
  topic = null;
  renderedRun = "";
  say("");
  roundTicket.done();
  runTicket.done();
  sessionStorage.removeItem(SESSION_KEY);
  screen = "welcome";
}
const poller = createPoller({
  identity: () => pollIdentity(session),
  fetchSnapshot: (sid) => api(`/sessions/${sid}`),
  onSnapshot: (current) => {
    adoptSession(current);
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
    if (!schemaSupported(cfg)) {
      app.innerHTML =
        "<h2>画面が新しくなりました。</h2><p>お手数ですが、ページを再読み込みしてください。</p>";
      return;
    }
    limits = withLimits(cfg.selection);
    const saved = sessionStorage.getItem(SESSION_KEY);
    if (saved) {
      try {
        // Screen, round, draft and commit state all come from the snapshot.
        adoptAll(await api(`/sessions/${saved}`));
        topic = session.run?.topic_id || null;
        screen = initialScreen(session, limits);
        renderedRun = JSON.stringify(session.run);
        if (session.run && session.run.status !== "done") poller.start();
        if (screen === "cards") await loadAllRounds().catch((e) => error(e.message));
      } catch {
        sessionStorage.removeItem(SESSION_KEY);
        session = null;
        draft = emptyDraft(aspectKeys());
      }
    }
    render();
  } catch (e) {
    app.innerHTML =
      "<h2>接続できませんでした。</h2><p>ローカルサーバーを確認し、ページを再読み込みしてください。</p>";
    error(e.message);
  }
})();
