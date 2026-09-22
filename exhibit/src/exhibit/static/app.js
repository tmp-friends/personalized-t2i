import {
  canRequestRound,
  commitBlocker,
  createPoller,
  createSelectionWriter,
  createTicket,
  currentRound,
  draftFromSnapshot,
  draftMatchesSnapshot,
  emptyDraft,
  feedbackVisible,
  gainIndex,
  GAIN_LABELS,
  GAIN_STEPS,
  initialScreen,
  likeAll,
  pollIdentity,
  removeCard,
  roundNotice,
  roundNumber,
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
  sample: "事前生成サンプル",
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
    // A sample is somebody else's preference; the visitor's draft is untouched.
    adoptSession(
      await api(`/sessions/${session.id}/sample`, "POST", {
        sample_id: document.querySelector("#sample-id").value,
      }),
    );
    renderedRun = JSON.stringify(session.run);
    say("");
    goto("compare");
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
  }${sampleControl()}</div>${heroArt()}</section><ol class="journey"><li><b>01</b><h3>好きな画像を選ぶ</h3><p>${limits.round_size}枚ずつ、最大${limits.max_rounds}回。合計${limits.min}〜${limits.max}枚</p></li><li><b>02</b><h3>お題を選ぶ</h3><p>一文は誰でも同じ</p></li><li><b>03</b><h3>見比べる</h3><p>パーソナライズなし・あり</p></li></ol>`;
  bind("start", () => start(true));
  bindSample();
}
async function start(show) {
  poller.stop();
  roundTicket.done();
  runTicket.done();
  adoptAll(await api("/sessions", "POST"));
  sessionStorage.setItem(SESSION_KEY, session.id);
  topic = null;
  say("");
  lastActive = Date.now();
  if (show) goto("cards");
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
function cardsScreen() {
  const round = currentRound(session);
  const ids = round?.card_ids || [];
  const blocker = commitBlocker(draft, limits);
  const more = canRequestRound(session, limits);
  const roundText = roundNotice(session, limits);
  app.innerHTML = `${steps(1)}<div class="topline"><div><h2>好きな画像を選んでください。</h2><p>人物ではなく、色・光・描き方・雰囲気の好みで選びます。好きなものがなければ、選ばずに次の候補へ進めます。</p></div><div class="counter" aria-label="選んだ枚数"><b>${draft.selection.length}</b><small>/ ${limits.min}〜${limits.max}枚</small></div></div>${noticeLine()}<div class="roundline"><span class="round-tag">${roundNumber(session)} / ${limits.max_rounds}回目</span><span>この回の候補 ${ids.length}枚</span>${
    roundText ? `<span class="round-note">${esc(roundText)}</span>` : ""
  }</div><div class="card-grid">${ids
    .map((id) => {
      const card = cardOf(id);
      if (!card) return "";
      const n = picked(id);
      return `<button class="card ${n >= 0 ? "selected" : ""}" data-card="${esc(id)}" aria-pressed="${n >= 0}" aria-label="${esc(card.label)}を${n >= 0 ? "選択解除" : "選ぶ"}"><img src="${esc(card.url)}" alt="${esc(card.label)}" loading="lazy">${n >= 0 ? `<span class="order">${n + 1}</span>` : ""}<span class="card-foot">${esc(card.subject_label || card.label)}</span></button>`;
    })
    .join("")}</div><div class="round-actions">${
    more
      ? '<button id="more" class="secondary">ほかの候補も見る <span>→</span></button>'
      : '<span class="note">候補はこれですべてです。</span>'
  }<span class="note">選ばなかった画像を「嫌い」とは扱いません。</span></div>${trayBlock(
    "選んだ画像",
    "それぞれ、どこが好きかを教えてください。前の回で選んだ画像もここで直せます。",
  )}<div class="action-row"><span class="note">${esc(
    BLOCK_TEXT(blocker) || `${draft.selection.length}枚を参照に使います。`,
  )}</span><button id="commit" class="primary" ${blocker ? "disabled" : ""}>これで決定 <span>→</span></button></div>`;
  on("card", ({ card }) => {
    const result = toggleCard(draft, card, limits);
    draft = result.draft;
    if (result.error) return error(result.error);
    edited();
  });
  bindPickRows();
  bind("more", nextRound);
  bind("commit", async () => {
    await save(true);
    const blocked = commitBlocker(draft, limits);
    if (blocked) return error(BLOCK_TEXT(blocked));
    say("");
    goto("topics");
  });
}
/** One request per intended round: a second click while in flight does nothing. */
async function nextRound() {
  if (!canRequestRound(session, limits)) return;
  await save(false);
  const button = document.querySelector("#more");
  const request_id = roundTicket.take();
  if (!request_id) return;
  if (button) button.disabled = true;
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
        roundTicket.done();
        say("");
        break;
      } catch (e) {
        if (e.status !== 409) throw e;
        adoptAll(await api(`/sessions/${session.id}`));
        if (!canRequestRound(session, limits)) {
          roundTicket.done();
          say("お見せできる画像はこれですべてです。");
          break;
        }
        if (attempt >= 1) throw e;
        await sleep(250);
      }
    }
  } catch (e) {
    roundTicket.fail(); // a retry stays the same request: never two rounds
    throw e;
  } finally {
    render();
    window.scrollTo(0, 0);
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
  app.innerHTML = `${steps(2)}<h2>お題を選んでください。</h2><p>お題の一文は誰でも同じです。渡すのは、選んだ画像の説明文だけです。</p>${noticeLine()}${refStrip()}<div class="topics">${cfg.topics
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
  )}</div><p class="kv"><b>設定</b>　alpha ${esc(p.alpha ?? cfg.policy?.alpha ?? "—")} / policy ${esc(run.policy_id || "—")} / pooled ${esc(policy.pooled_mode || "—")} / 参照単位 ${esc(policy.reference_unit || "—")}</p><p class="kv"><b>内容ハッシュ</b>　<code>${esc(run.personalization_hash || "—")}</code></p><p>生成モデル・seed・負のプロンプト・生成設定は、パーソナライズなしの4枚とまったく同じです。</p></details>`;
}
function feedbackBlock(run) {
  if (!feedbackVisible(run)) return "";
  const chosen = run.feedback?.preference || null;
  const options = [
    ["plain", "通常"],
    ["personal", "あなた向け"],
    ["tie", "同じくらい"],
  ];
  return `<section class="feedback"><div class="picked-head"><h3>どちらが好みですか？</h3><span>任意です。答えても答えなくても、画像は変わりません。</span></div><div class="seg" role="group" aria-label="どちらが好みか">${options
    .map(
      ([value, label]) =>
        `<button class="step ${chosen === value ? "on" : ""}" data-pref="${value}" aria-pressed="${chosen === value}">${label}</button>`,
    )
    .join("")}</div>${
    chosen
      ? '<p class="note">回答を受け取りました。いつでも選び直せます。</p>'
      : ""
  }</section>`;
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
  app.innerHTML = `${steps(3)}<h2>パーソナライズなし・ありを見比べてください。</h2><div class="callout"><b>${esc(topicLabel(run.topic_id))}</b><span>入力文も生成モデルも seed も同じです。違うのは、あなたの好みを参照したかどうかだけです。</span></div>${noticeLine()}${statusbox(
    run,
  )}<section class="row"><div class="comparison-label"><h3>パーソナライズなし</h3><span>好みを使わずに描いた4枚</span></div>${shots(
    run.plain,
    working,
  )}</section><section class="row mine"><div class="comparison-label"><h3>パーソナライズあり</h3><span>選んだ画像の好みを参照して描いた4枚</span><span class="badge ${sample ? "sample" : ""}">${esc(MODES[run.mode] || run.mode)}</span>${
    run.error ? `<span class="badge sample">${esc(run.error)}</span>` : ""
  }</div>${shots(run.personal, working)}${inputsPanel(run)}</section>${
    sample
      ? '<p class="error-message">事前生成のサンプルです。あなたの選択を反映した結果ではありません。あなたの好みで試すには、下の「好きな画像を選ぶ」へ進んでください。</p>'
      : ""
  }${feedbackBlock(run)}${adjustBlock(run)}<div class="action-row"><button id="reselect" class="secondary">好きな画像を選ぶ</button><button id="another" class="secondary" ${working ? "disabled" : ""}>別のお題で描く</button>${
    working ? '<button id="cancel" class="secondary">描くのを中止する</button>' : ""
  }<button id="finish" class="primary">体験を終了</button></div><p class="note">終了すると、この体験の選択・参照・生成画像は削除されます。${cfg.idle_seconds}秒の無操作でも終了します。${working ? "処理中は無操作リセットを止めています。" : ""}</p>`;
  bindAdjust(run);
  on("pref", ({ pref }) => act(() => answer(pref)));
  bind("reselect", () => {
    say("");
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
/** The optional, labelled answer about this finished comparison. */
async function answer(preference) {
  const run = session.run;
  if (!feedbackVisible(run)) return;
  try {
    adoptSession(
      await api(`/sessions/${session.id}/runs/${run.id}/feedback`, "PUT", {
        expected_revision: session.revision,
        preference,
      }),
    );
    say("");
  } catch (e) {
    if (e.status !== 409) throw e;
    adoptAll(await api(`/sessions/${session.id}`));
    say("この結果には回答できませんでした。画面を最新にしました。");
  }
  renderedRun = JSON.stringify(session.run);
  render();
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
