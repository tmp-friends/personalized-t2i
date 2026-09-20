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
  adjust = { basedOn: undefined, alpha: "mid", weights: {} },
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
const STEPS = [
  "01 好きな画像を選ぶ",
  "02 お題を選ぶ",
  "03 どちらが好き？",
  "04 答えと調整",
];
function steps(n) {
  return `<div class="stepbar">${STEPS.map(
    (label, i) =>
      `<span class="${i + 1 === n ? "active" : ""}">${esc(label)}</span>`,
  ).join('<i class="line"></i>')}</div>`;
}
function render() {
  resetButton.hidden = !session;
  (
    {
      welcome,
      cards: cardsScreen,
      topics: topicsScreen,
      blind: blindScreen,
      result: resultScreen,
    }[screen] || welcome
  )();
}

/* ---------------------------------------------------------------- welcome */
function sampleControl() {
  return cfg.samples?.length
    ? `<div class="sample-picker"><select id="sample-id" aria-label="事前生成サンプル">${cfg.samples
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
    adjust.basedOn = undefined;
    screen = "result";
    render();
    window.scrollTo(0, 0);
  });
}
function welcome() {
  const art = cfg.topics?.[0];
  app.innerHTML = `<section class="hero"><div><div class="eyebrow">FOUNDATION ENCODERS · YOUR TASTE</div><h1>同じ一文から、<br>あなたの一枚を。</h1><p class="intro">好きな画像を選ぶ。同じお題で、方式を伏せたまま見比べる。<br>答えを見たら、好みの反映を自分の手で操作する。<br>入力文も生成モデルも変えずに、絵がどこまで変わるかを確かめてください。</p><button class="primary" id="start" ${cfg.ready ? "" : "disabled"}>体験をはじめる <span>↗</span></button><p class="note">約3分 · 登録不要 · この端末の中だけで動作します</p><div class="welcome-small">${sampleControl()}</div>${
    cfg.ready
      ? ""
      : '<p class="note ready-note">画像を準備中です。準備が終わると体験できます。</p>'
  }</div><div class="hero-art">${
    art?.preview_url
      ? `<img src="${esc(art.preview_url)}" alt="${esc(art.label)}の通常生成画像">`
      : ""
  }<div class="floating-note"><small>SAME PROMPT, SAME MODEL</small>変えるのは、参照する好みと反映の強さだけ。</div><p class="image-caption">Illustrious XL v2.0 · 参照なしの事前生成画像</p></div></section><div class="journey"><div><b>01</b><span>好きな画像を選ぶ<small>16枚から3〜5枚</small></span></div><div><b>02</b><span>同じお題で比べる<small>方式は伏せたまま</small></span></div><div><b>03</b><span>反映を操作する<small>強さと参照を自分で</small></span></div></div>`;
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
  adjust = { basedOn: undefined, alpha: "mid", weights: {} };
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
  app.innerHTML = `${steps(1)}<div class="topline"><div><div class="eyebrow">PICK WHAT YOU LIKE</div><h2>好きな画像を選んでください。</h2><p>同じ人物が4つの描き方で並んでいます。人物ではなく、描き方の好みで選んでください。</p></div><div class="counter">${two(selection.length)}<small> / ${min}〜${max}枚</small></div></div><div class="card-grid">${order
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
              )}</div><code class="ref-en">${esc(refText(entry) || "（側面がすべて外れています）")}</code></div><button class="quiet" data-drop="${esc(entry.card_id)}">選択を外す ✕</button></div>`;
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
  app.innerHTML = `${steps(2)}<div class="eyebrow">SAME PROMPT FOR EVERYONE</div><h2>お題を選んでください。</h2><p>お題の一文は誰でも同じです。渡すのは、下の英語の説明文だけです。</p>${refStrip()}<div class="topics">${cfg.topics
    .map(
      (t) =>
        `<button class="topic ${t.id === topic ? "selected" : ""}" data-topic="${esc(t.id)}" aria-pressed="${t.id === topic}"><img src="${esc(t.preview_url)}" alt="${esc(t.label)}の参照なし生成サンプル" loading="lazy"><span>${esc(t.label)}</span></button>`,
    )
    .join(
      "",
    )}</div><div class="action-row"><button id="back" class="quiet">← ${session.run ? "答えに戻る" : "好きな画像を選び直す"}</button><button id="generate" class="primary">この好みで描く <span>↗</span></button></div><p class="note">参照なしの4枚と、同じ入力文・同じseedで描いたあなた向けの4枚を、方式を伏せて並べます。</p>`;
  on("topic", ({ topic: id }) => {
    topic = id;
    topicsScreen();
  });
  bind("back", () => {
    screen = session.run ? "result" : "cards";
    render();
    window.scrollTo(0, 0);
  });
  bind("generate", async () => {
    const button = document.querySelector("#generate");
    if (button) button.disabled = true;
    try {
      const weights = Object.fromEntries(
        selection.map((s) => [s.card_id, "normal"]),
      );
      session = await api(`/sessions/${session.id}/runs`, "POST", {
        topic_id: topic,
        alpha: "mid",
        weights,
        request_id: crypto.randomUUID(),
      });
      adjust = { basedOn: undefined, alpha: "mid", weights };
      screen = session.run?.blind?.revealed ? "result" : "blind";
      renderedRun = JSON.stringify(session.run);
      render();
      window.scrollTo(0, 0);
      watch();
    } finally {
      const b = document.querySelector("#generate");
      if (b) b.disabled = false;
    }
  });
}

/* ------------------------------------------------- 03 どちらが好き？ */
function statusbox(run) {
  const working = run.status !== "done";
  return `<div class="statusbox" role="status">${working ? '<span class="spinner"></span>' : "<span>✓</span>"}<p>${esc(run.message || (working ? "描いています…" : "できました。"))}${run.elapsed_seconds ? ` · ${esc(run.elapsed_seconds)}秒` : ""}</p></div>`;
}
function blindScreen() {
  const run = session.run;
  if (!run) {
    screen = selectionComplete(selection, cfg.selection) ? "topics" : "cards";
    return render();
  }
  if (run.blind?.revealed) {
    screen = "result";
    return render();
  }
  adoptSelection();
  const pairs = run.blind?.pairs || [];
  const allReady = pairs.length > 0 && pairs.every((p) => p.ready);
  const working = run.status !== "done";
  app.innerHTML = `${steps(3)}<div class="topline"><div><div class="eyebrow">WHICH ONE DO YOU LIKE?</div><h2>どちらが好きですか？</h2><p>${esc(topicLabel(run.topic_id))}。ラベルはありません。左右の並びはランダムです。</p></div><div class="counter">${two(run.blind?.answered || 0)}<small> / ${two(pairs.length)}</small></div></div>${statusbox(run)}<div class="pair-list">${pairs
    .map((pair) => {
      const answered = pair.pick != null;
      return `<section class="pair-row ${answered ? "answered" : ""}"><div class="pair-head"><span class="seedtag">seed ${esc(pair.seed)}</span><span class="pair-state">${pair.ready ? (answered ? (pair.pick === "tie" ? "決められない を記録しました" : "この対は回答済み") : "好きな方をえらんでください") : "描いています…"}</span></div><div class="pair-grid">${
        pair.ready
          ? pair.items
              .map(
                (item, i) =>
                  `<button class="pair ${pair.pick === item.token ? "chosen" : ""}" data-pair="${pair.index}" data-pick="${esc(item.token)}" aria-label="seed ${esc(pair.seed)} の候補${i === 0 ? "A" : "B"}をえらぶ"><img src="${esc(item.url)}" alt="seed ${esc(pair.seed)} の候補${i === 0 ? "A" : "B"}"><span class="pair-bottom"><b>${i === 0 ? "A" : "B"}</b><span>こちらが好き ↗</span></span></button>`,
              )
              .join("")
          : '<div class="placeholder">描いています…</div><div class="placeholder">描いています…</div>'
      }</div>${pair.ready ? `<button class="quiet tie ${pair.pick === "tie" ? "chosen" : ""}" data-pair="${pair.index}" data-pick="tie">決められない</button>` : ""}</section>`;
    })
    .join(
      "",
    )}</div><div class="action-row"><button id="cancel" class="secondary" ${working ? "" : "hidden"}>中止して好みを直す</button><button id="reveal" class="primary" ${allReady ? "" : "disabled"}>答えを見る <span>↗</span></button></div><p class="note">すべてに答えなくても構いません。4対そろったら「答えを見る」を押せます。${working ? "処理中は無操作リセットを止めています。" : ""}</p>`;
  on("pick", async ({ pair, pick }) => {
    session = await api(`/sessions/${session.id}/blind`, "POST", {
      pair_index: Number(pair),
      pick,
    });
    renderedRun = JSON.stringify(session.run);
    keepScroll(render);
  });
  bind("cancel", async () => {
    poller.stop();
    session = await api(`/sessions/${session.id}/cancel`, "POST");
    // Cancelling the first variant drops the run; the selection survives.
    adoptSelection();
    screen = session.run
      ? session.run.blind?.revealed
        ? "result"
        : "blind"
      : "cards";
    render();
    window.scrollTo(0, 0);
  });
  bind("reveal", async () => {
    session = await api(`/sessions/${session.id}/reveal`, "POST");
    adjust.basedOn = undefined;
    screen = "result";
    renderedRun = JSON.stringify(session.run);
    render();
    window.scrollTo(0, 0);
    if (session.run?.status !== "done") watch();
  });
}

/* --------------------------------------------------- 04 答えと調整 */
/** Past the card grid the server owns the selection (a sample run sets its own). */
function adoptSelection() {
  const theirs = session?.selection;
  if (theirs?.length && JSON.stringify(theirs) !== JSON.stringify(selection))
    selection = structuredClone(theirs);
}
function headline(run) {
  if (run.variants?.some((v) => v.mode === "sample"))
    return "事前生成のサンプルです。通常の4枚と、代表的な好みで描いた4枚を並べています。";
  const score = run.blind?.score;
  const total = run.blind?.pairs?.length || 4;
  if (!score || !score.answered)
    return "答えを見ましょう。どちらが好みを反映した画像かを、下に示します。";
  const personal = score.personal || 0;
  const decided = personal + (score.plain || 0);
  const notes = [];
  if (score.tie) notes.push(`${score.tie}対は「決められない」`);
  if (score.answered < total) notes.push(`${total - score.answered}対は未回答`);
  const tail = notes.length ? `（${notes.join("・")}）` : "";
  if (!decided)
    return `${score.answered}対すべてで「決められない」をえらびました。差は小さかった、ということです。`;
  const head = notes.length ? `好きな方をえらんだ${decided}対のうち` : `${decided}対のうち`;
  if (!personal)
    return `${head}、好みを反映した画像は1枚もえらばれませんでした。${tail}`;
  return `${head}、あなたが選んだ${personal}枚は、好みを反映した画像でした。${tail}`;
}
function variantLabel(v) {
  const bits = [`反映 ${esc(cfg.alpha_labels?.[v.alpha_key] || v.alpha_key)}`];
  const by = (key) =>
    Object.entries(v.weights || {})
      .filter(([, k]) => k === key)
      .map(([id]) => refName(id));
  const emphasis = by("emphasis"),
    excluded = by("exclude");
  if (emphasis.length) bits.push(`重視: ${emphasis.join("・")}`);
  if (excluded.length) bits.push(`外す: ${excluded.join("・")}`);
  if (!emphasis.length && !excluded.length) bits.push("参照はすべて通常");
  return bits.join(" · ");
}
function shots(images, count = 4) {
  return `<div class="image-grid">${Array.from({ length: count }, (_, i) => {
    const x = images?.[i];
    return x
      ? `<figure class="shot"><img src="${esc(x.url)}" alt="seed ${esc(x.seed)} の生成画像" loading="lazy"><figcaption><span>${two(i + 1)}</span><span>seed ${esc(x.seed)}</span></figcaption></figure>`
      : `<div class="placeholder">描いています…</div>`;
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
function inputsPanel(v) {
  const p = v.personalization;
  return `<details class="prompt-details" open><summary>この一枚に渡したもの</summary><p class="kv"><b>お題の一文（全員に共通・変えていません）</b></p><code class="block">${esc(v.prompt || "—")}</code><p class="kv"><b>参照した好み（選んだ画像に付けた確認済みの説明文）</b>　同じ説明文を複数の画像が持つと、その分だけ重みが大きくなります。</p><div class="ref-used">${refRows(
    p?.refs || [],
  )}</div><p class="kv"><b>反映の強さ alpha</b>　${esc(cfg.alpha_labels?.[v.alpha_key] || v.alpha_key)} = ${esc(p?.alpha ?? cfg.alphas?.[v.alpha_key] ?? "—")}</p><p>生成モデル・seed・負のプロンプト・生成設定は、参照なしの4枚とまったく同じです。</p></details>`;
}
function adjustPanel(run) {
  const working = run.status !== "done";
  const full = (run.variants?.length || 0) >= (cfg.max_variants || 3);
  const disabled = working || full;
  return `<section class="adjust"><div class="comparison-label"><h3>強さと参照を調整する</h3><span>押したときだけ描き直します</span></div><div class="adjust-row"><span class="adjust-name">反映の強さ</span><div class="seg" role="group" aria-label="反映の強さ">${Object.keys(
    cfg.alphas || {},
  )
    .map(
      (key) =>
        `<button class="${adjust.alpha === key ? "on" : ""}" data-alpha="${esc(key)}" aria-pressed="${adjust.alpha === key}">${esc(cfg.alpha_labels?.[key] || key)}<small>${esc(cfg.alphas[key])}</small></button>`,
    )
    .join("")}</div></div>${selection
    .map((entry) => {
      const card = cardOf(entry.card_id);
      const current = adjust.weights[entry.card_id] || "normal";
      return `<div class="adjust-row"><span class="adjust-name"><img src="${esc(card?.url || "")}" alt="">${esc(refName(entry.card_id))}</span><div class="seg" role="group" aria-label="${esc(refName(entry.card_id))}の重み">${Object.keys(
        cfg.weights || {},
      )
        .map(
          (key) =>
            `<button class="${current === key ? "on" : ""}" data-weight="${esc(key)}" data-card="${esc(entry.card_id)}" aria-pressed="${current === key}">${esc(cfg.weight_labels?.[key] || key)}</button>`,
        )
        .join("")}</div></div>`;
    })
    .join(
      "",
    )}<div class="action-row"><p class="note">反映を強くするほど良いわけではありません。お題とのバランスは、あなたが決めてください。${full ? `<br>描き直しはこの体験で${cfg.max_variants}回までです。` : ""}${working ? "<br>いま描いています。終わるまでお待ちください。" : ""}</p><button id="regen" class="primary" ${disabled ? "disabled" : ""}>この設定で描く <span>↗</span></button></div></section>`;
}
function resultScreen() {
  const run = session.run;
  if (!run) {
    screen = selectionComplete(selection, cfg.selection) ? "topics" : "cards";
    return render();
  }
  adoptSelection();
  syncAdjust(run);
  const variants = [...(run.variants || [])].reverse(); // newest first
  const sample =
    run.mode === "sample" || variants.some((v) => v.mode === "sample");
  const working = run.status !== "done";
  const canRetopic = ownSelection && selectionComplete(selection, cfg.selection);
  app.innerHTML = `${steps(4)}<div class="eyebrow">THE ANSWER</div><h2>${esc(headline(run))}</h2><div class="callout">入力文も生成モデルも変えていません。変えたのは、参照した好みと反映の強さだけです。</div>${
    run.status === "done" ? "" : statusbox(run)
  }<div class="comparison-label"><h3>通常（参照なし）</h3><span>${esc(topicLabel(run.topic_id))} · 好みを使わずに描いた4枚</span></div>${shots(
    run.plain,
  )}${variants
    .map(
      (v, i) =>
        `<div class="comparison-label ${i === 0 ? "newest" : ""}"><h3>好みを反映</h3><span>${variantLabel(v)}</span><span class="badge ${sample ? "sample" : ""}">${esc(MODES[sample ? "sample" : v.mode] || v.mode)}</span>${i === 0 && variants.length > 1 ? '<span class="badge new">最新</span>' : ""}${v.error ? `<span class="badge sample">${esc(v.error === "cancelled" ? "中止しました" : v.error)}</span>` : ""}</div><div class="variant ${i === 0 ? "newest" : ""}">${shots(v.images, 4)}${i === 0 ? inputsPanel(v) : ""}</div>`,
    )
    .join("")}${
    sample
      ? `<p class="error-message">事前生成のサンプルです。あなたの選択を反映した結果ではなく、強さと参照の調整もできません。${canRetopic ? "" : "ご自分の好みで試すには、最初から始めてください。"}</p>`
      : adjustPanel(run)
  }<div class="action-row">${
    canRetopic
      ? `<button id="another" class="secondary" ${working ? "disabled" : ""}>別のお題で描く</button>`
      : '<button id="restart" class="secondary">最初から始める</button>'
  }${working ? '<button id="cancel" class="secondary">描くのを中止する</button>' : ""}<button id="finish" class="primary">体験を終了 <span>↗</span></button></div><p class="note">終了すると、この体験の選択・参照・生成画像は削除されます。${cfg.idle_seconds}秒の無操作でも終了します。</p>`;
  on("alpha", ({ alpha }) => {
    adjust.alpha = alpha;
    keepScroll(resultScreen);
  });
  on("weight", ({ weight, card }) => {
    const next = { ...adjust.weights, [card]: weight };
    if (Object.values(next).every((k) => k === "exclude"))
      return error("参照をすべて外すことはできません。1件は残してください。");
    adjust.weights = next;
    keepScroll(resultScreen);
  });
  bind("regen", async () => {
    session = await api(`/sessions/${session.id}/runs`, "POST", {
      topic_id: run.topic_id,
      alpha: adjust.alpha,
      weights: adjust.weights,
      request_id: crypto.randomUUID(),
    });
    adjust.basedOn = undefined;
    renderedRun = JSON.stringify(session.run);
    render();
    window.scrollTo(0, 0);
    watch();
  });
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
    adoptSelection();
    if (!session.run) screen = "cards";
    renderedRun = JSON.stringify(session.run);
    render();
  });
  bind("finish", finish);
}
/** Seed the adjust controls from the newest variant, until the visitor edits them. */
function syncAdjust(run) {
  const v = run.variants?.[run.variants.length - 1];
  const id = v?.id ?? null;
  if (adjust.basedOn === id) return;
  const weights = { ...(v?.weights || {}) };
  for (const entry of selection)
    if (!weights[entry.card_id]) weights[entry.card_id] = "normal";
  adjust = { basedOn: id, alpha: v?.alpha_key || "mid", weights };
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
  adjust = { basedOn: undefined, alpha: "mid", weights: {} };
  sessionStorage.removeItem(SESSION_KEY);
  sessionStorage.removeItem(DRAFT_KEY);
  screen = "welcome";
}
const poller = createPoller({
  identity: () => pollIdentity(session),
  fetchSnapshot: (sid) => api(`/sessions/${sid}`),
  onSnapshot: (current) => {
    session = current;
    if (screen === "blind" && session.run?.blind?.revealed) screen = "result";
    const signature = JSON.stringify(session.run);
    if (signature !== renderedRun && (screen === "blind" || screen === "result")) {
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
