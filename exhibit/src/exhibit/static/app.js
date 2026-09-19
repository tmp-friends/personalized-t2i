import { createPoller, initialScreen } from "./session.mjs";
("use strict");
const app = document.querySelector("#app");
const resetButton = document.querySelector("#reset");
let cfg,
  session = null,
  screen = "welcome",
  edits = {},
  topic = "cat",
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
const title = (id) => cfg.topics.find((t) => t.id === id)?.label || id;
const image = (id, alt = "選択した画像") =>
  `<img src="/assets/pairs/${esc(id)}.png" alt="${esc(alt)}">`;
function error(text) {
  const box = document.querySelector("#error");
  box.textContent = text;
  box.hidden = false;
  setTimeout(() => (box.hidden = true), 6000);
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
function steps(n) {
  return `<div class="stepbar"><span class="${n === 1 ? "active" : ""}">01 好きな画像を選ぶ</span><i class="line"></i><span class="${n === 2 ? "active" : ""}">02 好みを確かめる</span><i class="line"></i><span class="${n === 3 ? "active" : ""}">03 あなた向けに描く</span></div>`;
}
function bind(id, fn) {
  document.getElementById(id)?.addEventListener("click", () => act(fn));
}
function render() {
  resetButton.hidden = !session;
  (({ welcome, choose, persona, topics, result })[screen] || welcome)();
}
function sampleControl() {
  return cfg.samples.length
    ? `<div class="sample-picker"><select id="sample-id" aria-label="事前生成サンプル">${cfg.samples.map((s) => `<option value="${esc(s.id)}">${esc(s.id.split("-")[0].toUpperCase())} · ${esc(title(s.topic_id))}</option>`).join("")}</select><button class="secondary" id="sample">サンプル体験</button></div>`
    : "";
}
function bindSample() {
  bind("sample", async () => {
    if (!session) await start(false);
    session = await api(`/sessions/${session.id}/sample`, "POST", {
      sample_id: document.querySelector("#sample-id").value,
    });
    screen = "result";
    render();
  });
}
function welcome() {
  app.innerHTML = `<section class="hero"><div><div class="eyebrow">FIND YOUR VISUAL TASTE</div><h1>あなたの「好き」を、<br>一枚の絵へ。</h1><p class="intro">5回、直感で画像を選ぶ。<br>見えてきた好みを、あなたの言葉で確かめる。<br>AIと一緒に、あなたらしい描き方を探しましょう。</p><button class="primary" id="start">体験をはじめる <span>↗</span></button><p class="note">約2〜3分 · 登録不要 · この端末の中で動作します</p><div class="welcome-small">${sampleControl()}</div>${cfg.ready ? "" : '<p class="note ready-note">画像を準備中です。準備完了後に体験できます。</p>'}</div><div class="hero-art"><img src="/assets/generic/cat-0.png" alt="窓辺に座る猫の生成画像"><div class="floating-note"><small>A LITTLE MORE YOU</small>柔らかな光。好きな色。あなたの一枚。</div><p class="image-caption">SDXL · 事前生成した通常画像</p></div></section><div class="journey"><div><b>01</b><span>直感で選ぶ<small>5つの画像ペアから</small></span></div><div><b>02</b><span>解釈を直す<small>違っていたら、変えていい</small></span></div><div><b>03</b><span>並べて見つける<small>最後に選ぶのは、あなた</small></span></div>`;
  bind("start", () => start(true));
  bindSample();
}
async function start(show) {
  poller.stop();
  session = await api("/sessions", "POST");
  sessionStorage.setItem("taste-session", session.id);
  edits = {};
  sessionStorage.removeItem("taste-edits");
  lastActive = Date.now();
  if (show) {
    screen = "choose";
    render();
  }
}
function choose() {
  const answered = new Set(session.choices.map((c) => c.pair_id));
  let pair = session.pairs.find((p) => !answered.has(p.id));
  const valid = session.choices.filter((c) => c.chosen_id).length;
  const retry = !pair;
  if (!pair && valid >= 3) {
    screen = "persona";
    render();
    return;
  }
  if (!pair)
    pair = session.pairs.find(
      (p) => !session.choices.find((c) => c.pair_id === p.id)?.chosen_id,
    );
  const n = retry ? session.pairs.indexOf(pair) + 1 : answered.size + 1;
  app.innerHTML = `${steps(1)}<div class="topline"><div><div class="eyebrow">${retry ? "もう少しだけ、教えてください" : "FOLLOW YOUR INSTINCT"}</div><h2>どちらが、より好きですか？</h2><p>${esc(pair.label)} · 正解はありません。直感でどうぞ。</p></div><div class="counter">0${n}<small> / 05</small></div></div>${retry ? `<p class="error-message">有効な選択が${valid}回でした。好みを組み立てるには3回以上の選択が必要です。決められない場合はサンプルを体験できます。</p>` : ""}<div class="pair-grid">${pair.image_ids.map((id, i) => `<button class="pair" data-choice="${esc(id)}" aria-label="${i === 0 ? "左" : "右"}の画像を選ぶ">${image(id, `${pair.label}・${i === 0 ? "左" : "右"}の画像`)}<span class="pair-bottom"><b>${i === 0 ? "A" : "B"}</b><span>こちらが好き ↗</span></span></button>`).join("")}</div><div class="choice-bottom"><div class="dots">${session.pairs.map((p) => `<span class="dot ${answered.has(p.id) ? "done" : ""}"></span>`).join("")}</div><button id="skip" class="quiet">${retry ? "別の画像ペアを見る" : "決められない →"}</button></div><p class="note">選ばなかった画像を「嫌い」とは扱いません。画像の左右はランダムです。</p>${retry ? sampleControl() : ""}`;
  app.querySelectorAll("[data-choice]").forEach((b) =>
    b.addEventListener("click", () =>
      act(async () => {
        session = await api(`/sessions/${session.id}/choices`, "POST", {
          pair_id: pair.id,
          chosen_id: b.dataset.choice,
        });
        render();
      }),
    ),
  );
  bind("skip", async () => {
    session = await api(`/sessions/${session.id}/choices`, "POST", {
      pair_id: pair.id,
      chosen_id: null,
    });
    if (retry) {
      const i = session.pairs.findIndex((p) => p.id === pair.id);
      session.pairs.push(...session.pairs.splice(i, 1));
    }
    render();
  });
  bindSample();
}
function persona() {
  const p = session.persona;
  app.innerHTML = `${steps(2)}<div class="eyebrow">A REFLECTION, NOT A DEFINITION</div><div class="persona-head"><h2>こんな「好き」が、見えてきました。</h2></div><p class="intro">これは選択からの仮説です。違っていたら直してください。</p><p class="note">画像特徴は事前にローカルAIが解析。ここでは確認済みの根拠を集計した簡易表示を使います。</p><div class="persona-list">${Object.entries(
    cfg.axes,
  )
    .map(([key, axis]) => {
      const a = p.axes[key];
      const value = Object.hasOwn(edits, key) ? edits[key] : a.value;
      const ev = p.evidence.filter((e) => a.evidence_ids.includes(e.id));
      return `<div class="persona-row"><div class="axis-name">${esc(axis.label)}</div><div><div class="axis-value">${value ? esc(axis.values[value][0]) : a.conflict ? "複数の傾向があり、まだ分かりません" : "今回は反映しない / まだ分かりません"}</div><span class="evidence-text">${a.count > 1 ? `複数の選択で見られた傾向 · ${a.count}件の根拠` : a.count === 1 ? "1回だけの仮説 · 1件の根拠" : "根拠が足りないため推定していません"}</span>${ev.length ? `<details><summary>選択の根拠を見る</summary>${ev.map((e) => `<div class="evidence">${image(e.chosen_id)}<p>${esc(e.text_ja || e.text)}</p></div>`).join("")}</details>` : ""}</div><select data-axis="${key}" aria-label="${esc(axis.label)}の反映方法"><option value="__inferred" ${!Object.hasOwn(edits, key) ? "selected" : ""}>推定した好みを使う</option><option value="__off" ${Object.hasOwn(edits, key) && edits[key] === null ? "selected" : ""}>今回は反映しない</option>${Object.entries(
        axis.values,
      )
        .map(
          ([v, label]) =>
            `<option value="${v}" ${edits[key] === v ? "selected" : ""}>${esc(label[0])} に訂正</option>`,
        )
        .join("")}</select></div>`;
    })
    .join(
      "",
    )}</div><div class="action-row"><p class="note">OFF・訂正した項目は、生成と推薦の両方に同じように反映します。<br>年齢や性格などは推定しません。</p><button id="to-topics" class="primary">お題を選ぶ <span>→</span></button></div>`;
  app.querySelectorAll("[data-axis]").forEach((s) =>
    s.addEventListener("change", () => {
      if (s.value === "__inferred") delete edits[s.dataset.axis];
      else edits[s.dataset.axis] = s.value === "__off" ? null : s.value;
      sessionStorage.setItem(
        "taste-edits",
        JSON.stringify({ sid: session.id, edits }),
      );
      persona();
    }),
  );
  bind("to-topics", () => {
    screen = "topics";
    render();
    window.scrollTo(0, 0);
  });
}
function chips(context) {
  const prefs =
    context?.preferences ||
    Object.fromEntries(
      Object.entries(session.persona?.axes || {}).map(([k, v]) => [
        k,
        Object.hasOwn(edits, k) ? edits[k] : v.value,
      ]),
    );
  return `<div class="context-chips">${
    Object.entries(prefs)
      .filter(([, v]) => v)
      .map(([k, v]) => `<span>${esc(cfg.axes[k].values[v][0])}</span>`)
      .join("") || "<span>好みを反映しない通常生成</span>"
  }</div>`;
}
function topics() {
  app.innerHTML = `${steps(3)}<div class="eyebrow">SAME SCENE, YOUR PERSPECTIVE</div><h2>何を、描いてみましょう。</h2><p>お題はそのままに、色や光、描き方に好みを反映します。</p>${chips()}<div class="topics">${cfg.topics.map((t) => `<button class="topic ${t.id === topic ? "selected" : ""}" data-topic="${t.id}" aria-pressed="${t.id === topic}"><img src="/assets/generic/${t.id}-0.png" alt="${esc(t.label)}の通常生成サンプル"><span>${esc(t.label)}</span></button>`).join("")}</div><div class="action-row"><button id="back" class="quiet">← 好みを直す</button><button id="generate" class="primary">この好みで描く <span>↗</span></button></div><p class="note">通常側4枚と、同じseedで生成するあなた向け4枚を比較します。<br>現在は ZIPP-style の実生成モードです。PIGRewardの自動推薦は検証中のため使用していません。</p>`;
  app.querySelectorAll("[data-topic]").forEach((b) =>
    b.addEventListener("click", () => {
      topic = b.dataset.topic;
      topics();
    }),
  );
  bind("back", () => {
    screen = "persona";
    render();
  });
  bind("generate", async () => {
    document.querySelector("#generate").disabled = true;
    try {
      session = await api(`/sessions/${session.id}/runs`, "POST", {
        topic_id: topic,
        edits,
        request_id: crypto.randomUUID(),
      });
      screen = "result";
      render();
      watch();
    } finally {
      const b = document.querySelector("#generate");
      if (b) b.disabled = false;
    }
  });
}
const modes = {
  live: "この場で生成",
  "exact-cache": "同一条件のキャッシュ",
  sample: "事前生成サンプル",
  generic: "通常画像のみ",
};
function cards(images, personal, run) {
  return `<div class="image-grid">${Array.from({ length: 4 }, (_, i) => {
    const x = images[i];
    if (!x)
      return `<div class="placeholder" aria-label="${i + 1}枚目を待っています">0${i + 1}</div>`;
    return `<button class="candidate ${run.selected_id === x.id ? "chosen" : ""}" data-pick="${esc(x.id)}" ${run.status === "done" ? "" : "disabled"} aria-label="${personal ? "あなた向け" : "通常"}の画像${i + 1}を選ぶ"><img src="${esc(x.url)}" alt="${esc(title(run.topic_id))}・${personal ? "好みを反映" : "通常"}・${i + 1}枚目">${x.id === run.winner_id ? '<span class="recommend">AIのおすすめ</span>' : ""}<span class="caption"><span>0${i + 1}</span><span>seed ${esc(x.seed)}</span></span></button>`;
  }).join("")}</div>`;
}
function result() {
  const run = session.run;
  if (!run) {
    screen = "persona";
    render();
    return;
  }
  const working = run.status !== "done";
  app.innerHTML = `${steps(3)}<div class="topline"><div><div class="eyebrow">YOUR TASTE, SIDE BY SIDE</div><h2>${esc(title(run.topic_id))}。どの一枚が好きですか？</h2><p>上下の同じ位置は、同じseed。描き方の違いを見比べてください。</p></div><span class="badge ${run.mode === "sample" ? "sample" : ""}">${esc(modes[run.mode] || run.mode)}</span></div>${chips(run.context)}${run.mode === "sample" ? `<p class="error-message">事前生成サンプルです。あなたの回答を反映した結果ではありません。</p><div class="sample-choices">${run.sample_choices.map((c) => image(c.chosen_id)).join("")}</div>` : ""}<div class="statusbox" role="status">${working ? '<span class="spinner"></span>' : "<span>✓</span>"}<p>${esc(run.message)}${run.elapsed_seconds ? ` · ${run.elapsed_seconds}秒` : ""}</p></div><div class="comparison-label"><h3>好みを使わずに描く</h3><span>通常の描き方 · 事前生成した4枚</span></div>${cards(run.generic, false, run)}${run.mode !== "generic" ? `<div class="comparison-label"><h3>${run.mode === "sample" ? "代表的な好みで描く" : "あなたの好みで描く"}</h3><span>${esc(modes[run.mode])} · ${run.personalized.length} / 4枚</span></div>${cards(run.personalized, true, run)}` : ""}${run.winner_id ? `<p class="note">モデルが最終比較で評価した軸: ${esc(run.judgments.at(-1)?.dimensions?.join(" / "))}</p><details><summary>推薦理由の原文を見る</summary><p>${esc(run.reason)}</p></details>` : ""}<details class="prompt-details"><summary>描き方と実際のプロンプトを見る</summary><p><b>通常：</b>${esc(run.generic_prompt || "準備中")}</p><p><b>好みを反映：</b>${esc(run.personalized_prompt || (run.mode === "generic" ? "実行しません" : "準備中"))}</p><p>両方とも元のお題の被写体・場所・時間を固定し、表現を補足します。</p><p>共通context: ${esc(run.context.hash)}</p></details>${run.selected_id ? `<p class="selection-message">${run.selected_id === "none" ? "「しっくりくる画像はなかった」を記録しました。" : "あなたの一枚を選びました。"} 感じ方は、いつでも変わっていい。</p>` : ""}<div class="action-row"><button id="edit-again" class="secondary">${working ? "中止して好みを直す" : "好みを直してもう一度"}</button>${!working ? '<button id="none" class="quiet">しっくりくる画像はなかった</button>' : ""}<button id="finish" class="primary">体験を終了 <span>↗</span></button></div>${!working ? sampleControl() : ""}<p class="note">終了すると、この体験の選択・好み・生成画像は削除されます。${working ? "処理中は無操作リセットを停止しています。" : "90秒の無操作でも終了します。"}</p>`;
  app.querySelectorAll("[data-pick]").forEach((b) =>
    b.addEventListener("click", () =>
      act(async () => {
        session = await api(`/sessions/${session.id}/selection`, "POST", {
          image_id: b.dataset.pick,
        });
        render();
      }),
    ),
  );
  bind("none", async () => {
    session = await api(`/sessions/${session.id}/selection`, "POST", {
      image_id: "none",
    });
    render();
  });
  bind("finish", finish);
  bind("edit-again", async () => {
    if (
      run.mode === "sample" &&
      session.choices.filter((c) => c.chosen_id).length < 3
    ) {
      await finish();
      await start(true);
      return;
    }
    poller.stop();
    session = await api(`/sessions/${session.id}/cancel`, "POST");
    screen = "persona";
    render();
  });
  bindSample();
}
async function finish() {
  poller.stop();
  if (session) {
    try {
      await api(`/sessions/${session.id}`, "DELETE");
    } catch (e) {
      if (e.status !== 404) throw e;
    }
  }
  session = null;
  sessionStorage.removeItem("taste-session");
  sessionStorage.removeItem("taste-edits");
  edits = {};
  screen = "welcome";
  render();
  window.scrollTo(0, 0);
}
const poller = createPoller({
  identity: () =>
    session?.run
      ? {
          sid: session.id,
          job: session.run.id,
          context: session.run.context.hash,
        }
      : null,
  fetchSnapshot: (sid) => api(`/sessions/${sid}`),
  onSnapshot: (current) => {
    session = current;
    const signature = JSON.stringify(session.run);
    if (signature !== renderedRun && screen === "result") {
      renderedRun = signature;
      render();
    }
  },
  onExpired: () => {
    session = null;
    sessionStorage.removeItem("taste-session");
    sessionStorage.removeItem("taste-edits");
    screen = "welcome";
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
  if (
    session &&
    !pending &&
    (!session.run || session.run.status === "done") &&
    Date.now() - lastActive > cfg.idle_seconds * 1000
  )
    act(finish);
  if (session?.run && session.run.status !== "done") lastActive = Date.now();
}, 1000);
(async () => {
  try {
    cfg = await api("/config");
    const saved = sessionStorage.getItem("taste-session");
    if (saved) {
      try {
        session = await api(`/sessions/${saved}`);
        screen = initialScreen(session);
        try {
          const draft = JSON.parse(
            sessionStorage.getItem("taste-edits") || "null",
          );
          if (draft?.sid === session.id) edits = draft.edits;
        } catch {}
        if (session.run && session.run.status !== "done") watch();
      } catch {
        sessionStorage.removeItem("taste-session");
      }
    }
    render();
  } catch (e) {
    app.innerHTML =
      "<h2>接続できませんでした。</h2><p>ローカルサーバーを確認し、ページを再読み込みしてください。</p>";
    error(e.message);
  }
})();
