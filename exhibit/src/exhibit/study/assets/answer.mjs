// The offline answer page. It knows two images per question and nothing else:
// no method names, no reference texts, no prompts, no readable file names.

const $ = (selector) => document.querySelector(selector);
const el = (tag, attrs = {}, children = []) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "text") node.textContent = value;
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
};

const CHOICES = [
  ["A", "左（A）"],
  ["B", "右（B）"],
  ["tie", "同じくらい"],
];
const KEPT = [
  ["both", "どちらも保たれている"],
  ["A", "左（A）だけ保たれている"],
  ["B", "右（B）だけ保たれている"],
  ["neither", "どちらも保たれていない"],
];
const BREAKAGE = [
  ["none", "どちらにもない"],
  ["A", "左（A）にある"],
  ["B", "右（B）にある"],
  ["both", "どちらにもある"],
];

const state = { phase: "intro", index: 0, answers: {}, fidelity: {} };

const answered = () => PAGE.pairs.filter((pair) => state.answers[pair.pair_id]).length;
const checked = () =>
  PAGE.pairs.filter((pair) => {
    const value = state.fidelity[pair.pair_id];
    return value && value.subject_kept && value.breakage;
  }).length;

function shots(pair) {
  const grid = el("div", { class: "shots" });
  for (const [side, image] of [
    ["A", pair.a],
    ["B", pair.b],
  ]) {
    const figure = el("figure", { class: "shot" });
    figure.append(
      el("img", { src: image, alt: `${side}` }),
      el("figcaption", { text: side }),
    );
    grid.append(figure);
  }
  return grid;
}

function buttons(options, current, onPick, attribute) {
  const row = el("div", { class: "chips" });
  for (const [value, label] of options) {
    const button = el("button", {
      class: "chip",
      type: "button",
      [attribute]: value,
      "aria-pressed": String(current === value),
      text: label,
    });
    button.addEventListener("click", () => onPick(value));
    row.append(button);
  }
  return row;
}

function renderIntro() {
  const main = $("#main");
  main.replaceChildren();
  const panel = el("div", { class: "panel" });
  panel.append(
    el("h2", { text: "はじめに" }),
    el("p", {
      class: "sub",
      text: `${PAGE.pairs.length}組の画像を1組ずつ見て、自分の好みに近い方を選んでください。どちらとも言えないときは「同じくらい」で構いません。`,
    }),
    el("p", {
      class: "sub",
      text: "画像がどの方式で作られたかは表示されません。作り方を推測する必要はありません。",
    }),
    el("p", { class: "sub", text: PAGE.storage_note }),
  );
  const start = el("button", {
    id: "start",
    class: "primary",
    type: "button",
    text: "はじめる",
  });
  start.addEventListener("click", () => {
    state.phase = "preference";
    render();
  });
  panel.append(start);
  main.append(panel);
}

function renderPreference() {
  const main = $("#main");
  main.replaceChildren();
  const pair = PAGE.pairs[state.index];
  const panel = el("div", { class: "panel" });
  panel.append(
    el("p", {
      class: "round-tag",
      text: `${state.index + 1} / ${PAGE.pairs.length} 組`,
    }),
    el("h2", { text: "どちらが自分の好みに近いですか" }),
    shots(pair),
    buttons(
      CHOICES,
      state.answers[pair.pair_id],
      (value) => {
        state.answers[pair.pair_id] = value;
        if (state.index < PAGE.pairs.length - 1) state.index += 1;
        render();
      },
      "data-choice",
    ),
  );
  const nav = el("div", { class: "actions" });
  const back = el("button", {
    id: "back",
    class: "secondary",
    type: "button",
    text: "前へ",
    disabled: state.index ? null : "disabled",
  });
  back.addEventListener("click", () => {
    state.index = Math.max(0, state.index - 1);
    render();
  });
  const forward = el("button", {
    id: "forward",
    class: "secondary",
    type: "button",
    text: "次へ",
    disabled: state.index < PAGE.pairs.length - 1 ? null : "disabled",
  });
  forward.addEventListener("click", () => {
    state.index = Math.min(PAGE.pairs.length - 1, state.index + 1);
    render();
  });
  nav.append(back, forward);
  const next = el("button", {
    id: "to-fidelity",
    class: "primary",
    type: "button",
    text: "次の確認にすすむ",
    disabled: answered() === PAGE.pairs.length ? null : "disabled",
  });
  next.addEventListener("click", () => {
    state.phase = "fidelity";
    state.index = 0;
    render();
  });
  nav.append(next);
  if (answered() !== PAGE.pairs.length)
    nav.append(
      el("p", {
        class: "note",
        text: `未回答が${PAGE.pairs.length - answered()}組あります。`,
      }),
    );
  main.append(panel, nav);
}

function renderFidelity() {
  const main = $("#main");
  main.replaceChildren();
  const pair = PAGE.pairs[state.index];
  const value = state.fidelity[pair.pair_id] || {};
  const panel = el("div", { class: "panel" });
  panel.append(
    el("p", {
      class: "round-tag",
      text: `確認 ${state.index + 1} / ${PAGE.pairs.length} 組`,
    }),
    el("h2", { text: "描かれている人物と場面は保たれていますか" }),
    el("p", {
      class: "sub",
      text: "これは好みの質問とは別の確認です。好みの回答には影響しません。",
    }),
    shots(pair),
    buttons(
      KEPT,
      value.subject_kept,
      (choice) => {
        state.fidelity[pair.pair_id] = { ...value, subject_kept: choice };
        render();
      },
      "data-kept",
    ),
    el("h2", { text: "目立つ崩れはありますか" }),
    buttons(
      BREAKAGE,
      value.breakage,
      (choice) => {
        state.fidelity[pair.pair_id] = { ...value, breakage: choice };
        if (state.index < PAGE.pairs.length - 1) state.index += 1;
        render();
      },
      "data-breakage",
    ),
  );
  const nav = el("div", { class: "actions" });
  const back = el("button", {
    id: "back",
    class: "secondary",
    type: "button",
    text: "前へ",
    disabled: state.index ? null : "disabled",
  });
  back.addEventListener("click", () => {
    state.index = Math.max(0, state.index - 1);
    render();
  });
  const forward = el("button", {
    id: "forward",
    class: "secondary",
    type: "button",
    text: "次へ",
    disabled: state.index < PAGE.pairs.length - 1 ? null : "disabled",
  });
  forward.addEventListener("click", () => {
    state.index = Math.min(PAGE.pairs.length - 1, state.index + 1);
    render();
  });
  const done = el("button", {
    id: "finish",
    class: "primary",
    type: "button",
    text: "回答を保存する",
    disabled: checked() === PAGE.pairs.length ? null : "disabled",
  });
  done.addEventListener("click", () => {
    state.phase = "done";
    render();
  });
  nav.append(back, forward, done);
  main.append(panel, nav);
}

function exportValue() {
  return {
    schema_version: 1,
    study_id: PAGE.study_id,
    participant_id: PAGE.participant_id,
    answers: PAGE.pairs.map((pair) => ({
      pair_id: pair.pair_id,
      choice: state.answers[pair.pair_id] || null,
    })),
    fidelity: PAGE.pairs.map((pair) => ({
      pair_id: pair.pair_id,
      subject_kept: (state.fidelity[pair.pair_id] || {}).subject_kept || null,
      breakage: (state.fidelity[pair.pair_id] || {}).breakage || null,
    })),
  };
}

function renderDone() {
  const main = $("#main");
  main.replaceChildren();
  const text = JSON.stringify(exportValue(), null, 2) + "\n";
  const name = `${PAGE.study_id}-${PAGE.participant_id}-answers.json`;
  const panel = el("div", { class: "panel" });
  panel.append(
    el("h2", { text: "回答を保存してください" }),
    el("p", {
      class: "sub",
      text: `下のボタンで ${name} を保存し、実施者に渡してください。保存できない場合は、枠内の文字をすべてコピーして同じ名前で保存してください。`,
    }),
  );
  const save = el("button", {
    id: "save",
    class: "primary",
    type: "button",
    text: "JSONを保存",
  });
  save.addEventListener("click", () => {
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = el("a", { href: url, download: name });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  panel.append(save, el("textarea", { id: "export", readonly: "readonly", rows: "14" }));
  main.append(panel);
  $("#export").value = text;
}

function render() {
  if (state.phase === "intro") renderIntro();
  else if (state.phase === "preference") renderPreference();
  else if (state.phase === "fidelity") renderFidelity();
  else renderDone();
}

render();
