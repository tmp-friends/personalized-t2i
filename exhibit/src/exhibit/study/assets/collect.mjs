// The offline collection page: one participant builds one PreferenceSnapshot per
// condition and exports it as JSON. No network, no server, no stored identity
// beyond the anonymous participant id the operator dictates.

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

const state = {
  typed: "",
  participant: null,
  order: null,
  step: 0,
  condition: null,
  selection: [],
  shown: [],
  rounds: [],
  startedAt: 0,
  exported: [],
};

const conditionOf = (name) => STUDY.conditions.find((item) => item.condition === name);
const limitsOf = (condition) => condition.selection;
const orderedConditions = () =>
  state.order === "legacy_first"
    ? ["legacy_pool", "new_pool"]
    : state.order === "new_first"
      ? ["new_pool", "legacy_pool"]
      : [STUDY.conditions[0].condition];

const picked = (cardId) => state.selection.find((item) => item.card_id === cardId);

function blocker() {
  const limits = limitsOf(state.condition);
  if (state.selection.length < limits.min)
    return `あと${limits.min - state.selection.length}枚選んでください。`;
  if (state.selection.length > limits.max)
    return `選べるのは${limits.max}枚までです。`;
  const unanswered = state.selection.find((item) => !item.aspects.length);
  if (unanswered) return "好きなところを1つ以上えらんでください。";
  return null;
}

/* ------------------------------------------------------------------ rounds */

function visibleCards() {
  const condition = state.condition;
  if (condition.flow === "single_screen") return condition.catalog.cards;
  const round = state.rounds[state.rounds.length - 1];
  const byId = Object.fromEntries(condition.catalog.cards.map((c) => [c.id, c]));
  return (round?.card_ids || []).map((cardId) => byId[cardId]);
}

function requestRound() {
  const condition = state.condition;
  const round = nextRound(
    condition.catalog,
    { selection: state.selection },
    {
      shownIds: state.shown,
      roundIndex: state.rounds.length,
      keyTable: condition.key_table,
      selection: condition.selection,
    },
  );
  state.rounds.push(round);
  state.shown = [...state.shown, ...round.card_ids];
  return round;
}

const canAskRound = () => {
  const condition = state.condition;
  if (condition.flow === "single_screen") return false;
  if (state.rounds.length >= condition.selection.max_rounds) return false;
  return state.shown.length < condition.catalog.cards.length;
};

/* ------------------------------------------------------------------- views */

function startCondition(name) {
  state.condition = conditionOf(name);
  state.selection = [];
  state.shown = [];
  state.rounds = [];
  state.startedAt = Date.now();
  if (state.condition.flow === "single_screen")
    state.shown = state.condition.catalog.cards.map((card) => card.id);
  else requestRound();
  render();
}

function renderCard(card) {
  const chosen = Boolean(picked(card.id));
  const node = el("button", {
    class: `card${chosen ? " selected" : ""}`,
    type: "button",
    "data-card": card.id,
    "aria-pressed": String(chosen),
  });
  node.append(
    el("img", { src: `images/${card.id}.png`, alt: card.label, loading: "lazy" }),
    el("span", { class: "card-label", text: card.label }),
  );
  node.addEventListener("click", () => toggleCard(card));
  return node;
}

function toggleCard(card) {
  const limits = limitsOf(state.condition);
  if (picked(card.id)) {
    state.selection = state.selection.filter((item) => item.card_id !== card.id);
  } else if (state.selection.length >= limits.max) {
    $("#notice").textContent = `選べるのは${limits.max}枚までです。`;
    return;
  } else {
    state.selection = [
      ...state.selection,
      {
        card_id: card.id,
        strength: 1,
        // The v1 flow used every phrase unless the visitor dropped one; the new
        // flow never turns an unanswered card into "every aspect".
        aspects: state.condition.aspects_default_on ? [...STUDY.aspects] : [],
      },
    ];
  }
  $("#notice").textContent = "";
  render();
}

function renderPick(entry, card) {
  const row = el("div", {
    class: `pick${entry.aspects.length ? "" : " needs"}`,
    id: `pick-${card.id}`,
  });
  row.append(
    el("img", { src: `images/${card.id}.png`, alt: card.label }),
    el("div", { class: "pick-label", text: card.label }),
  );
  const chips = el("div", { class: "chips" });
  for (const aspect of STUDY.aspects) {
    const on = entry.aspects.includes(aspect);
    const chip = el("button", {
      class: "chip",
      type: "button",
      "data-aspect": aspect,
      "data-target": card.id,
      "aria-pressed": String(on),
      text: card.aspects_ja[aspect],
    });
    chip.addEventListener("click", () => {
      entry.aspects = on
        ? entry.aspects.filter((item) => item !== aspect)
        : [...new Set([...entry.aspects, aspect])].sort();
      render();
    });
    chips.append(chip);
  }
  const all = el("button", {
    class: "chip all",
    type: "button",
    "data-likeall": card.id,
    text: "全部好き",
  });
  all.addEventListener("click", () => {
    entry.aspects = [...STUDY.aspects].sort();
    render();
  });
  chips.append(all);
  row.append(chips);

  if (state.condition.strength) {
    const steps = el("div", { class: "steps" });
    for (const [value, label] of [
      [1, "好き"],
      [2, "とても好き"],
    ]) {
      const step = el("button", {
        class: "step",
        type: "button",
        "data-strength": value,
        "data-target": card.id,
        "aria-pressed": String(entry.strength === value),
        text: label,
      });
      step.addEventListener("click", () => {
        entry.strength = value;
        render();
      });
      steps.append(step);
    }
    row.append(steps);
  }
  const drop = el("button", { class: "drop", type: "button", text: "選択を外す" });
  drop.addEventListener("click", () => {
    state.selection = state.selection.filter((item) => item.card_id !== card.id);
    render();
  });
  row.append(drop);
  return row;
}

function snapshot() {
  const gains = Object.fromEntries(STUDY.aspects.map((aspect) => [aspect, 1]));
  const value = {
    schema_version: 1,
    study_id: STUDY.study_id,
    study_kind: STUDY.study_kind,
    participant_id: state.participant,
    catalog_id: state.condition.catalog.catalog_id,
    selection: [...state.selection]
      .map((entry) => ({
        card_id: entry.card_id,
        strength: entry.strength,
        aspects: [...entry.aspects].sort(),
      }))
      .sort((a, b) => (a.card_id < b.card_id ? -1 : a.card_id > b.card_id ? 1 : 0)),
    aspect_gains: gains,
    shown_ids: [...state.shown],
    round_count: state.condition.flow === "single_screen" ? 1 : state.rounds.length,
    elapsed_ms: Date.now() - state.startedAt,
  };
  if (STUDY.study_kind === "elicitation") {
    value.collection_condition = state.condition.condition;
    value.presented_index = state.step;
  }
  return value;
}

function download(name, text) {
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = el("a", { href: url, download: name });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function finish() {
  const value = snapshot();
  const text = JSON.stringify(value, null, 2) + "\n";
  const name = `${STUDY.study_id}-${value.participant_id}${
    value.collection_condition ? "-" + value.collection_condition : ""
  }.json`;
  state.exported.push({ name, value });
  state.step += 1;
  render(text, name);
}

/* ------------------------------------------------------------------ render */

function renderIntro() {
  const main = $("#main");
  main.replaceChildren();
  const form = el("div", { class: "panel" });
  form.append(
    el("h2", { text: "参加者ID" }),
    el("p", {
      class: "sub",
      text: `実施者から渡された匿名IDを入力してください（形式: ${STUDY.participant_pattern_label}）。名前やメールアドレスは入力しないでください。`,
    }),
  );
  const input = el("input", {
    id: "participant",
    type: "text",
    autocomplete: "off",
    spellcheck: "false",
    placeholder: STUDY.participant_placeholder,
  });
  // A re-render must not silently drop what the participant already typed.
  input.value = state.typed;
  input.addEventListener("input", () => {
    state.typed = input.value;
  });
  form.append(input);
  if (STUDY.study_kind === "elicitation") {
    form.append(
      el("h2", { text: "実施順" }),
      el("p", { class: "sub", text: "実施者の指示どおりに選んでください。" }),
    );
    const orders = el("div", { class: "chips" });
    for (const [value, label] of [
      ["legacy_first", "旧 → 新"],
      ["new_first", "新 → 旧"],
    ]) {
      const button = el("button", {
        class: "chip",
        type: "button",
        "data-order": value,
        "aria-pressed": String(state.order === value),
        text: label,
      });
      button.addEventListener("click", () => {
        state.order = value;
        render();
      });
      orders.append(button);
    }
    form.append(orders);
  }
  const start = el("button", { class: "primary", type: "button", text: "はじめる" });
  start.addEventListener("click", () => {
    const value = (state.typed = input.value.trim());
    if (!new RegExp(STUDY.participant_pattern).test(value)) {
      $("#notice").textContent = `参加者IDの形式が違います（${STUDY.participant_pattern_label}）。`;
      return;
    }
    if (STUDY.study_kind === "elicitation" && !state.order) {
      $("#notice").textContent = "実施順を選んでください。";
      return;
    }
    state.participant = value;
    $("#notice").textContent = "";
    startCondition(orderedConditions()[0]);
  });
  form.append(start);
  const stored = el("div", { class: "panel stored" });
  stored.append(el("h2", { text: "保存される項目" }));
  const list = el("ul");
  for (const item of STUDY.stored_fields) list.append(el("li", { text: item }));
  stored.append(list, el("p", { class: "sub", text: STUDY.storage_note }));
  main.append(form, stored);
}

function renderCollect() {
  const main = $("#main");
  main.replaceChildren();
  const condition = state.condition;
  const limits = limitsOf(condition);
  const head = el("div", { class: "panel" });
  head.append(
    el("h2", { text: condition.title }),
    el("p", { class: "sub", text: condition.instruction }),
  );
  const round = state.rounds[state.rounds.length - 1];
  if (condition.flow !== "single_screen")
    head.append(
      el("p", {
        class: "round-tag",
        text: `${state.rounds.length} / ${condition.selection.max_rounds} 回目 · ${
          state.selection.length
        } / ${limits.max} 枚選択中`,
      }),
    );
  else
    head.append(
      el("p", {
        class: "round-tag",
        text: `${state.selection.length} / ${limits.max} 枚選択中`,
      }),
    );
  if (round?.shortfall_reason)
    head.append(
      el("p", { class: "sub", text: "お見せできる画像はこれで最後です。" }),
    );
  main.append(head);

  const grid = el("div", { class: "card-grid" });
  for (const card of visibleCards()) grid.append(renderCard(card));
  main.append(grid);

  if (state.selection.length) {
    const tray = el("div", { class: "panel" });
    tray.append(
      el("h2", { text: "選んだ画像の、好きなところ" }),
      el("p", { class: "sub", text: condition.aspect_instruction }),
    );
    const byId = Object.fromEntries(condition.catalog.cards.map((c) => [c.id, c]));
    for (const entry of state.selection)
      tray.append(renderPick(entry, byId[entry.card_id]));
    main.append(tray);
  }

  const actions = el("div", { class: "actions" });
  if (canAskRound()) {
    const more = el("button", {
      id: "more",
      class: "secondary",
      type: "button",
      text: "ほかの候補も見る",
    });
    more.addEventListener("click", () => {
      requestRound();
      render();
    });
    actions.append(more);
  }
  const reason = blocker();
  const commit = el("button", {
    id: "commit",
    class: "primary",
    type: "button",
    text: "この内容で決定",
    disabled: reason ? "disabled" : null,
  });
  commit.addEventListener("click", finish);
  actions.append(commit);
  if (reason) actions.append(el("p", { class: "note", text: reason }));
  main.append(actions);
}

function renderDone(text, name) {
  const main = $("#main");
  main.replaceChildren();
  const panel = el("div", { class: "panel" });
  const remaining = orderedConditions().slice(state.step);
  panel.append(
    el("h2", { text: "この内容を保存してください" }),
    el("p", {
      class: "sub",
      text: `下のボタンで ${name} を保存し、実施者に渡してください。保存できない場合は、枠内の文字をすべてコピーして同じ名前で保存してください。`,
    }),
  );
  const save = el("button", { class: "primary", type: "button", text: "JSONを保存" });
  save.addEventListener("click", () => download(name, text));
  panel.append(save);
  panel.append(el("textarea", { id: "export", readonly: "readonly", rows: "14" }));
  main.append(panel);
  if (remaining.length) {
    const next = el("button", {
      id: "next-condition",
      class: "secondary",
      type: "button",
      text: "次の選び方にすすむ",
    });
    next.addEventListener("click", () => startCondition(remaining[0]));
    main.append(next);
  } else {
    main.append(el("p", { class: "note", text: "これで終わりです。ありがとうございました。" }));
  }
  $("#export").value = text;
}

function render(text, name) {
  if (!state.participant) renderIntro();
  else if (text) renderDone(text, name);
  else renderCollect();
}

render();
