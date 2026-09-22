// Dependency-free stand-in for the FAN exhibition backend, used to drive the real
// frontend (exhibit/src/exhibit/static) in a browser without a GPU or the Python app.
// It implements the API table of docs/superpowers/specs/2026-09-21-fan-personalization-improvement-design.md
// §8.1 and the snapshot shape exhibit/src/exhibit/service.py publishes: revisions,
// multi-round elicitation with a request-id ledger, the 409/422 split, the run
// lifecycle with exact-cache, optional labelled feedback and samples that leave the
// visitor's draft alone. It holds one session in memory, fakes generation progress
// (one image per PAIR_MS) and draws every image as an SVG on the fly.
//
//   node exhibit/tests/mock_api.mjs [--port 8811] [--pair-ms 500] [--idle 90]
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const STATIC = path.join(HERE, "..", "src", "exhibit", "static");
const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const PORT = Number(arg("port", 8811));
const PAIR_MS = Number(arg("pair-ms", 500));
const IDLE_SECONDS = Number(arg("idle", 90));
const SEEDS = [230923, 230924, 230925, 230926];
const ALPHA = 0.5;
const TAIL = "absurdres, highres.";
const SELECTION = { min: 3, max: 10, round_size: 12, max_rounds: 3 };
const ASPECT_KEYS = ["color", "lighting", "texture", "mood"];
const STRENGTHS = [1, 2];
const GAINS = [0.5, 1, 2];
const SUBJECT_CAP = 3;
const RELAXED_CAP = 4;
const EXPLORE_SLOTS = 4;
const DIVERSITY_PENALTY = 0.25;
const POLICY = {
  policy_id: "legacy_exhibit",
  alpha: ALPHA,
  skip: -2,
  skip_pa: [0, 1, 2, 3, 4, 5, 6, 7],
  use_attn_mask: false,
  pooled_mode: "plain",
  profiling: { mode: "all" },
  reference_unit: "aspect_phrase",
};

/* ------------------------------------------------------------- catalog */
const SUBJECTS = [
  { id: "girl", label: "街角の少女", hue: 18 },
  { id: "student", label: "図書館の青年", hue: 210 },
  { id: "traveler", label: "草原の旅人", hue: 120 },
  { id: "barista", label: "カフェの店員", hue: 300 },
];
// Design §6.1: four axes x four levels, and the deterministic 16-profile layout.
const AXES = {
  color: [
    "warm color palette, amber tones",
    "cool color palette, blue and teal tones",
    "muted colors, low saturation",
    "vivid colors, high saturation",
  ],
  lighting: [
    "soft diffused light",
    "hard directional light, strong shadows",
    "backlighting, rim light",
    "even lighting, gentle shadows",
  ],
  texture: [
    "watercolor painting, soft brushwork",
    "cel shading, clean lineart",
    "flat color, minimal shading",
    "painterly texture, detailed shading",
  ],
  mood: [
    "calm atmosphere",
    "cheerful expression, lively atmosphere",
    "serious atmosphere",
    "dreamlike atmosphere",
  ],
};
const AXES_JA = {
  color: ["暖かい色", "涼しい色", "落ち着いた色", "鮮やかな色"],
  lighting: ["やわらかい光", "強い光と影", "逆光の縁取り", "均一な光"],
  texture: ["水彩のような筆づかい", "くっきりした線と塗り", "平たい塗り", "緻密な陰影"],
  mood: ["静かな雰囲気", "にぎやかな雰囲気", "真剣な雰囲気", "夢のような雰囲気"],
};
const MUL2 = [0, 2, 3, 1];
const CARDS = SUBJECTS.flatMap((subject) => {
  const cards = [];
  for (let color = 0; color < 4; color++)
    for (let lighting = 0; lighting < 4; lighting++) {
      const levels = {
        color,
        lighting,
        texture: color ^ lighting,
        mood: color ^ MUL2[lighting],
      };
      const profile_id = `c${color}-l${lighting}-t${levels.texture}-m${levels.mood}`;
      const aspects = Object.fromEntries(
        ASPECT_KEYS.map((key) => [key, AXES[key][levels[key]]]),
      );
      const aspects_ja = Object.fromEntries(
        ASPECT_KEYS.map((key) => [key, AXES_JA[key][levels[key]]]),
      );
      const profile_label = `${aspects_ja.color}・${aspects_ja.texture}`;
      cards.push({
        id: `${subject.id}-${profile_id}`,
        subject_id: subject.id,
        profile_id,
        label: `${subject.label} · ${profile_label}`,
        subject_label: subject.label,
        profile_label,
        axis_levels: levels,
        aspects,
        aspects_ja,
        url: `/assets/cards-v2/${subject.id}-${profile_id}.png`,
      });
    }
  return cards;
});
const TOPICS = [
  { id: "cat", label: "窓辺で猫と過ごす少女", prompt: "1girl with a cat by the window" },
  { id: "tokyo", label: "夜の街を歩く", prompt: "1girl walking through a neon city at night" },
  { id: "rain", label: "雨上がりの帰り道", prompt: "1girl on the way home after the rain" },
  { id: "forest", label: "朝の森を抜ける", prompt: "1girl walking through a morning forest" },
  { id: "cafe", label: "静かなカフェの午後", prompt: "1girl in a quiet cafe in the afternoon" },
  { id: "lighthouse", label: "灯台のある岬", prompt: "1girl at a cape with a lighthouse" },
];
const cardOf = (id) => CARDS.find((c) => c.id === id);
const subjectOf = (id) => SUBJECTS.find((s) => s.id === cardOf(id)?.subject_id);
const topicOf = (id) => TOPICS.find((t) => t.id === id);
const targetPrompt = (topicId) =>
  `masterpiece, best quality, ${topicOf(topicId)?.prompt ?? topicId}, ${TAIL}`;
const hashOf = (value) =>
  crypto.createHash("sha256").update(JSON.stringify(value)).digest("hex");
const CATALOG_HASH = hashOf(CARDS).slice(0, 32);
const CONFIG = {
  schema_version: 2,
  catalog_id: "catalog-v2",
  catalog_hash: CATALOG_HASH,
  cards: CARDS,
  aspects: { color: "色", lighting: "光", texture: "描画", mood: "雰囲気" },
  topics: TOPICS.map((t) => ({
    id: t.id,
    label: t.label,
    preview_url: `/assets/generic/${t.id}-0.png`,
  })),
  alpha: ALPHA,
  policy: {
    policy_id: POLICY.policy_id,
    policy_hash: hashOf(POLICY).slice(0, 32),
    alpha: ALPHA,
    pooled_mode: POLICY.pooled_mode,
    reference_unit: POLICY.reference_unit,
    profiling: POLICY.profiling,
  },
  strengths: STRENGTHS,
  aspect_gains: GAINS,
  selection: SELECTION,
  idle_seconds: IDLE_SECONDS,
  timeout_seconds: 120,
  samples: [
    {
      id: "s1-cat",
      topic_id: "cat",
      label: "S1 · 窓辺で猫と過ごす少女",
      preview_url: "/assets/cards-v2/girl-c0-l0-t0-m0.png",
    },
    { id: "s2-tokyo", topic_id: "tokyo", label: "S2 · 夜の街を歩く", preview_url: null },
  ],
  ready: true,
};
const POLICY_HASH = CONFIG.policy.policy_hash;

/* ------------------------------------------------------------- drawing */
const esc = (s) =>
  String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
function svg({ hue, sat = 62, light = 46, top, bottom, seed = 0 }) {
  const a = `hsl(${hue} ${sat}% ${light}%)`;
  const b = `hsl(${(hue + 40) % 360} ${Math.max(20, sat - 22)}% ${Math.max(12, light - 28)}%)`;
  const r = 120 + ((seed * 37) % 90);
  // 1024x1280 (4:5 portrait), matching the real generation size.
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1280" width="1024" height="1280">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${a}"/><stop offset="1" stop-color="${b}"/></linearGradient></defs>
<rect width="1024" height="1280" fill="url(#g)"/>
<circle cx="${360 + ((seed * 53) % 300)}" cy="${500 + ((seed * 29) % 280)}" r="${r * 2}" fill="#ffffff22"/>
<circle cx="${720 - ((seed * 17) % 240)}" cy="${900 - ((seed * 41) % 250)}" r="${r * 1.2}" fill="#00000033"/>
<rect x="0" y="0" width="1024" height="1280" fill="none" stroke="#ffffff30" stroke-width="4"/>
<text x="48" y="104" font-family="monospace" font-size="56" fill="#fff" opacity="0.92">${esc(top)}</text>
<text x="48" y="1216" font-family="monospace" font-size="40" fill="#fff" opacity="0.75">${esc(bottom)}</text>
<text x="976" y="1216" text-anchor="end" font-family="monospace" font-size="30" fill="#fff" opacity="0.45">1024x1280</text>
</svg>`;
}

/* ------------------------------------------------------- round policy */
const AXIS_PAIRS = [];
for (let i = 0; i < ASPECT_KEYS.length; i++)
  for (let j = i + 1; j < ASPECT_KEYS.length; j++) AXIS_PAIRS.push([i, j]);
const levelsOf = (cardId) => ASPECT_KEYS.map((key) => cardOf(cardId).axis_levels[key]);
function addCounts(counts, levels) {
  levels.forEach((level, axis) => {
    const key = `a${axis}:${level}`;
    counts[key] = (counts[key] || 0) + 1;
  });
  for (const [left, right] of AXIS_PAIRS) {
    const key = `p${left}-${right}:${levels[left]}-${levels[right]}`;
    counts[key] = (counts[key] || 0) + 1;
  }
}
function cost(counts, levels) {
  let total = 0;
  levels.forEach((level, axis) => (total += counts[`a${axis}:${level}`] || 0));
  for (const [left, right] of AXIS_PAIRS)
    total += counts[`p${left}-${right}:${levels[left]}-${levels[right]}`] || 0;
  return total;
}
const match = (a, b) =>
  a.reduce((n, level, axis) => n + (level === b[axis] ? 1 : 0), 0) / a.length;
function similarity(levels, preferences) {
  const total = preferences.reduce((sum, p) => sum + p.strength, 0);
  if (!total) return 0;
  const score = preferences.reduce(
    (sum, p) =>
      sum +
      (p.strength *
        p.axes.reduce((n, axis) => n + (levels[axis] === p.levels[axis] ? 1 : 0), 0)) /
        p.axes.length,
    0,
  );
  return score / total;
}
const tiebreak = (seed, roundIndex, cardId) =>
  crypto.createHash("sha256").update(`${seed}:${roundIndex}:${cardId}`).digest("hex");
function plan(pool, preferences, { counts, roundSize, cap, keys }) {
  const remaining = [...pool];
  const local = { ...counts };
  const picks = [];
  const used = {};
  const explore = preferences.length ? Math.min(EXPLORE_SLOTS, roundSize) : roundSize;
  for (const [slot, target] of [
    ["similar", roundSize - explore],
    ["explore", explore],
  ]) {
    for (let n = 0; n < target; n++) {
      let chosen = null,
        best = null;
      const taken = slot === "similar" ? picks.map((p) => levelsOf(p.card_id)) : [];
      for (const cardId of remaining) {
        const subject = cardOf(cardId).subject_id;
        if ((used[subject] || 0) >= cap) continue;
        const levels = levelsOf(cardId);
        let key;
        if (slot === "similar") {
          const penalty = taken.reduce((max, other) => Math.max(max, match(levels, other)), 0);
          key = [
            -(similarity(levels, preferences) - DIVERSITY_PENALTY * penalty),
            keys[cardId],
          ];
        } else key = [cost(local, levels), keys[cardId]];
        if (best === null || key[0] < best[0] || (key[0] === best[0] && key[1] < best[1])) {
          chosen = cardId;
          best = key;
        }
      }
      if (chosen === null) break;
      picks.push({ card_id: chosen, slot });
      remaining.splice(remaining.indexOf(chosen), 1);
      used[cardOf(chosen).subject_id] = (used[cardOf(chosen).subject_id] || 0) + 1;
      addCounts(local, levelsOf(chosen));
    }
  }
  return picks;
}
function nextRound(state) {
  const roundIndex = state.rounds.length;
  const seen = new Set(state.shown);
  const counts = {};
  for (const cardId of state.shown) addCounts(counts, levelsOf(cardId));
  const pool = CARDS.map((c) => c.id).filter((id) => !seen.has(id));
  const keys = Object.fromEntries(
    pool.map((id) => [id, tiebreak(state.seed, roundIndex, id)]),
  );
  const preferences = state.selection
    .filter((entry) => entry.aspects.length)
    .map((entry) => ({
      levels: levelsOf(entry.card_id),
      axes: entry.aspects.map((aspect) => ASPECT_KEYS.indexOf(aspect)),
      strength: entry.strength,
    }));
  const options = { counts, roundSize: SELECTION.round_size, keys };
  let picks = plan(pool, preferences, { ...options, cap: SUBJECT_CAP });
  if (picks.length < SELECTION.round_size) {
    const widened = plan(pool, preferences, { ...options, cap: RELAXED_CAP });
    if (widened.length > picks.length) picks = widened;
  }
  let shortfall = null;
  if (picks.length < SELECTION.round_size)
    shortfall = !pool.length
      ? "no_unseen_cards"
      : pool.length < SELECTION.round_size
        ? "insufficient_unseen_cards"
        : "subject_cap_limit";
  // Same-subject cards stay adjacent, in the order they were picked.
  const order = [...new Set(picks.map((p) => cardOf(p.card_id).subject_id))];
  const ordered = order.flatMap((subject) =>
    picks.filter((p) => cardOf(p.card_id).subject_id === subject),
  );
  const record = {
    round_index: roundIndex,
    card_ids: ordered.map((p) => p.card_id),
    cards: ordered,
    shortfall_reason: shortfall,
  };
  return { round_id: hashOf([state.seed, record]).slice(0, 16), ...record };
}

/* --------------------------------------------------------------- state */
let session = null;
const token = () => crypto.randomBytes(8).toString("hex");
const isInt = (value) => Number.isInteger(value);

class Refused extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
  }
}
const refuse = (status, detail) => {
  throw new Refused(status, detail);
};

/** Design §5.3: one entry per distinct phrase, weight = strength x aspect gain. */
function mergedRefs(state) {
  const merged = new Map();
  for (const entry of state.selection) {
    const card = cardOf(entry.card_id);
    for (const aspect of ASPECT_KEYS) {
      if (!entry.aspects.includes(aspect)) continue;
      const text = card.aspects[aspect];
      const item = merged.get(text) || { text, weight: 0, card_ids: [], aspects: [] };
      item.weight = Math.round((item.weight + entry.strength * state.aspect_gains[aspect]) * 100) / 100;
      if (!item.card_ids.includes(entry.card_id)) item.card_ids.push(entry.card_id);
      if (!item.aspects.includes(aspect)) item.aspects.push(aspect);
      merged.set(text, item);
    }
  }
  return [...merged.values()]
    .map((item) => ({
      ref_id: hashOf(item.text).slice(0, 16),
      text: item.text,
      weight: item.weight,
      card_ids: [...item.card_ids].sort(),
      aspects: [...item.aspects].sort(),
    }))
    .sort((a, b) => (a.text < b.text ? -1 : a.text > b.text ? 1 : 0));
}
function personalizationOf(state, topicId) {
  const refs = mergedRefs(state);
  const hash = hashOf({
    refs,
    prompt: targetPrompt(topicId),
    policy: POLICY,
    catalog: CATALOG_HASH,
  }).slice(0, 32);
  return {
    refs,
    alpha: ALPHA,
    policy_id: POLICY.policy_id,
    policy_hash: POLICY_HASH,
    personalization_hash: hash,
    hash,
    effective_policy: POLICY,
  };
}
const busy = () => Boolean(session?.run && session.run.status_at() !== "done");
function personalHue(state) {
  if (!state.selection.length) return 210;
  const hues = state.selection.map((entry) => subjectOf(entry.card_id).hue);
  return Math.round(hues.reduce((a, b) => a + b, 0) / hues.length);
}
function progress(run) {
  if (run.mode !== "live") return SEEDS.length;
  return Math.max(0, Math.min(SEEDS.length, Math.floor((Date.now() - run.startedAt) / PAIR_MS)));
}
function imagesOf(run, kind) {
  const count = kind === "plain" ? SEEDS.length : progress(run);
  return SEEDS.slice(0, count).map((seed, i) => {
    const relative =
      kind === "plain" ? `${run.id}/plain-${i}.png` : `${run.id}/personal/personal-${i}.png`;
    return {
      id: `${kind}-${i}`,
      seed,
      sha256: hashOf([run.id, kind, i]),
      relative_path: relative,
      url: `/api/sessions/${session.id}/images/${relative}`,
      ...(kind === "plain"
        ? { prompt: targetPrompt(run.topic_id) }
        : {
            policy_id: run.mode === "sample" ? null : POLICY.policy_id,
            policy_hash: run.mode === "sample" ? null : POLICY_HASH,
            personalization_hash: run.personalization.hash,
          }),
    };
  });
}
function makeRun(topicId, requestId, state) {
  const personalization = personalizationOf(state, topicId);
  const cacheKey = `${topicId}:${personalization.hash}`;
  const cached = state.cache[cacheKey];
  const run = {
    id: `run-${token().slice(0, 6)}`,
    request_id: requestId,
    topic_id: topicId,
    mode: cached ? "exact-cache" : "live",
    startedAt: Date.now(),
    cacheKey,
    preference_revision: state.revision,
    preference: {
      revision: state.revision,
      catalog_id: CONFIG.catalog_id,
      catalog_hash: CATALOG_HASH,
      selection: JSON.parse(JSON.stringify(state.selection)),
      aspect_gains: { ...state.aspect_gains },
    },
    personalization,
    feedback: null,
    status_at() {
      return this.mode === "live" && progress(this) < SEEDS.length ? "generating" : "done";
    },
  };
  return run;
}
function publicRun(state) {
  const run = state.run;
  if (!run) return null;
  const status = run.status_at();
  const done = status === "done";
  if (done && run.mode === "live") state.cache[run.cacheKey] = true;
  const message =
    run.mode === "sample"
      ? "代表的な選択から事前に生成したサンプルです。あなたの選択を反映した結果ではありません。"
      : run.mode === "exact-cache"
        ? "同じ好み・同じお題で、この体験中に生成した結果です。"
        : done
          ? "4枚ができました。"
          : `${progress(run)} / ${SEEDS.length}枚ができました。`;
  return {
    id: run.id,
    topic_id: run.topic_id,
    status,
    mode: run.mode,
    message,
    elapsed_seconds: done ? Math.max(1, Math.round((Date.now() - run.startedAt) / 1000)) : 0,
    error: null,
    prompt: targetPrompt(run.topic_id),
    plain: imagesOf(run, "plain"),
    personal: imagesOf(run, "personal"),
    preference_revision: run.preference_revision,
    preference: run.preference,
    policy_id: run.mode === "sample" ? null : POLICY.policy_id,
    policy_hash: run.mode === "sample" ? null : POLICY_HASH,
    personalization_hash: run.personalization.hash,
    personalization: run.personalization,
    feedback: run.feedback,
    timings: done ? { generation: { wall_seconds: 25.1 } } : {},
  };
}
function roundAvailable(state) {
  if (state.rounds.length >= SELECTION.max_rounds) return false;
  const seen = new Set(state.shown);
  return CARDS.some((card) => !seen.has(card.id));
}
function snapshot() {
  const state = session;
  return {
    id: state.id,
    revision: state.revision,
    catalog_id: CONFIG.catalog_id,
    catalog_hash: CATALOG_HASH,
    committed: state.committed,
    selection: JSON.parse(JSON.stringify(state.selection)),
    aspect_gains: { ...state.aspect_gains },
    rounds: state.rounds.map((round) => ({
      round_id: round.round_id,
      round_index: round.round_index,
      card_ids: [...round.card_ids],
      cards: round.cards.map((card) => ({ ...card })),
      shortfall_reason: round.shortfall_reason,
    })),
    shown_ids: [...state.shown],
    next_round_available: roundAvailable(state),
    run: publicRun(state),
  };
}
function serveRound(state) {
  const record = nextRound(state);
  state.rounds.push(record);
  state.shown.push(...record.card_ids);
  return record;
}

/* ---------------------------------------------------------- validation */
function ledger(state, requestId, kind, payload) {
  if (typeof requestId !== "string" || !requestId || requestId.length > 100)
    refuse(422, "Invalid request ID");
  const payloadHash = hashOf({ kind, payload });
  const entry = state.ledger[requestId];
  if (entry && entry.payload_hash !== payloadHash)
    refuse(409, "Request ID was reused with different input");
  return { payloadHash, entry };
}
function guardBusy(state) {
  if (state.run && state.run.status_at() !== "done")
    refuse(409, "生成中です。終わるまで選択は変えられません。");
}
function guardRevision(state, expected) {
  if (!isInt(expected)) refuse(422, "expected_revision must be an integer");
  if (expected !== state.revision)
    refuse(409, "画面が古くなっています。最新の状態を読み込んでください。");
}
const sameKeys = (value, keys) =>
  value &&
  typeof value === "object" &&
  !Array.isArray(value) &&
  Object.keys(value).length === keys.length &&
  keys.every((key) => key in value);
function normalizeSelection(body, state) {
  if (!sameKeys(body, ["expected_revision", "cards", "aspect_gains", "commit"]))
    refuse(422, "Preference payload has unknown or missing fields");
  if (typeof body.commit !== "boolean") refuse(422, "commit must be a boolean");
  const commit = body.commit;
  if (!Array.isArray(body.cards)) refuse(422, "cards must be a list");
  const lowest = commit ? SELECTION.min : 0;
  if (body.cards.length < lowest || body.cards.length > SELECTION.max)
    refuse(422, `${lowest}〜${SELECTION.max}枚を選んでください。`);
  const gains = body.aspect_gains;
  if (!sameKeys(gains, ASPECT_KEYS)) refuse(422, "aspect_gains must name every aspect");
  for (const key of ASPECT_KEYS)
    if (typeof gains[key] !== "number" || !GAINS.includes(gains[key]))
      refuse(422, "Invalid aspect_gains");
  const shown = new Set(state.shown);
  const seen = new Set();
  const selection = body.cards.map((entry) => {
    if (!sameKeys(entry, ["card_id", "strength", "aspects"]))
      refuse(422, "Invalid selection entry");
    if (typeof entry.card_id !== "string" || !cardOf(entry.card_id) || seen.has(entry.card_id))
      refuse(422, "Unknown or duplicate card");
    if (!isInt(entry.strength) || !STRENGTHS.includes(entry.strength))
      refuse(422, "strength must be 1 or 2");
    if (
      !Array.isArray(entry.aspects) ||
      entry.aspects.some((a) => !ASPECT_KEYS.includes(a)) ||
      new Set(entry.aspects).size !== entry.aspects.length ||
      (!entry.aspects.length && commit)
    )
      refuse(422, "Invalid aspects");
    if (!shown.has(entry.card_id)) refuse(422, "まだ表示していない画像は選べません。");
    seen.add(entry.card_id);
    return {
      card_id: entry.card_id,
      strength: entry.strength,
      aspects: [...entry.aspects].sort(),
    };
  });
  selection.sort((a, b) => (a.card_id < b.card_id ? -1 : a.card_id > b.card_id ? 1 : 0));
  return {
    selection,
    aspect_gains: Object.fromEntries(ASPECT_KEYS.map((key) => [key, gains[key]])),
    committed: commit,
  };
}

/* -------------------------------------------------------------- server */
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".woff2": "font/woff2",
  ".svg": "image/svg+xml; charset=utf-8",
};
function send(res, status, body, type = "application/json", cache = "no-store") {
  const payload = type.startsWith("application/json") ? JSON.stringify(body) : body;
  res.writeHead(status, {
    "content-type": type,
    "cache-control": cache,
    "x-content-type-options": "nosniff",
  });
  res.end(payload);
}
const fail = (res, status, detail) => send(res, status, { detail });
const ok = (res) => send(res, 200, snapshot());
function image(res, opts) {
  // Cards and finished frames are immutable here; let the browser keep them so a
  // re-render does not refetch every picture.
  send(res, 200, svg(opts), "image/svg+xml; charset=utf-8", "max-age=120");
}
function readBody(req) {
  return new Promise((resolve) => {
    let raw = "";
    req.on("data", (c) => (raw += c));
    req.on("end", () => {
      try {
        resolve(raw ? JSON.parse(raw) : {});
      } catch {
        resolve({});
      }
    });
  });
}
function serveStatic(res, name) {
  const file = path.join(STATIC, path.basename(name));
  if (!fs.existsSync(file)) return fail(res, 404, "not found");
  const binary = path.extname(file) === ".woff2";
  send(
    res,
    200,
    fs.readFileSync(file, binary ? null : "utf8"),
    TYPES[path.extname(file)] || "text/plain",
  );
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  const p = url.pathname;
  const m = req.method;
  try {
    /* ---- static + assets ---- */
    if (p === "/" && m === "GET") return serveStatic(res, "index.html");
    if (p.startsWith("/static/") && m === "GET") return serveStatic(res, p.slice(8));
    if (p === "/favicon.ico") {
      res.writeHead(204, { "cache-control": "no-store" });
      return res.end();
    }
    if (p === "/fallback")
      return send(res, 200, "<h1>fallback</h1>", "text/html; charset=utf-8");
    if (p === "/tech")
      return send(res, 200, "<h1>tech</h1>", "text/html; charset=utf-8");
    if (p.startsWith("/assets/cards-v2/")) {
      const card = cardOf(path.basename(p, ".png"));
      if (!card) return fail(res, 404, "no card");
      return image(res, {
        hue: (subjectOf(card.id).hue + card.axis_levels.color * 24) % 360,
        sat: [55, 55, 24, 82][card.axis_levels.color],
        light: [50, 44, 40, 52][card.axis_levels.lighting],
        top: card.subject_id.toUpperCase(),
        bottom: card.profile_id,
        seed: CARDS.indexOf(card) + 1,
      });
    }
    if (p.startsWith("/assets/generic/")) {
      const [topicId, n] = path.basename(p, ".png").split("-");
      return image(res, {
        hue: 205,
        sat: 14,
        light: 40,
        top: `PLAIN · ${topicId}`,
        bottom: `seed ${SEEDS[Number(n) || 0]}`,
        seed: Number(n) + 3,
      });
    }

    /* ---- api ---- */
    if (p === "/api/config" && m === "GET") return send(res, 200, CONFIG);
    if (p === "/api/health" && m === "GET")
      return send(res, 200, {
        status: "ok",
        gpu_busy: busy(),
        mode: "fan-live",
        offline: true,
      });
    if (p === "/api/sessions" && m === "POST") {
      if (session) return fail(res, 409, "ほかの体験が進行中です。");
      session = {
        id: `sess-${token().slice(0, 8)}`,
        seed: token(),
        revision: 0,
        selection: [],
        aspect_gains: Object.fromEntries(ASPECT_KEYS.map((key) => [key, 1])),
        committed: false,
        rounds: [],
        shown: [],
        ledger: {},
        cache: {},
        run: null,
      };
      serveRound(session);
      return ok(res);
    }
    const parts = p.split("/").filter(Boolean); // api sessions <sid> ...
    if (parts[0] !== "api" || parts[1] !== "sessions") return fail(res, 404, "not found");
    const sid = parts[2];
    if (!session || session.id !== sid)
      return fail(res, 404, "体験が終了しました。最初から始めてください。");
    const state = session;
    const tail = parts.slice(3);
    const body = m === "GET" || m === "DELETE" ? {} : await readBody(req);

    if (!tail.length && m === "GET") return ok(res);
    if (!tail.length && m === "DELETE") {
      session = null;
      return send(res, 200, { ok: true, releasing_gpu: false });
    }
    if (tail[0] === "images") {
      const run = state.run;
      if (!run) return fail(res, 404, "no run");
      const name = path.basename(tail[tail.length - 1], ".png");
      const index = Number(name.split("-").pop());
      if (!Number.isInteger(index) || index < 0 || index >= SEEDS.length)
        return fail(res, 404, "no image");
      if (name.startsWith("personal"))
        return image(res, {
          hue: personalHue(state),
          top: `FAN · alpha ${ALPHA}`,
          bottom: `seed ${SEEDS[index]}`,
          seed: index + 9,
        });
      return image(res, {
        hue: 205,
        sat: 14,
        light: 40,
        top: "PLAIN",
        bottom: `seed ${SEEDS[index]}`,
        seed: index + 3,
      });
    }
    if (tail[0] === "touch" && m === "POST") return send(res, 200, { ok: true });

    if (tail[0] === "selection" && m === "PUT") {
      guardBusy(state);
      guardRevision(state, body.expected_revision);
      const content = normalizeSelection(body, state);
      const current = {
        selection: state.selection,
        aspect_gains: state.aspect_gains,
        committed: state.committed,
      };
      if (hashOf(content) !== hashOf(current)) {
        Object.assign(state, JSON.parse(JSON.stringify(content)));
        state.revision += 1;
        // The finished comparison belongs to the previous preference.
        state.run = null;
      }
      return ok(res);
    }
    if (tail[0] === "rounds" && m === "POST") {
      if (!sameKeys(body, ["request_id", "expected_revision"]))
        refuse(422, "Round payload has unknown or missing fields");
      const { entry, payloadHash } = ledger(state, body.request_id, "round", {
        expected_revision: body.expected_revision,
      });
      if (entry) return ok(res); // the same request never shows a card twice
      guardBusy(state);
      guardRevision(state, body.expected_revision);
      if (state.rounds.length >= SELECTION.max_rounds)
        refuse(409, `選択は最大${SELECTION.max_rounds}回までです。`);
      const seen = new Set(state.shown);
      if (!CARDS.some((card) => !seen.has(card.id)))
        refuse(409, "お見せできる画像がもうありません。");
      const record = serveRound(state);
      state.ledger[body.request_id] = {
        kind: "round",
        payload_hash: payloadHash,
        round_id: record.round_id,
      };
      return ok(res);
    }
    if (tail[0] === "runs" && tail.length === 1 && m === "POST") {
      if (!sameKeys(body, ["topic_id", "request_id", "expected_revision"]))
        refuse(422, "Run payload has unknown or missing fields");
      const { entry, payloadHash } = ledger(state, body.request_id, "run", {
        topic_id: body.topic_id,
        expected_revision: body.expected_revision,
      });
      if (entry) {
        if (state.run && state.run.id === entry.run_id) return ok(res);
        refuse(409, "この生成はすでに終わっています。");
      }
      if (!topicOf(body.topic_id)) refuse(422, "Unknown topic");
      guardRevision(state, body.expected_revision);
      if (!state.committed) refuse(422, `${SELECTION.min}枚以上を選んで決定してください。`);
      if (state.run && state.run.status_at() !== "done") refuse(409, "処理中です。");
      state.run = makeRun(body.topic_id, body.request_id, state);
      state.ledger[body.request_id] = {
        kind: "run",
        payload_hash: payloadHash,
        run_id: state.run.id,
      };
      return ok(res);
    }
    if (tail[0] === "runs" && tail.length === 3 && tail[2] === "feedback" && m === "PUT") {
      if (!sameKeys(body, ["expected_revision", "preference"]))
        refuse(422, "Feedback payload has unknown or missing fields");
      if (!["plain", "personal", "tie"].includes(body.preference))
        refuse(422, "Unknown preference");
      if (!isInt(body.expected_revision))
        refuse(422, "expected_revision must be an integer");
      const run = state.run;
      if (
        !run ||
        run.id !== tail[1] ||
        run.mode === "sample" ||
        run.status_at() !== "done" ||
        run.preference_revision !== body.expected_revision ||
        body.expected_revision !== state.revision
      )
        refuse(409, "この結果には回答できません。");
      run.feedback = {
        preference: body.preference,
        preference_revision: run.preference_revision,
        personalization_hash: run.personalization.hash,
      };
      return ok(res);
    }
    if (tail[0] === "cancel" && m === "POST") {
      // Cancelling an unfinished comparison drops the whole run.
      if (state.run && state.run.status_at() !== "done") state.run = null;
      return ok(res);
    }
    if (tail[0] === "sample" && m === "POST") {
      if (busy()) refuse(409, "処理中です。");
      const sample = CONFIG.samples.find((s) => s.id === body.sample_id);
      if (!sample) refuse(422, "Sample unavailable or inconsistent");
      // Somebody else's preference: draft, revision and rounds stay untouched.
      const shown = ["girl-c0-l0-t0-m0", "student-c1-l1-t0-m3", "traveler-c2-l0-t2-m2"];
      const borrowed = {
        selection: shown.map((card_id) => ({
          card_id,
          strength: 1,
          aspects: [...ASPECT_KEYS].sort(),
        })),
        aspect_gains: Object.fromEntries(ASPECT_KEYS.map((key) => [key, 1])),
        revision: 0,
        cache: {}, // a sample never touches this session's cache
      };
      const run = makeRun(sample.topic_id, `sample-${sample.id}`, borrowed);
      run.mode = "sample";
      run.preference_revision = null;
      run.preference = { ...run.preference, source: "sample", sample_id: sample.id };
      state.run = run;
      return ok(res);
    }
    return fail(res, 404, "not found");
  } catch (e) {
    if (e instanceof Refused) return fail(res, e.status, e.message);
    return fail(res, 500, String(e?.stack || e));
  }
});
server.listen(PORT, "127.0.0.1", () =>
  console.log(`mock api on http://127.0.0.1:${PORT} (pair ${PAIR_MS}ms)`),
);
