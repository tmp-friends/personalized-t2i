// Dependency-free stand-in for the FAN exhibition backend, used to drive the real
// frontend (exhibit/src/exhibit/static) in a browser without a GPU or the Python app.
// It implements the API table of docs/superpowers/specs/2026-09-21-fan-exhibition-demo-design.md
// §5 and the snapshot shapes of §4.1, holds one session in memory, fakes generation
// progress (one image per PAIR_MS) and draws every image as an SVG on the fly.
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

const SUBJECTS = [
  { id: "girl", label: "街角の少女", prompt: "1girl, standing on a street corner" },
  { id: "student", label: "図書館の青年", prompt: "1boy, reading in a library" },
  { id: "traveler", label: "草原の旅人", prompt: "1girl, traveler on a grassland" },
  { id: "barista", label: "カフェの店員", prompt: "1boy, barista behind a counter" },
];
const PROFILES = [
  {
    id: "warm_soft",
    label: "あたたかく、やわらかく",
    hue: 28,
    aspects: {
      color: "warm color palette, amber and orange tones",
      lighting: "soft lighting, gentle shadows",
      texture: "watercolor painting, soft brushwork, painterly texture",
      mood: "calm atmosphere",
    },
    aspects_ja: {
      color: "暖かなオレンジ系の色",
      lighting: "やわらかい光",
      texture: "水彩のような筆づかい",
      mood: "静かな雰囲気",
    },
  },
  {
    id: "cool_clean",
    label: "涼しく、くっきり",
    hue: 196,
    aspects: {
      color: "cool color palette, blue and teal tones",
      lighting: "soft diffused light",
      texture: "cel shading, clean lineart, anime coloring",
      mood: "calm atmosphere",
    },
    aspects_ja: {
      color: "青緑系の涼しい色",
      lighting: "拡散したやわらかい光",
      texture: "くっきりした線と塗り",
      mood: "静かな雰囲気",
    },
  },
  {
    id: "dramatic",
    label: "劇的な光と影",
    hue: 278,
    aspects: {
      color: "muted colors, dark background",
      lighting: "dramatic lighting, strong shadows, rim light",
      texture: "detailed shading, painterly texture",
      mood: "serious atmosphere",
    },
    aspects_ja: {
      color: "落ち着いた色と暗い背景",
      lighting: "強い光と影",
      texture: "細かな陰影",
      mood: "真剣な雰囲気",
    },
  },
  {
    id: "vivid_lively",
    label: "鮮やかで、にぎやか",
    hue: 338,
    aspects: {
      color: "vivid colors, saturated colors",
      lighting: "bright lighting, sparkle",
      texture: "cel shading, flat color",
      mood: "cheerful expression, lively atmosphere",
    },
    aspects_ja: {
      color: "鮮やかで濃い色",
      lighting: "明るくきらめく光",
      texture: "平たく明快な塗り",
      mood: "楽しい雰囲気",
    },
  },
];
const ASPECT_KEYS = ["color", "lighting", "texture", "mood"];
const TOPICS = [
  { id: "cat", label: "窓辺で猫と過ごす少女", prompt: "1girl with a cat by the window" },
  { id: "tokyo", label: "夜の街を歩く", prompt: "1girl walking through a neon city at night" },
  { id: "rain", label: "雨上がりの帰り道", prompt: "1girl on the way home after the rain" },
  { id: "forest", label: "朝の森を抜ける", prompt: "1girl walking through a morning forest" },
  { id: "cafe", label: "静かなカフェの午後", prompt: "1girl in a quiet cafe in the afternoon" },
  { id: "lighthouse", label: "灯台のある岬", prompt: "1girl at a cape with a lighthouse" },
];
const CARDS = SUBJECTS.flatMap((s) =>
  PROFILES.map((p) => ({
    id: `${s.id}-${p.id}`,
    subject_id: s.id,
    profile_id: p.id,
    label: `${s.label} · ${p.label}`,
    subject_label: s.label,
    profile_label: p.label,
    aspects: { ...p.aspects },
    aspects_ja: { ...p.aspects_ja },
    url: `/assets/cards/${s.id}-${p.id}.png`,
  })),
);
const CONFIG = {
  cards: CARDS,
  aspects: { color: "色", lighting: "光", texture: "描画", mood: "雰囲気" },
  topics: TOPICS.map((t) => ({
    id: t.id,
    label: t.label,
    preview_url: `/assets/generic/${t.id}-0.png`,
  })),
  alpha: ALPHA,
  selection: { min: 3, max: 5 },
  idle_seconds: IDLE_SECONDS,
  timeout_seconds: 120,
  samples: [
    {
      id: "s1-cat",
      topic_id: "cat",
      label: "S1 · 窓辺で猫と過ごす少女",
      preview_url: "/assets/cards/girl-warm_soft.png",
    },
    { id: "s2-tokyo", topic_id: "tokyo", label: "S2 · 夜の街を歩く", preview_url: null },
  ],
  ready: true,
};
const cardOf = (id) => CARDS.find((c) => c.id === id);
const profileOf = (cardId) => PROFILES.find((p) => p.id === cardOf(cardId)?.profile_id);
const topicOf = (id) => TOPICS.find((t) => t.id === id);
const targetPrompt = (topicId) =>
  `masterpiece, best quality, ${topicOf(topicId)?.prompt ?? topicId}, ${TAIL}`;

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

/* --------------------------------------------------------------- state */
let session = null;
const token = () => crypto.randomBytes(8).toString("hex");
const shuffle = (xs) => {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = crypto.randomInt(i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};
/**
 * One entry per distinct aspect phrase, merged across the selected cards:
 * a phrase weighs as many cards as share it, aspects turned off never appear.
 */
function mergedRefs(selection) {
  const out = new Map();
  const weight = 1;
  for (const entry of selection) {
    const card = cardOf(entry.card_id);
    const off = entry.aspects_off || [];
    for (const aspect of ASPECT_KEYS) {
      if (off.includes(aspect)) continue;
      const text = card.aspects[aspect];
      const hit = out.get(text);
      if (!hit) out.set(text, { text, weight, aspect, card_ids: [entry.card_id] });
      else {
        hit.weight = Math.round((hit.weight + weight) * 100) / 100;
        if (!hit.card_ids.includes(entry.card_id)) hit.card_ids.push(entry.card_id);
      }
    }
  }
  return [...out.values()];
}
const hashOf = (obj) =>
  crypto.createHash("sha256").update(JSON.stringify(obj)).digest("hex").slice(0, 16);
/** Personalised images get the mean hue of the selected cards; plain stays neutral. */
function personalHue() {
  if (!session.selection.length) return 210;
  const hues = session.selection.map((entry) => profileOf(entry.card_id).hue);
  return Math.round(hues.reduce((a, b) => a + b, 0) / hues.length);
}
function progress(run) {
  if (run.mode === "sample") return 4;
  return Math.max(0, Math.min(4, Math.floor((Date.now() - run.startedAt) / PAIR_MS)));
}
const runStatus = () => (progress(session.run) < 4 ? "generating" : "done");
function makeRun(topic_id, request_id) {
  return {
    id: `run-${token().slice(0, 6)}`,
    topic_id,
    request_id,
    mode: "live",
    startedAt: Date.now(),
  };
}
function personalImages(run) {
  return SEEDS.slice(0, progress(run)).map((seed, i) => ({
    id: `personal-${i}`,
    seed,
    url: `/api/sessions/${session.id}/images/personal/personal-${i}.png`,
    relative_path: `runs/${run.id}/personal/personal-${i}.png`,
    sha256: hashOf([run.id, i]),
  }));
}
function snapshot() {
  const base = {
    id: session.id,
    card_order: session.card_order,
    selection: session.selection,
    run: null,
  };
  const run = session.run;
  if (!run) return base;
  const status = runStatus();
  const refs = mergedRefs(session.selection);
  base.run = {
    id: run.id,
    topic_id: run.topic_id,
    mode: run.mode || "live",
    status,
    message:
      status === "done"
        ? "4枚できました。"
        : `${progress(run)} / 4枚ができました。`,
    elapsed_seconds: Math.round((Date.now() - run.startedAt) / 1000),
    error: null,
    prompt: targetPrompt(run.topic_id),
    plain: SEEDS.map((seed, i) => ({
      id: `plain-${i}`,
      seed,
      url: `/api/sessions/${session.id}/images/plain-${i}.png`,
      relative_path: `runs/${run.id}/plain-${i}.png`,
      prompt: targetPrompt(run.topic_id),
    })),
    personal: personalImages(run),
    personalization: { alpha: ALPHA, sample_size: 0, hash: hashOf([refs, ALPHA]), refs },
    timings: { generation: { wall_seconds: 25.1 } },
  };
  return base;
}

/* -------------------------------------------------------------- server */
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".woff2": "font/woff2",
};
function send(res, status, body, type = "application/json") {
  const payload = type.startsWith("application/json") ? JSON.stringify(body) : body;
  res.writeHead(status, {
    "content-type": type,
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
  });
  res.end(payload);
}
const fail = (res, status, detail) => send(res, status, { detail });
const ok = (res) => send(res, 200, snapshot());
function image(res, opts) {
  send(res, 200, svg(opts), "image/svg+xml; charset=utf-8");
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
  send(res, 200, fs.readFileSync(file, binary ? null : "utf8"), TYPES[path.extname(file)] || "text/plain");
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
    if (p.startsWith("/assets/cards/")) {
      const card = cardOf(path.basename(p, ".png"));
      if (!card) return fail(res, 404, "no card");
      const prof = PROFILES.find((x) => x.id === card.profile_id);
      return image(res, {
        hue: prof.hue,
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
        gpu_busy: Boolean(session?.run && runStatus() !== "done"),
        mode: "fan-live",
        offline: true,
      });
    if (p === "/api/sessions" && m === "POST") {
      if (session) return fail(res, 409, "ほかの体験が進行中です。");
      session = {
        id: `sess-${token().slice(0, 8)}`,
        card_order: shuffle(CARDS.map((c) => c.id)),
        selection: [],
        run: null,
      };
      return ok(res);
    }
    const parts = p.split("/").filter(Boolean); // api sessions <sid> ...
    if (parts[0] !== "api" || parts[1] !== "sessions") return fail(res, 404, "not found");
    const sid = parts[2];
    if (!session || session.id !== sid)
      return fail(res, 404, "セッションが見つかりません。");
    const tail = parts.slice(3);
    const body = m === "GET" || m === "DELETE" ? {} : await readBody(req);

    if (!tail.length && m === "GET") return ok(res);
    if (!tail.length && m === "DELETE") {
      session = null;
      return send(res, 200, { ok: true });
    }
    if (tail[0] === "images") {
      const run = session.run;
      if (!run) return fail(res, 404, "no run");
      if (tail.length === 3) {
        if (tail[1] !== "personal") return fail(res, 404, "no image");
        const i = Number(path.basename(tail[2], ".png").split("-").pop());
        return image(res, {
          hue: personalHue(),
          top: `FAN · alpha ${ALPHA}`,
          bottom: `seed ${SEEDS[i]}`,
          seed: i + 9,
        });
      }
      const i = Number(path.basename(tail[1], ".png").split("-").pop());
      return image(res, {
        hue: 205,
        sat: 14,
        light: 40,
        top: "PLAIN",
        bottom: `seed ${SEEDS[i]}`,
        seed: i + 3,
      });
    }
    if (tail[0] === "touch" && m === "POST") return send(res, 200, { ok: true });
    if (tail[0] === "selection" && m === "PUT") {
      if (session.run && runStatus() !== "done")
        return fail(res, 409, "生成中は変更できません。");
      const cards = body.cards || [];
      if (cards.length < CONFIG.selection.min || cards.length > CONFIG.selection.max)
        return fail(res, 409, "3〜5枚を選んでください。");
      for (const c of cards) {
        if (!cardOf(c.card_id)) return fail(res, 409, `不明なカード: ${c.card_id}`);
        if ((c.aspects_off || []).length >= ASPECT_KEYS.length)
          return fail(res, 409, "側面をすべて外すことはできません。");
      }
      session.selection = cards.map((c) => ({
        card_id: c.card_id,
        aspects_off: c.aspects_off || [],
      }));
      session.run = null;
      return ok(res);
    }
    if (tail[0] === "runs" && m === "POST") {
      const { topic_id, request_id } = body;
      if (!topicOf(topic_id)) return fail(res, 409, "不明なお題です。");
      if (!session.selection.length) return fail(res, 409, "先に画像を選んでください。");
      const run = session.run;
      if (run?.request_id === request_id) return ok(res);
      if (run && runStatus() !== "done") return fail(res, 409, "いま描いています。");
      // Every request is a fresh comparison; the finished one is replaced.
      session.run = makeRun(topic_id, request_id);
      return ok(res);
    }
    if (tail[0] === "cancel" && m === "POST") {
      // Cancelling an unfinished comparison drops the whole run.
      if (session.run && runStatus() !== "done") session.run = null;
      return ok(res);
    }
    if (tail[0] === "sample" && m === "POST") {
      const sample = CONFIG.samples.find((s) => s.id === body.sample_id);
      if (!sample) return fail(res, 404, "no sample");
      session.selection = ["girl-warm_soft", "student-cool_clean", "traveler-warm_soft"].map(
        (card_id) => ({ card_id, aspects_off: [] }),
      );
      session.run = makeRun(sample.topic_id, sample.id);
      session.run.mode = "sample";
      return ok(res);
    }
    return fail(res, 404, "not found");
  } catch (e) {
    return fail(res, 500, String(e?.stack || e));
  }
});
server.listen(PORT, "127.0.0.1", () =>
  console.log(`mock api on http://127.0.0.1:${PORT} (pair ${PAIR_MS}ms)`),
);
