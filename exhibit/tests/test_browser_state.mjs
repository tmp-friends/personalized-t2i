import test from "node:test";
import assert from "node:assert/strict";
import {
  canRequestRound,
  commitBlocker,
  createPoller,
  createSelectionWriter,
  createSerialWriter,
  createTicket,
  currentRound,
  draftFromSnapshot,
  draftMatchesSnapshot,
  emptyDraft,
  feedbackVisible,
  gainIndex,
  GAIN_STEPS,
  initialScreen,
  likeAll,
  pollIdentity,
  removeCard,
  roundNotice,
  roundNumber,
  schemaSupported,
  selectionComplete,
  selectionPayload,
  setGain,
  setStrength,
  toggleAspect,
  toggleCard,
} from "../src/exhibit/static/session.mjs";

const LIMITS = { min: 3, max: 10, round_size: 12, max_rounds: 3 };
const ASPECTS = ["color", "lighting", "texture", "mood"];
const ids = (n, prefix = "c") =>
  Array.from({ length: n }, (_, i) => `${prefix}${String(i).padStart(2, "0")}`);
const chosen = (n, extra = {}) =>
  ids(n).map((card_id) => ({
    card_id,
    strength: 1,
    aspects: ["color"],
    ...extra,
  }));
const gains = (over = {}) => ({
  color: 1,
  lighting: 1,
  texture: 1,
  mood: 1,
  ...over,
});
const round = (index, cards, extra = {}) => ({
  round_id: `r${index}`,
  round_index: index,
  card_ids: cards,
  cards: cards.map((card_id) => ({ card_id, slot: "explore" })),
  shortfall_reason: null,
  ...extra,
});
const snapshot = (extra = {}) => ({
  id: "s1",
  revision: 0,
  catalog_id: "catalog-v2",
  catalog_hash: "abc",
  committed: false,
  selection: [],
  aspect_gains: gains(),
  rounds: [round(0, ids(12))],
  shown_ids: ids(12),
  next_round_available: true,
  run: null,
  ...extra,
});
const run = (extra = {}) => ({
  id: "j1",
  topic_id: "cat",
  status: "generating",
  mode: "live",
  error: null,
  preference_revision: 1,
  personal: [],
  feedback: null,
  ...extra,
});

function harness(start = { sid: "s1", run: "j1", revision: 1, sessionRevision: 1 }) {
  let current = start,
    resolve,
    reject,
    callback;
  const received = [],
    expired = [];
  const poller = createPoller({
    identity: () => current,
    fetchSnapshot: () =>
      new Promise((a, b) => {
        resolve = a;
        reject = b;
      }),
    onSnapshot: (s) => received.push(s),
    onExpired: () => expired.push(true),
    onError: () => {},
    schedule: (fn) => {
      callback = fn;
      return 1;
    },
    unschedule: () => {},
  });
  return {
    poller,
    received,
    expired,
    change: (v) => (current = v),
    tick: () => callback(),
    resolve: (v) => resolve(v),
    reject: (v) => reject(v),
  };
}

/* ------------------------------------------------------------- schema */

test("a config from another schema stops the client instead of guessing", () => {
  assert.equal(schemaSupported({ schema_version: 2 }), true);
  for (const value of [1, 3, undefined, null, "2"])
    assert.equal(schemaSupported({ schema_version: value }), false);
});

/* --------------------------------------------------- restore on reload */

test("a reload restores round, draft and commit state from the snapshot alone", () => {
  const server = snapshot({
    revision: 4,
    committed: true,
    selection: [
      { card_id: "c01", strength: 2, aspects: ["texture", "color"] },
      { card_id: "c14", strength: 1, aspects: ["mood"] },
      { card_id: "c07", strength: 1, aspects: ["color"] },
    ],
    aspect_gains: gains({ texture: 2, mood: 0.5 }),
    rounds: [round(0, ids(12)), round(1, ids(12, "d"))],
    shown_ids: [...ids(12), ...ids(12, "d")],
  });
  const draft = draftFromSnapshot(server, ASPECTS);
  // Sorted by card id and with canonical aspects, exactly as the server stores it.
  assert.deepEqual(
    draft.selection.map((entry) => entry.card_id),
    ["c01", "c07", "c14"],
  );
  assert.deepEqual(draft.selection[0].aspects, ["color", "texture"]);
  assert.equal(draft.selection[0].strength, 2);
  assert.deepEqual(draft.aspect_gains, gains({ texture: 2, mood: 0.5 }));
  assert.equal(draft.committed, true);
  assert.equal(draftMatchesSnapshot(draft, server, ASPECTS), true);
  // Only the latest round is on screen; both rounds count toward the maximum.
  assert.equal(roundNumber(server), 2);
  assert.deepEqual(currentRound(server).card_ids, ids(12, "d"));
  assert.equal(initialScreen(server, LIMITS), "topics");
});

test("an incomplete or unanswered draft resumes on the card grid", () => {
  assert.equal(initialScreen(null, LIMITS), "welcome");
  assert.equal(initialScreen(snapshot(), LIMITS), "cards");
  assert.equal(
    initialScreen(snapshot({ committed: true, selection: chosen(2) }), LIMITS),
    "cards",
  );
  // Committed by an older client, but a card has no answered aspect: still editing.
  assert.equal(
    initialScreen(
      snapshot({
        committed: true,
        selection: [...chosen(2), { card_id: "c02", strength: 1, aspects: [] }],
      }),
      LIMITS,
    ),
    "cards",
  );
  assert.equal(
    initialScreen(snapshot({ committed: true, selection: chosen(3) }), LIMITS),
    "topics",
  );
  assert.equal(initialScreen(snapshot({ run: run() }), LIMITS), "compare");
  assert.equal(
    initialScreen(snapshot({ run: run({ status: "done" }) }), LIMITS),
    "compare",
  );
});

test("a short round and the end of the pool are explained in plain words", () => {
  assert.equal(roundNotice(snapshot(), LIMITS), null);
  assert.match(
    roundNotice(
      snapshot({
        rounds: [round(0, ids(4), { shortfall_reason: "insufficient_unseen_cards" })],
      }),
      LIMITS,
    ),
    /4枚/,
  );
  assert.match(
    roundNotice(
      snapshot({
        rounds: [round(0, ids(9), { shortfall_reason: "subject_cap_limit" })],
      }),
      LIMITS,
    ),
    /9枚/,
  );
  assert.equal(
    roundNotice(snapshot({ next_round_available: false }), LIMITS),
    "お見せできる画像はこれで最後です。",
  );
  assert.equal(canRequestRound(snapshot(), LIMITS), true);
  assert.equal(canRequestRound(snapshot({ next_round_available: false }), LIMITS), false);
  // Three rounds are the maximum even if the server still has cards.
  assert.equal(
    canRequestRound(
      snapshot({ rounds: [round(0, ids(12)), round(1, ids(12)), round(2, ids(12))] }),
      LIMITS,
    ),
    false,
  );
});

/* ------------------------------------------------------- draft editing */

test("a newly selected card starts at 「全部好き」 and can be narrowed", () => {
  let draft = emptyDraft(ASPECTS);
  ({ draft } = toggleCard(draft, "c01", LIMITS, ASPECTS));
  assert.deepEqual(draft.selection, [
    { card_id: "c01", strength: 1, aspects: ["color", "lighting", "mood", "texture"] },
  ]);
  for (const aspect of ["color", "lighting", "mood"])
    draft = toggleAspect(draft, "c01", aspect, ASPECTS);
  assert.deepEqual(draft.selection[0].aspects, ["texture"]);
  draft = likeAll(draft, "c01", ASPECTS);
  assert.deepEqual(draft.selection[0].aspects, ["color", "lighting", "mood", "texture"]);
  draft = toggleAspect(draft, "c01", "color", ASPECTS);
  assert.deepEqual(draft.selection[0].aspects, ["lighting", "mood", "texture"]);
  draft = setStrength(draft, "c01", 2);
  assert.equal(draft.selection[0].strength, 2);
  assert.equal(setStrength(draft, "c01", 3).selection[0].strength, 2);
  draft = removeCard(draft, "c01");
  assert.deepEqual(draft.selection, []);
});

test("commit is blocked, with the card named, while an aspect is unanswered", () => {
  const ready = {
    selection: chosen(3),
    aspect_gains: gains(),
    committed: false,
  };
  assert.equal(commitBlocker(ready, LIMITS), null);
  const unanswered = {
    ...ready,
    selection: [...chosen(2), { card_id: "c09", strength: 1, aspects: [] }],
  };
  assert.deepEqual(commitBlocker(unanswered, LIMITS), {
    code: "no_aspects",
    missing: 0,
    card_id: "c09",
  });
  assert.deepEqual(commitBlocker({ ...ready, selection: chosen(1) }, LIMITS), {
    code: "too_few",
    missing: 2,
    card_id: null,
  });
  assert.equal(selectionComplete(chosen(3), LIMITS), true);
  assert.equal(selectionComplete(chosen(2), LIMITS), false);
});

test("the selection cap is explained instead of silently dropping a card", () => {
  let draft = { selection: chosen(10), aspect_gains: gains(), committed: false };
  const refused = toggleCard(draft, "c99", LIMITS);
  assert.equal(refused.draft.selection.length, 10);
  assert.match(refused.error, /10枚/);
  // Deselecting always works, and then there is room again.
  const dropped = toggleCard(refused.draft, "c00", LIMITS);
  assert.equal(dropped.error, null);
  assert.equal(dropped.draft.selection.length, 9);
  const added = toggleCard(dropped.draft, "c99", LIMITS);
  assert.equal(added.error, null);
  assert.equal(added.draft.selection.length, 10);
  assert.equal(selectionComplete(added.draft.selection, LIMITS), true);
});

test("an aspect gain is an absolute 0.5 / 1 / 2 step, never a nudge", () => {
  let draft = emptyDraft(ASPECTS);
  assert.deepEqual(GAIN_STEPS, [0.5, 1, 2]);
  assert.equal(gainIndex(draft.aspect_gains.color), 1);
  draft = setGain(draft, "color", 2);
  assert.equal(draft.aspect_gains.color, 2);
  assert.equal(gainIndex(draft.aspect_gains.color), 2);
  // Pressing 「強める」 twice does not reach 4: the value is the step itself.
  draft = setGain(draft, "color", 2);
  assert.equal(draft.aspect_gains.color, 2);
  draft = setGain(draft, "color", 0.5);
  assert.equal(draft.aspect_gains.color, 0.5);
  assert.equal(gainIndex(draft.aspect_gains.color), 0);
  assert.equal(setGain(draft, "color", 3).aspect_gains.color, 0.5);
  assert.equal(setGain(draft, "unknown", 2).aspect_gains.color, 0.5);
  // The gains a server snapshot cannot explain fall back to 「そのまま」.
  assert.deepEqual(
    draftFromSnapshot(snapshot({ aspect_gains: gains({ color: 4 }) }), ASPECTS)
      .aspect_gains,
    gains(),
  );
});

test("the selection payload is a full replace carrying the expected revision", () => {
  const draft = {
    selection: [{ card_id: "c01", strength: 2, aspects: ["color", "mood"] }],
    aspect_gains: gains({ mood: 0.5 }),
    committed: false,
  };
  assert.deepEqual(selectionPayload(draft, 7, true), {
    expected_revision: 7,
    cards: [{ card_id: "c01", strength: 2, aspects: ["color", "mood"] }],
    aspect_gains: gains({ mood: 0.5 }),
    commit: true,
  });
});

/* ---------------------------------------------------------- feedback */

test("the optional answer is offered only for the visitor's own finished run", () => {
  assert.equal(feedbackVisible(run({ status: "done" })), true);
  assert.equal(feedbackVisible(null), false);
  assert.equal(feedbackVisible(run()), false, "still generating");
  assert.equal(
    feedbackVisible(run({ status: "done", error: "GPU error" })),
    false,
    "a failed run is not asked about",
  );
  assert.equal(
    feedbackVisible(run({ status: "done", mode: "sample", preference_revision: null })),
    false,
    "a sample is somebody else's preference",
  );
  assert.equal(
    feedbackVisible(run({ status: "done", preference_revision: null })),
    false,
  );
});

/* ----------------------------------------------------------- identity */

test("the poll identity carries the session, the run and the preference revision", () => {
  assert.equal(pollIdentity(null), null);
  assert.equal(pollIdentity({ id: "s1", revision: 2, run: null }), null);
  assert.deepEqual(pollIdentity({ id: "s1", revision: 2, run: run() }), {
    sid: "s1",
    run: "j1",
    revision: 1,
    sessionRevision: 2,
  });
});

/* ------------------------------------------------------------- poller */

test("an old run response never replaces a newer run in the same session", async () => {
  const h = harness();
  h.poller.start();
  const old = h.tick();
  h.change({ sid: "s1", run: "j2", revision: 1, sessionRevision: 1 });
  h.resolve({ id: "s1", revision: 1, run: run({ status: "done" }) });
  await old;
  assert.deepEqual(h.received, []);
});

test("a snapshot for an older preference revision is discarded", async () => {
  const h = harness();
  h.poller.start();
  const first = h.tick();
  // The server answers with the same run but the preference it was started from
  // has been replaced; the screen must not adopt it.
  h.resolve({
    id: "s1",
    revision: 2,
    run: run({ status: "done", preference_revision: 0 }),
  });
  await first;
  assert.deepEqual(h.received, []);

  const second = h.tick();
  h.resolve({ id: "s1", revision: 1, run: run({ status: "done" }) });
  await second;
  assert.equal(h.received.length, 1);
  assert.equal(h.received[0].run.preference_revision, 1);
});

test("a late 404 from a finished session cannot expire its successor", async () => {
  const h = harness();
  h.poller.start();
  const old = h.tick();
  h.poller.stop();
  h.change({ sid: "s2", run: "j3", revision: 0, sessionRevision: 0 });
  h.poller.start();
  h.reject({ status: 404 });
  await old;
  assert.deepEqual(h.expired, []);
});

test("a 404 for the session being polled expires it exactly once", async () => {
  const h = harness();
  h.poller.start();
  const first = h.tick();
  h.reject({ status: 404 });
  await first;
  assert.deepEqual(h.expired, [true]);
  await h.tick(); // polling has stopped; the scheduled callback is inert
  assert.deepEqual(h.expired, [true]);
});

test("polling accepts the current result and never overlaps requests", async () => {
  const h = harness();
  h.poller.start();
  const first = h.tick();
  const result = { id: "s1", revision: 1, run: run() };
  h.resolve(result);
  await first;
  assert.deepEqual(h.received, [result]);
  // still generating, so the poller keeps going with a second, non-overlapping request
  const second = h.tick();
  const done = { id: "s1", revision: 1, run: run({ status: "done" }) };
  h.resolve(done);
  await second;
  assert.deepEqual(h.received, [result, done]);
  await h.tick(); // done stops the poller
  assert.deepEqual(h.received, [result, done]);
});

/* ------------------------------------------------------------ tickets */

test("double-clicking 「ほかの候補も見る」 asks for one round, and a retry reuses its id", () => {
  let made = 0;
  const ticket = createTicket(() => `req-${++made}`);
  const first = ticket.take();
  assert.equal(first, "req-1");
  assert.equal(ticket.take(), null, "the second click while in flight asks for nothing");
  assert.equal(made, 1);
  // The request failed: the retry is the same request, so the server replays it
  // instead of serving (and consuming) another round.
  ticket.fail();
  assert.equal(ticket.take(), "req-1");
  assert.equal(made, 1);
  ticket.done();
  assert.equal(ticket.take(), "req-2", "the next intended round is a new request");
});

/* --------------------------------------------------------- write path */

test("selection writes are serialised and pending edits coalesce", async () => {
  const sent = [];
  let release;
  const writer = createSerialWriter(async (value) => {
    sent.push(value);
    await new Promise((done) => (release = done));
    return { ok: value };
  });
  const first = writer.push("a");
  const second = writer.push("b");
  const third = writer.push("c");
  assert.deepEqual(sent, ["a"], "only one write is in flight");
  release();
  assert.deepEqual(await first, { ok: "a" });
  // "b" was superseded by "c" before anything left the browser.
  release();
  assert.deepEqual(await second, { ok: "c" });
  assert.deepEqual(await third, { ok: "c" });
  assert.deepEqual(sent, ["a", "c"]);
});

test("a stale revision adopts the server snapshot and resends with it", async () => {
  const server = { revision: 3 };
  const payloads = [];
  const adopted = [];
  const writer = createSelectionWriter({
    revision: () => server.revision,
    put: async (payload) => {
      payloads.push(payload);
      if (payload.expected_revision !== 3)
        throw Object.assign(new Error("画面が古くなっています。"), { status: 409 });
      return { id: "s1", revision: 4, committed: payload.commit };
    },
    refetch: async () => snapshot({ revision: 3, selection: chosen(3) }),
    adopt: (value) => {
      adopted.push(value);
      server.revision = value.revision;
    },
    wait: async () => {},
  });
  server.revision = 1; // the screen is one revision behind the server
  const result = await writer.push({
    draft: { selection: chosen(3), aspect_gains: gains(), committed: false },
    commit: true,
  });
  assert.deepEqual(
    payloads.map((p) => p.expected_revision),
    [1, 3],
  );
  assert.equal(adopted.length, 1, "the server snapshot is adopted once");
  assert.equal(adopted[0].revision, 3);
  assert.equal(result.revision, 4);
});

test("a write that keeps failing surfaces the server's message", async () => {
  const busy = Object.assign(new Error("生成中です。"), { status: 409 });
  const writer = createSelectionWriter({
    revision: () => 0,
    put: async () => {
      throw busy;
    },
    refetch: async () => snapshot(),
    adopt: () => {},
    wait: async () => {},
  });
  await assert.rejects(
    writer.push({ draft: emptyDraft(ASPECTS), commit: false }),
    /生成中です。/,
  );
});

test("a write refused for any other reason is not retried", async () => {
  let calls = 0;
  const writer = createSelectionWriter({
    revision: () => 0,
    put: async () => {
      calls++;
      throw Object.assign(new Error("Invalid aspects"), { status: 422 });
    },
    refetch: async () => snapshot(),
    adopt: () => {},
    wait: async () => {},
  });
  await assert.rejects(
    writer.push({ draft: emptyDraft(ASPECTS), commit: false }),
    /Invalid aspects/,
  );
  assert.equal(calls, 1);
});
