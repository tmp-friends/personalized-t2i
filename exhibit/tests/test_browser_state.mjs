import test from "node:test";
import assert from "node:assert/strict";
import {
  createPoller,
  initialScreen,
  pollIdentity,
  selectionComplete,
} from "../src/exhibit/static/session.mjs";

const LIMITS = { min: 3, max: 5 };
const pick = (n) =>
  Array.from({ length: n }, (_, i) => ({ card_id: `c${i}`, aspects_off: [] }));
const run = (extra = {}) => ({
  id: "j1",
  topic_id: "cat",
  status: "generating",
  blind: { revealed: false, pairs: [], answered: 0 },
  variants: [{ id: "v0" }],
  ...extra,
});

function harness(start = { sid: "s1", run: "j1", variant: "v0", revealed: false }) {
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

/* ------------------------------------------------------------ screens */

test("an incomplete selection resumes on the card grid", () => {
  for (const n of [0, 1, 2])
    assert.equal(
      initialScreen({ id: "s1", selection: pick(n), run: null }, LIMITS),
      "cards",
    );
  assert.equal(
    initialScreen({ id: "s1", selection: pick(6), run: null }, LIMITS),
    "cards",
    "more cards than the maximum is not a usable selection",
  );
});

test("a usable selection with no run resumes on the topic picker", () => {
  for (const n of [3, 4, 5])
    assert.equal(
      initialScreen({ id: "s1", selection: pick(n), run: null }, LIMITS),
      "topics",
    );
  assert.equal(selectionComplete(pick(3), LIMITS), true);
  assert.equal(selectionComplete(pick(2), LIMITS), false);
});

test("a run that has not been revealed resumes on the blind comparison", () => {
  assert.equal(
    initialScreen({ id: "s1", selection: pick(3), run: run() }, LIMITS),
    "blind",
  );
  assert.equal(
    initialScreen(
      { id: "s1", selection: pick(3), run: run({ status: "done" }) },
      LIMITS,
    ),
    "blind",
    "a finished run is still blind until the visitor reveals it",
  );
});

test("a revealed run resumes on the answer and adjust screen", () => {
  const revealed = run({
    status: "done",
    blind: { revealed: true, pairs: [], answered: 4, score: { personal: 3 } },
  });
  assert.equal(
    initialScreen({ id: "s1", selection: pick(3), run: revealed }, LIMITS),
    "result",
  );
  assert.equal(initialScreen(null, LIMITS), "welcome");
});

/* ----------------------------------------------------------- identity */

test("the poll identity carries the session, run, newest variant and reveal state", () => {
  assert.equal(pollIdentity(null), null);
  assert.equal(pollIdentity({ id: "s1", run: null }), null);
  assert.deepEqual(
    pollIdentity({
      id: "s1",
      run: run({ variants: [{ id: "v0" }, { id: "v1" }] }),
    }),
    { sid: "s1", run: "j1", variant: "v1", revealed: false },
  );
  assert.deepEqual(
    pollIdentity({
      id: "s1",
      run: run({ blind: { revealed: true, pairs: [] } }),
    }),
    { sid: "s1", run: "j1", variant: "v0", revealed: true },
  );
});

/* ------------------------------------------------------------- poller */

test("an old run response never replaces a newer run in the same session", async () => {
  const h = harness();
  h.poller.start();
  const old = h.tick();
  h.change({ sid: "s1", run: "j2", variant: "v0", revealed: false });
  h.resolve({ id: "s1", run: run({ status: "done" }) });
  await old;
  assert.deepEqual(h.received, []);
});

test("a snapshot from before a new variant was requested is ignored", async () => {
  const h = harness();
  h.poller.start();
  const old = h.tick();
  h.change({ sid: "s1", run: "j1", variant: "v1", revealed: true });
  h.resolve({ id: "s1", run: run({ status: "done" }) });
  await old;
  assert.deepEqual(h.received, []);
});

test("a snapshot that has forgotten the reveal never hides the answer again", async () => {
  const h = harness({ sid: "s1", run: "j1", variant: "v0", revealed: true });
  h.poller.start();
  const stale = h.tick();
  h.resolve({ id: "s1", run: run({ status: "done" }) }); // revealed: false
  await stale;
  assert.deepEqual(h.received, []);
});

test("a late 404 from a finished session cannot expire its successor", async () => {
  const h = harness();
  h.poller.start();
  const old = h.tick();
  h.poller.stop();
  h.change({ sid: "s2", run: "j3", variant: "v0", revealed: false });
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
  const result = { id: "s1", run: run() };
  h.resolve(result);
  await first;
  assert.deepEqual(h.received, [result]);
  // still generating, so the poller keeps going with a second, non-overlapping request
  const second = h.tick();
  const done = { id: "s1", run: run({ status: "done" }) };
  h.resolve(done);
  await second;
  assert.deepEqual(h.received, [result, done]);
  await h.tick(); // done stops the poller
  assert.deepEqual(h.received, [result, done]);
});

test("a blind pick keeps the same identity, so its poll results still arrive", async () => {
  const picked = run({
    blind: { revealed: false, pairs: [{ index: 0, pick: "abc" }], answered: 1 },
  });
  assert.deepEqual(pollIdentity({ id: "s1", run: picked }), {
    sid: "s1",
    run: "j1",
    variant: "v0",
    revealed: false,
  });
  const h = harness();
  h.poller.start();
  const first = h.tick();
  const result = { id: "s1", run: picked };
  h.resolve(result);
  await first;
  assert.deepEqual(h.received, [result]);
});
