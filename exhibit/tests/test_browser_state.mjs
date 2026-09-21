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
  personal: [],
  ...extra,
});

function harness(start = { sid: "s1", run: "j1" }) {
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

test("any run, generating or done, resumes on the comparison", () => {
  assert.equal(
    initialScreen({ id: "s1", selection: pick(3), run: run() }, LIMITS),
    "compare",
  );
  assert.equal(
    initialScreen(
      { id: "s1", selection: pick(3), run: run({ status: "done" }) },
      LIMITS,
    ),
    "compare",
  );
  assert.equal(initialScreen(null, LIMITS), "welcome");
});

/* ----------------------------------------------------------- identity */

test("the poll identity carries the session and run", () => {
  assert.equal(pollIdentity(null), null);
  assert.equal(pollIdentity({ id: "s1", run: null }), null);
  assert.deepEqual(pollIdentity({ id: "s1", run: run() }), { sid: "s1", run: "j1" });
});

/* ------------------------------------------------------------- poller */

test("an old run response never replaces a newer run in the same session", async () => {
  const h = harness();
  h.poller.start();
  const old = h.tick();
  h.change({ sid: "s1", run: "j2" });
  h.resolve({ id: "s1", run: run({ status: "done" }) });
  await old;
  assert.deepEqual(h.received, []);
});

test("a late 404 from a finished session cannot expire its successor", async () => {
  const h = harness();
  h.poller.start();
  const old = h.tick();
  h.poller.stop();
  h.change({ sid: "s2", run: "j3" });
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
