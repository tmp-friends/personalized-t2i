import test from "node:test";
import assert from "node:assert/strict";
import { createPoller, initialScreen } from "../src/exhibit/static/session.mjs";

function harness() {
  let current = { sid: "s1", job: "j1", context: "c1" },
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

test("an old job response never replaces a newer job in the same session", async () => {
  const h = harness();
  h.poller.start();
  const old = h.tick();
  h.change({ sid: "s1", job: "j2", context: "c2" });
  h.resolve({
    id: "s1",
    run: { id: "j1", context: { hash: "c1" }, status: "done" },
  });
  await old;
  assert.deepEqual(h.received, []);
});

test("a late 404 from a finished session cannot expire its successor", async () => {
  const h = harness();
  h.poller.start();
  const old = h.tick();
  h.poller.stop();
  h.change({ sid: "s2", job: "j3", context: "c3" });
  h.poller.start();
  h.reject({ status: 404 });
  await old;
  assert.deepEqual(h.expired, []);
});

test("polling accepts the current result and never overlaps requests", async () => {
  const h = harness();
  h.poller.start();
  const first = h.tick();
  const result = {
    id: "s1",
    run: { id: "j1", context: { hash: "c1" }, status: "done" },
  };
  h.resolve(result);
  await first;
  assert.deepEqual(h.received, [result]);
});

test("reload after zero or two valid answers restores unanswered choices", () => {
  for (const n of [0, 2])
    assert.equal(
      initialScreen({
        run: null,
        choices: Array.from({ length: 5 }, (_, i) => ({
          chosen_id: i < n ? "a" : null,
        })),
      }),
      "choose",
    );
  assert.equal(
    initialScreen({
      run: null,
      choices: Array.from({ length: 5 }, () => ({ chosen_id: "a" })),
    }),
    "persona",
  );
});
