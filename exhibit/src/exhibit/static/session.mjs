// Pure session-state helpers, shared by app.js and exhibit/tests/test_browser_state.mjs.
// They keep old network responses from reviving deleted sessions, cancelled runs or
// snapshots that predate the visitor's latest action.

export const DEFAULT_SELECTION = { min: 3, max: 5 };

/** A selection is usable once it holds between `min` and `max` cards. */
export function selectionComplete(selection, limits = DEFAULT_SELECTION) {
  const n = Array.isArray(selection) ? selection.length : 0;
  const min = limits?.min ?? DEFAULT_SELECTION.min;
  const max = limits?.max ?? DEFAULT_SELECTION.max;
  return n >= min && n <= max;
}

/** Which screen a restored (sessionStorage) session should resume on. */
export function initialScreen(session, limits = DEFAULT_SELECTION) {
  if (!session) return "welcome";
  if (session.run) return "compare";
  return selectionComplete(session.selection, limits) ? "topics" : "cards";
}

/**
 * The identity a poll request is tied to. A response is only accepted while all
 * of these still hold: same session, same run. Anything the visitor does in the
 * meantime invalidates in-flight polls.
 */
export function pollIdentity(session) {
  const run = session?.run;
  if (!session?.id || !run) return null;
  return { sid: session.id, run: run.id ?? null };
}

export function createPoller({
  identity,
  fetchSnapshot,
  onSnapshot,
  onExpired,
  onError,
  schedule = (fn) => setTimeout(fn, 500),
  unschedule = (id) => clearTimeout(id),
}) {
  let generation = 0,
    timer = null;
  const same = (a, b) =>
    Boolean(a) &&
    Boolean(b) &&
    a.sid === b.sid &&
    a.run === b.run;
  // A snapshot from another session/run is older than what the screen shows.
  const fresh = (requested, result) =>
    Boolean(result) &&
    result.id === requested.sid &&
    (result.run?.id ?? null) === requested.run;
  function stop() {
    generation++;
    if (timer !== null) unschedule(timer);
    timer = null;
  }
  function start() {
    stop();
    const epoch = generation;
    // Schedule after the previous request settles, never concurrently.
    const again = () => {
      if (epoch === generation) timer = schedule(tick);
    };
    async function tick() {
      const requested = identity();
      if (epoch !== generation) return;
      if (!requested?.sid) return;
      let result;
      try {
        result = await fetchSnapshot(requested.sid);
      } catch (error) {
        if (epoch !== generation) return;
        if (!same(requested, identity())) return again();
        if (error.status === 404) {
          stop();
          onExpired();
          return;
        }
        onError(error);
        return again();
      }
      if (epoch !== generation) return;
      if (!same(requested, identity()) || !fresh(requested, result))
        return again();
      onSnapshot(result);
      if (result.run?.status === "done") {
        stop();
        return;
      }
      again();
    }
    timer = schedule(tick);
  }
  return { start, stop };
}
