// Keep old network responses from reviving completed sessions or cancelled jobs.
export function initialScreen(session) {
  if (session.run) return "result";
  const valid = session.choices.filter((c) => c.chosen_id).length;
  return session.choices.length >= 5 && valid >= 3 ? "persona" : "choose";
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
    a && b && a.sid === b.sid && a.job === b.job && a.context === b.context;
  function stop() {
    generation++;
    if (timer !== null) unschedule(timer);
    timer = null;
  }
  function start() {
    stop();
    const epoch = generation;
    async function tick() {
      const requested = identity();
      if (epoch !== generation || !requested?.sid) return;
      try {
        const result = await fetchSnapshot(requested.sid);
        if (epoch !== generation || !same(requested, identity())) return;
        if (
          result.id !== requested.sid ||
          result.run?.id !== requested.job ||
          result.run?.context?.hash !== requested.context
        )
          return;
        onSnapshot(result);
        if (result.run.status === "done") {
          stop();
          return;
        }
      } catch (error) {
        if (epoch !== generation || !same(requested, identity())) return;
        if (error.status === 404) {
          stop();
          onExpired();
          return;
        }
        onError(error);
      }
      // Schedule after the previous request settles, never concurrently.
      if (epoch === generation) timer = schedule(tick);
    }
    timer = schedule(tick);
  }
  return { start, stop };
}
