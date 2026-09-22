// Pure session-state helpers, shared by app.js and exhibit/tests/test_browser_state.mjs.
//
// The server snapshot (GET /api/sessions/{sid}) is the single source of truth: every
// helper here either derives what the screen should show from a snapshot, or turns a
// local edit into the next full-replace selection payload. Nothing here touches the
// DOM or the network, so the whole state machine is testable under `node --test`.

/** The `/api/config` schema this client was written for. */
export const SCHEMA_VERSION = 2;
export const DEFAULT_SELECTION = { min: 3, max: 10, round_size: 12, max_rounds: 3 };
export const ASPECT_KEYS = ["color", "lighting", "texture", "mood"];
/** Absolute, three-step aspect gain: never a relative nudge. */
export const GAIN_STEPS = [0.5, 1, 2];
export const GAIN_LABELS = ["弱める", "そのまま", "強める"];
export const STRENGTH_LABELS = { 1: "好き", 2: "とても好き" };

const byCard = (a, b) => (a.card_id < b.card_id ? -1 : a.card_id > b.card_id ? 1 : 0);

export function withLimits(limits) {
  const merged = { ...DEFAULT_SELECTION };
  for (const key of Object.keys(DEFAULT_SELECTION)) {
    const value = limits?.[key];
    if (Number.isInteger(value) && value > 0) merged[key] = value;
  }
  return merged;
}

/** A client written for another schema must not reinterpret the new payloads. */
export function schemaSupported(config) {
  return config?.schema_version === SCHEMA_VERSION;
}

const canonicalAspects = (aspects, keys = ASPECT_KEYS) =>
  [...new Set(Array.isArray(aspects) ? aspects : [])]
    .filter((aspect) => keys.includes(aspect))
    .sort();

const canonicalGains = (gains, keys = ASPECT_KEYS) =>
  Object.fromEntries(
    keys.map((key) => [key, GAIN_STEPS.includes(gains?.[key]) ? gains[key] : 1]),
  );

/** An untouched draft: no cards, every aspect gain at 「そのまま」. */
export function emptyDraft(keys = ASPECT_KEYS) {
  return { selection: [], aspect_gains: canonicalGains({}, keys), committed: false };
}

/**
 * The working copy the screen edits, rebuilt from the server snapshot. After a
 * reload this is the only thing that decides what the visitor sees selected.
 */
export function draftFromSnapshot(snapshot, keys = ASPECT_KEYS) {
  if (!snapshot) return emptyDraft(keys);
  return {
    selection: (Array.isArray(snapshot.selection) ? snapshot.selection : [])
      .map((entry) => ({
        card_id: entry.card_id,
        strength: entry.strength === 2 ? 2 : 1,
        aspects: canonicalAspects(entry.aspects, keys),
      }))
      .sort(byCard),
    aspect_gains: canonicalGains(snapshot.aspect_gains, keys),
    committed: Boolean(snapshot.committed),
  };
}

/** Normalized content, in the same shape the server compares for a revision bump. */
export function draftContent(draft) {
  return JSON.stringify({
    selection: (draft?.selection || [])
      .map((entry) => ({
        card_id: entry.card_id,
        strength: entry.strength,
        aspects: [...entry.aspects],
      }))
      .sort(byCard),
    aspect_gains: draft?.aspect_gains || {},
    committed: Boolean(draft?.committed),
  });
}

export const sameDraft = (a, b) => draftContent(a) === draftContent(b);

/** True when the server already holds exactly this content at this commit state. */
export function draftMatchesSnapshot(draft, snapshot, keys = ASPECT_KEYS) {
  return sameDraft(draft, draftFromSnapshot(snapshot, keys));
}

/** The full-replace body of `PUT /api/sessions/{sid}/selection`. */
export function selectionPayload(draft, revision, commit) {
  return {
    expected_revision: revision,
    cards: (draft?.selection || []).map((entry) => ({
      card_id: entry.card_id,
      strength: entry.strength,
      aspects: [...entry.aspects],
    })),
    aspect_gains: { ...(draft?.aspect_gains || {}) },
    commit: Boolean(commit),
  };
}

/** Card count only; aspects are checked separately by `commitBlocker`. */
export function selectionComplete(selection, limits = DEFAULT_SELECTION) {
  const lim = withLimits(limits);
  const n = Array.isArray(selection) ? selection.length : 0;
  return n >= lim.min && n <= lim.max;
}

/**
 * Why 「決定」 is refused, or null when the draft can be committed. A card whose
 * aspects are still unanswered blocks the commit and names itself.
 */
export function commitBlocker(draft, limits = DEFAULT_SELECTION) {
  const lim = withLimits(limits);
  const selection = draft?.selection || [];
  if (selection.length < lim.min)
    return { code: "too_few", missing: lim.min - selection.length, card_id: null };
  if (selection.length > lim.max)
    return { code: "too_many", missing: 0, card_id: null };
  const unanswered = selection.find((entry) => !entry.aspects?.length);
  if (unanswered)
    return { code: "no_aspects", missing: 0, card_id: unanswered.card_id };
  return null;
}

/** Which screen a snapshot resumes on; rounds, draft and commit all come from it. */
export function initialScreen(snapshot, limits = DEFAULT_SELECTION) {
  if (!snapshot) return "welcome";
  if (snapshot.run) return "compare";
  if (snapshot.committed && !commitBlocker(draftFromSnapshot(snapshot), limits))
    return "topics";
  return "cards";
}

/* ------------------------------------------------------------------ rounds */

/** Only the latest round is on screen; earlier picks live in the tray. */
export function currentRound(snapshot) {
  const rounds = snapshot?.rounds;
  return Array.isArray(rounds) && rounds.length ? rounds[rounds.length - 1] : null;
}

export function roundNumber(snapshot) {
  const round = currentRound(snapshot);
  if (!round) return 0;
  return Number.isInteger(round.round_index)
    ? round.round_index + 1
    : snapshot.rounds.length;
}

export function canRequestRound(snapshot, limits = DEFAULT_SELECTION) {
  const lim = withLimits(limits);
  if (!snapshot?.next_round_available) return false;
  return (snapshot.rounds?.length || 0) < lim.max_rounds;
}

const SHORTFALL = {
  no_unseen_cards: () => "お見せできる画像はこれですべてです。",
  insufficient_unseen_cards: (n) => `残りの画像が少ないため、この回は${n}枚です。`,
  subject_cap_limit: (n) =>
    `同じ人物ばかりにならないように、この回は${n}枚にしています。`,
};

/** A plain-language note about a short round or the end of the pool. */
export function roundNotice(snapshot, limits = DEFAULT_SELECTION) {
  const round = currentRound(snapshot);
  if (!round) return null;
  const shortfall = SHORTFALL[round.shortfall_reason];
  if (shortfall) return shortfall((round.card_ids || []).length);
  if (!canRequestRound(snapshot, limits))
    return "お見せできる画像はこれで最後です。";
  return null;
}

/* ------------------------------------------------------------- draft edits */

const replace = (draft, selection) => ({ ...draft, selection: selection.sort(byCard) });
const edit = (draft, cardId, change) =>
  replace(
    draft,
    (draft.selection || []).map((entry) =>
      entry.card_id === cardId ? { ...entry, ...change(entry) } : entry,
    ),
  );

/** A newly selected card starts at 「全部好き」; the visitor narrows it from there. */
export function toggleCard(draft, cardId, limits = DEFAULT_SELECTION, keys = ASPECT_KEYS) {
  const lim = withLimits(limits);
  const selection = draft.selection || [];
  if (selection.some((entry) => entry.card_id === cardId))
    return {
      draft: replace(
        draft,
        selection.filter((entry) => entry.card_id !== cardId),
      ),
      error: null,
    };
  if (selection.length >= lim.max)
    return {
      draft,
      error: `選べるのは${lim.max}枚までです。ほかの画像を外すと、この画像を選べます。`,
    };
  return {
    draft: replace(draft, [
      ...selection,
      { card_id: cardId, strength: 1, aspects: canonicalAspects(keys, keys) },
    ]),
    error: null,
  };
}

export function removeCard(draft, cardId) {
  return replace(
    draft,
    (draft.selection || []).filter((entry) => entry.card_id !== cardId),
  );
}

export function setStrength(draft, cardId, strength) {
  if (strength !== 1 && strength !== 2) return draft;
  return edit(draft, cardId, () => ({ strength }));
}

export function toggleAspect(draft, cardId, aspect, keys = ASPECT_KEYS) {
  if (!keys.includes(aspect)) return draft;
  return edit(draft, cardId, (entry) => ({
    aspects: entry.aspects.includes(aspect)
      ? entry.aspects.filter((item) => item !== aspect)
      : canonicalAspects([...entry.aspects, aspect], keys),
  }));
}

/** 「全部好き」: fill every aspect back in after the visitor narrowed them. */
export function likeAll(draft, cardId, keys = ASPECT_KEYS) {
  return edit(draft, cardId, () => ({ aspects: canonicalAspects(keys, keys) }));
}

/** Absolute gain: 弱める / そのまま / 強める map to 0.5 / 1 / 2, not to a nudge. */
export function setGain(draft, aspect, value) {
  if (!GAIN_STEPS.includes(value) || !(aspect in (draft.aspect_gains || {})))
    return draft;
  return { ...draft, aspect_gains: { ...draft.aspect_gains, [aspect]: value } };
}

export const gainIndex = (value) => {
  const index = GAIN_STEPS.indexOf(value);
  return index < 0 ? 1 : index;
};

/* ---------------------------------------------------------------- feedback */

/**
 * The optional, labelled answer is offered only for the visitor's own finished,
 * error-free comparison. Samples and failed runs never ask.
 */
export function feedbackVisible(run) {
  return Boolean(
    run &&
      run.id &&
      run.mode !== "sample" &&
      run.status === "done" &&
      !run.error &&
      run.preference_revision !== null &&
      run.preference_revision !== undefined,
  );
}

/* ---------------------------------------------------------------- identity */

/**
 * What a poll response must still match: the same session, the same run and the
 * same preference. A snapshot for an older revision or another run is discarded.
 */
export function pollIdentity(session) {
  const run = session?.run;
  if (!session?.id || !run) return null;
  return {
    sid: session.id,
    run: run.id ?? null,
    revision: run.preference_revision ?? null,
    sessionRevision: session.revision ?? null,
  };
}

const IDENTITY_KEYS = ["sid", "run", "revision", "sessionRevision"];

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
    Boolean(a) && Boolean(b) && IDENTITY_KEYS.every((key) => a[key] === b[key]);
  // A snapshot from another session, run or preference revision is not newer.
  const fresh = (requested, result) =>
    Boolean(result) &&
    result.id === requested.sid &&
    (result.run?.id ?? null) === requested.run &&
    (result.run?.preference_revision ?? null) === requested.revision &&
    (result.revision ?? null) === requested.sessionRevision;
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
      if (!same(requested, identity()) || !fresh(requested, result)) return again();
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

/* ----------------------------------------------------------------- tickets */

/**
 * One request id per intended action. A second click while the first request is
 * in flight gets nothing, and a retry after a failure reuses the same id, so the
 * server's ledger can replay it instead of serving another round.
 */
export function createTicket(makeId) {
  let id = null,
    busy = false;
  return {
    get busy() {
      return busy;
    },
    get id() {
      return id;
    },
    take() {
      if (busy) return null;
      if (id === null) id = makeId();
      busy = true;
      return id;
    },
    /** The request failed: keep the id so a retry is the same request. */
    fail() {
      busy = false;
      return id;
    },
    /** The action finished: the next intended action needs a new id. */
    done() {
      busy = false;
      id = null;
    },
  };
}

/**
 * Serialised, coalescing writer: one selection PUT is in flight at a time and
 * queued edits collapse into the latest draft, which is what gets sent next.
 */
export function createSerialWriter(send) {
  let running = false,
    pending = null;
  async function pump() {
    if (running) return;
    running = true;
    try {
      while (pending) {
        const job = pending;
        pending = null;
        try {
          const result = await send(job.value);
          for (const waiter of job.waiters) waiter.resolve(result);
        } catch (error) {
          for (const waiter of job.waiters) waiter.reject(error);
        }
      }
    } finally {
      running = false;
    }
  }
  return {
    get busy() {
      return running || Boolean(pending);
    },
    push(value) {
      return new Promise((resolve, reject) => {
        const waiters = pending ? pending.waiters : [];
        pending = { value, waiters: [...waiters, { resolve, reject }] };
        pump();
      });
    },
  };
}

export const WRITE_RETRY_MS = [0, 300, 700, 1500, 2500];

/**
 * The selection writer: full-replace PUTs, one at a time, with the revision read
 * at send time. A 409 (stale revision, or a worker that still owns the GPU just
 * after cancel/done) adopts the snapshot the server returns and retries with it.
 */
export function createSelectionWriter({
  revision,
  put,
  refetch,
  adopt,
  wait = () => Promise.resolve(),
  delays = WRITE_RETRY_MS,
}) {
  return createSerialWriter(async ({ draft, commit }) => {
    let last = null;
    for (let attempt = 0; attempt < delays.length; attempt++) {
      if (delays[attempt]) await wait(delays[attempt]);
      try {
        return await put(selectionPayload(draft, revision(), commit));
      } catch (error) {
        if (error?.status !== 409) throw error;
        last = error;
        adopt(await refetch());
      }
    }
    throw last;
  });
}
