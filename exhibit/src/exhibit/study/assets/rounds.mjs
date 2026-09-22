// A faithful port of `exhibit.elicitation.next_round` for the offline study page.
//
// The study pages have no server, so the round policy of design §7.2 has to run in
// the browser. Only one part of the Python original cannot be reproduced here: the
// sha256 tie-break. The builder precomputes it for every (round_index, card_id) and
// embeds it as `keyTable`, so this file is pure arithmetic over the same catalog
// levels, in the same order. `exhibit/tests/test_preference_study.py` drives both
// implementations over the same fixtures and compares the card ids.

export const ASPECT_KEYS = ["color", "lighting", "texture", "mood"];
export const AXIS_PAIRS = [
  [0, 1],
  [0, 2],
  [0, 3],
  [1, 2],
  [1, 3],
  [2, 3],
];
export const SUBJECT_CAP = 3;
export const RELAXED_SUBJECT_CAP = 4;
export const EXPLORE_SLOTS = 4;
export const DIVERSITY_PENALTY = 0.25;
export const DEFAULT_SELECTION = { min: 3, max: 10, round_size: 12, max_rounds: 3 };

export const cardLevels = (card) => ASPECT_KEYS.map((axis) => card.axis_levels[axis]);

function addCounts(counts, levels) {
  for (let axis = 0; axis < levels.length; axis++) {
    const key = `a${axis}:${levels[axis]}`;
    counts[key] = (counts[key] || 0) + 1;
  }
  for (const [left, right] of AXIS_PAIRS) {
    const key = `p${left}${right}:${levels[left]}:${levels[right]}`;
    counts[key] = (counts[key] || 0) + 1;
  }
}

/** Cumulative exposure of the card's four levels and its six level pairs. */
function cost(counts, levels) {
  let total = 0;
  for (let axis = 0; axis < levels.length; axis++)
    total += counts[`a${axis}:${levels[axis]}`] || 0;
  for (const [left, right] of AXIS_PAIRS)
    total += counts[`p${left}${right}:${levels[left]}:${levels[right]}`] || 0;
  return total;
}

function match(levels, other) {
  let same = 0;
  for (let axis = 0; axis < ASPECT_KEYS.length; axis++)
    if (levels[axis] === other[axis]) same += 1;
  return same / ASPECT_KEYS.length;
}

/** Strength-weighted mean over explicitly liked aspects only. */
function similarity(levels, preferences) {
  let total = 0;
  for (const entry of preferences) total += entry.strength;
  if (!total) return 0.0;
  let score = 0;
  for (const entry of preferences) {
    let same = 0;
    for (const axis of entry.axes) if (levels[axis] === entry.levels[axis]) same += 1;
    score += (entry.strength * same) / entry.axes.length;
  }
  return score / total;
}

const lower = (a, b) => (a[0] !== b[0] ? a[0] < b[0] : a[1] < b[1]);

function plan(pool, preferences, { counts, roundSize, cap, levels, keys, subjects }) {
  const remaining = [...pool];
  const seen = { ...counts };
  const picks = [];
  const used = {};
  const explore = preferences.length ? Math.min(EXPLORE_SLOTS, roundSize) : roundSize;
  for (const [slot, target] of [
    ["similar", roundSize - explore],
    ["explore", explore],
  ]) {
    for (let taken = 0; taken < target; taken++) {
      let chosen = null;
      let best = null;
      const already =
        slot === "similar" ? picks.map((pick) => levels[pick.card_id]) : [];
      for (const cardId of remaining) {
        if ((used[subjects[cardId]] || 0) >= cap) continue;
        let key;
        if (slot === "similar") {
          let penalty = 0.0;
          for (const other of already)
            penalty = Math.max(penalty, match(levels[cardId], other));
          const score = similarity(levels[cardId], preferences);
          key = [-(score - DIVERSITY_PENALTY * penalty), keys[cardId]];
        } else {
          key = [cost(seen, levels[cardId]), keys[cardId]];
        }
        if (best === null || lower(key, best)) {
          chosen = cardId;
          best = key;
        }
      }
      if (chosen === null) break;
      picks.push({ card_id: chosen, slot });
      remaining.splice(remaining.indexOf(chosen), 1);
      used[subjects[chosen]] = (used[subjects[chosen]] || 0) + 1;
      addCounts(seen, levels[chosen]);
    }
  }
  return picks;
}

/** Same-subject cards stay adjacent, in the deterministic order they were picked. */
function bySubject(picks, subjects) {
  const order = [...new Set(picks.map((pick) => subjects[pick.card_id]))];
  return order.flatMap((subject) =>
    picks.filter((pick) => subjects[pick.card_id] === subject),
  );
}

function preferencesOf(snapshot, cards) {
  const preferences = [];
  for (const entry of snapshot?.selection || []) {
    const card = cards[entry.card_id];
    if (!card) throw new Error("Unknown or duplicate card");
    const axes = [...new Set(entry.aspects || [])]
      .filter((aspect) => ASPECT_KEYS.includes(aspect))
      .sort()
      .map((aspect) => ASPECT_KEYS.indexOf(aspect));
    if (!axes.length) continue;
    preferences.push({ levels: cardLevels(card), axes, strength: entry.strength });
  }
  return preferences;
}

/** Pick the next round of cards; `roundIndex` counts from 0. */
export function nextRound(
  catalog,
  snapshot,
  { shownIds = [], roundIndex = 0, keyTable = {}, selection = DEFAULT_SELECTION },
) {
  const config = { ...DEFAULT_SELECTION, ...(selection || {}) };
  if (!Number.isInteger(roundIndex) || roundIndex < 0)
    throw new Error("roundIndex must be a non-negative integer");
  if (roundIndex >= config.max_rounds)
    throw new Error(`More than ${config.max_rounds} rounds were requested`);

  const cards = Object.fromEntries(catalog.cards.map((card) => [card.id, card]));
  const known = Object.fromEntries(
    (catalog.all_cards || catalog.cards).map((card) => [card.id, card]),
  );
  const preferences = preferencesOf(snapshot, cards);

  const counts = {};
  const shown = [];
  for (const cardId of shownIds) {
    if (!known[cardId]) throw new Error("Unknown shown card");
    if (shown.includes(cardId)) continue;
    shown.push(cardId);
    addCounts(counts, cardLevels(known[cardId]));
  }
  const pool = catalog.cards
    .map((card) => card.id)
    .filter((cardId) => !shown.includes(cardId));
  const levels = {};
  const subjects = {};
  const keys = {};
  for (const cardId of pool) {
    levels[cardId] = cardLevels(cards[cardId]);
    subjects[cardId] = cards[cardId].subject_id;
    keys[cardId] = (keyTable[roundIndex] || {})[cardId];
    if (typeof keys[cardId] !== "string")
      throw new Error(`missing tie-break key: ${roundIndex}:${cardId}`);
  }

  const shared = { counts, roundSize: config.round_size, levels, keys, subjects };
  let picks = plan(pool, preferences, { ...shared, cap: SUBJECT_CAP });
  let cap = SUBJECT_CAP;
  if (picks.length < config.round_size) {
    const widened = plan(pool, preferences, { ...shared, cap: RELAXED_SUBJECT_CAP });
    if (widened.length > picks.length) {
      picks = widened;
      cap = RELAXED_SUBJECT_CAP;
    }
  }

  let shortfall = null;
  if (picks.length < config.round_size) {
    if (!pool.length) shortfall = "no_unseen_cards";
    else if (pool.length < config.round_size) shortfall = "insufficient_unseen_cards";
    else shortfall = "subject_cap_limit";
  }
  const ordered = bySubject(picks, subjects);
  return {
    round_index: roundIndex,
    shown_ids: shown,
    card_ids: ordered.map((pick) => pick.card_id),
    cards: ordered,
    round_size: config.round_size,
    subject_cap: cap,
    subject_cap_relaxed: cap !== SUBJECT_CAP,
    shortfall_reason: shortfall,
  };
}
