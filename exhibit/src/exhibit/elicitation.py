"""Reproducible round candidates and the normalized preference payload.

Pure CPU: the round policy only reads catalog axis levels, never an image model.
"""

from __future__ import annotations

import hashlib
from itertools import combinations

from .domain import ASPECTS, canonical_aspects, digest, valid_gain, valid_strength

ALGORITHM_VERSION = "coverage-similarity-1"
# Design §6.2; the demo config is migrated separately and may still carry v1 limits.
SELECTION = {"min": 3, "max": 10, "round_size": 12, "max_rounds": 3}
SUBJECT_CAP = 3
RELAXED_SUBJECT_CAP = 4
EXPLORE_SLOTS = 4
DIVERSITY_PENALTY = 0.25
AXIS_PAIRS = tuple(combinations(range(len(ASPECTS)), 2))
SELECTION_FIELDS = {"card_id", "strength", "aspects"}
PAYLOAD_FIELDS = {"cards", "aspect_gains"}


class PreferenceError(ValueError):
    """A preference payload the server refuses; the API answers 422."""


class RoundError(ValueError):
    """A round that cannot be served at all; the API answers 409."""


def _selection_config(value):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("selection config must be an object")
    config = {key: value.get(key, default) for key, default in SELECTION.items()}
    if any(type(config[key]) is not int or config[key] < 1 for key in config):
        raise ValueError("selection config values must be positive integers")
    if config["min"] > config["max"]:
        raise ValueError("selection min must not exceed max")
    return config


def _catalog(catalog):
    if (
        not isinstance(catalog, dict)
        or not isinstance(catalog.get("cards"), list)
        or not isinstance(catalog.get("all_cards"), list)
        or not isinstance(catalog.get("catalog_id"), str)
        or not isinstance(catalog.get("catalog_hash"), str)
    ):
        raise TypeError("catalog must be a loaded catalog")
    return {card["id"]: card for card in catalog["cards"]}


def _levels(card):
    return tuple(card["axis_levels"][axis] for axis in ASPECTS)


def _add_counts(counts, levels):
    for axis, level in enumerate(levels):
        counts[(axis, level)] = counts.get((axis, level), 0) + 1
    for left, right in AXIS_PAIRS:
        key = (left, right, levels[left], levels[right])
        counts[key] = counts.get(key, 0) + 1


def _cost(counts, levels):
    """Cumulative exposure of the card's four levels and its six level pairs."""
    return sum(counts.get((axis, level), 0) for axis, level in enumerate(levels)) + sum(
        counts.get((left, right, levels[left], levels[right]), 0)
        for left, right in AXIS_PAIRS
    )


def _match(levels, other):
    return sum(1 for axis in range(len(ASPECTS)) if levels[axis] == other[axis]) / len(
        ASPECTS
    )


def _similarity(levels, preferences):
    """Strength-weighted mean over explicitly liked aspects only."""
    total = sum(strength for _, _, strength in preferences)
    if not total:
        return 0.0
    score = sum(
        strength * sum(1 for axis in axes if levels[axis] == other[axis]) / len(axes)
        for other, axes, strength in preferences
    )
    return score / total


def _tiebreak(session_seed, round_index, card_id):
    value = f"{session_seed}:{round_index}:{card_id}".encode()
    return hashlib.sha256(value).hexdigest()


def _preferences(snapshot, cards):
    """Liked cards as (levels, liked axis indexes, strength); unanswered dropped."""
    if not isinstance(snapshot, dict):
        raise PreferenceError("snapshot must be an object")
    selection = snapshot.get("selection")
    if selection is None:
        selection = []
    if not isinstance(selection, list):
        raise PreferenceError("selection must be a list")
    preferences, seen = [], set()
    for entry in selection:
        if not isinstance(entry, dict) or set(entry) != SELECTION_FIELDS:
            raise PreferenceError("Invalid selection entry")
        card = (
            cards.get(entry["card_id"]) if isinstance(entry["card_id"], str) else None
        )
        if card is None or entry["card_id"] in seen:
            raise PreferenceError("Unknown or duplicate card")
        if not valid_strength(entry["strength"]):
            raise PreferenceError("strength must be 1 or 2")
        try:
            aspects = canonical_aspects(entry["aspects"], allow_empty=True)
        except ValueError as exc:
            raise PreferenceError(str(exc)) from exc
        seen.add(entry["card_id"])
        if not aspects:
            continue
        indexes = [ASPECTS.index(aspect) for aspect in aspects]
        preferences.append((_levels(card), indexes, float(entry["strength"])))
    return preferences


def _plan(pool, preferences, *, counts, round_size, cap, levels, keys, subjects):
    """Greedy similar-then-explore picks under a per-subject cap."""
    remaining, counts, picks, used = list(pool), dict(counts), [], {}
    explore = min(EXPLORE_SLOTS, round_size) if preferences else round_size
    for slot, target in (("similar", round_size - explore), ("explore", explore)):
        for _ in range(target):
            chosen, best = None, None
            taken = (
                [levels[pick["card_id"]] for pick in picks] if slot == "similar" else []
            )
            for card_id in remaining:
                if used.get(subjects[card_id], 0) >= cap:
                    continue
                if slot == "similar":
                    penalty = max(
                        (_match(levels[card_id], other) for other in taken), default=0.0
                    )
                    score = _similarity(levels[card_id], preferences)
                    key = (-(score - DIVERSITY_PENALTY * penalty), keys[card_id])
                else:
                    key = (_cost(counts, levels[card_id]), keys[card_id])
                if best is None or key < best:
                    chosen, best = card_id, key
            if chosen is None:
                break
            picks.append({"card_id": chosen, "slot": slot})
            remaining.remove(chosen)
            used[subjects[chosen]] = used.get(subjects[chosen], 0) + 1
            _add_counts(counts, levels[chosen])
    return picks


def _by_subject(picks, subjects):
    """Same-subject cards stay adjacent, in the deterministic order they were picked."""
    order = list(dict.fromkeys(subjects[pick["card_id"]] for pick in picks))
    return [
        pick
        for subject in order
        for pick in picks
        if subjects[pick["card_id"]] == subject
    ]


def next_round(
    catalog, snapshot, *, shown_ids, round_index, session_seed, selection=None
):
    """Pick the next round of cards; `round_index` counts from 0."""
    config = _selection_config(selection)
    if type(round_index) is not int or round_index < 0:
        raise RoundError("round_index must be a non-negative integer")
    if round_index >= config["max_rounds"]:
        raise RoundError(f"More than {config['max_rounds']} rounds were requested")
    if not isinstance(session_seed, str) or not session_seed:
        raise RoundError("session_seed is required")
    if not isinstance(shown_ids, list) or any(
        not isinstance(card_id, str) for card_id in shown_ids
    ):
        raise RoundError("shown_ids must be a list of card ids")

    cards = _catalog(catalog)
    known = {card["id"]: card for card in catalog["all_cards"]}
    if not isinstance(snapshot, dict):
        raise PreferenceError("snapshot must be an object")
    if any(
        snapshot.get(key, catalog[key]) != catalog[key]
        for key in ("catalog_id", "catalog_hash")
    ):
        raise PreferenceError("Stale catalog hash")
    preferences = _preferences(snapshot, cards)

    counts, shown = {}, []
    for card_id in shown_ids:
        if card_id not in known:
            raise RoundError("Unknown shown card")
        if card_id not in shown:
            shown.append(card_id)
            _add_counts(counts, _levels(known[card_id]))
    seen = set(shown)
    pool = [card["id"] for card in catalog["cards"] if card["id"] not in seen]
    levels = {card_id: _levels(cards[card_id]) for card_id in pool}
    subjects = {card_id: cards[card_id]["subject_id"] for card_id in pool}
    keys = {card_id: _tiebreak(session_seed, round_index, card_id) for card_id in pool}

    arguments = {
        "counts": counts,
        "round_size": config["round_size"],
        "levels": levels,
        "keys": keys,
        "subjects": subjects,
    }
    picks = _plan(pool, preferences, cap=SUBJECT_CAP, **arguments)
    cap = SUBJECT_CAP
    if len(picks) < config["round_size"]:
        widened = _plan(pool, preferences, cap=RELAXED_SUBJECT_CAP, **arguments)
        if len(widened) > len(picks):
            picks, cap = widened, RELAXED_SUBJECT_CAP

    shortfall = None
    if len(picks) < config["round_size"]:
        if not pool:
            shortfall = "no_unseen_cards"
        elif len(pool) < config["round_size"]:
            shortfall = "insufficient_unseen_cards"
        else:
            shortfall = "subject_cap_limit"

    ordered = _by_subject(picks, subjects)
    identity = {
        "round_index": round_index,
        "algorithm_version": ALGORITHM_VERSION,
        "session_seed": session_seed,
        "catalog_id": catalog["catalog_id"],
        "catalog_hash": catalog["catalog_hash"],
        "shown_ids": shown,
        "card_ids": [pick["card_id"] for pick in ordered],
        "cards": ordered,
        "round_size": config["round_size"],
        "subject_cap": cap,
        "subject_cap_relaxed": cap != SUBJECT_CAP,
        "shortfall_reason": shortfall,
    }
    return {"round_id": digest(identity), **identity}


def normalize_preferences(payload, catalog, *, commit, selection=None):
    """Canonicalize `{cards, aspect_gains}` into a server-owned snapshot body."""
    config = _selection_config(selection)
    cards = _catalog(catalog)
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_FIELDS:
        raise PreferenceError("Preference payload has unknown or missing fields")
    entries = payload["cards"]
    if not isinstance(entries, list):
        raise PreferenceError("cards must be a list")
    lowest = config["min"] if commit else 0
    if not lowest <= len(entries) <= config["max"]:
        raise PreferenceError(f"{lowest}〜{config['max']}枚を選んでください。")

    gains = payload["aspect_gains"]
    if not isinstance(gains, dict) or set(gains) != set(ASPECTS):
        raise PreferenceError("aspect_gains must name every aspect")
    if any(not valid_gain(gain) for gain in gains.values()):
        raise PreferenceError("Invalid aspect_gains")

    normalized, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != SELECTION_FIELDS:
            raise PreferenceError("Invalid selection entry")
        card_id = entry["card_id"]
        if not isinstance(card_id, str) or card_id not in cards or card_id in seen:
            raise PreferenceError("Unknown or duplicate card")
        if not valid_strength(entry["strength"]):
            raise PreferenceError("strength must be 1 or 2")
        try:
            aspects = canonical_aspects(entry["aspects"], allow_empty=not commit)
        except ValueError as exc:
            raise PreferenceError(str(exc)) from exc
        seen.add(card_id)
        normalized.append(
            {"card_id": card_id, "strength": entry["strength"], "aspects": aspects}
        )
    return {
        "catalog_id": catalog["catalog_id"],
        "catalog_hash": catalog["catalog_hash"],
        "selection": sorted(normalized, key=lambda item: item["card_id"]),
        "aspect_gains": {aspect: float(gains[aspect]) for aspect in ASPECTS},
    }
