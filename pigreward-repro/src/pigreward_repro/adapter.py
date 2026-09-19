"""Independent parser for the public model-card output; no invented score scale."""

import random
import re

PARSER_VERSION = "card-v1"
ROW = re.compile(
    r"^([A-Za-z][A-Za-z /-]{1,48}):\s*(.+?)\s*\(Score 1:\s*(-?\d+(?:\.\d+)?),\s*Score 2:\s*(-?\d+(?:\.\d+)?)\)\s*$",
    re.MULTILINE,
)


def parse_judgment(raw, candidate_ids, *, finished):
    result = {
        "status": "inconclusive",
        "winner_id": None,
        "candidate_ids": list(candidate_ids),
        "raw": raw,
        "parser_version": PARSER_VERSION,
        "dimensions": [],
        "reason": "",
    }
    if len(candidate_ids) != 2 or len(set(candidate_ids)) != 2 or not finished:
        return result
    # Consume the entire accepted format; unknown rows may contain conflicts.
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if ROW.fullmatch(line):
            continue
        if re.fullmatch(r"Score [12]:\s*-?\d+(?:\.\d+)?", line):
            continue
        if re.fullmatch(r"Image [12] is better[.!]?", line, re.IGNORECASE):
            continue
        return result
    endings = re.findall(r"\bImage ([12]) is better\b", raw, re.IGNORECASE)
    rows = ROW.findall(raw)
    totals = [
        re.findall(rf"^Score {i}:\s*(-?\d+(?:\.\d+)?)\s*$", raw, re.MULTILINE)
        for i in (1, 2)
    ]
    if (
        len(endings) != 1
        or not rows
        or not re.search(r"Image [12] is better[.!]?\s*$", raw, re.IGNORECASE)
    ):
        return result
    if re.search(r"\b(tie|equally good|inconclusive)\b", raw, re.IGNORECASE):
        return result
    if any(len(t) != 1 for t in totals):
        return result
    scores = [float(t[0]) for t in totals]
    sums = [sum(float(row[i]) for row in rows) for i in (2, 3)]
    if any(abs(a - b) > 0.01 for a, b in zip(scores, sums)):
        return result
    winner = int(endings[0]) - 1
    if scores[winner] <= scores[1 - winner]:
        return result
    return {
        **result,
        "status": "valid",
        "winner_id": candidate_ids[winner],
        "dimensions": [r[0] for r in rows],
        "reason": " ".join(r[1] for r in rows)[:1200],
        "scores": scores,
        "axis_comparisons": [
            {"axis": r[0], "reason": r[1], "scores": [float(r[2]), float(r[3])]}
            for r in rows
        ],
    }


def tournament(candidate_ids, compare, *, seed):
    if len(candidate_ids) != 4 or len(set(candidate_ids)) != 4:
        raise ValueError("Four distinct candidates required")
    ordered = list(candidate_ids)
    random.Random(seed).shuffle(ordered)
    records = []
    winners = []
    for a, b in [(ordered[0], ordered[1]), (ordered[2], ordered[3])]:
        judgment = compare(a, b)
        records.append({**judgment, "candidate_ids": [a, b]})
        if judgment.get("status") != "valid" or judgment.get("winner_id") not in (a, b):
            return {"winner_id": None, "judgments": records, "status": "manual"}
        winners.append(judgment["winner_id"])
    judgment = compare(*winners)
    records.append({**judgment, "candidate_ids": winners})
    valid = judgment.get("status") == "valid" and judgment.get("winner_id") in winners
    return {
        "winner_id": judgment["winner_id"] if valid else None,
        "judgments": records,
        "status": "recommended" if valid else "manual",
        "reason": judgment.get("reason", "") if valid else "",
    }
