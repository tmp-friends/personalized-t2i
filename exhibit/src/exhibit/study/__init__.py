"""Blind preference study: collection page, participants, manifest, answer pages.

Nothing here invents a participant, an answer, or a win rate. Without
``participants.json`` only the collection page and the Japanese instructions are
produced, and the status stays ``not_generated``.
"""

from __future__ import annotations

import copy
import json
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..catalog import load_catalog
from ..domain import ASPECTS, digest

# The round policy is shared with the exhibit; the study page must not fork it.
from ..elicitation import _tiebreak, normalize_preferences
from ..evaluation import participant_bootstrap_interval

ASSETS = Path(__file__).resolve().parent / "assets"
PARTICIPANT_ID = re.compile(r"^p[0-9]{2,4}$")
PARTICIPANT_ID_LABEL = "p と2〜4桁の数字（例 p001）"
CONDITIONS = {"encoder": ("new_pool",), "elicitation": ("legacy_pool", "new_pool")}
COMPARISON_ROLES = {
    "candidate_vs_legacy": ("candidate", "legacy"),
    "own_vs_other": ("own", "other"),
    "new_pool_vs_legacy_pool": ("new_pool", "legacy_pool"),
}
ROLE_POLICY = {
    "candidate": "candidate",
    "own": "candidate",
    "other": "candidate",
    "legacy": "legacy",
    "new_pool": "encoder",
    "legacy_pool": "encoder",
}
ROLE_CONDITION = {
    "candidate": "new_pool",
    "legacy": "new_pool",
    "own": "new_pool",
    "other": "new_pool",
    "new_pool": "new_pool",
    "legacy_pool": "legacy_pool",
}
OTHER_ROLES = {"other"}
STORED_FIELDS = (
    "匿名の参加者ID（実施者が割り当てた記号のみ）",
    "選んだ画像のID、好きなところ（色・光・描き方・雰囲気）、好きさの強さ",
    "提示された画像のIDと、選び終わるまでにかかった時間",
)
STORAGE_NOTE = (
    "名前・メールアドレス・端末の情報は保存しません。"
    "保存されるのはこの端末に書き出すJSONファイルだけで、通信は行いません。"
)


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


class StudyError(ValueError):
    """An operator-facing refusal; the CLI prints it and exits non-zero."""


def study_dir(config):
    return Path(config["study"]["study_dir"])


def catalogs_for(config, *, loader=None):
    """Load the reviewed catalog of every condition this kind collects."""
    loader = loader or load_catalog
    study = config["study"]
    result = {"new_pool": loader(study["catalog_id"], reviewed_only=True)}
    if study["study_kind"] == "elicitation":
        result["legacy_pool"] = loader(study["legacy_catalog_id"], reviewed_only=True)
    return result


def _selection_for(config, condition):
    study = config["study"]
    return copy.deepcopy(
        study["legacy_selection"] if condition == "legacy_pool" else study["selection"]
    )


# --------------------------------------------------------------- collection


def key_table(catalog, *, session_seed, rounds):
    """Precompute `elicitation._tiebreak` so the page needs no hash in JavaScript."""
    return {
        str(index): {
            card["id"]: _tiebreak(session_seed, index, card["id"])
            for card in catalog["cards"]
        }
        for index in range(rounds)
    }


def inline_module(text):
    """Turn an ES module into inlinable source; `export` is the only difference."""
    lines = []
    for line in text.splitlines():
        if re.match(r"^export\s*\{", line):
            continue
        lines.append(re.sub(r"^export\s+", "", line))
    return "\n".join(lines)


def _page(template, *, title, subtitle, data, script):
    html = (ASSETS / template).read_text()
    for token, value in (
        ("__TITLE__", title),
        ("__SUBTITLE__", subtitle),
        ("__STYLE__", (ASSETS / "study.css").read_text()),
        ("__DATA__", data),
        ("__SCRIPT__", script),
    ):
        html = html.replace(token, value)
    return html


def _page_json(value):
    """Embed data in a classic script without ever closing the script element."""
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    return text.replace("</", "<\\/")


def collection_condition(config, condition, catalog):
    """One condition of the collection page: catalog, limits, and the round keys."""
    study = config["study"]
    limits = _selection_for(config, condition)
    legacy = condition == "legacy_pool"
    return {
        "condition": condition,
        "flow": "single_screen" if legacy else "rounds",
        "aspects_default_on": legacy,
        "strength": not legacy,
        "selection": limits,
        "title": "気に入った画像をえらぶ"
        if not legacy
        else "気に入った画像をえらぶ（一覧から）",
        "instruction": (
            f"好みに近い画像を{limits['min']}〜{limits['max']}枚えらんでください。"
            + (
                "一覧はこの1画面だけです。"
                if legacy
                else f"「ほかの候補も見る」で最大{limits['max_rounds']}回まで別の画像を見られます。"
            )
        ),
        "aspect_instruction": (
            "画像ごとに、好きなところを外してください。"
            if legacy
            else "画像ごとに、好きなところを1つ以上えらんでください。すべて好きなら「全部好き」を押してください。"
        ),
        "catalog": {
            "catalog_id": catalog["catalog_id"],
            "cards": [
                {
                    "id": card["id"],
                    "subject_id": card["subject_id"],
                    "axis_levels": card["axis_levels"],
                    "label": card["label"],
                    "aspects_ja": card["aspects_ja"],
                }
                for card in catalog["cards"]
            ],
        },
        "key_table": key_table(
            catalog,
            session_seed=f"{study['study_id']}:{condition}",
            rounds=limits["max_rounds"],
        ),
    }


def build_collection_page(config, catalogs):
    """The collection page plus the card images it needs beside it."""
    study = config["study"]
    conditions = [
        collection_condition(config, condition, catalogs[condition])
        for condition in CONDITIONS[study["study_kind"]]
    ]
    data = {
        "study_id": study["study_id"],
        "study_kind": study["study_kind"],
        "aspects": list(ASPECTS),
        "participant_pattern": PARTICIPANT_ID.pattern,
        "participant_pattern_label": PARTICIPANT_ID_LABEL,
        "participant_placeholder": "p001",
        "stored_fields": list(STORED_FIELDS),
        "storage_note": STORAGE_NOTE,
        "conditions": conditions,
    }
    script = "\n".join(
        inline_module((ASSETS / name).read_text())
        for name in ("rounds.mjs", "collect.mjs")
    )
    html = _page(
        "collect.html",
        title="好みの画像をえらぶ",
        subtitle="このページは通信しません。回答は最後にJSONファイルとして保存します。",
        data="globalThis.STUDY = " + _page_json(data) + ";",
        script=script,
    )
    images = {}
    for condition in CONDITIONS[study["study_kind"]]:
        catalog = catalogs[condition]
        for card in catalog["cards"]:
            images[f"images/{card['id']}.png"] = card["path"]
    return {"html": html, "images": images}


# ------------------------------------------------------------- participants


def _condition_of(record, kind):
    condition = record.get("collection_condition")
    if kind == "elicitation":
        if condition not in CONDITIONS[kind]:
            raise StudyError(
                "collection_condition must be legacy_pool or new_pool for an elicitation study"
            )
        return condition
    if condition not in (None, "new_pool"):
        raise StudyError("an encoder study collects only the new_pool condition")
    return "new_pool"


def _metrics(record):
    value = {}
    for key in ("elapsed_ms", "round_count"):
        number = record.get(key)
        if number is None:
            continue
        if type(number) is not int or number < 0:
            raise StudyError(f"{key} must be a non-negative integer")
        value[key] = number
    shown = record.get("shown_ids")
    if shown is not None:
        if not isinstance(shown, list) or any(
            not isinstance(item, str) for item in shown
        ):
            raise StudyError("shown_ids must be a list of card ids")
        value["shown_count"] = len(set(shown))
    return value


def validate_export(record, config, catalogs, *, source="export"):
    """One participant export: anonymous, self-consistent, and catalog-bound."""
    study = config["study"]
    if not isinstance(record, dict):
        raise StudyError("export must be a JSON object")
    missing = [
        key
        for key in ("participant_id", "catalog_id", "selection", "aspect_gains")
        if key not in record
    ]
    if missing:
        raise StudyError("missing required fields: " + ", ".join(missing))
    participant = record["participant_id"]
    if not isinstance(participant, str) or not PARTICIPANT_ID.fullmatch(participant):
        raise StudyError(f"participant_id must match {PARTICIPANT_ID.pattern}")
    if record.get("study_id") not in (None, study["study_id"]):
        raise StudyError("export belongs to another study_id")
    condition = _condition_of(record, study["study_kind"])
    catalog = catalogs[condition]
    if record["catalog_id"] != catalog["catalog_id"]:
        raise StudyError(f"unknown catalog for {condition}: {record['catalog_id']!r}")
    payload = {"cards": record["selection"], "aspect_gains": record["aspect_gains"]}
    try:
        snapshot = normalize_preferences(
            payload,
            catalog,
            commit=True,
            selection=_selection_for(config, condition),
        )
    except (TypeError, ValueError) as error:
        raise StudyError(f"invalid selection: {error}") from error
    # The client never supplies the catalog hash; the server attaches its own.
    return {
        "participant_id": participant,
        "collection_condition": condition,
        "presented_index": record.get("presented_index"),
        "catalog_id": catalog["catalog_id"],
        "catalog_hash": catalog["catalog_hash"],
        "snapshot": {"revision": 0, **snapshot},
        "metrics": _metrics(record),
        "source": source,
    }


def merge_participants(config, exports, *, existing=None, catalogs=None):
    """Validate exports and merge them; a differing repeat is a conflict, not an update."""
    catalogs = catalogs or catalogs_for(config)
    study = config["study"]
    records = copy.deepcopy((existing or {}).get("records", []))
    index = {
        (item["participant_id"], item["collection_condition"]): item for item in records
    }
    accepted, rejected, unchanged = [], [], []
    for source, raw in exports:
        try:
            record = validate_export(raw, config, catalogs, source=str(source))
        except StudyError as error:
            rejected.append({"source": str(source), "reason": str(error)})
            continue
        key = (record["participant_id"], record["collection_condition"])
        previous = index.get(key)
        if previous is not None:
            same = {k: v for k, v in previous.items() if k != "source"} == {
                k: v for k, v in record.items() if k != "source"
            }
            (unchanged if same else rejected).append(
                {
                    "source": str(source),
                    "participant_id": record["participant_id"],
                    "collection_condition": record["collection_condition"],
                    **(
                        {}
                        if same
                        else {"reason": "conflicting record for this participant"}
                    ),
                }
            )
            continue
        index[key] = record
        records.append(record)
        accepted.append(
            {
                "source": str(source),
                "participant_id": record["participant_id"],
                "collection_condition": record["collection_condition"],
            }
        )
    records.sort(
        key=lambda item: (item["participant_id"], item["collection_condition"])
    )
    wanted = set(CONDITIONS[study["study_kind"]])
    by_participant = {}
    for record in records:
        by_participant.setdefault(record["participant_id"], set()).add(
            record["collection_condition"]
        )
    incomplete = [
        {"participant_id": participant, "missing": sorted(wanted - conditions)}
        for participant, conditions in sorted(by_participant.items())
        if conditions != wanted
    ]
    document = {
        "schema_version": 1,
        "study_id": study["study_id"],
        "study_kind": study["study_kind"],
        "updated_at": _now(),
        "records": records,
    }
    report = {
        "accepted": accepted,
        "unchanged": unchanged,
        "rejected": rejected,
        "participants": len(by_participant),
        "complete_participants": len(by_participant) - len(incomplete),
        "incomplete": incomplete,
    }
    return document, report


def complete_participants(config, participants):
    """Participants whose every condition was collected, in a stable order."""
    wanted = set(CONDITIONS[config["study"]["study_kind"]])
    grouped = {}
    for record in participants.get("records", []):
        grouped.setdefault(record["participant_id"], {})[
            record["collection_condition"]
        ] = record
    return {
        participant: conditions
        for participant, conditions in sorted(grouped.items())
        if set(conditions) == wanted
    }


# ------------------------------------------------------------------ manifest


def derangement(participant_ids, seed):
    """A deterministic permutation with no participant reading their own profile."""
    order = sorted(participant_ids)
    if len(order) < 2:
        raise StudyError(
            "a self-assignment-free permutation needs at least 2 participants"
        )
    generator = random.Random(digest({"seed": seed, "participants": order}))
    shuffled = list(order)
    for _ in range(1000):
        generator.shuffle(shuffled)
        if all(left != right for left, right in zip(order, shuffled)):
            return dict(zip(order, shuffled))
    rotated = order[1:] + order[:1]
    return dict(zip(order, rotated))


def _variant_id(study_id, participant, condition, policy_hash):
    return digest(
        {
            "study_id": study_id,
            "participant_id": participant,
            "condition": condition,
            "policy_hash": policy_hash,
        }
    )[:32]


def _pair_id(study_id, participant, comparison, topic_id, seed):
    return digest(
        {
            "study_id": study_id,
            "participant_id": participant,
            "comparison": comparison,
            "topic_id": topic_id,
            "seed": seed,
        }
    )[:32]


def _balanced_sides(count, seed):
    if count % 2:
        raise StudyError("a comparison needs an even number of pairs to balance sides")
    values = ["left"] * (count // 2) + ["right"] * (count // 2)
    random.Random(digest(seed)).shuffle(values)
    return values


def build_manifest(config, participants, *, catalogs=None):
    """Fix every pair, side, and order before a single image exists."""
    study = config["study"]
    catalogs = catalogs or catalogs_for(config)
    kind = study["study_kind"]
    complete = complete_participants(config, participants)
    minimum = study["participants"]["pilot_minimum"]
    if len(complete) < 2:
        raise StudyError(
            "study manifest needs at least 2 participants; no image job was produced"
        )
    if len(complete) < minimum:
        raise StudyError(
            f"study manifest needs at least {minimum} participants with every condition collected, "
            f"got {len(complete)}; no image job was produced "
            f"(conclusions additionally need {study['participants']['main_minimum']})"
        )
    for participant, conditions in complete.items():
        for condition, record in conditions.items():
            if record["catalog_hash"] != catalogs[condition]["catalog_hash"]:
                raise StudyError(
                    f"participant {participant} was collected against another {condition} catalog"
                )

    assignment = (
        derangement(list(complete), study["derangement_seed"])
        if any("other" in COMPARISON_ROLES[name] for name in study["comparisons"])
        else {}
    )
    variants = {}

    def variant_for(participant, role):
        condition = ROLE_CONDITION[role]
        owner = assignment[participant] if role in OTHER_ROLES else participant
        policy = study["policies"][ROLE_POLICY[role]]
        variant_id = _variant_id(
            study["study_id"], owner, condition, policy["policy_hash"]
        )
        variants.setdefault(
            variant_id,
            {
                "variant_id": variant_id,
                "participant_id": owner,
                "condition": condition,
                "policy_id": policy["policy_id"],
                "policy_hash": policy["policy_hash"],
                "effective_policy": copy.deepcopy(policy["effective_policy"]),
                "catalog_id": complete[owner][condition]["catalog_id"],
                "catalog_hash": complete[owner][condition]["catalog_hash"],
                "snapshot": copy.deepcopy(complete[owner][condition]["snapshot"]),
            },
        )
        return variant_id

    pairs, keys, order = [], {}, {}
    topics = {topic["id"]: topic for topic in study["topics"]}
    for participant in complete:
        participant_pairs = []
        for comparison in study["comparisons"]:
            subject, baseline = COMPARISON_ROLES[comparison]
            rows = [
                (topic_id, seed)
                for topic_id in study["comparisons"][comparison]["topics"]
                for seed in study["comparisons"][comparison]["seeds"]
            ]
            sides = _balanced_sides(
                len(rows),
                {
                    "study_id": study["study_id"],
                    "participant_id": participant,
                    "comparison": comparison,
                },
            )
            for (topic_id, seed), side in zip(rows, sides):
                if topic_id not in topics:
                    raise StudyError(f"unknown study topic: {topic_id}")
                pair_id = _pair_id(
                    study["study_id"], participant, comparison, topic_id, seed
                )
                subject_variant = variant_for(participant, subject)
                baseline_variant = variant_for(participant, baseline)
                left, right = (
                    (subject, baseline) if side == "left" else (baseline, subject)
                )
                left_variant, right_variant = (
                    (subject_variant, baseline_variant)
                    if side == "left"
                    else (baseline_variant, subject_variant)
                )
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "participant_id": participant,
                        "comparison": comparison,
                        "topic_id": topic_id,
                        "seed": seed,
                    }
                )
                keys[pair_id] = {
                    "pair_id": pair_id,
                    "participant_id": participant,
                    "comparison": comparison,
                    "topic_id": topic_id,
                    "seed": seed,
                    "subject_role": subject,
                    "baseline_role": baseline,
                    "subject_side": "A" if side == "left" else "B",
                    "sides": {
                        "A": {"role": left, "variant_id": left_variant},
                        "B": {"role": right, "variant_id": right_variant},
                    },
                }
                participant_pairs.append(pair_id)
        shuffled = list(participant_pairs)
        random.Random(
            digest(
                {
                    "study_id": study["study_id"],
                    "participant_id": participant,
                    "purpose": "presentation_order",
                }
            )
        ).shuffle(shuffled)
        order[participant] = shuffled

    jobs = {}
    for key in keys.values():
        for side in key["sides"].values():
            job_key = (side["variant_id"], key["topic_id"], key["seed"])
            jobs.setdefault(
                job_key,
                {
                    "variant_id": side["variant_id"],
                    "topic_id": key["topic_id"],
                    "seed": key["seed"],
                },
            )
    job_list = sorted(
        jobs.values(),
        key=lambda item: (item["variant_id"], item["topic_id"], item["seed"]),
    )
    maximum = config.get("limits", {}).get("max_images", 512)
    if len(job_list) > maximum:
        raise StudyError(
            f"study needs {len(job_list)} images and exceeds the {maximum} image limit; "
            "split the participants into separate studies"
        )

    identity = {
        "schema_version": 1,
        "study_id": study["study_id"],
        "study_kind": kind,
        "participants": sorted(complete),
        "assignment": assignment,
        "assignment_seed": study["derangement_seed"],
        "participants_required": copy.deepcopy(study["participants"]),
        "bootstrap": copy.deepcopy(study["bootstrap"]),
        "policies": copy.deepcopy(study["policies"]),
        "comparisons": copy.deepcopy(study["comparisons"]),
        "topics": copy.deepcopy(study["topics"]),
        "generation": copy.deepcopy(config["generation"]),
        "variants": [variants[key] for key in sorted(variants)],
        "pairs": sorted(pairs, key=lambda item: item["pair_id"]),
        "presentation_order": order,
        "jobs": job_list,
    }
    manifest = {
        **identity,
        "study_hash": digest(identity),
        "created_at": _now(),
        "image_count": len(job_list),
    }
    key_document = {
        "schema_version": 1,
        "study_id": study["study_id"],
        "study_hash": manifest["study_hash"],
        "note": "方式と左右の対応表。集計器だけが読む。参加者には渡さない。",
        "pairs": keys,
    }
    return manifest, key_document


# -------------------------------------------------------------- answer pages


def image_name(study_id, participant, pair_id, side):
    return (
        digest(
            {
                "study_id": study_id,
                "participant_id": participant,
                "pair_id": pair_id,
                "side": side,
            }
        )[:32]
        + ".png"
    )


def build_answer_pages(config, manifest, keys, images, directory):
    """One opaque page per participant; the method never reaches the client."""
    study = config["study"]
    directory = Path(directory)
    missing = []
    written = []
    for participant in manifest["participants"]:
        payload_pairs = []
        copies = {}
        for pair_id in manifest["presentation_order"][participant]:
            key = keys["pairs"][pair_id]
            entry = {"pair_id": pair_id}
            for side in ("A", "B"):
                variant = key["sides"][side]["variant_id"]
                source = images.get(f"{variant}:{key['topic_id']}:{key['seed']}")
                if not source or not Path(source).is_file():
                    missing.append({"pair_id": pair_id, "side": side})
                    continue
                name = image_name(study["study_id"], participant, pair_id, side)
                copies[name] = source
                entry["a" if side == "A" else "b"] = f"images/{name}"
            if "a" in entry and "b" in entry:
                payload_pairs.append(entry)
        if not payload_pairs:
            continue
        page_dir = directory / "answer" / participant
        (page_dir / "images").mkdir(parents=True, exist_ok=True)
        for name, source in copies.items():
            shutil.copyfile(source, page_dir / "images" / name)
        data = {
            "study_id": study["study_id"],
            "participant_id": participant,
            "pairs": payload_pairs,
            "storage_note": (
                "保存されるのは、匿名の参加者IDと各組の回答だけです。"
                "このページは通信しません。"
            ),
        }
        html = _page(
            "answer.html",
            title="2枚の画像をくらべる",
            subtitle="このページは通信しません。回答は最後にJSONファイルとして保存します。",
            data="globalThis.PAGE = " + _page_json(data) + ";",
            script=inline_module((ASSETS / "answer.mjs").read_text()),
        )
        write_text(page_dir / "index.html", html)
        written.append(
            {
                "participant_id": participant,
                "path": str((page_dir / "index.html").relative_to(directory)),
                "pairs": len(payload_pairs),
            }
        )
    return {"pages": written, "missing_images": missing}


# ------------------------------------------------------------------ scoring


def _score(choice, subject_side):
    if choice == "tie":
        return 0.5
    if choice in ("A", "B"):
        return 1.0 if choice == subject_side else 0.0
    return None


def read_answers(directory):
    """Read every answer export, keeping the reason a file was left out."""
    directory = Path(directory)
    rows, excluded = [], []
    if not directory.is_dir():
        return rows, excluded
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            excluded.append({"file": path.name, "reason": f"unreadable: {error}"})
            continue
        rows.append({"file": path.name, "value": value})
    return rows, excluded


def unstarted_summary(directory):
    """A study whose manifest does not exist yet: 未実施, with no invented number."""
    directory = Path(directory)
    status = read_json(directory / "status.json")
    return {
        "schema_version": 1,
        "study_id": status["study_id"],
        "study_kind": status["study_kind"],
        "study_hash": None,
        "generated_at": _now(),
        "participants_in_manifest": 0,
        "participants_with_answers": 0,
        "participants_required": copy.deepcopy(status["participants_required"]),
        "status": "not_started",
        "reasons": ["study_manifest_missing", status["comparison_images"]],
        "comparisons": {},
        "excluded_files": [],
        "duplicate_answers": [],
        "answer_problems": {},
        "fidelity": {
            "answered_rows": 0,
            "counts": {"subject_kept": {}, "breakage": {}},
            "note": "忠実度の確認であり、好みの勝率には加えない。",
        },
        "default_policy_change": {
            "eligible": False,
            "requires": "両方の比較が improved で、主評価の最低人数を満たすこと",
            "default_policy_id": "legacy_exhibit",
            "note": "この集計はfan-policies.jsonを変更しない。",
        },
    }


def summarize_study(directory, *, participants=None):
    """Average inside a participant first, then across participants."""
    directory = Path(directory)
    if not (directory / "manifest.json").is_file():
        return unstarted_summary(directory)
    manifest = read_json(directory / "manifest.json")
    keys = read_json(directory / "keys.json")
    study_id = manifest["study_id"]
    kind = manifest["study_kind"]
    rows, excluded = read_answers(directory / "answers")
    known = set(manifest["participants"])
    answers = {}
    duplicates = []
    for row in rows:
        value = row["value"]
        if not isinstance(value, dict):
            excluded.append({"file": row["file"], "reason": "not a JSON object"})
            continue
        if value.get("study_id") != study_id:
            excluded.append({"file": row["file"], "reason": "another study_id"})
            continue
        participant = value.get("participant_id")
        if participant not in known:
            excluded.append({"file": row["file"], "reason": "unknown participant_id"})
            continue
        if participant in answers:
            excluded.append(
                {
                    "file": row["file"],
                    "reason": "a second answer file for this participant",
                }
            )
            continue
        seen, choices, voided = set(), {}, set()
        problems = []
        for item in value.get("answers", []):
            if not isinstance(item, dict):
                problems.append({"reason": "invalid answer row"})
                continue
            pair_id = item.get("pair_id")
            if not isinstance(pair_id, str) or pair_id not in keys["pairs"]:
                problems.append({"pair_id": pair_id, "reason": "unknown pair_id"})
                continue
            if keys["pairs"][pair_id]["participant_id"] != participant:
                problems.append(
                    {"pair_id": pair_id, "reason": "pair of another participant"}
                )
                continue
            if pair_id in seen:
                duplicates.append({"participant_id": participant, "pair_id": pair_id})
                if pair_id in voided or choices.get(pair_id) != item.get("choice"):
                    # Two different answers to one pair: neither is the answer.
                    voided.add(pair_id)
                    choices.pop(pair_id, None)
                    reason = "conflicting duplicate answer"
                else:
                    reason = "duplicate answer"
                if {"pair_id": pair_id, "reason": reason} not in problems:
                    problems.append({"pair_id": pair_id, "reason": reason})
                continue
            seen.add(pair_id)
            choices[pair_id] = item.get("choice")
        answers[participant] = {
            "file": row["file"],
            "choices": choices,
            "problems": problems,
            "fidelity": value.get("fidelity", []),
        }

    comparisons = {}
    for comparison in manifest["comparisons"]:
        subject, baseline = COMPARISON_ROLES[comparison]
        pairs = [pair for pair in manifest["pairs"] if pair["comparison"] == comparison]
        by_participant = {}
        unanswered, missing = [], []
        for pair in pairs:
            participant = pair["participant_id"]
            record = answers.get(participant)
            if record is None:
                missing.append(pair["pair_id"])
                continue
            if pair["pair_id"] not in record["choices"]:
                missing.append(pair["pair_id"])
                continue
            value = _score(
                record["choices"][pair["pair_id"]],
                keys["pairs"][pair["pair_id"]]["subject_side"],
            )
            if value is None:
                unanswered.append(pair["pair_id"])
                continue
            by_participant.setdefault(participant, []).append(value)
        expected = {}
        for pair in pairs:
            expected[pair["participant_id"]] = (
                expected.get(pair["participant_id"], 0) + 1
            )
        # Only a participant who answered every pair of this comparison is a unit;
        # a few favourable answers never stand in for a whole person.
        means, incomplete = {}, {}
        for participant, values in sorted(by_participant.items()):
            if len(values) == expected[participant]:
                means[participant] = sum(values) / len(values)
            else:
                incomplete[participant] = {
                    "answered": len(values),
                    "expected": expected[participant],
                    "partial_mean": sum(values) / len(values),
                }
        summary = {
            "comparison": comparison,
            "subject_role": subject,
            "baseline_role": baseline,
            "pair_count": len(pairs),
            "answered_pairs": sum(len(values) for values in by_participant.values()),
            "unanswered_pairs": sorted(unanswered),
            "missing_pairs": sorted(missing),
            "participants_scored": len(means),
            "participant_means": means,
            "incomplete_participants": incomplete,
            "mean": None,
            "interval": None,
            "conclusion": "undetermined",
            "reasons": [],
        }
        required = manifest["participants_required"]["main_minimum"]
        if incomplete:
            summary["reasons"].append("incomplete_participants_excluded")
        if not means:
            summary["reasons"].append(
                "no_complete_participant" if by_participant else "no_answers"
            )
        else:
            bootstrap = manifest["bootstrap"]
            summary["mean"] = sum(means.values()) / len(means)
            summary["interval"] = participant_bootstrap_interval(
                means,
                draws=bootstrap["draws"],
                seed=bootstrap["seed"],
                confidence=bootstrap["confidence"],
            )
            if len(means) < required:
                summary["reasons"].append("insufficient_participants")
            elif summary["mean"] > 0.5 and summary["interval"]["lower"] > 0.5:
                summary["conclusion"] = "improved"
            else:
                summary["conclusion"] = "not_confirmed"
        comparisons[comparison] = summary

    # Only the encoder study compares policies; both of its comparisons must agree.
    eligible = (
        kind == "encoder"
        and set(comparisons) == {"candidate_vs_legacy", "own_vs_other"}
        and all(item["conclusion"] == "improved" for item in comparisons.values())
    )
    result = {
        "schema_version": 1,
        "study_id": study_id,
        "study_kind": kind,
        "study_hash": manifest["study_hash"],
        "generated_at": _now(),
        "participants_in_manifest": len(manifest["participants"]),
        "participants_with_answers": len(answers),
        "participants_required": copy.deepcopy(manifest["participants_required"]),
        "status": "not_started" if not answers else "collected",
        "comparisons": comparisons,
        "excluded_files": excluded,
        "duplicate_answers": duplicates,
        "answer_problems": {
            participant: record["problems"]
            for participant, record in sorted(answers.items())
            if record["problems"]
        },
        "fidelity": _fidelity(manifest, keys, answers),
        "default_policy_change": {
            "eligible": eligible,
            "requires": "両方の比較が improved で、主評価の最低人数を満たすこと",
            "default_policy_id": manifest["policies"]
            .get("legacy", {})
            .get("policy_id", "legacy_exhibit"),
            "note": "この集計はfan-policies.jsonを変更しない。",
        },
    }
    if kind == "elicitation":
        result["collection_effect"] = _collection_effect(
            manifest, comparisons, participants
        )
    return result


def _fidelity(manifest, keys, answers):
    """Kept apart from the preference score on purpose."""
    counts = {"subject_kept": {}, "breakage": {}}
    rows = 0
    for participant, record in answers.items():
        for item in record["fidelity"] if isinstance(record["fidelity"], list) else []:
            if not isinstance(item, dict):
                continue
            pair_id = item.get("pair_id")
            key = keys["pairs"].get(pair_id)
            if key is None or key["participant_id"] != participant:
                continue
            rows += 1
            for field in ("subject_kept", "breakage"):
                value = item.get(field)
                if value is None:
                    continue
                label = value
                if value in ("A", "B"):
                    label = "subject" if value == key["subject_side"] else "baseline"
                counts[field][label] = counts[field].get(label, 0) + 1
    return {
        "answered_rows": rows,
        "counts": counts,
        "note": "忠実度の確認であり、好みの勝率には加えない。",
    }


def _collection_effect(manifest, comparisons, participants):
    """Design §9.7: the effect of the pool plus the whole selection flow."""
    metrics = {}
    for record in (participants or {}).get("records", []):
        condition = record["collection_condition"]
        bucket = metrics.setdefault(
            condition,
            {
                "participants": 0,
                "mean_elapsed_ms": None,
                "mean_selection_count": None,
                "mean_explicit_aspects": None,
                "_elapsed": [],
                "_selection": [],
                "_aspects": [],
            },
        )
        bucket["participants"] += 1
        if "elapsed_ms" in record.get("metrics", {}):
            bucket["_elapsed"].append(record["metrics"]["elapsed_ms"])
        selection = record["snapshot"]["selection"]
        bucket["_selection"].append(len(selection))
        bucket["_aspects"].append(sum(len(item["aspects"]) for item in selection))
    for bucket in metrics.values():
        for name, values in (
            ("mean_elapsed_ms", bucket.pop("_elapsed")),
            ("mean_selection_count", bucket.pop("_selection")),
            ("mean_explicit_aspects", bucket.pop("_aspects")),
        ):
            bucket[name] = sum(values) / len(values) if values else None
    comparison = comparisons.get("new_pool_vs_legacy_pool", {})
    return {
        "by_condition": metrics,
        # The rate comes only from participants who answered every pair, while the
        # selection metrics come from everyone who was collected: different scopes.
        "preference_rate_new_pool": comparison.get("mean"),
        "participants_scored": comparison.get("participants_scored", 0),
        "incomplete_participants": sorted(
            comparison.get("incomplete_participants", {})
        ),
        "effect_label": "pool + selection flow",
        "note": "差は「pool＋選択フロー全体」の効果であり、枚数増加だけの効果とは呼ばない。",
        "scope_note": (
            "選択の指標は収集できた全員分、選好率は全組に回答した参加者のみ。"
            "一部だけ回答した参加者は選好率に含めず、incomplete_participants に残す。"
        ),
    }


NOT_GENERATED = "not_generated"
NOT_GENERATED_JA = "比較画像未生成"


def status_document(config, *, participants=None, manifest=None, images=0, pages=0):
    """What exists right now; an absent stage is named, never filled in."""
    study = config["study"]
    complete = len(complete_participants(config, participants)) if participants else 0
    if images and manifest and images >= manifest["image_count"]:
        images_state = "generated"
    elif images:
        images_state = "partial"
    else:
        images_state = NOT_GENERATED
    return {
        "schema_version": 1,
        "study_id": study["study_id"],
        "study_kind": study["study_kind"],
        "study_dir": study["study_dir"],
        "updated_at": _now(),
        "collection_page": "collect/index.html",
        "instructions": "README.md",
        "participants": complete,
        "participants_required": copy.deepcopy(study["participants"]),
        "study_manifest": "manifest.json" if manifest else None,
        "planned_images": manifest["image_count"] if manifest else None,
        "generated_images": images,
        "comparison_images": images_state,
        "comparison_images_ja": (
            NOT_GENERATED_JA if images_state == NOT_GENERATED else images_state
        ),
        "answer_pages": pages,
    }


def instructions_markdown(config, *, status):
    """The Japanese 実施説明 an operator follows from collection to summary."""
    study = config["study"]
    kind = study["study_kind"]
    conditions = "、".join(CONDITIONS[kind])
    minimum = study["participants"]
    return "\n".join(
        [
            f"# 選好評価の実施手順（{kind} study）",
            "",
            f"- study_id: `{study['study_id']}`",
            f"- 保存先: `{study['study_dir']}`",
            f"- 収集する条件: {conditions}",
            (
                f"- 人数: pilot {minimum['pilot_minimum']}人（操作確認のみ）、"
                f"本評価 {minimum['main_minimum']}人以上（結論はこちらが必要）"
            ),
            f"- 比較画像: {status['comparison_images_ja']}",
            "",
            "## 0. 原則",
            "",
            "- 参加者の回答を実施者や agent が代わりに作らない。未取得は未取得のまま残す。",
            (
                "- 参加者IDは匿名の記号だけを使う（形式: "
                f"{PARTICIPANT_ID_LABEL}）。氏名・メールアドレスは受け取らない。"
            ),
            (
                "- ページは通信しない。ブラウザーで `collect/index.html` を直接開くか、"
                "`python -m http.server` のような静的配信で開く。"
            ),
            "",
            "## 1. 好みの収集",
            "",
            "1. 参加者ごとに未使用のIDを決め、`collect/index.html` を開いてもらう。",
            *(
                [
                    (
                        "2. 実施順は半数を「旧 → 新」、残り半数を「新 → 旧」にする"
                        "（p001, p003, … は旧→新、p002, p004, … は新→旧）。"
                    ),
                    "3. 2つの条件それぞれでJSONを保存してもらい、2ファイルを受け取る。",
                ]
                if kind == "elicitation"
                else ["2. 選び終えたらJSONを保存してもらい、1ファイルを受け取る。"]
            ),
            "",
            "```bash",
            "PYTHONPATH=exhibit/src exhibit/.venv/bin/python \\",
            "  exhibit/scripts/build_preference_study.py --config exhibit/configs/fan-evaluation.json \\",
            f"  --study-kind {kind} --merge /path/to/*.json",
            "```",
            "",
            "受理・却下の内訳は `merge-report.json`、確定した入力は `participants.json` に残る。",
            "",
            "## 2. 比較画像の生成",
            "",
            "人数が足りたら manifest を作り、GPU で生成する。",
            "",
            "```bash",
            "PYTHONPATH=exhibit/src exhibit/.venv/bin/python \\",
            "  exhibit/scripts/build_preference_study.py --config exhibit/configs/fan-evaluation.json \\",
            f"  --study-kind {kind}",
            "PYTHONPATH=exhibit/src exhibit/.venv/bin/python \\",
            "  exhibit/scripts/evaluate_fan.py study --config exhibit/configs/fan-evaluation.json \\",
            f"  --study-kind {kind} --resume",
            "```",
            "",
            "## 3. 回答",
            "",
            (
                "画像ができたら、もう一度 build_preference_study.py を実行すると "
                "`answer/<participant_id>/index.html` ができる。"
            ),
            "参加者には自分のフォルダーだけを渡す。`manifest.json` と `keys.json` は渡さない。",
            "回答JSONは `answers/` に集める。",
            "",
            "## 4. 集計",
            "",
            "```bash",
            "PYTHONPATH=exhibit/src exhibit/.venv/bin/python \\",
            f"  exhibit/scripts/summarize_preference_study.py --study {study['study_dir']}",
            "```",
            "",
            (
                "勝ち1・同点0.5・負け0を参加者内で平均し、その後で参加者間平均を取る。"
                "95%区間は参加者単位のbootstrap "
                f"{study['bootstrap']['draws']}回（seed={study['bootstrap']['seed']}）。"
            ),
            (
                "- その比較の全組に回答した参加者だけを平均に入れる。"
                "一部だけ回答した参加者は `incomplete_participants` に人数・回答数・"
                "その範囲の平均を残し、黙って捨てない。"
            ),
            (
                "- 同じ組に違う回答が2件あるとその組は無効になり、"
                "`conflicting duplicate answer` として記録する"
                "（同じ回答の重複は1件として数える）。無効になった組がある参加者は未完了扱い。"
            ),
            "回答が1件もなければ集計は「未実施」と書く。仮の勝率は作らない。",
            "",
            "## 参加者に渡してよいファイル",
            "",
            "- `collect/index.html` と `collect/images/`",
            "- `answer/<participant_id>/`（自分の分だけ）",
            "",
            "## 渡してはいけないファイル",
            "",
            "- `manifest.json`（方式・割り当て）、`keys.json`（方式と左右の対応表）",
            "- `participants.json`、ほかの参加者の `answer/` フォルダー",
            "",
        ]
    )


CONCLUSION_JA = {
    "improved": "改善を確認",
    "not_confirmed": "改善を確認できず",
    "undetermined": "未確定",
}


def summary_markdown(summary):
    """A short Japanese report; an empty study says 未実施 and nothing else."""
    lines = [
        f"# 選好評価の集計（{summary['study_kind']}）",
        "",
        f"- study_id: `{summary['study_id']}`",
        f"- study_hash: `{summary['study_hash'] or '未作成'}`",
        f"- 集計時刻: {summary['generated_at']}",
        (
            f"- 参加者: manifest {summary['participants_in_manifest']}人 / 回答あり "
            f"{summary['participants_with_answers']}人"
            f"（主評価の最低人数 {summary['participants_required']['main_minimum']}人）"
        ),
        "",
    ]
    if summary["status"] == "not_started":
        lines += [
            "## 結果",
            "",
            "**未実施**。回答ファイルが1件もないため、勝率も区間も算出しない。",
            "",
        ]
    else:
        lines += ["## 比較ごとの結果", ""]
        for name, item in summary["comparisons"].items():
            mean = item["mean"]
            interval = item["interval"]
            lines.append(f"### {name}")
            lines.append("")
            lines.append(
                f"- 判定: **{CONCLUSION_JA[item['conclusion']]}**"
                + (f"（{'・'.join(item['reasons'])}）" if item["reasons"] else "")
            )
            lines.append(
                "- 参加者内平均→参加者間平均: "
                + ("未算出" if mean is None else f"{mean:.3f}")
                + f"（{item['participants_scored']}人）"
            )
            if interval:
                lines.append(
                    f"- 95%区間（参加者bootstrap {interval['draws']}回, seed={interval['seed']}）: "
                    f"{interval['lower']:.3f} 〜 {interval['upper']:.3f}"
                )
            lines.append(
                f"- 回答済み {item['answered_pairs']} / {item['pair_count']} 組"
                f"、未回答 {len(item['unanswered_pairs'])} 組、欠損 {len(item['missing_pairs'])} 組"
            )
            incomplete = item["incomplete_participants"]
            lines.append(
                "- 平均に入れるのは全組に回答した参加者だけ。"
                + (
                    f"一部だけ回答した参加者 {len(incomplete)}人は平均・区間から除外し、"
                    "`incomplete_participants` に残す（黙って捨てない）。"
                    if incomplete
                    else "この比較では未完了の参加者はいない。"
                )
            )
            for participant, value in sorted(incomplete.items()):
                lines.append(
                    f"  - {participant}: {value['answered']} / {value['expected']} 組"
                    f"（この範囲だけの平均 {value['partial_mean']:.3f}、集計には使わない）"
                )
            lines.append("")
    change = summary["default_policy_change"]
    lines += [
        "## 既定値の変更条件",
        "",
        f"- 条件: {change['requires']}",
        f"- 現時点の該当: {'満たす' if change['eligible'] else '満たさない'}",
        f"- 既定 policy は `{change['default_policy_id']}` のまま。{change['note']}",
        "",
        "## 欠損・除外",
        "",
        f"- 除外ファイル: {len(summary['excluded_files'])}件",
        f"- 重複回答: {len(summary['duplicate_answers'])}件",
        f"- 回答内の問題: {len(summary['answer_problems'])}人分",
        (
            "- 同じ組に違う回答が2件あるときは、その組を無効にして "
            "`conflicting duplicate answer` として記録する。"
            "同じ回答の重複は1件として数え、`duplicate answer` として記録する。"
        ),
        (
            "- 無効・未回答になった組を持つ参加者はその比較で未完了になり、"
            "平均には入らず人数と回答数だけが残る。"
        ),
        "",
        "## 忠実度の確認",
        "",
        f"- 回答行: {summary['fidelity']['answered_rows']}件。{summary['fidelity']['note']}",
        "",
    ]
    if "collection_effect" in summary:
        effect = summary["collection_effect"]
        lines += [
            "## 収集フローの効果",
            "",
            f"- 効果のラベル: {effect['effect_label']}",
            f"- {effect['note']}",
            "",
        ]
        lines.append(f"- {effect['scope_note']}")
        rate = effect["preference_rate_new_pool"]
        lines.append(
            "- 新poolの選好率: "
            + ("未算出" if rate is None else f"{rate:.3f}")
            + f"（全組に回答した {effect['participants_scored']}人、"
            f"未完了 {len(effect['incomplete_participants'])}人）"
        )
        for condition, item in effect["by_condition"].items():
            lines.append(
                f"- {condition}: {item['participants']}人、平均選択枚数 "
                f"{item['mean_selection_count']}、平均の明示側面数 "
                f"{item['mean_explicit_aspects']}、平均所要 {item['mean_elapsed_ms']}ms"
            )
        lines.append("")
    return "\n".join(lines)
