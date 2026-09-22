#!/usr/bin/env python3
"""Assemble docs/reports/fan-personalization/strength/ from the strength runs.

Every declared experiment of ``configs/fan-strength.json`` is reported from the
artefacts that are actually on disk: a declared experiment without a run is
未実施, a registered run without ``records.json`` is 実行中. Nothing is
fabricated -- no score, no judgement, no human answer.

The report holds three things: a Markdown summary with side-by-side contact
sheets, a blinded pair set an AI judge can answer, and ``review.html`` for a
blind human review. Judge answers are merged only from the files given with
``--judge-answers``.

``--extra-run <dir>=<label>`` renders the same sheets for a run of the formal
screen / refine / heldout chain, which carries no ``display`` block. Those runs
are for looking at (plan item P3) and never enter the judge set.

    PYTHONPATH=exhibit/src exhibit/.venv/bin/python \
        exhibit/scripts/build_strength_report.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from exhibit.evaluation import (
    list_strength_experiments,
    load_strength_config,
)
from PIL import Image, ImageDraw, ImageFont

REPO = PROJECT.parent
DEFAULT_CONFIG = PROJECT / "configs/fan-strength.json"
DEFAULT_OUTPUTS = PROJECT / "outputs/fan-evaluation/strength"
DEFAULT_REPORT = REPO / "docs/reports/fan-personalization/strength"

NOT_RUN, RUNNING, DONE = "未実施", "実行中", "完了"
VERDICT = {"pass": "合格", "fail": "不合格", "unmeasured": "未計測"}

TILE = (320, 400)
OVERVIEW_TILE = (192, 240)
PAIR_TILE = (448, 560)
PAIR_GUTTER = 8
HEADER_H = 24
CAPTION_H = 34
OVERVIEW_CAPTION_H = 30
JPEG_QUALITY = 85
DASH = "—"


# ------------------------------------------------------------------ 小さな道具


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def number(value, digits=4):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return DASH
    return f"{value:+.{digits}f}"


def plain_number(value, digits=4):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return DASH
    return f"{value:.{digits}f}"


def interval(value):
    """A bootstrap row as `mean [lower, upper]`; absent stays absent."""
    if not isinstance(value, dict):
        return ""
    lower, upper = value.get("lower"), value.get("upper")
    if lower is None or upper is None:
        return ""
    return f" [{number(lower)}, {number(upper)}]"


def ascii_text(value):
    """Captions use PIL's default font, so keep them to plain ASCII."""
    text = "" if value is None else str(value)
    return "".join(char if 32 <= ord(char) < 127 else "?" for char in text)


def relative_to_repo(path):
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def timestamp():
    return datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S JST")


def pair_identifier(experiment_id, job_id):
    raw = f"{experiment_id}:{job_id}".encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow: a fixed bitmap face
        return ImageFont.load_default()


def history_text(history):
    """Every reference of one history with its weight, in fixture order."""
    return " / ".join(
        f"{ref.get('text', '')}（重み {plain_number(ref.get('weight'), 1)}）"
        for ref in history.get("refs", [])
    )


# ---------------------------------------------------------------- 画像の組み立て


def tile_image(path, size):
    """One image at an exact size; a missing file stays visibly missing."""
    try:
        image = Image.open(path).convert("RGB")
    except (OSError, ValueError):
        image = Image.new("RGB", size, (48, 48, 48))
        draw = ImageDraw.Draw(image)
        draw.text((8, size[1] // 2 - 8), "MISSING IMAGE", font=font(16), fill=(230, 230, 230))
        return image
    return image.resize(size, Image.LANCZOS)


def draw_caption(canvas, box, lines, *, size=13):
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, fill=(20, 20, 20))
    face = font(size)
    for index, line in enumerate(lines[:2]):
        draw.text(
            (box[0] + 6, box[1] + 4 + index * (size + 2)),
            ascii_text(line),
            font=face,
            fill=(240, 240, 240),
        )


def contact_sheet(columns, header, *, tile=TILE, caption_height=CAPTION_H):
    """Columns left to right, each with a two-line caption strip underneath."""
    width = tile[0] * len(columns)
    height = HEADER_H + tile[1] + caption_height
    canvas = Image.new("RGB", (max(width, tile[0]), height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), ascii_text(header), font=font(14), fill=(235, 235, 235))
    for index, column in enumerate(columns):
        left = index * tile[0]
        canvas.paste(tile_image(column["path"], tile), (left, HEADER_H))
        draw_caption(
            canvas,
            (left, HEADER_H + tile[1], left + tile[0] - 1, height - 1),
            column["caption"],
        )
    return canvas


def pair_sheet(left_path, right_path):
    """Two tiles labelled A and B; which is the candidate is not visible here."""
    width = PAIR_TILE[0] * 2 + PAIR_GUTTER
    canvas = Image.new("RGB", (width, PAIR_TILE[1]), (255, 255, 255))
    canvas.paste(tile_image(left_path, PAIR_TILE), (0, 0))
    canvas.paste(tile_image(right_path, PAIR_TILE), (PAIR_TILE[0] + PAIR_GUTTER, 0))
    draw = ImageDraw.Draw(canvas)
    face = font(34)
    for label, left in (("A", 0), ("B", PAIR_TILE[0] + PAIR_GUTTER)):
        draw.rectangle((left + 8, 8, left + 52, 52), fill=(255, 255, 255))
        draw.text((left + 20, 12), label, font=face, fill=(20, 20, 20))
    return canvas


def save_jpeg(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=JPEG_QUALITY)
    return path


# -------------------------------------------------------------- 実行結果の探索


def run_finished_at(checkpoint):
    runs = (checkpoint or {}).get("runs") or []
    if not runs:
        return (checkpoint or {}).get("created_at") or ""
    return runs[-1].get("finished_at") or runs[-1].get("started_at") or ""


def load_run(directory):
    manifest = read_json(directory / "manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("identity"), dict):
        return None
    checkpoint = read_json(directory / "checkpoint.json", {})
    return {
        "directory": directory,
        "manifest": manifest,
        "checkpoint": checkpoint if isinstance(checkpoint, dict) else {},
        "records": read_json(directory / "records.json"),
        "metrics": read_json(directory / "metrics.json"),
        "decision": read_json(directory / "decision.json"),
        "finished_at": run_finished_at(checkpoint),
    }


def discover_runs(outputs):
    """Group registered runs by their declared experiment id; newest wins."""
    found = {}
    outputs = Path(outputs)
    if not outputs.is_dir():
        return found
    for directory in sorted(outputs.iterdir()):
        if not directory.is_dir():
            continue
        run = load_run(directory)
        if run is None:
            continue
        experiment_id = (run["manifest"].get("display") or {}).get("experiment_id")
        if not experiment_id:
            continue
        found.setdefault(experiment_id, []).append(run)
    result = {}
    for experiment_id, runs in found.items():
        runs.sort(key=lambda item: (item["finished_at"], item["directory"].name))
        result[experiment_id] = {"run": runs[-1], "superseded": runs[:-1]}
    return result


# ------------------------------------------------------------------ 索引と集計


def index_records(records):
    """Role-keyed lookups; an absent records.json leaves every lookup empty."""
    plain, legacy, candidate = {}, {}, {}
    for record in records or []:
        key_seed = (record.get("topic_id"), record.get("seed"))
        if record.get("role") == "plain":
            plain[key_seed] = record
        elif record.get("role") == "legacy":
            legacy[(record.get("topic_id"), record.get("history_id"), record.get("seed"))] = record
        elif record.get("role") == "candidate":
            candidate[
                (
                    record.get("policy_hash"),
                    record.get("topic_id"),
                    record.get("history_id"),
                    record.get("seed"),
                )
            ] = record
    return {"plain": plain, "legacy": legacy, "candidate": candidate}


def policy_columns(config, manifest):
    """The run's candidate policies in declared order, plus the ones it lacks.

    A policy declared after the run was made has no images and no scores, so it
    is named as missing rather than shown as an empty column.
    """
    labels = manifest.get("policy_labels") or {}
    effective = {
        item["policy_hash"]: item["effective_policy"]
        for item in manifest.get("identity", {}).get("policies", [])
    }
    columns, seen, missing = [], set(), []
    for item in (config or {}).get("policies", []):
        policy_hash = item["policy_hash"]
        seen.add(policy_hash)
        if policy_hash not in effective:
            missing.append(item.get("policy_id") or policy_hash[:12])
            continue
        columns.append(
            {
                "policy_id": item.get("policy_id") or labels.get(policy_hash, policy_hash[:12]),
                "policy_hash": policy_hash,
                "effective_policy": effective[policy_hash],
                "declared": True,
            }
        )
    for policy_hash, policy in effective.items():
        if policy_hash in seen:
            continue
        columns.append(
            {
                "policy_id": labels.get(policy_hash, policy_hash[:12]),
                "policy_hash": policy_hash,
                "effective_policy": policy,
                "declared": False,
            }
        )
    return columns, missing


def delta_caption(record):
    if not isinstance(record, dict):
        return "no record"
    delta = record.get("delta_vs_legacy") or {}
    if not delta:
        status = record.get("status", "unmeasured")
        return f"vs legacy: {status}"
    return (
        f"dHist {number(delta.get('history_score'))}"
        f" dTgt {number(delta.get('target_score'))}"
    )


def case_columns(run, lookups, columns, topic_id, history_id, seed):
    """plain | legacy_exhibit | each candidate policy, in that fixed order."""
    images = run["directory"] / "images"
    labels = run["manifest"].get("policy_labels") or {}
    legacy_hash = run["manifest"]["identity"].get("legacy_policy", {}).get("policy_hash")
    jobs = {}
    for job in run["manifest"]["identity"].get("jobs", []):
        role = job.get("role")
        column = job.get("policy_hash") if role == "candidate" else role
        jobs[(column, job.get("topic_id"), job.get("history_id"), job.get("seed"))] = job[
            "job_id"
        ]
    plain_job = jobs.get(("plain", topic_id, None, seed))
    legacy_job = jobs.get(("legacy", topic_id, history_id, seed))
    result = [
        {
            "path": images / f"{plain_job}.png" if plain_job else images / "missing.png",
            "caption": ["plain (no personalization)", "baseline"],
            "job_id": plain_job,
        },
        {
            "path": images / f"{legacy_job}.png" if legacy_job else images / "missing.png",
            "caption": [
                ascii_text(labels.get(legacy_hash, "legacy_exhibit")),
                "reference policy",
            ],
            "job_id": legacy_job,
        },
    ]
    for column in columns:
        job_id = jobs.get((column["policy_hash"], topic_id, history_id, seed))
        record = lookups["candidate"].get(
            (column["policy_hash"], topic_id, history_id, seed)
        )
        result.append(
            {
                "path": images / f"{job_id}.png" if job_id else images / "missing.png",
                "caption": [ascii_text(column["policy_id"]), delta_caption(record)],
                "job_id": job_id,
                "policy_id": column["policy_id"],
                "policy_hash": column["policy_hash"],
                "record": record,
            }
        )
    return result


def build_sheets(run, lookups, columns, directory):
    """One contact sheet per (topic, history, seed) plus a compact overview."""
    identity = run["manifest"]["identity"]
    topics = [topic["id"] for topic in identity.get("topics", [])]
    histories = [history["id"] for history in identity.get("histories", [])]
    seeds = list(identity.get("seeds", []))
    sheets = []
    for topic_id in topics:
        for history_id in histories:
            for seed in seeds:
                cells = case_columns(run, lookups, columns, topic_id, history_id, seed)
                header = f"topic={topic_id}  history={history_id}  seed={seed}"
                name = f"sheet-{topic_id}-{history_id}-seed{seed}.jpg"
                save_jpeg(contact_sheet(cells, header), directory / name)
                sheets.append(
                    {
                        "file": name,
                        "topic_id": topic_id,
                        "history_id": history_id,
                        "seed": seed,
                        "columns": len(cells),
                    }
                )
    overview = None
    if topics and histories and seeds:
        topic_id = "cat" if "cat" in topics else topics[0]
        history_id = "warm" if "warm" in histories else histories[0]
        seed = seeds[0]
        cells = case_columns(run, lookups, columns, topic_id, history_id, seed)
        header = f"overview  topic={topic_id}  history={history_id}  seed={seed}"
        save_jpeg(
            contact_sheet(
                cells,
                header,
                tile=OVERVIEW_TILE,
                caption_height=OVERVIEW_CAPTION_H,
            ),
            directory / "overview.jpg",
        )
        overview = {
            "file": "overview.jpg",
            "topic_id": topic_id,
            "history_id": history_id,
            "seed": seed,
        }
    return sheets, overview


# ---------------------------------------------------------------- 判定用ペア


def build_judge_material(experiment_id, run, lookups, columns, directory):
    """A blinded plain-vs-candidate pair per candidate record, plus its key."""
    identity = run["manifest"]["identity"]
    targets = {topic["id"]: topic.get("target_text", "") for topic in identity.get("topics", [])}
    histories = {history["id"]: history for history in identity.get("histories", [])}
    labels = {column["policy_hash"]: column["policy_id"] for column in columns}
    images = run["directory"] / "images"
    plain_jobs = {
        (job["topic_id"], job["seed"]): job["job_id"]
        for job in identity.get("jobs", [])
        if job.get("role") == "plain"
    }
    entries = []
    for job in identity.get("jobs", []):
        if job.get("role") != "candidate":
            continue
        plain_job = plain_jobs.get((job.get("topic_id"), job.get("seed")))
        candidate_path = images / f"{job['job_id']}.png"
        plain_path = images / f"{plain_job}.png" if plain_job else None
        if plain_path is None or not plain_path.is_file() or not candidate_path.is_file():
            continue
        entries.append(
            {
                "pair_id": pair_identifier(experiment_id, job["job_id"]),
                "job_id": job["job_id"],
                "plain_path": plain_path,
                "candidate_path": candidate_path,
                "policy_hash": job.get("policy_hash"),
                "policy_id": labels.get(job.get("policy_hash"), job.get("policy_hash", "")[:12]),
                "topic_id": job.get("topic_id"),
                "history_id": job.get("history_id"),
                "seed": job.get("seed"),
            }
        )
    entries.sort(key=lambda item: item["pair_id"])
    generator = random.Random(0)
    tasks, key = [], {}
    for entry in entries:
        candidate_left = generator.random() < 0.5
        left = entry["candidate_path"] if candidate_left else entry["plain_path"]
        right = entry["plain_path"] if candidate_left else entry["candidate_path"]
        name = f"pairs/{entry['pair_id']}.jpg"
        save_jpeg(pair_sheet(left, right), directory / name)
        history = histories.get(entry["history_id"], {})
        tasks.append(
            {
                "pair_id": entry["pair_id"],
                "image": name,
                "history_id": entry["history_id"],
                "history_text": history_text(history),
                "target_text": targets.get(entry["topic_id"], ""),
            }
        )
        key[entry["pair_id"]] = {
            "candidate_side": "A" if candidate_left else "B",
            "policy_id": entry["policy_id"],
            "policy_hash": entry["policy_hash"],
            "topic_id": entry["topic_id"],
            "history_id": entry["history_id"],
            "seed": entry["seed"],
            "job_id": entry["job_id"],
        }
    write_json(directory / "tasks.json", {"experiment_id": experiment_id, "tasks": tasks})
    write_json(directory / "key.json", key)
    return tasks, key


def load_answer_files(paths):
    """Judge answers exactly as given; an unreadable file is reported, not guessed."""
    judges, problems = [], []
    for name in paths:
        value = read_json(Path(name))
        if not isinstance(value, dict) or not isinstance(value.get("answers"), dict):
            problems.append(f"読めない回答ファイル: {name}")
            continue
        judges.append(
            {
                "judge": str(value.get("judge") or Path(name).stem),
                "kind": "human" if value.get("kind") == "human" else "ai",
                "source": str(name),
                "answers": value["answers"],
            }
        )
    return judges, problems


def merge_judgements(key, judges):
    """Win rate vs plain, target retention and inter-judge agreement, per policy."""
    policies = {}
    for entry in key.values():
        policies.setdefault(
            entry["policy_id"],
            {
                "policy_id": entry["policy_id"],
                "policy_hash": entry["policy_hash"],
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "decided": 0,
                "win_rate": None,
                "target_yes": 0,
                "target_answers": 0,
                "target_kept_rate": None,
            },
        )
    preferences = {}
    used_judges = []
    for judge in judges:
        touched = False
        for pair_id, answer in judge["answers"].items():
            entry = key.get(pair_id)
            if entry is None or not isinstance(answer, dict):
                continue
            touched = True
            row = policies[entry["policy_id"]]
            side = entry["candidate_side"]
            other = "B" if side == "A" else "A"
            preference = answer.get("preference")
            if preference == side:
                row["wins"] += 1
                preferences.setdefault(pair_id, []).append("candidate")
            elif preference == other:
                row["losses"] += 1
                preferences.setdefault(pair_id, []).append("plain")
            elif preference == "tie":
                row["ties"] += 1
                preferences.setdefault(pair_id, []).append("tie")
            kept = answer.get("target_kept")
            if isinstance(kept, dict) and kept.get(side) in {"yes", "partly", "no"}:
                row["target_answers"] += 1
                if kept[side] == "yes":
                    row["target_yes"] += 1
        if touched:
            used_judges.append(judge["judge"])
    for row in policies.values():
        row["decided"] = row["wins"] + row["losses"]
        if row["decided"]:
            row["win_rate"] = row["wins"] / row["decided"]
        if row["target_answers"]:
            row["target_kept_rate"] = row["target_yes"] / row["target_answers"]
    multi = {pair: values for pair, values in preferences.items() if len(values) > 1}
    agreed = sum(1 for values in multi.values() if len(set(values)) == 1)
    agreement = {
        "pairs_with_multiple_judges": len(multi),
        "pairs_in_agreement": agreed,
        "rate": (agreed / len(multi)) if multi else None,
    }
    return {
        "judges": used_judges,
        "answered_pairs": len(preferences),
        "pair_count": len(key),
        "policies": [policies[name] for name in sorted(policies)],
        "agreement": agreement,
    }


# ------------------------------------------------------------------- Markdown


def policy_rows(columns, run):
    """One row per policy, richest first by mean Δhistory against legacy."""
    metrics = (run or {}).get("metrics") or {}
    by_policy = metrics.get("policies") or {}
    decision = (run or {}).get("decision") or {}
    verdicts = {item.get("policy_hash"): item for item in decision.get("candidates") or []}
    rows = []
    for column in columns:
        summary = by_policy.get(column["policy_hash"]) or {}
        bootstrap = summary.get("bootstrap_95") or {}
        candidate = verdicts.get(column["policy_hash"]) or {}
        policy = column["effective_policy"] or {}
        profiling = policy.get("profiling") or {}
        profile = profiling.get("mode", DASH)
        if profiling.get("value") is not None:
            profile = f"{profile} {profiling['value']}"
        rows.append(
            {
                "policy_id": column["policy_id"],
                "alpha": policy.get("alpha"),
                "skip_pa": policy.get("skip_pa"),
                "pooled_mode": policy.get("pooled_mode", DASH),
                "profiling": profile,
                "use_attn_mask": policy.get("use_attn_mask"),
                "embed_gain": policy.get("embed_gain"),
                "hidden_norm": policy.get("hidden_norm"),
                "history_vs_legacy": summary.get("mean_history_score_delta_vs_legacy"),
                "history_vs_legacy_ci": bootstrap.get("history_delta_vs_legacy"),
                "target_vs_legacy": summary.get("mean_target_score_delta_vs_legacy"),
                "target_vs_legacy_ci": bootstrap.get("target_delta_vs_legacy"),
                "history_vs_plain": summary.get("mean_history_score_delta_vs_plain"),
                "target_vs_plain": summary.get("mean_target_score_delta_vs_plain"),
                "status": candidate.get("status"),
                "reasons": candidate.get("reasons") or [],
                "by_history": summary.get("by_history") or {},
            }
        )
    rows.sort(
        key=lambda row: (
            row["history_vs_legacy"] is None,
            -(row["history_vs_legacy"] or 0.0),
            row["policy_id"],
        )
    )
    return rows


def policy_table(rows):
    head = (
        "| policy_id | alpha | skip_pa | pooled | profiling | mask | embed_gain "
        "| hidden_norm | Δhistory vs legacy | Δtarget vs legacy | Δhistory vs plain "
        "| Δtarget vs plain | 判定 |"
    )
    lines = [head, "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        verdict = VERDICT.get(row["status"], NOT_RUN)
        if row["reasons"]:
            verdict += "（" + ", ".join(str(item) for item in row["reasons"][:4]) + "）"
        skip_pa = row["skip_pa"]
        skip = DASH if skip_pa is None else "[" + ",".join(str(item) for item in skip_pa) + "]"
        lines.append(
            "| `{policy}` | {alpha} | {skip} | {pooled} | {profile} | {mask} | {gain} "
            "| {norm} | {history}{history_ci} | {target}{target_ci} | {history_plain} "
            "| {target_plain} | {verdict} |".format(
                policy=row["policy_id"],
                alpha=plain_number(row["alpha"], 2),
                skip=skip,
                pooled=row["pooled_mode"],
                profile=row["profiling"],
                mask="あり" if row["use_attn_mask"] else "なし",
                gain=plain_number(row["embed_gain"], 2),
                norm=cell(row["hidden_norm"]),
                history=number(row["history_vs_legacy"]),
                history_ci=interval(row["history_vs_legacy_ci"]),
                target=number(row["target_vs_legacy"]),
                target_ci=interval(row["target_vs_legacy_ci"]),
                history_plain=number(row["history_vs_plain"]),
                target_plain=number(row["target_vs_plain"]),
                verdict=verdict,
            )
        )
    return lines


def history_table(rows, histories):
    if not histories:
        return []
    lines = [
        "| policy_id | " + " | ".join(histories) + " |",
        "|---" * (len(histories) + 1) + "|",
    ]
    for row in rows:
        cells = []
        for history_id in histories:
            summary = row["by_history"].get(history_id) or {}
            cells.append(number(summary.get("mean_history_score_delta_vs_legacy")))
        lines.append(f"| `{row['policy_id']}` | " + " | ".join(cells) + " |")
    return lines


def status_line(run):
    if run is None:
        return NOT_RUN, "この実験はまだ実行されていません。"
    if not run["records"]:
        jobs = (run["checkpoint"].get("jobs") or {}).values()
        done = sum(1 for state in jobs if state.get("status") == "done")
        return RUNNING, f"画像 {done} / {len(jobs)} 枚まで生成済み。records.json はまだありません。"
    counts = ((run["metrics"] or {}).get("status_counts")) or {}
    total = len(run["records"])
    return (
        DONE,
        "計測 {measured} / 失敗 {failed} / 未計測 {unmeasured}（レコード {total} 件）".format(
            measured=counts.get("measured", 0),
            failed=counts.get("failed", 0),
            unmeasured=counts.get("unmeasured", 0),
            total=total,
        ),
    )


def judge_section(experiment_id, tasks, merged):
    lines = ["### AI 判定（plain との 2 枚比較）", ""]
    if not tasks:
        lines += ["未実施。判定用のペア画像がまだありません。", ""]
        return lines
    lines += [
        (
            f"判定素材: `{experiment_id}/judge/tasks.json`（{len(tasks)} ペア）、"
            f"画像は `{experiment_id}/judge/pairs/`。正解は "
            f"`{experiment_id}/judge/key.json` にあり、tasks.json には入っていません。"
        ),
        "",
    ]
    if merged is None or not merged["judges"]:
        lines += [
            "回答は未実施。回答 JSON を `--judge-answers` に渡すと集計します。",
            "",
        ]
        return lines
    lines += [
        f"回答者: {', '.join(merged['judges'])}（回答済み {merged['answered_pairs']} / {merged['pair_count']} ペア）",
        "",
        "| policy_id | plain に対する勝率 | n（引き分け除く） | 引き分け | target 維持率（はい） | n |",
        "|---|---|---|---|---|---|",
    ]
    for row in merged["policies"]:
        lines.append(
            "| `{policy}` | {rate} | {n} | {ties} | {kept} | {kept_n} |".format(
                policy=row["policy_id"],
                rate=plain_number(row["win_rate"]) if row["win_rate"] is not None else DASH,
                n=row["decided"],
                ties=row["ties"],
                kept=plain_number(row["target_kept_rate"])
                if row["target_kept_rate"] is not None
                else DASH,
                kept_n=row["target_answers"],
            )
        )
    agreement = merged["agreement"]
    rate = agreement["rate"]
    lines += [
        "",
        "判定者間の一致率: {rate}（複数人が答えたペア {pairs} 件のうち {agreed} 件で全員一致）".format(
            rate=plain_number(rate) if rate is not None else DASH,
            pairs=agreement["pairs_with_multiple_judges"],
            agreed=agreement["pairs_in_agreement"],
        ),
        "",
    ]
    return lines


def human_section(experiment_id, tasks, merged):
    lines = ["### 人による評価", ""]
    if not tasks:
        lines += ["未実施。評価用のペア画像がまだありません。", ""]
        return lines
    if merged is None or not merged["judges"]:
        lines += [
            "未実施。人の回答を代わりに作ることはしません。",
            "",
            "1. `review.html` をブラウザで開く（ファイルを直接開けます）。",
            "2. 氏名を入れると、ペアの並びがその氏名から決まる順に入れ替わります。",
            "3. 各ペアで「参照の好みに近いのはどちら？」と「お題(target)を保っているか」に答える。",
            (
                "4. 「回答を書き出す」で JSON を保存し、"
                "`build_strength_report.py --judge-answers <保存した JSON>` に渡す。"
            ),
            "",
        ]
        return lines
    lines += [
        f"回答者: {', '.join(merged['judges'])}（回答済み {merged['answered_pairs']} / {merged['pair_count']} ペア）",
        "",
        "| policy_id | plain に対する勝率 | n（引き分け除く） | target 維持率（はい） | n |",
        "|---|---|---|---|---|",
    ]
    for row in merged["policies"]:
        lines.append(
            "| `{policy}` | {rate} | {n} | {kept} | {kept_n} |".format(
                policy=row["policy_id"],
                rate=plain_number(row["win_rate"]) if row["win_rate"] is not None else DASH,
                n=row["decided"],
                kept=plain_number(row["target_kept_rate"])
                if row["target_kept_rate"] is not None
                else DASH,
                kept_n=row["target_answers"],
            )
        )
    lines.append("")
    return lines


def experiment_index(experiment_id, run, sheets, overview, columns):
    identity = run["manifest"]["identity"]
    lines = [
        f"# {experiment_id} 画像一覧",
        "",
        "列の並びは左から plain、legacy_exhibit、"
        + "、".join(f"`{column['policy_id']}`" for column in columns)
        + " です。",
        "",
        "## 参照（履歴）",
        "",
    ]
    for history in identity.get("histories", []):
        lines += [
            f"### {history['id']}",
            "",
            "| ref_id | 重み | 参照テキスト |",
            "|---|---|---|",
        ]
        for ref in history.get("refs", []):
            lines.append(
                "| `{ref}` | {weight} | {text} |".format(
                    ref=ref.get("ref_id", ""),
                    weight=plain_number(ref.get("weight"), 1),
                    text=str(ref.get("text", "")).replace("|", "\\|"),
                )
            )
        lines.append("")
    lines += ["## お題", "", "| topic | target_text |", "|---|---|"]
    for topic in identity.get("topics", []):
        lines.append(
            f"| `{topic['id']}` | {str(topic.get('target_text', '')).replace('|', chr(92) + '|')} |"
        )
    lines += ["", "## シート", ""]
    if overview:
        lines.append(f"- [overview.jpg](overview.jpg) — 概観（{overview['topic_id']} / {overview['history_id']} / seed {overview['seed']}）")
    for sheet in sheets:
        lines.append(
            "- [{file}]({file}) — topic {topic} / history {history} / seed {seed}（{columns} 列）".format(
                file=sheet["file"],
                topic=sheet["topic_id"],
                history=sheet["history_id"],
                seed=sheet["seed"],
                columns=sheet["columns"],
            )
        )
    lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------------------- GALLERY


def shorten(value, limit=60):
    text = str(value if value is not None else "")
    return text if len(text) <= limit else text[:limit] + "…"


def cell(value):
    """One table cell; a list becomes `[a,b]`, a missing value stays a dash."""
    if value is None:
        return DASH
    if isinstance(value, bool):
        return "あり" if value else "なし"
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value) + "]"
    if isinstance(value, dict):
        mode = value.get("mode", DASH)
        return f"{mode} {value['value']}" if value.get("value") is not None else str(mode)
    # An empty string is a real setting (e5 drops the negative prompt), not a gap.
    return str(value) if str(value) else "（空文字列）"


def settings_table(rows):
    """Every column of a sheet, left to right, with the settings that made it."""
    head = (
        "| 条件 | alpha | skip | skip_pa | pooled_mode | profiling | use_attn_mask "
        "| embed_gain | hidden_norm | reference_unit |"
    )
    lines = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    for name, policy in rows:
        if policy is None:
            lines.append(f"| {name} |" + f" {DASH} |" * 9)
            continue
        lines.append(
            "| {name} | {alpha} | {skip} | {skip_pa} | {pooled} | {profiling} "
            "| {mask} | {gain} | {norm} | {unit} |".format(
                name=name,
                alpha=plain_number(policy.get("alpha"), 2),
                skip=cell(policy.get("skip")),
                skip_pa=cell(policy.get("skip_pa")),
                pooled=cell(policy.get("pooled_mode")),
                profiling=cell(policy.get("profiling")),
                mask=cell(policy.get("use_attn_mask")),
                gain=plain_number(policy.get("embed_gain"), 2),
                norm=cell(policy.get("hidden_norm")),
                unit=cell(policy.get("reference_unit")),
            )
        )
    return lines


def gallery_settings_rows(gallery):
    identity = gallery["identity"]
    rows = [("plain（個人化なし）", None)]
    legacy = identity.get("legacy_policy", {}).get("effective_policy")
    if legacy:
        rows.append((f"`{gallery['legacy_label']}`", legacy))
    for column in gallery["columns"]:
        rows.append((f"`{column['policy_id']}`", column["effective_policy"]))
    return rows


def refs_table(history):
    lines = ["| ref_id | 重み | 参照テキスト |", "|---|---|---|"]
    for ref in history.get("refs", []):
        lines.append(
            "| `{ref}` | {weight} | {text} |".format(
                ref=ref.get("ref_id", ""),
                weight=plain_number(ref.get("weight"), 1),
                text=str(ref.get("text", "")).replace("|", "\\|"),
            )
        )
    return lines


def sheet_figure(label, sheet):
    """A named sheet, embedded by the same relative path the README uses."""
    alt = (
        f"topic={sheet['topic_id']} history={sheet['history_id']} seed={sheet['seed']}"
    )
    return [
        f"**topic={sheet['topic_id']} / history={sheet['history_id']} / seed={sheet['seed']}**",
        "",
        f"![{alt}]({label}/{sheet['file']})",
        "",
    ]


def gallery_history_sections(gallery):
    """Per history: its references, the first seed inline, the rest folded away."""
    identity = gallery["identity"]
    label = gallery["label"]
    seeds = list(identity.get("seeds", []))
    by_case = {
        (sheet["topic_id"], sheet["history_id"], sheet["seed"]): sheet
        for sheet in gallery["sheets"]
    }
    lines, inline, folded = [], 0, 0
    for history in identity.get("histories", []):
        lines += [f"### 履歴 `{history['id']}`", ""] + refs_table(history) + [""]
        for index, seed in enumerate(seeds):
            sheets = [
                by_case[(topic["id"], history["id"], seed)]
                for topic in identity.get("topics", [])
                if (topic["id"], history["id"], seed) in by_case
            ]
            if not sheets:
                continue
            if index == 0:
                for sheet in sheets:
                    lines += sheet_figure(label, sheet)
                inline += len(sheets)
                continue
            # GitHub renders images inside <details> only with the blank lines.
            lines += [
                f"<details><summary>seed {seed} のシート（{len(sheets)} 枚）</summary>",
                "",
            ]
            for sheet in sheets:
                lines += sheet_figure(label, sheet)
            lines += ["</details>", ""]
            folded += len(sheets)
    return lines, inline, folded


def gallery_experiment_section(gallery):
    lines = [f"## {gallery['title']}", ""]
    if gallery["description"]:
        lines += [gallery["description"], ""]
    if gallery["plan_items"]:
        lines.append("- plan 項目: " + ", ".join(gallery["plan_items"]))
    lines += [f"- experiment hash: `{gallery['hash']}`", ""]
    lines += ["### 条件の設定", ""] + settings_table(gallery_settings_rows(gallery))
    lines.append("")
    if gallery["override"]:
        lines += [
            "この実験だけ生成設定を上書きしています。",
            "",
            "| 項目 | 値 |",
            "|---|---|",
        ]
        for key, value in gallery["override"].items():
            lines.append(f"| `{key}` | {cell(value) if not isinstance(value, dict) else json.dumps(value, ensure_ascii=False)} |")
        lines.append("")
    if gallery["overview"]:
        lines += [
            f"![{gallery['label']} の概観]({gallery['label']}/{gallery['overview']['file']})",
            "",
        ]
    history_lines, inline, folded = gallery_history_sections(gallery)
    lines += history_lines
    return lines, inline, folded


def generation_section(galleries):
    """The production generation block, taken from a run that did not override it."""
    lines = ["## 共通の生成設定", ""]
    base = next(
        (item["identity"].get("generation") for item in galleries if not item["override"]),
        None,
    )
    if base is None:
        base = galleries[0]["identity"].get("generation") if galleries else None
    if not base:
        lines += ["実行された run がないので、生成設定は読み取れません。", ""]
        return lines
    pipeline = base.get("pipeline_config") or {}
    vae = base.get("vae") or {}
    scheduler = base.get("scheduler", DASH)
    kwargs = base.get("scheduler_kwargs") or {}
    if kwargs:
        scheduler += " " + json.dumps(kwargs, ensure_ascii=False)
    rows = [
        ("モデル", f"`{base.get('model', DASH)}` @ `{shorten(base.get('revision'), 12)}`"),
        ("checkpoint", f"`{base.get('checkpoint', DASH)}`"),
        ("pipeline 設定", f"`{pipeline.get('model', DASH)}`"),
        ("VAE", f"`{vae.get('model', DASH)}`"),
        ("scheduler", scheduler),
        ("steps", cell(base.get("steps"))),
        ("guidance_scale", cell(base.get("guidance_scale"))),
        ("解像度", f"{base.get('width', DASH)}×{base.get('height', DASH)}"),
        ("precision", cell(base.get("precision"))),
        ("positive_prompt_tail", f"`{base.get('positive_prompt_tail', DASH)}`"),
        ("negative prompt", shorten(base.get("negative_prompt"))),
    ]
    lines += ["| 項目 | 値 |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    lines.append("")

    topics, histories = {}, {}
    for gallery in galleries:
        for topic in gallery["identity"].get("topics", []):
            topics.setdefault(topic["id"], topic)
        for history in gallery["identity"].get("histories", []):
            histories.setdefault(history["id"], history)
    if topics:
        lines += [
            "### お題（topic）",
            "",
            "| topic | generation_prompt（先頭のみ） | target_text |",
            "|---|---|---|",
        ]
        for topic in topics.values():
            lines.append(
                "| `{id}` | {prompt} | {target} |".format(
                    id=topic["id"],
                    prompt=shorten(topic.get("generation_prompt"), 70),
                    target=str(topic.get("target_text", "")).replace("|", "\\|"),
                )
            )
        lines.append("")
    if histories:
        lines += ["### 履歴（history）と参照語句", ""]
        for history in histories.values():
            lines += [f"#### `{history['id']}`", ""] + refs_table(history) + [""]
    return lines


def paper_regime_section(report_dir):
    """A2 is written by hand beside the report; it is quoted, never rebuilt."""
    lines = ["## A2: 論文レジーム（base SDXL 1.0）", ""]
    directory = report_dir / "a2-paper-regime"
    report = read_json(directory / "report.json")
    if not isinstance(report, dict):
        lines += [
            "未実施。`a2-paper-regime/report.json` がありません。",
            "",
        ]
        return lines
    settings = report.get("settings") or {}
    lines += [
        (
            "FAN 本来の土俵（base SDXL 1.0、自然文プロンプト、自然文参照）での上限確認です。"
            "生成設定が違うので exhibit の rules は当てはめません。"
        ),
        "",
        "| 項目 | 値 |",
        "|---|---|",
    ]
    for name in (
        "model",
        "size",
        "steps",
        "guidance_scale",
        "skip",
        "skip_pa",
        "use_attn_mask",
        "sample_size",
        "negative_prompt",
    ):
        lines.append(f"| `{name}` | {cell(settings.get(name))} |")
    lines += [
        f"| alpha | {cell(report.get('alphas'))} |",
        f"| seeds | {cell(report.get('seeds'))} |",
        "",
    ]
    topics = report.get("topics") or []
    if topics:
        lines += ["| topic | プロンプト |", "|---|---|"]
        lines += [
            "| `{id}` | {prompt} |".format(
                id=topic.get("id"), prompt=str(topic.get("prompt", "")).replace("|", "\\|")
            )
            for topic in topics
        ]
        lines.append("")
    for history in report.get("histories") or []:
        lines += [f"#### `{history.get('id')}`", ""] + refs_table(history) + [""]
    if (directory / "contact-sheet.jpg").is_file():
        lines += ["![A2 のコンタクトシート](a2-paper-regime/contact-sheet.jpg)", ""]
    images = report.get("images") or []
    if images:
        head = (
            "| 画像 | topic | history | alpha | pixel MAE (vs alpha 0) "
            "| cos hidden | cos pooled |"
        )
        lines += [head, "|---|---|---|---|---|---|---|"]
        for image in images:
            cosine = image.get("cosine_vs_alpha0") or {}
            lines.append(
                "| `{file}` | {topic} | {history} | {alpha} | {mae} | {hidden} | {pooled} |".format(
                    file=image.get("file", ""),
                    topic=image.get("topic", DASH),
                    history=image.get("history") or DASH,
                    alpha=plain_number(image.get("alpha"), 2),
                    mae=plain_number(image.get("pixel_mae_vs_alpha0")),
                    hidden=plain_number(cosine.get("hidden")),
                    pooled=plain_number(cosine.get("pooled")),
                )
            )
        lines.append("")
    return lines


def gallery(entries, extras, report_dir):
    """Every condition with the settings that made it and the images it made."""
    galleries = [entry["gallery"] for entry in entries if entry["gallery"]]
    extra_galleries = [extra["gallery"] for extra in extras]
    lines = [
        "# FAN 強度実験 · 設定と画像の一覧",
        "",
        f"生成日時: {timestamp()}",
        "",
        (
            "1 枚のシートは 1 ケース（topic × history × seed）です。列は左から "
            "plain（個人化なし）、legacy_exhibit（現行既定）、設定表の順に並ぶ候補 policy。"
            "各タイルの下の帯に policy_id と、そのケースでの legacy_exhibit に対する "
            "Δhistory（dHist）・Δtarget（dTgt）が入っています。"
            "タイルは 1024×1280 の生成画像を 320×400 に縮小したものです。"
        ),
        "",
        "判定と数値は [README.md](README.md) にあります。",
        "",
    ]
    lines += generation_section(galleries + extra_galleries)
    counts = {}
    for entry in entries:
        if entry["gallery"] is None:
            lines += [
                f"## `{entry['experiment_id']}`",
                "",
                f"**{entry['state']}** — 画像がないので一覧はありません。",
                "",
            ]
            counts[entry["experiment_id"]] = {"inline": 0, "details": 0}
            continue
        section, inline, folded = gallery_experiment_section(entry["gallery"])
        lines += section
        counts[entry["experiment_id"]] = {"inline": inline, "details": folded}
    for extra in extra_galleries:
        section, inline, folded = gallery_experiment_section(extra)
        lines += section
        counts[extra["label"]] = {"inline": inline, "details": folded}
    lines += paper_regime_section(report_dir)
    return "\n".join(lines), counts


def readme(entries, extras, problems, config_path):
    lines = [
        "# FAN 強度実験 · 結果レポート",
        "",
        f"生成日時: {timestamp()}",
        "",
        "`build_strength_report.py` がディスク上の成果物だけから作ります。",
        "実行されていない実験は **未実施** であり、成功でも失敗でもありません。",
        "",
        "条件ごとの設定と生成画像の一覧は [GALLERY.md](GALLERY.md) にあります。",
        "",
        "## 指標の読み方",
        "",
        "- `target_score`: 生成画像とお題文（target_text）の CLIP コサイン類似度。主題を保っているか。",
        "- `history_score`: 生成画像と参照語句の CLIP コサイン類似度を、参照の重みで加重平均したもの。好みに寄ったか。",
        "- `Δ` は既定 policy `legacy_exhibit` との差。`vs plain` は個人化なしの画像との差。",
        "- 合否は既存 rules: target の床 `Δtarget ≥ -0.01`、history の改善 `Δhistory ≥ +0.005`。",
        "- `rules.binding` が false の実験（生成設定を変えたもの）は参考値で、policy の採否には使いません。",
        "- 数値は小数第 4 位まで。角括弧は履歴クラスタのブートストラップ 95% 区間。",
        "",
        "## 判定・評価の回答形式",
        "",
        (
            "AI 判定も人による評価も、同じ形の JSON を "
            "`--judge-answers` に渡します。`kind` が `human` のファイルは"
            "「人による評価」に、それ以外は「AI 判定」に集計されます"
            "（`review.html` の書き出しは自動で `human` になります）。"
        ),
        "",
        "```json",
        "{",
        '  "judge": "judge-a",',
        '  "kind": "ai",',
        '  "answers": {',
        '    "<pair_id>": {',
        '      "preference": "A",',
        '      "target_kept": {"A": "yes", "B": "partly"},',
        '      "note": ""',
        "    }",
        "  }",
        "}",
        "```",
        "",
        (
            "`preference` は `A` / `B` / `tie`、`target_kept` は `yes` / `partly` / `no`。"
            "`pair_id` は各実験の `judge/tasks.json` にあります。"
            "どちらが個人化した画像かは `judge/key.json` にだけ書いてあり、"
            "回答する側には見えません。"
        ),
        "",
    ]
    if problems:
        lines += ["## 入力の問題", ""] + [f"- {item}" for item in problems] + [""]
    lines += ["## 実験一覧", "", "| 実験 | 状態 | 種別 | rules 拘束 | 内訳 |", "|---|---|---|---|---|"]
    for entry in entries:
        lines.append(
            "| `{experiment}` | {state} | {kind} | {binding} | {detail} |".format(
                experiment=entry["experiment_id"],
                state=entry["state"],
                kind=entry["kind"],
                binding="あり" if entry["binding"] else "なし",
                detail=entry["detail"],
            )
        )
    lines.append("")
    for entry in entries:
        lines += entry["section"]
    if extras:
        lines += [
            "## 参考: 正式チェーンの run",
            "",
            (
                "`fan-strength.json` の実験ではなく、screen / refine / heldout の"
                "正式チェーンの run です。目視 (P3) のためにシートだけを作ります。"
                "判定ペアと `review.html` には入りません。"
            ),
            "",
        ]
        for extra in extras:
            lines += extra["section"]
    lines += [
        "## 見方",
        "",
        (
            "- シート画像は 1 ケース（topic × history × seed）を 1 枚にまとめたものです。"
            "左から plain（個人化なし）、legacy_exhibit（現行既定）、"
            "宣言順の候補 policy が並びます。"
        ),
        (
            "- 各タイルの下の帯に policy_id と、そのケースでの legacy_exhibit に対する "
            "Δhistory（dHist）・Δtarget（dTgt）が入っています。帯は ASCII のみです。"
        ),
        (
            "- 同じ行の画像は seed も生成設定も同じで、encoder の policy だけが違います。"
            "違いが見えない場合、その policy はそのケースで効いていません。"
        ),
        (
            "- 各実験の `index.md` に参照語句の全文があります。"
            "どの好みを再現しようとしたのかはそこで確認してください。"
        ),
        (
            "- `judge/pairs/` の画像は plain と候補の 2 枚で、左右はペアごとに入れ替えてあります。"
            "どちらが候補かは `judge/key.json` にだけ書いてあります。"
        ),
        "",
        "## 再実行",
        "",
        "```bash",
    ]
    for entry in entries:
        lines.append(
            "PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py "
            f"strength --config {relative_to_repo(config_path)} --experiment {entry['experiment_id']} --resume"
        )
    lines += [
        "PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_strength_report.py",
        "PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_strength_report.py \\",
        "  --judge-answers <回答1>.json <回答2>.json",
        "```",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ review.html


REVIEW_TEMPLATE = """<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FAN 強度実験 · ブラインド評価</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#141414;color:#eee;font-family:system-ui,sans-serif;line-height:1.7}
main{max-width:1000px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:20px} h2{font-size:16px;margin:0 0 4px}
.box{background:#1e1e1e;border:1px solid #333;border-radius:8px;padding:16px;margin-bottom:16px}
img{width:100%;height:auto;border-radius:6px;display:block}
label{display:block;margin:6px 0 2px;font-size:14px;color:#bbb}
input,textarea{width:100%;box-sizing:border-box;background:#111;color:#eee;
border:1px solid #444;border-radius:6px;padding:8px;font:inherit}
button{background:#d98d6a;color:#1a1a1a;border:0;border-radius:6px;padding:10px 16px;
font:inherit;font-weight:700;cursor:pointer}
button.ghost{background:#333;color:#eee}
.row{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 8px}
.row button{background:#2a2a2a;color:#eee;font-weight:400;padding:6px 14px;border:1px solid #444}
.row button[aria-pressed="true"]{background:#d98d6a;color:#1a1a1a;font-weight:700}
.bar{position:sticky;top:0;background:#141414;border-bottom:1px solid #333;padding:10px 0;z-index:2}
.meta{font-size:13px;color:#999}
</style></head><body><main>
<h1>FAN 強度実験 · ブラインド評価</h1>
<div class="box">
<p>2 枚の画像 A・B を見て、2 つの質問に答えてください。どちらが個人化した画像かは伏せてあります。
並び順は氏名ごとに変わります。回答はこのブラウザに保存され、いつでも続きから再開できます。</p>
<label for="name">氏名（回答ファイルに残ります）</label>
<input id="name" placeholder="例: reviewer-01">
<p class="meta" id="progress">氏名を入れると始まります。</p>
<div class="row"><button id="export">回答を書き出す</button>
<button class="ghost" id="reset">この氏名の回答を消す</button></div>
</div>
<div id="list"></div>
</main>
<script id="pairs" type="application/json">__PAIRS__</script>
<script>
const PAIRS = JSON.parse(document.getElementById("pairs").textContent);
const PREF = [["A","A のほう"],["B","B のほう"],["tie","同じ"]];
const KEPT = [["yes","はい"],["partly","一部"],["no","いいえ"]];
let answers = {}, name = "";
function seedOf(text){let h=2166136261>>>0;for(let i=0;i<text.length;i++){
h^=text.charCodeAt(i);h=Math.imul(h,16777619)>>>0;}return h>>>0;}
function rng(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);
t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};}
function shuffled(text){const order=PAIRS.slice();const next=rng(seedOf(text));
for(let i=order.length-1;i>0;i--){const j=Math.floor(next()*(i+1));
[order[i],order[j]]=[order[j],order[i]];}return order;}
function storageKey(){return "fan-strength-review:"+name;}
function load(){try{answers=JSON.parse(localStorage.getItem(storageKey())||"{}");}
catch(error){answers={};}}
function save(){try{localStorage.setItem(storageKey(),JSON.stringify(answers));}
catch(error){}}
function entry(id){if(!answers[id]){answers[id]={preference:null,target_kept:{A:null,B:null},note:""};}
return answers[id];}
function done(){return Object.values(answers).filter(function(a){
return a.preference&&a.target_kept.A&&a.target_kept.B;}).length;}
function progress(){document.getElementById("progress").textContent=
name?("回答済み "+done()+" / "+PAIRS.length+" ペア"):"氏名を入れると始まります。";}
function group(options, current, onPick){const row=document.createElement("div");
row.className="row";const made=[];options.forEach(function(option){
const button=document.createElement("button");button.type="button";
button.textContent=option[1];
button.setAttribute("aria-pressed",String(current===option[0]));
button.onclick=function(){onPick(option[0]);made.forEach(function(other,index){
other.setAttribute("aria-pressed",String(options[index][0]===option[0]));});};
row.appendChild(button);made.push(button);});return row;}
function render(){const list=document.getElementById("list");list.textContent="";
if(!name){return;}
shuffled(name).forEach(function(pair,index){const answer=entry(pair.pair_id);
const box=document.createElement("div");box.className="box";
const head=document.createElement("h2");
head.textContent=(index+1)+" / "+PAIRS.length;box.appendChild(head);
const meta=document.createElement("p");meta.className="meta";
meta.textContent="参照の好み: "+pair.history_text;box.appendChild(meta);
const target=document.createElement("p");target.className="meta";
target.textContent="お題(target): "+pair.target_text;box.appendChild(target);
const image=document.createElement("img");image.loading="lazy";image.src=pair.image;
image.alt=pair.pair_id;box.appendChild(image);
const q1=document.createElement("label");
q1.textContent="参照の好みに近いのはどちら？";box.appendChild(q1);
box.appendChild(group(PREF,answer.preference,function(value){
answer.preference=value;save();progress();}));
["A","B"].forEach(function(side){const q=document.createElement("label");
q.textContent="お題(target)を保っているか（"+side+"）";box.appendChild(q);
box.appendChild(group(KEPT,answer.target_kept[side],function(value){
answer.target_kept[side]=value;save();progress();}));});
const note=document.createElement("label");note.textContent="気づいたこと（任意）";
box.appendChild(note);const field=document.createElement("textarea");field.rows=2;
field.value=answer.note||"";field.oninput=function(){answer.note=field.value;save();};
box.appendChild(field);list.appendChild(box);});}
document.getElementById("name").oninput=function(event){
name=event.target.value.trim();if(name){load();}else{answers={};}render();progress();};
document.getElementById("reset").onclick=function(){if(!name){return;}
answers={};save();render();progress();};
document.getElementById("export").onclick=function(){
if(!name){alert("先に氏名を入れてください。");return;}
const payload={judge:name,kind:"human",source:"review.html",answers:answers};
const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});
const link=document.createElement("a");link.href=URL.createObjectURL(blob);
link.download="review-"+name.replace(/[^A-Za-z0-9_-]/g,"_")+".json";link.click();
URL.revokeObjectURL(link.href);};
progress();
</script></body></html>
"""


def review_html(pairs):
    payload = json.dumps(pairs, ensure_ascii=False).replace("</", "<\\/")
    return REVIEW_TEMPLATE.replace("__PAIRS__", payload)


# ----------------------------------------------------------------------- main


LABEL_CHARACTERS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def parse_extra_runs(values, taken):
    """`<run directory>=<label>` pairs; an unusable one is named, not guessed."""
    parsed, problems, seen = [], [], set()
    for value in values:
        directory, separator, label = str(value).partition("=")
        label = label.strip()
        if not separator or not label:
            problems.append(f"`--extra-run` の書式は `<run ディレクトリ>=<ラベル>` です: {value}")
            continue
        if set(label) - LABEL_CHARACTERS:
            problems.append(f"ラベルに使えない文字があります: {label}")
            continue
        if label in taken or label in seen:
            problems.append(f"ラベルが実験 id か別の run と重なっています: {label}")
            continue
        run = load_run(Path(directory))
        if run is None:
            problems.append(f"run を読めません（manifest.json がない）: {directory}")
            continue
        seen.add(label)
        parsed.append((label, run))
    return parsed, problems


def build_extra_entry(label, run, report_dir):
    """A formal-chain run: the same sheets and policy table, no judge pairs."""
    identity = run["manifest"]["identity"]
    decision = run["decision"] or {}
    stage = decision.get("stage") or identity.get("phase") or DASH
    directory = report_dir / label
    directory.mkdir(parents=True, exist_ok=True)
    lookups = index_records(run["records"])
    columns, _ = policy_columns(None, run["manifest"])
    sheets, overview = build_sheets(run, lookups, columns, directory)
    (directory / "index.md").write_text(
        experiment_index(label, run, sheets, overview, columns)
    )
    state, detail = status_line(run)
    counts = {}
    for candidate in decision.get("candidates") or []:
        name = VERDICT.get(candidate.get("status"), candidate.get("status", "?"))
        counts[name] = counts.get(name, 0) + 1
    lines = [
        f"### `{label}`",
        "",
        f"- 段階: {stage}",
        f"- 状態: **{state}** — {detail}",
        f"- experiment hash: `{run['manifest'].get('experiment_hash', '')}`",
        "- 判定: {status}（{counts}）".format(
            status=decision.get("status") or NOT_RUN,
            counts=", ".join(f"{name} {count}" for name, count in sorted(counts.items()))
            or "候補なし",
        ),
        f"- 出力: `{relative_to_repo(run['directory'])}`",
        f"- 画像: [{label}/index.md]({label}/index.md)（シート {len(sheets)} 枚）",
        "",
    ]
    if overview:
        lines += [f"![{label} の概観]({label}/{overview['file']})", ""]
    lines += policy_table(policy_rows(columns, run)) + [""]
    return {
        "label": label,
        "stage": stage,
        "sheets": len(sheets),
        "section": lines,
        "gallery": {
            "label": label,
            "title": f"`{label}`（{stage}）",
            "description": "正式チェーンの run。目視 (P3) 用。",
            "plan_items": [],
            "hash": run["manifest"].get("experiment_hash", ""),
            "identity": identity,
            "legacy_label": (run["manifest"].get("policy_labels") or {}).get(
                identity.get("legacy_policy", {}).get("policy_hash"), "legacy_exhibit"
            ),
            "columns": columns,
            "sheets": sheets,
            "overview": overview,
            "override": {},
        },
    }


def build_experiment_entry(experiment_id, description, config, found, report_dir, judges):
    """One declared experiment: sheets, judge material and its README section."""
    entry = {
        "experiment_id": experiment_id,
        "kind": (config or {}).get("experiment_kind", "policy"),
        "binding": bool(((config or {}).get("rules") or {}).get("binding", True)),
        "state": NOT_RUN,
        "detail": "未実行",
        "pairs": [],
        "section": [],
        "judge_summary": None,
        "gallery": None,
    }
    record = found.get(experiment_id)
    run = record["run"] if record else None
    state, detail = status_line(run)
    entry["state"], entry["detail"] = state, detail
    lines = [f"## `{experiment_id}`", "", description or "（説明なし）", ""]
    plan_items = (config or {}).get("plan_items") or []
    lines.append("- plan 項目: " + (", ".join(plan_items) if plan_items else DASH))
    lines.append(f"- 種別: {entry['kind']}（rules 拘束: {'あり' if entry['binding'] else 'なし'}）")
    lines.append(f"- 状態: **{state}** — {detail}")
    if run is None:
        lines += [
            "",
            (
                "まだ実行されていないので、数値も画像もありません。"
                "下の「再実行」のコマンドで生成してください。"
            ),
            "",
        ]
        entry["section"] = lines
        return entry

    lines.append(f"- experiment hash: `{run['manifest'].get('experiment_hash', '')}`")
    lines.append(f"- 出力: `{relative_to_repo(run['directory'])}`")
    if record["superseded"]:
        lines.append(
            "- 同じ実験 id の古い実行: "
            + ", ".join(f"`{item['directory'].name}`" for item in record["superseded"])
        )
    override = (run["manifest"].get("display") or {}).get("generation_override") or {}
    if override:
        lines.append(f"- 生成設定の上書き: `{json.dumps(override, ensure_ascii=False)}`")

    directory = report_dir / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    lookups = index_records(run["records"])
    columns, missing = policy_columns(config, run["manifest"])
    sheets, overview = build_sheets(run, lookups, columns, directory)
    tasks, key = build_judge_material(
        experiment_id, run, lookups, columns, directory / "judge"
    )
    (directory / "index.md").write_text(
        experiment_index(experiment_id, run, sheets, overview, columns)
    )
    lines.append(f"- 画像: [{experiment_id}/index.md]({experiment_id}/index.md)（シート {len(sheets)} 枚）")
    lines.append("")
    if overview:
        lines += [
            f"![{experiment_id} の概観]({experiment_id}/{overview['file']})",
            "",
        ]
    if missing:
        lines += [
            "宣言されているが今回の実行に含まれない policy: "
            + ", ".join(f"`{item}`" for item in missing),
            "",
        ]
    rows = policy_rows(columns, run)
    lines += ["### policy 別", ""] + policy_table(rows) + [""]
    histories = [history["id"] for history in run["manifest"]["identity"].get("histories", [])]
    table = history_table(rows, histories)
    if table:
        lines += ["### 履歴別 Δhistory vs legacy", ""] + table + [""]

    entry["gallery"] = {
        "label": experiment_id,
        "title": f"`{experiment_id}`",
        "description": description or "",
        "plan_items": plan_items,
        "hash": run["manifest"].get("experiment_hash", ""),
        "identity": run["manifest"]["identity"],
        "legacy_label": (run["manifest"].get("policy_labels") or {}).get(
            run["manifest"]["identity"].get("legacy_policy", {}).get("policy_hash"),
            "legacy_exhibit",
        ),
        "columns": columns,
        "sheets": sheets,
        "overview": overview,
        "override": override,
    }
    entry["pairs"] = [
        {**task, "experiment_id": experiment_id, "image": f"{experiment_id}/judge/{task['image']}"}
        for task in tasks
    ]
    ai_judges = [judge for judge in judges if judge["kind"] != "human"]
    human_judges = [judge for judge in judges if judge["kind"] == "human"]
    merged_ai = merge_judgements(key, ai_judges) if key and ai_judges else None
    merged_human = merge_judgements(key, human_judges) if key and human_judges else None
    lines += judge_section(experiment_id, tasks, merged_ai)
    lines += human_section(experiment_id, tasks, merged_human)
    summary = {
        "experiment_id": experiment_id,
        "pair_count": len(tasks),
        "ai": merged_ai,
        "human": merged_human,
    }
    write_json(directory / "judge/summary.json", summary)
    entry["judge_summary"] = summary
    entry["section"] = lines
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--judge-answers",
        nargs="*",
        default=[],
        metavar="PATH",
        help="判定結果の JSON。渡さなければ判定は未実施のまま。",
    )
    parser.add_argument(
        "--extra-run",
        action="append",
        default=[],
        metavar="DIR=LABEL",
        help="正式チェーンの run ディレクトリ。目視用のシートだけを作ります。",
    )
    args = parser.parse_args(argv)

    report_dir = Path(args.report)
    report_dir.mkdir(parents=True, exist_ok=True)
    judges, problems = load_answer_files(args.judge_answers)
    found = discover_runs(args.outputs)
    declared = list_strength_experiments(args.config)

    entries = []
    for experiment_id, description in declared.items():
        try:
            config = load_strength_config(args.config, experiment_id)
        except (OSError, TypeError, ValueError) as error:
            config = None
            problems.append(f"`{experiment_id}` の設定を読めません: {error}")
        entries.append(
            build_experiment_entry(
                experiment_id, description, config, found, report_dir, judges
            )
        )
    for experiment_id in sorted(set(found) - set(declared)):
        problems.append(
            f"宣言されていない実験の実行があります: `{experiment_id}` "
            f"(`{found[experiment_id]['run']['directory'].name}`)"
        )

    extra_runs, extra_problems = parse_extra_runs(args.extra_run, set(declared))
    problems += extra_problems
    extras = [
        build_extra_entry(label, run, report_dir) for label, run in extra_runs
    ]

    pairs = [pair for entry in entries for pair in entry["pairs"]]
    (report_dir / "review.html").write_text(review_html(pairs))
    (report_dir / "README.md").write_text(
        readme(entries, extras, problems, args.config)
    )
    page, gallery_counts = gallery(entries, extras, report_dir)
    (report_dir / "GALLERY.md").write_text(page)

    result = {
        "report": str(report_dir / "README.md"),
        "gallery": str(report_dir / "GALLERY.md"),
        "review": str(report_dir / "review.html"),
        "pairs": len(pairs),
        "judges": [judge["judge"] for judge in judges],
        "problems": problems,
        "experiments": {entry["experiment_id"]: entry["state"] for entry in entries},
        "extra_runs": {extra["label"]: extra["sheets"] for extra in extras},
        "gallery_images": gallery_counts,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
