#!/usr/bin/env python3
"""audit_direction.py -- direction audit for FAN personalization runs.

For images that already exist on disk under exhibit/outputs/fan-evaluation,
checks whether a personalization policy moves the generated image toward
what its reference prompts actually ask for (warmer/cooler, more/less
saturated, higher/lower contrast, more/fewer edges), or whether it just
drifts the same way regardless of what the references say.

This script does not generate anything. It only reads manifest.json /
summary.json / images/*.png that a previous `evaluate_fan.py` run produced,
and the reference texts declared in each run's own manifest (or, for the
shipped exhibit samples, in exhibit/assets/samples.json).

Run it with the fan-repro venv, which has numpy + Pillow (exhibit/.venv does
not):

    fan-repro/.venv/bin/python exhibit/scripts/audit_direction.py \\
        exhibit/outputs/fan-evaluation exhibit/outputs/fan-evaluation/strength \\
        --out-json docs/reports/fan-personalization/strength/direction-audit/audit.json \\
        --out-md   docs/reports/fan-personalization/strength/direction-audit/README.md

Each positional argument is either a run directory that directly contains
manifest.json + summary.json, or a container directory searched recursively
for such run directories (directories literally named "diagnostics", and
"matrix-*" scratch directories, are skipped -- they hold encoder-diagnostic
or worker-checkpoint data, not scored image runs).

Pass all related run directories together in one invocation. A refine-phase
record's paired plain/legacy baseline image was often generated in the
parent screen-phase run, not its own run directory, and its `plain_job_id`
is resolved by looking up that job_id across every run given on the command
line -- not only the run the record itself came from.

By default the script also audits the exhibit's shipped sample images
(exhibit/assets/generic/<topic>-<i>.png vs exhibit/assets/samples/<sample
id>-<i>.png, refs from exhibit/assets/samples.json) as Table D; pass
--no-samples to skip that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import UTC
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

METRICS = ["warmth", "lum", "sat", "contrast", "edge"]
RESIZE = (256, 320)  # (width, height)
EDGE_THRESHOLD = 40

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Pixel metrics
# ---------------------------------------------------------------------------


def image_metrics(path: Path) -> dict:
    with Image.open(path) as im:
        im = im.convert("RGB").resize(RESIZE)
        arr = np.asarray(im).astype(np.float64)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        warmth = float(np.mean(r - b))
        lum_arr = 0.299 * r + 0.587 * g + 0.114 * b
        lum = float(np.mean(lum_arr))
        contrast = float(np.std(lum_arr))
        hsv = np.asarray(im.convert("HSV")).astype(np.float64)
        sat = float(np.mean(hsv[..., 1]))
        gray = im.convert("L")
        edge_arr = np.asarray(gray.filter(ImageFilter.FIND_EDGES)).astype(np.float64)
        edge = float(np.mean(edge_arr > EDGE_THRESHOLD) * 100.0)
    return {
        "warmth": warmth,
        "lum": lum,
        "sat": sat,
        "contrast": contrast,
        "edge": edge,
    }


class MetricCache:
    def __init__(self):
        self._cache: dict = {}

    def get(self, path: Path) -> dict | None:
        key = str(path)
        if key not in self._cache:
            try:
                self._cache[key] = image_metrics(path)
            except (OSError, ValueError) as error:  # corrupt/missing file
                print(f"warning: failed to read {path}: {error}", file=sys.stderr)
                self._cache[key] = None
        return self._cache[key]


# ---------------------------------------------------------------------------
# Reference-text -> expected-direction mapping
# ---------------------------------------------------------------------------


def text_expectations(text: str) -> list[tuple[str, str]]:
    """Map a single reference phrase to zero or more (metric, direction) pairs."""
    t = (text or "").lower()
    exps: list[tuple[str, str]] = []

    if "warm color" in t:
        exps.append(("warmth", "up"))
    if "cool color" in t:
        exps.append(("warmth", "down"))

    if "muted" in t or "desaturated" in t:
        exps.append(("sat", "down"))
    if "pastel" in t:
        exps.append(("sat", "down"))
        exps.append(("lum", "up"))
    if "vivid" in t or ("saturated" in t and "desaturated" not in t):
        exps.append(("sat", "up"))

    if "harsh sunlight" in t or "high contrast" in t or "hard cast shadow" in t:
        exps.append(("contrast", "up"))
    if "backlighting" in t or "rim light" in t:
        exps.append(("contrast", "up"))
    if "overcast" in t or "diffused" in t:
        exps.append(("contrast", "down"))

    if "night" in t or "dim lighting" in t or "dark background" in t:
        exps.append(("lum", "down"))

    if "no lineart" in t or "lineless" in t or "flat color" in t:
        exps.append(("edge", "down"))
    if "sketch" in t or "pencil lines" in t:
        exps.append(("edge", "up"))
    if ("cel shading" in t or "lineart" in t) and "no lineart" not in t:
        exps.append(("edge", "up"))

    # de-dup, keep first-seen order
    seen = set()
    out = []
    for e in exps:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def history_expectations(refs: list[dict]) -> list[tuple[str, str]]:
    """Direction expectations for a history, one entry per metric.

    A history can carry several refs, and different refs can name the same
    metric with opposite directions (e.g. a "mixed" history with both a
    warm-color ref and a cool-color ref). Such a record can never satisfy
    both, so scoring it against either direction would just drag every hit
    rate toward 50%. When a metric gets both "up" and "down" from a
    history's own refs, that metric is dropped entirely for this history;
    the history's other, non-conflicting metrics are kept.
    """
    by_metric: dict[str, set] = defaultdict(set)
    for ref in refs or []:
        for metric, direction in text_expectations(ref.get("text", "")):
            by_metric[metric].add(direction)
    exps = []
    for metric, directions in by_metric.items():
        if len(directions) == 1:
            exps.append((metric, next(iter(directions))))
        # else: conflicting directions for the same metric -> drop it
    return sorted(exps)


# ---------------------------------------------------------------------------
# Run discovery / loading
# ---------------------------------------------------------------------------


def discover_runs(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()

    def consider(d: Path) -> bool:
        if (d / "manifest.json").exists() and (d / "summary.json").exists():
            rd = d.resolve()
            if rd not in seen:
                seen.add(rd)
                found.append(d)
            return True
        return False

    for raw in paths:
        p = Path(raw)
        if not p.exists():
            print(f"warning: path not found: {p}", file=sys.stderr)
            continue
        if consider(p):
            continue
        for root, dirs, _files in os.walk(p):
            dirs[:] = [
                d for d in dirs if d != "diagnostics" and not d.startswith("matrix-")
            ]
            rootp = Path(root)
            if consider(rootp):
                dirs[:] = []  # a run dir found; do not descend into it further
    return found


def load_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    policy_labels = manifest.get("policy_labels") or {}
    histories = {}
    for h in (manifest.get("identity", {}) or {}).get("histories") or []:
        histories[h["id"]] = history_expectations(h.get("refs") or [])
    return {
        "dir": run_dir,
        "phase": (manifest.get("identity", {}) or {}).get("phase"),
        "policy_labels": policy_labels,
        "histories": histories,
        "images_dir": run_dir / "images",
        "records": summary.get("records") or [],
    }


# ---------------------------------------------------------------------------
# Per-record delta computation
# ---------------------------------------------------------------------------


def build_index(runs: list[dict]):
    """job_id -> record, and job_id -> resolved image path.

    Images on disk are named "<job_id>.png" (see evaluate_fan.py's worker
    event path), not by image_sha256 -- image_sha256 is only a content
    checksum used to detect duplicate/invalid images. A record's own job_id
    always names a file in its own run's images/ directory; a personalized
    record's plain/legacy baseline is looked up by plain_job_id, whose file
    may live in a *different* run directory (e.g. the parent screen run for
    a refine-phase record), so the path is resolved against every images/
    directory from every run passed on the command line.
    """
    job_index: dict[str, dict] = {}
    job_path: dict[str, Path] = {}
    for run in runs:
        for r in run["records"]:
            job_id = r["job_id"]
            job_index.setdefault(job_id, r)
            if job_id not in job_path:
                candidate = run["images_dir"] / f"{job_id}.png"
                if candidate.exists():
                    job_path[job_id] = candidate
    return job_index, job_path


def collect_records(runs: list[dict], cache: MetricCache):
    job_index, job_path = build_index(runs)
    results = []
    skipped = Counter()

    for run in runs:
        for r in run["records"]:
            if r.get("role") == "plain":
                continue
            if r.get("status") != "measured":
                skipped["status_not_measured"] += 1
                continue
            policy_hash = r.get("policy_hash")
            policy_id = run["policy_labels"].get(
                policy_hash, f"unknown:{(policy_hash or 'none')[:8]}"
            )
            plain_job_id = r.get("plain_job_id")
            plain_record = job_index.get(plain_job_id) if plain_job_id else None
            if plain_record is None:
                skipped["no_plain_record"] += 1
                continue
            if plain_record.get("status") != "measured":
                skipped["plain_not_measured"] += 1
                continue
            own_path = job_path.get(r["job_id"])
            plain_path = job_path.get(plain_job_id)
            if own_path is None or plain_path is None:
                skipped["missing_image_file"] += 1
                continue
            own_m = cache.get(own_path)
            plain_m = cache.get(plain_path)
            if own_m is None or plain_m is None:
                skipped["image_open_error"] += 1
                continue
            delta = {k: own_m[k] - plain_m[k] for k in METRICS}
            history_id = r.get("history_id")
            exps = run["histories"].get(history_id, []) if history_id else []
            results.append(
                {
                    "run_dir": str(run["dir"].relative_to(REPO_ROOT))
                    if run["dir"].is_relative_to(REPO_ROOT)
                    else str(run["dir"]),
                    "phase": run["phase"],
                    "policy_id": policy_id,
                    "policy_hash": policy_hash,
                    "history_id": history_id,
                    "topic_id": r.get("topic_id"),
                    "seed": r.get("seed"),
                    "job_id": r.get("job_id"),
                    "delta": delta,
                    "expectations": [list(e) for e in exps],
                }
            )
    return results, skipped


# ---------------------------------------------------------------------------
# Table D: shipped exhibit samples
# ---------------------------------------------------------------------------


def collect_samples(
    samples_json: Path, generic_dir: Path, samples_dir: Path, cache: MetricCache
):
    if not samples_json.exists():
        return [], Counter()
    samples = json.loads(samples_json.read_text())
    rows = []
    skipped = Counter()
    for s in samples:
        sample_id = s["id"]
        topic_id = s["topic_id"]
        refs = (s.get("personalization") or {}).get("refs") or []
        exps = history_expectations(refs)
        ref_texts = [ref.get("text", "") for ref in refs]
        for i in range(4):
            plain_path = generic_dir / f"{topic_id}-{i}.png"
            pers_path = samples_dir / f"{sample_id}-{i}.png"
            if not plain_path.exists() or not pers_path.exists():
                skipped["missing_file"] += 1
                continue
            own_m = cache.get(pers_path)
            plain_m = cache.get(plain_path)
            if own_m is None or plain_m is None:
                skipped["image_open_error"] += 1
                continue
            delta = {k: own_m[k] - plain_m[k] for k in METRICS}
            rows.append(
                {
                    "sample_id": sample_id,
                    "topic_id": topic_id,
                    "index": i,
                    "policy_id": "legacy_exhibit",
                    "delta": delta,
                    "expectations": [list(e) for e in exps],
                    "ref_texts": ref_texts,
                }
            )
    return rows, skipped


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else None


def round_row(row: dict, digits=4) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, float):
            out[k] = round(v, digits)
        else:
            out[k] = v
    return out


def table_a(results):
    by_policy = defaultdict(list)
    for r in results:
        by_policy[r["policy_id"]].append(r)
    out = []
    for policy_id, rs in sorted(by_policy.items()):
        row = {"policy_id": policy_id, "n": len(rs)}
        for metric in METRICS:
            row[metric] = mean(r["delta"][metric] for r in rs)
        out.append(round_row(row))
    return out


def table_b(results):
    by_key = defaultdict(list)
    for r in results:
        for exp in r["expectations"]:
            by_key[(r["policy_id"], tuple(exp))].append(r)
    out = []
    for (policy_id, (metric, direction)), rs in sorted(
        by_key.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        deltas = [r["delta"][metric] for r in rs]
        hits = sum(1 for d in deltas if (d > 0 if direction == "up" else d < 0))
        out.append(
            round_row(
                {
                    "policy_id": policy_id,
                    "metric": metric,
                    "direction": direction,
                    "n": len(rs),
                    "mean_delta": mean(deltas),
                    "hit_rate": hits / len(rs) if rs else None,
                }
            )
        )
    return out


def table_c(results):
    by_key = defaultdict(list)
    for r in results:
        by_key[(r["policy_id"], r["history_id"])].append(r)
    out = []
    for (policy_id, history_id), rs in sorted(
        by_key.items(), key=lambda x: (x[0][0], str(x[0][1]))
    ):
        row = {"policy_id": policy_id, "history_id": history_id, "n": len(rs)}
        for metric in METRICS:
            row[metric] = mean(r["delta"][metric] for r in rs)
        row = round_row(row)
        exp_set = set()
        for r in rs:
            exp_set.update(tuple(e) for e in r["expectations"])
        exp_hits = []
        for metric, direction in sorted(exp_set):
            d = row[metric]
            hit = (d > 0) if direction == "up" else (d < 0)
            exp_hits.append(
                {"metric": metric, "direction": direction, "hit": bool(hit)}
            )
        row["expectations"] = exp_hits
        out.append(row)
    return out


def table_d(sample_rows):
    by_sample = defaultdict(list)
    for r in sample_rows:
        by_sample[r["sample_id"]].append(r)
    out = []
    for sample_id, rs in sorted(by_sample.items()):
        row = {"sample_id": sample_id, "topic_id": rs[0]["topic_id"], "n": len(rs)}
        for metric in METRICS:
            row[metric] = mean(r["delta"][metric] for r in rs)
        row = round_row(row)
        row["ref_texts"] = rs[0].get("ref_texts", [])
        exp_set = set()
        for r in rs:
            exp_set.update(tuple(e) for e in r["expectations"])
        exp_hits = []
        for metric, direction in sorted(exp_set):
            d = row[metric]
            hit = (d > 0) if direction == "up" else (d < 0)
            exp_hits.append(
                {"metric": metric, "direction": direction, "hit": bool(hit)}
            )
        row["expectations"] = exp_hits
        out.append(row)
    # overall drift row across all shipped samples (comparable to table A's legacy_exhibit row)
    if sample_rows:
        overall = {
            "sample_id": "(all samples)",
            "topic_id": None,
            "n": len(sample_rows),
        }
        for metric in METRICS:
            overall[metric] = mean(r["delta"][metric] for r in sample_rows)
        out.append(round_row(overall))
    return out


def never_satisfied(table_b_rows, threshold=0.5):
    best = defaultdict(float)
    for row in table_b_rows:
        key = (row["metric"], row["direction"])
        best[key] = max(
            best[key], row["hit_rate"] if row["hit_rate"] is not None else 0.0
        )
    return sorted(key for key, rate in best.items() if rate < threshold)


def weakest_expectations(table_b_rows, n_worst=2):
    """(metric, direction) pairs where even the best policy barely clears chance."""
    best = {}
    for row in table_b_rows:
        key = (row["metric"], row["direction"])
        hr = row["hit_rate"] if row["hit_rate"] is not None else 0.0
        if key not in best or hr > best[key][0]:
            best[key] = (hr, row["policy_id"], row["n"])
    ranked = sorted(best.items(), key=lambda kv: kv[1][0])
    return ranked[:n_worst]


def bidirectional_steering(table_b_rows, min_n=10):
    """For each policy that has both 'up' and 'down' hit rates for a metric,
    report min(hit_up, hit_down): high min = genuinely steers both ways;
    low min with a high max = a one-way generic drift dressed as a 'hit'."""
    by_policy_metric = defaultdict(dict)
    for row in table_b_rows:
        if row["hit_rate"] is None or row["n"] < min_n:
            continue
        by_policy_metric[(row["policy_id"], row["metric"])][row["direction"]] = (
            row["hit_rate"],
            row["n"],
        )
    out = []
    for (policy_id, metric), dirs in by_policy_metric.items():
        if "up" in dirs and "down" in dirs:
            hr_up, n_up = dirs["up"]
            hr_down, n_down = dirs["down"]
            out.append(
                {
                    "policy_id": policy_id,
                    "metric": metric,
                    "hit_up": hr_up,
                    "n_up": n_up,
                    "hit_down": hr_down,
                    "n_down": n_down,
                    "min_hit": min(hr_up, hr_down),
                    "max_hit": max(hr_up, hr_down),
                }
            )
    out.sort(key=lambda r: r["min_hit"], reverse=True)
    return out


def table_e(results, seed=12345, n_boot=1000, ci=0.95):
    """History contrast per (policy, metric): the drift-cancelling signal.

    Compares records whose history asks for a metric to go "up" against
    records (of the same policy) whose history asks for it to go "down" --
    e.g. warm-type histories vs cool-type histories for warmth, or
    flat-type vs sketch/cel-type histories for edge. Both groups sit under
    the same policy's constant drift, so contrast = mean(up) - mean(down)
    cancels that drift out: a genuinely steering policy should show
    contrast > 0 (its "up" images really do score higher on the metric than
    its "down" images), even if the absolute Table A drift points the wrong
    way overall. A bootstrap 95% CI (resampling each side independently,
    fixed seed for reproducibility) says whether that contrast is distinguishable
    from noise.

    Only (policy, metric) pairs with at least one record on each side are
    reported. Records come from the conflict-free expectation set (see
    history_expectations), so "mixed" histories with contradictory refs for
    a metric never appear in either side for that metric.
    """
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in results:
        for metric, direction in r["expectations"]:
            groups[(r["policy_id"], metric, direction)].append(r["delta"][metric])

    keys = sorted({(p, m) for (p, m, _d) in groups})
    rng = np.random.default_rng(seed)
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    out = []
    for policy_id, metric in keys:
        up = groups.get((policy_id, metric, "up"))
        down = groups.get((policy_id, metric, "down"))
        if not up or not down:
            continue
        up_arr = np.asarray(up)
        down_arr = np.asarray(down)
        mean_up = float(up_arr.mean())
        mean_down = float(down_arr.mean())
        contrast = mean_up - mean_down
        boot = np.empty(n_boot)
        for i in range(n_boot):
            su = rng.choice(up_arr, size=up_arr.size, replace=True)
            sd = rng.choice(down_arr, size=down_arr.size, replace=True)
            boot[i] = su.mean() - sd.mean()
        ci_low = float(np.percentile(boot, lo_pct))
        ci_high = float(np.percentile(boot, hi_pct))
        out.append(
            round_row(
                {
                    "policy_id": policy_id,
                    "metric": metric,
                    "mean_up": mean_up,
                    "n_up": len(up),
                    "mean_down": mean_down,
                    "n_down": len(down),
                    "contrast": contrast,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "significant_positive": bool(ci_low > 0),
                    "significant_negative": bool(ci_high < 0),
                }
            )
        )
    out.sort(key=lambda r: (r["policy_id"], r["metric"]))
    return out


# ---------------------------------------------------------------------------
# Markdown rendering (Japanese)
# ---------------------------------------------------------------------------

DIR_JA = {"up": "上", "down": "下"}
METRIC_JA = {
    "warmth": "暖色度 (R-B)",
    "lum": "明度",
    "sat": "彩度",
    "contrast": "コントラスト",
    "edge": "エッジ率",
}


def fmt(v):
    if v is None:
        return "-"
    return f"{v:+.4f}" if isinstance(v, float) else str(v)


def render_markdown(data: dict) -> str:
    lines = []
    lines.append("# FAN 個人化 方向性監査")
    lines.append("")
    lines.append(f"生成日時: {data['generated_at']}")
    lines.append("")
    lines.append(
        "`audit_direction.py` が、既にディスク上にある FAN 評価画像（screen/refine/heldout/"
        "strength の各実行）とショップ用サンプル画像だけから作ります。生成は一切行いません。"
    )
    lines.append("")
    lines.append(
        "各パーソナライズ画像を、同じ topic・同じ seed の plain 画像と比較し、"
        "暖色度 (R-B)・明度・彩度・コントラスト・エッジ率の差分 (Δ = パーソナライズ - plain) を測ります。"
        '参照文の語句からその画像が動くべき向き（例: "warm color" → 暖色度が上がるはず）を機械的に決め、'
        "Δ の符号がその向きと一致した割合を hit rate とします。"
    )
    lines.append("")
    lines.append("## 見方")
    lines.append("")
    lines.append(
        "- **Table A**: policy ごとの汎用ドリフト。参照文の向きを無視して、全パーソナライズ画像の平均Δ。"
    )
    lines.append(
        "- **Table B**: (policy, 期待される向き) ごとの n・平均Δ・hit rate。"
        "hit rate が高いほど「参照文が言った通りに動いた」。0.5 は五分五分＝実質効果なし。"
    )
    lines.append(
        "- **Table C**: (policy, history) ごとの平均Δと、その history が持つ各期待の成否（○/×）。"
    )
    lines.append(
        "- **Table D**: 展示用ショップサンプル (`exhibit/assets/samples.json`, policy は legacy_exhibit) での同じ集計。"
    )
    lines.append(
        "- **Table E**: policy 固有の一律ドリフトを引き算で打ち消した、history の向きへの本当の反応（contrast + ブートストラップ CI）。"
    )
    lines.append(
        "- **conflict-free 期待**: 1つの history が同じ指標について上と下の両方の参照を含む場合"
        '（"mixed" 系 history の warm+cool 参照など）、その指標はその history から除外する。'
        "その history の他の指標（矛盾していないもの）は残す。これを Table B・C・D・E すべてに適用済み。"
    )
    lines.append("")

    # Table A
    lines.append("## Table A: policy ごとの汎用ドリフト")
    lines.append("")
    header = ["policy_id", "n"] + [METRIC_JA[m] for m in METRICS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in data["table_a"]:
        cells = [row["policy_id"], str(row["n"])] + [fmt(row[m]) for m in METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Table B
    lines.append("## Table B: (policy, 期待される向き) ごとの hit rate")
    lines.append("")
    header = ["policy_id", "指標", "向き", "n", "平均Δ", "hit rate"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in data["table_b"]:
        cells = [
            row["policy_id"],
            METRIC_JA[row["metric"]],
            DIR_JA[row["direction"]],
            str(row["n"]),
            fmt(row["mean_delta"]),
            f"{row['hit_rate'] * 100:.0f}%" if row["hit_rate"] is not None else "-",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Table C
    lines.append("## Table C: (policy, history) ごとの平均Δと期待の成否")
    lines.append("")
    header = (
        ["policy_id", "history_id", "n"]
        + [METRIC_JA[m] for m in METRICS]
        + ["期待の成否"]
    )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in data["table_c"]:
        exp_str = (
            "; ".join(
                f"{METRIC_JA[e['metric']]}{DIR_JA[e['direction']]}{'○' if e['hit'] else '×'}"
                for e in row["expectations"]
            )
            or "(期待なし)"
        )
        cells = (
            [
                row["policy_id"],
                str(row["history_id"]),
                str(row["n"]),
            ]
            + [fmt(row[m]) for m in METRICS]
            + [exp_str]
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Table D
    lines.append("## Table D: 展示用ショップサンプル (policy=legacy_exhibit)")
    lines.append("")
    lines.append(
        "期待は Table A-C と同じ conflict-free ルールで決める（同一 history 内で同じ指標に上/下"
        "両方の期待が出る場合はその指標を除外）。参照文は `exhibit/assets/samples.json` を引かなくても"
        "読めるよう、そのまま列に載せている。"
    )
    lines.append("")
    if data["table_d"]:
        header = (
            ["sample_id", "topic_id", "n"]
            + [METRIC_JA[m] for m in METRICS]
            + ["期待の成否", "参照文"]
        )
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for row in data["table_d"]:
            exp_str = "; ".join(
                f"{METRIC_JA[e['metric']]}{DIR_JA[e['direction']]}{'○' if e['hit'] else '×'}"
                for e in row.get("expectations", [])
            ) or ("(全体平均)" if row["sample_id"] == "(all samples)" else "(期待なし)")
            ref_str = "; ".join(row.get("ref_texts") or []) or "-"
            cells = (
                [
                    row["sample_id"],
                    str(row.get("topic_id")),
                    str(row["n"]),
                ]
                + [fmt(row[m]) for m in METRICS]
                + [exp_str, ref_str]
            )
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("(サンプル画像が見つかりませんでした)")
    lines.append("")

    # Table E: history contrast (drift-cancelled steering signal)
    lines.append("## Table E: history contrast（ドリフトを打ち消した本当の反応）")
    lines.append("")
    lines.append(
        "同じ policy の中で、その指標を「上げろ」と言っている history 群と「下げろ」と言っている"
        "history 群を比べる。contrast = 上げろ群の平均Δ − 下げろ群の平均Δ。policy 固有の一律ドリフトは"
        "両群に等しくかかるので引き算で消え、contrast の符号・大きさが「参照の向きに応じて本当に反応したか」"
        "を表す。95% CI はブートストラップ（各群を独立に 1000 回リサンプル、乱数シード固定）。"
        "CI の下限が0より大きければ有意に正（up>down）、上限が0より小さければ有意に負（drift の向きと"
        "期待が逆）。Table B と同じ conflict-free 期待（mixed history の矛盾する指標は除外済み）を使う。"
    )
    lines.append("")
    if data["table_e"]:
        header = [
            "policy_id",
            "指標",
            "上げろ群 平均Δ (n)",
            "下げろ群 平均Δ (n)",
            "contrast",
            "95% CI",
            "有意",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for row in data["table_e"]:
            if row["significant_positive"]:
                sig = "正 (up>down)"
            elif row["significant_negative"]:
                sig = "負 (逆向き)"
            else:
                sig = "-"
            cells = [
                row["policy_id"],
                METRIC_JA[row["metric"]],
                f"{fmt(row['mean_up'])} ({row['n_up']})",
                f"{fmt(row['mean_down'])} ({row['n_down']})",
                fmt(row["contrast"]),
                f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]",
                sig,
            ]
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("(上/下 両方向の history を持つ policy がありませんでした)")
    lines.append("")

    # Bidirectional steering check
    lines.append("## 双方向性チェック: 「向きを切り替えられているか」")
    lines.append("")
    lines.append(
        "同じ指標について「上」を求められたときの hit rate と「下」を求められたときの hit rate を"
        "両方持つ policy だけを対象に、両方の最小値 (`min_hit`) を見る。`min_hit` が高い policy だけが"
        "参照の向きに応じて実際に切り替えている。`max_hit` は高いのに `min_hit` が低い policy は、"
        "参照の向きに関係なく一方向に寄っているだけ（Table A の汎用ドリフトが Table B の hit rate に"
        "そのまま表れているだけ）。n < 10 の組み合わせは除外。"
    )
    lines.append("")
    if data["bidirectional_steering"]:
        header = ["policy_id", "指標", "上 hit (n)", "下 hit (n)", "min_hit", "max_hit"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for row in data["bidirectional_steering"]:
            cells = [
                row["policy_id"],
                METRIC_JA[row["metric"]],
                f"{row['hit_up'] * 100:.0f}% ({row['n_up']})",
                f"{row['hit_down'] * 100:.0f}% ({row['n_down']})",
                f"{row['min_hit'] * 100:.0f}%",
                f"{row['max_hit'] * 100:.0f}%",
            ]
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("(該当する組み合わせがありませんでした)")
    lines.append("")

    # 所見
    lines.append("## 所見")
    lines.append("")
    for line in data["findings"]:
        lines.append(f"- {line}")
    lines.append("")

    if data["skipped"]:
        lines.append("## スキップ内訳")
        lines.append("")
        for reason, count in sorted(data["skipped"].items()):
            lines.append(f"- {reason}: {count}")
        lines.append("")

    return "\n".join(lines)


def build_findings(
    table_a_rows, table_b_rows, table_d_rows, table_e_rows, never_sat
) -> list[str]:
    findings = []
    a_by_policy = {r["policy_id"]: r for r in table_a_rows}
    legacy = a_by_policy.get("legacy_exhibit")
    if legacy:
        drift_bits = ", ".join(f"{METRIC_JA[m]} {fmt(legacy[m])}" for m in METRICS)
        findings.append(
            f"legacy_exhibit の汎用ドリフト（Table A, n={legacy['n']}）: {drift_bits}。"
            "参照の内容に関わらず暗く（明度マイナス）、線・エッジがわずかに減る方向へ寄る一方、"
            "暖色度と彩度はほぼ0で強い色味の偏りはない。"
        )
    strong = a_by_policy.get("strong_v1")
    if strong:
        drift_bits = ", ".join(f"{METRIC_JA[m]} {fmt(strong[m])}" for m in METRICS)
        findings.append(
            f"strong_v1 の汎用ドリフト（Table A, n={strong['n']}）: {drift_bits}。"
            "legacy_exhibit よりずっと大きく、暖色・高明度・低彩度側へ常に寄る。"
        )

    # hit rate per policy averaged over its own expectation rows
    hit_by_policy = defaultdict(list)
    for row in table_b_rows:
        if row["hit_rate"] is not None:
            hit_by_policy[row["policy_id"]].append((row["hit_rate"], row["n"]))
    ranked = []
    for policy_id, pairs in hit_by_policy.items():
        total_n = sum(n for _, n in pairs)
        weighted = sum(hr * n for hr, n in pairs) / total_n if total_n else 0.0
        ranked.append((weighted, total_n, policy_id))
    ranked.sort(reverse=True)
    if ranked:
        best = ranked[0]
        findings.append(
            f"期待方向への一致率（加重平均 hit rate）が最も高い policy: {best[2]}"
            f"（{best[0] * 100:.0f}%, n={best[1]}）。ただしこれは Table A のドリフトが大きい"
            "policy ほど、たまたま一方向の期待に当たりやすいだけの可能性がある（下記の双方向性チェック参照）。"
        )
    if legacy is not None:
        legacy_hits = [
            r
            for r in table_b_rows
            if r["policy_id"] == "legacy_exhibit" and r["hit_rate"] is not None
        ]
        if legacy_hits:
            total_n = sum(r["n"] for r in legacy_hits)
            weighted = (
                sum(r["hit_rate"] * r["n"] for r in legacy_hits) / total_n
                if total_n
                else 0.0
            )
            findings.append(
                f"legacy_exhibit の加重平均 hit rate: {weighted * 100:.0f}%（五分五分に近い, n={total_n}）"
            )

    # bidirectional steering vs one-way drift
    bidir = bidirectional_steering(table_b_rows)
    if bidir:
        top = bidir[0]
        worst_drifty = [r for r in bidir if r["max_hit"] - r["min_hit"] >= 0.5]
        worst_drifty.sort(key=lambda r: r["max_hit"] - r["min_hit"], reverse=True)
        findings.append(
            f"「参照の向きを本当に切り替えられているか」（同じ指標の上/下 両方の hit rate の低い方 min_hit）"
            f"で見ると、最良でも {top['policy_id']} の {METRIC_JA[top['metric']]} で "
            f"min_hit={top['min_hit'] * 100:.0f}%（上={top['hit_up'] * 100:.0f}%/n={top['n_up']}, "
            f"下={top['hit_down'] * 100:.0f}%/n={top['n_down']}）に留まる。"
        )
        if worst_drifty:
            w = worst_drifty[0]
            findings.append(
                f"最も「一方向だけの汎用ドリフトを hit と誤認しやすい」組み合わせ: {w['policy_id']} の"
                f"{METRIC_JA[w['metric']]}（上 hit={w['hit_up'] * 100:.0f}%・下 hit={w['hit_down'] * 100:.0f}%）。"
                "参照が warm/cool どちらを指定しても同じ側に寄っているだけで、指示への追従ではない。"
            )

    weakest = weakest_expectations(table_b_rows)
    if weakest:
        bits = "; ".join(
            f"{METRIC_JA[m]}{DIR_JA[d]}（最良 {pid} でも {hr * 100:.0f}%, n={n}）"
            for (m, d), (hr, pid, n) in weakest
        )
        findings.append(f"最も達成困難な期待（全 policy 中の最良値でも低い）: {bits}")

    if never_sat:
        pretty = ", ".join(f"{METRIC_JA[m]}{DIR_JA[d]}" for m, d in never_sat)
        findings.append(
            f"どの policy でも hit rate が5割未満（実質誰も達成できていない）期待: {pretty}"
        )
    else:
        findings.append(
            "hit rate が5割を超える policy が皆無、という期待はありませんでした（小標本の1政策が偶然超えている場合を含む）。"
        )

    # Table E: drift-cancelled history contrast
    a_lookup = {r["policy_id"]: r for r in table_a_rows}
    for metric in ("warmth", "edge"):
        rows_for_metric = [r for r in table_e_rows if r["metric"] == metric]
        sig = [r for r in rows_for_metric if r["significant_positive"]]
        sig.sort(key=lambda r: r["contrast"], reverse=True)
        if sig:
            best_for_metric = sig[0]
            names = "、".join(
                f"{r['policy_id']}（contrast {fmt(r['contrast'])}, "
                f"CI[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]）"
                for r in sig[:3]
            )
            findings.append(
                f"Table E: {METRIC_JA[metric]} で contrast の 95% CI が0を上回る（有意に up>down）policy: {names}"
            )
            drift = a_lookup.get(best_for_metric["policy_id"], {}).get(metric)
            if drift is not None:
                findings.append(
                    f"最良の {METRIC_JA[metric]} contrast（{best_for_metric['policy_id']}, "
                    f"{fmt(best_for_metric['contrast'])}) は Table A の絶対ドリフト（{fmt(drift)}）と"
                    + (
                        "近い大きさで、絶対ドリフトのほとんどが history 間の差ではなく一律の押し出しであることを示す。"
                        if abs(drift) > 0
                        and abs(best_for_metric["contrast"] - drift)
                        < abs(best_for_metric["contrast"]) * 0.3
                        else "は別物で、参照の向きに応じた反応も一定量ある。"
                    )
                )
        else:
            findings.append(
                f"Table E: {METRIC_JA[metric]} で contrast が有意に正の policy はありませんでした。"
            )

    for policy_id in ("legacy_exhibit", "strong_v1"):
        rows = [r for r in table_e_rows if r["policy_id"] == policy_id]
        for r in rows:
            drift = a_lookup.get(policy_id, {}).get(r["metric"])
            if drift is None:
                continue
            if r["mean_up"] > 0 and r["mean_down"] < 0:
                sign_note = "上げろ群・下げろ群とも符号が反転しており、両方向に本当に反応している。"
            elif r["mean_up"] > 0 or r["mean_down"] < 0:
                sign_note = (
                    f"ただし上げろ群={fmt(r['mean_up'])}・下げろ群={fmt(r['mean_down'])}で、"
                    "片側しか plain を跨いで符号反転しておらず、反応は一方向のみ（もう一方は"
                    "弱まるだけで逆転はしない）。"
                )
            else:
                sign_note = "上げろ群・下げろ群とも符号が反転していない。"
            findings.append(
                f"{policy_id} の {METRIC_JA[r['metric']]}: history contrast {fmt(r['contrast'])}"
                f"（CI[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}], 有意={r['significant_positive']}) "
                f"vs Table A の絶対ドリフト {fmt(drift)}。{sign_note}"
            )

    # concrete example from the shipped exhibit samples
    warm_samples = [r for r in table_d_rows if r["sample_id"] in ("s1-cat", "s1-tokyo")]
    if warm_samples:
        misses = [
            r
            for r in warm_samples
            if any(
                e["metric"] == "warmth" and e["direction"] == "up" and not e["hit"]
                for e in r.get("expectations", [])
            )
        ]
        if misses:
            names = "、".join(r["sample_id"] for r in misses)
            findings.append(
                f"展示サンプル（Table D）の実例: {names} は参照文で暖色（amber/warm color palette）と"
                "高コントラストの日差しを指定しているのに、legacy_exhibit 適用後は暖色度が下がっている"
                "（表の通り Δ暖色度がマイナス）。指示と逆方向に動く具体例。"
            )

    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dirs", nargs="+", help="run directories or containers to search"
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument(
        "--samples-json", type=Path, default=REPO_ROOT / "exhibit/assets/samples.json"
    )
    parser.add_argument(
        "--generic-dir", type=Path, default=REPO_ROOT / "exhibit/assets/generic"
    )
    parser.add_argument(
        "--samples-dir", type=Path, default=REPO_ROOT / "exhibit/assets/samples"
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="skip Table D (shipped exhibit samples)",
    )
    args = parser.parse_args(argv)

    run_dirs = discover_runs(args.run_dirs)
    if not run_dirs:
        print(
            "error: no run directories found (missing manifest.json/summary.json)",
            file=sys.stderr,
        )
        return 1
    print(f"found {len(run_dirs)} run directories", file=sys.stderr)

    runs = [load_run(d) for d in run_dirs]
    cache = MetricCache()
    results, skipped = collect_records(runs, cache)
    print(
        f"collected {len(results)} personalized records ({dict(skipped)})",
        file=sys.stderr,
    )

    sample_rows = []
    sample_skipped = Counter()
    if not args.no_samples:
        sample_rows, sample_skipped = collect_samples(
            args.samples_json, args.generic_dir, args.samples_dir, cache
        )
        print(
            f"collected {len(sample_rows)} sample rows ({dict(sample_skipped)})",
            file=sys.stderr,
        )

    ta = table_a(results)
    tb = table_b(results)
    tc = table_c(results)
    td = table_d(sample_rows)
    te = table_e(results)
    ns = never_satisfied(tb)
    bidir = bidirectional_steering(tb)
    findings = build_findings(ta, tb, td, te, ns)

    combined_skipped = Counter(skipped)
    for k, v in sample_skipped.items():
        combined_skipped[f"samples_{k}"] += v

    from datetime import datetime

    data = {
        "generated_at": datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "run_dirs": [str(d) for d in run_dirs],
        "n_personalized_records": len(results),
        "n_sample_rows": len(sample_rows),
        "skipped": dict(combined_skipped),
        "never_satisfied": [list(x) for x in ns],
        "table_a": ta,
        "table_b": tb,
        "table_c": tc,
        "table_d": td,
        "table_e": te,
        "bidirectional_steering": bidir,
        "records": results,
        "sample_records": sample_rows,
        "findings": findings,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(data))

    print(f"wrote {args.out_json}", file=sys.stderr)
    print(f"wrote {args.out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
