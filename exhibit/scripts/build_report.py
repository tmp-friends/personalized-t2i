#!/usr/bin/env python3
"""Assemble docs/reports/fan-personalization/ from what is actually on disk.

The three completion states of spec §0/§11 stay separate: 実装完了 / 実機検証 /
精度実証. Every block is rendered as 未実施 / 不合格 / 合格 from exactly one
artifact, with its missing rate and its reasons. A missing artifact is never a
success, and nothing from the old ``outputs/fan-probe`` or
``docs/reports/fan-demo`` is quoted as a current result.

    PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_report.py
"""

import argparse
import base64
import html
import io
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from exhibit.catalog import load_catalog
from exhibit.config import (
    ASSETS,
    CONFIG,
    FAN_POLICIES,
    OUTPUTS,
    REPO,
    read_json,
    write_json,
)
from exhibit.domain import (
    LEGACY_POLICY_ID,
    build_legacy_personalization,
    digest,
    file_hash,
    legacy_policy,
)
from PIL import Image

REPORT = REPO / "docs/reports/fan-personalization"
EVALUATION = OUTPUTS / "fan-evaluation"
V2_REVIEW = REPO / "exhibit/configs/cards-v2-review.json"
EXPERIMENT_DIR = re.compile(r"^[0-9a-f]{64}$")
NOT_RUN, FAILED, PASSED = "未実施", "不合格", "合格"
ORDER = {PASSED: 0, NOT_RUN: 1, FAILED: 2}
PHASES = ("screen", "refine", "heldout")
E = html.escape


def block(state, *, reasons=(), detail=None, missing_rate=None, source=None):
    """One judged artifact. `missing_rate` is None when nothing was measurable."""
    return {
        "state": state,
        "reasons": list(reasons),
        "detail": detail or {},
        "missing_rate": missing_rate,
        "source": str(source) if source else None,
    }


def combine(blocks):
    """A section is only as complete as its weakest block; failure outranks all."""
    states = [item["state"] for item in blocks.values()]
    if not states:
        return NOT_RUN
    return max(states, key=lambda state: ORDER[state])


def read_text(path):
    try:
        return Path(path).read_text()
    except OSError:
        return None


def relative(path):
    try:
        return str(Path(path).relative_to(REPO))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------- 実装完了


def parse_pytest(log):
    if not log:
        return None, None
    passed = re.search(r"(\d+) passed", log)
    failed = re.search(r"(\d+) failed", log)
    return (
        int(passed.group(1)) if passed else None,
        int(failed.group(1)) if failed else 0,
    )


def parse_node(log):
    """node --test prints "ℹ pass N" (spec) or "# pass N" (TAP)."""
    if not log:
        return None, None
    passed = re.search(r"[ℹ#]\s*pass\s+(\d+)", log)
    failed = re.search(r"[ℹ#]\s*fail\s+(\d+)", log)
    return (
        int(passed.group(1)) if passed else None,
        int(failed.group(1)) if failed else None,
    )


def tests_block():
    python_log = read_text(OUTPUTS / "all-tests.log")
    node_log = read_text(OUTPUTS / "js-tests.log")
    py_passed, py_failed = parse_pytest(python_log)
    node_passed, node_failed = parse_node(node_log)
    detail = {
        "python_passed": py_passed,
        "python_failed": py_failed,
        "node_passed": node_passed,
        "node_failed": node_failed,
    }
    if py_passed is None and node_passed is None:
        return block(
            NOT_RUN,
            reasons=["test_logs_missing"],
            detail=detail,
            source=OUTPUTS / "all-tests.log",
        )
    reasons = []
    if py_passed is None:
        reasons.append("python_log_missing")
    if node_passed is None:
        reasons.append("node_log_missing")
    if py_failed:
        reasons.append(f"python_failed:{py_failed}")
    if node_failed:
        reasons.append(f"node_failed:{node_failed}")
    state = PASSED if not reasons else (FAILED if py_failed or node_failed else NOT_RUN)
    return block(state, reasons=reasons, detail=detail, source=OUTPUTS)


def preflight_block():
    data = read_json(OUTPUTS / "preflight.json", None)
    if not isinstance(data, dict):
        return block(
            NOT_RUN,
            reasons=["preflight_not_run"],
            source=OUTPUTS / "preflight.json",
        )
    models = data.get("models")
    detail = {
        "catalog_id": data.get("catalog_id"),
        "reviewed_cards": data.get("reviewed_cards"),
        "cards": data.get("cards"),
        "fixed_images": data.get("fixed_images"),
        "generic_images": data.get("generic_images"),
        "samples": data.get("samples"),
        "models_checked": bool(models),
        "models_ready": models.get("ready") if isinstance(models, dict) else None,
    }
    errors = list(data.get("errors") or [])
    if isinstance(models, dict):
        errors.extend(models.get("errors") or [])
    state = PASSED if data.get("ready") and not errors else FAILED
    return block(
        state, reasons=errors, detail=detail, source=OUTPUTS / "preflight.json"
    )


def legacy_invariant_block():
    """The v1 samples must still rebuild to the hash stored beside their images."""
    samples = read_json(ASSETS / "samples.json", None)
    if not isinstance(samples, list) or not samples:
        return block(
            NOT_RUN, reasons=["samples_missing"], source=ASSETS / "samples.json"
        )
    mismatched, rebuilt = [], {}
    for sample in samples:
        sample_id = sample.get("id") if isinstance(sample, dict) else None
        try:
            value = build_legacy_personalization(sample["selection"])["hash"]
        except (KeyError, TypeError, ValueError) as error:
            mismatched.append(f"{sample_id}:{type(error).__name__}")
            continue
        rebuilt[sample_id] = value
        if value != (sample.get("personalization") or {}).get("hash"):
            mismatched.append(f"{sample_id}:hash_changed")
    baseline = read_json(OUTPUTS / "preparation/legacy-baseline/baseline.json", None)
    frozen = []
    if isinstance(baseline, dict):
        # The frozen v1 assets must be untouched; demo.json itself moved on purpose.
        for path, expected in sorted((baseline.get("images") or {}).items()):
            target = REPO / path
            if not target.is_file() or file_hash(target) != expected:
                frozen.append(path)
        for path, expected in sorted((baseline.get("files") or {}).items()):
            if path.startswith("exhibit/configs/"):
                continue
            target = REPO / path
            if not target.is_file() or file_hash(target) != expected:
                frozen.append(path)
    detail = {
        "samples": len(samples),
        "sample_hashes": rebuilt,
        "policy_id": LEGACY_POLICY_ID,
        "effective_policy": legacy_policy(),
        "frozen_assets_checked": bool(baseline),
        "frozen_assets_changed": frozen,
    }
    reasons = mismatched + [f"changed:{path}" for path in frozen]
    return block(
        PASSED if not reasons else FAILED,
        reasons=reasons,
        detail=detail,
        source=ASSETS / "samples.json",
    )


# --------------------------------------------------------------------- 実機検証


def encoding_block():
    """The newest exact-policy encoding diagnostic, per policy."""
    reports = sorted(
        (EVALUATION / "diagnostics").glob("*/report.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not reports:
        return block(NOT_RUN, reasons=["encoding_not_run"], source=EVALUATION)
    path = reports[-1]
    report = read_json(path, None)
    if not isinstance(report, dict) or not isinstance(report.get("policies"), dict):
        return block(NOT_RUN, reasons=["encoding_report_unreadable"], source=path)
    policies, failing = {}, []
    for policy_hash, value in report["policies"].items():
        eligibility = value.get("eligibility") or {}
        name = value.get("policy_id") or policy_hash[:12]
        policies[policy_hash] = {
            "policy_id": name,
            "passed": bool(eligibility.get("passed")),
            "failed_checks": sorted(
                {
                    check
                    for topics in (eligibility.get("failures") or {}).values()
                    for checks in topics.values()
                    for check in checks
                }
            ),
        }
        if not eligibility.get("passed"):
            failing.append(name)
    restoration = report.get("exception_restoration") or {}
    detail = {
        "policies": policies,
        "policy_count": len(policies),
        "passed_count": sum(1 for item in policies.values() if item["passed"]),
        "fan_restored_after_exception": restoration.get("fan_restored_saved_forward"),
        "versions": report.get("versions"),
        "seconds": report.get("seconds"),
    }
    # §11 B asks whether the comparison ran, not whether every candidate passed.
    # A candidate that fails the numerical gate is a recorded result; it blocks
    # adoption in 精度実証, not the diagnostic itself. The exhibit's own policy
    # failing, or a patch left in place, is a broken run.
    reasons = [f"detected_numerical_failure:{name}" for name in sorted(failing)]
    broken = [
        f"exhibit_policy_failed:{name}"
        for name in sorted(failing)
        if name == LEGACY_POLICY_ID
    ]
    if restoration and not restoration.get("fan_restored_saved_forward"):
        broken.append("patch_not_restored_after_exception")
    return block(
        FAILED if broken else PASSED,
        reasons=broken + reasons,
        detail=detail,
        missing_rate=0.0 if policies else None,
        source=path,
    )


def browser_block(report_dir):
    path = Path(report_dir) / "browser-evidence.json"
    data = read_json(path, None)
    if not isinstance(data, dict) or not data:
        return block(NOT_RUN, reasons=["browser_check_not_run"], source=path)
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else data
    errors = list(summary.get("errors") or [])
    console = list(summary.get("console") or [])
    external = list(summary.get("external_requests") or [])
    failed = list(summary.get("failed_requests") or [])
    detail = {
        "url": data.get("url"),
        "mock": data.get("mock"),
        "checks": len(summary.get("checks") or []),
        "page_errors": len(errors),
        "console_messages": len(console),
        "external_requests": len(external),
        "failed_requests": len(failed),
        "generate_wait_seconds": summary.get("generate_wait_seconds"),
    }
    reasons = []
    if errors:
        reasons.append(f"page_errors:{len(errors)}")
    if console:
        reasons.append(f"console_messages:{len(console)}")
    if external:
        reasons.append(f"external_requests:{len(external)}")
    if failed:
        reasons.append(f"failed_requests:{len(failed)}")
    if data.get("mock"):
        # A mock run proves the screen, not the machine; §11 B wants the server.
        reasons.append("mock_api_only")
    state = PASSED if not reasons else (NOT_RUN if data.get("mock") else FAILED)
    return block(state, reasons=reasons, detail=detail, source=path)


def rehearsal_block():
    path = OUTPUTS / "rehearsal.json"
    data = read_json(path, None)
    if not isinstance(data, dict) or not data.get("sessions"):
        return block(NOT_RUN, reasons=["rehearsal_not_run"], source=path)
    sessions = data["sessions"]
    successes = data.get("successes", 0)
    over = data.get("runs_over_deadline") or []
    detail = {
        "sessions": sessions,
        "successes": successes,
        "runs": data.get("runs"),
        "runs_measured": data.get("runs_measured"),
        "median_seconds": data.get("median_seconds"),
        "p95_seconds": data.get("p95_seconds"),
        "max_seconds": data.get("max_seconds"),
        "deadline_seconds": data.get("deadline_seconds"),
        "peak_vram_mib": data.get("peak_vram_mib"),
        "exact_cache_runs": data.get("exact_cache_runs"),
        "cancelled_runs": data.get("cancelled_runs"),
    }
    reasons = []
    if successes != sessions:
        reasons.append(f"failed_sessions:{sessions - successes}")
    reasons.extend(
        f"session_{item.get('index')}:{item.get('failed_step')}"
        for item in (data.get("failures") or [])
    )
    if over:
        reasons.append(f"runs_over_deadline:{len(over)}")
    missing = (sessions - successes) / sessions if sessions else None
    return block(
        PASSED if not reasons else FAILED,
        reasons=reasons,
        detail=detail,
        missing_rate=missing,
        source=path,
    )


def catalog_v2_block():
    """Generated, reviewed and why; the review file is the only proof of review."""
    findings = read_json(OUTPUTS / "preparation/catalog-v2/review-findings.json", None)
    manifest = read_json(ASSETS / "catalog-v2.json", None)
    generated = len((manifest or {}).get("images") or {}) if manifest else 0
    review = read_json(V2_REVIEW, {}) or {}
    reviewed_entries = sum(
        1
        for value in review.values()
        if isinstance(value, dict) and value.get("reviewed")
    )
    total, eligible = None, None
    try:
        catalog = load_catalog("catalog-v2", reviewed_only=True)
        total = len(catalog["all_cards"])
        eligible = len(catalog["cards"])
    except (OSError, TypeError, ValueError) as error:
        total, eligible = None, None
        catalog_error = f"catalog_unreadable:{error}"
    else:
        catalog_error = None
    detail = {
        "generated": generated,
        "expected": total,
        "reviewed_entries": reviewed_entries,
        "usable_cards": eligible,
        "findings_present": isinstance(findings, dict),
        "decision": (findings or {}).get("decision"),
        "visual_findings": (findings or {}).get("visual_findings"),
        "clip_probe": (findings or {}).get("clip_probe"),
        "checked_by": (findings or {}).get("checked_by"),
        "checked_at": (findings or {}).get("checked_at"),
    }
    reasons = []
    if catalog_error:
        reasons.append(catalog_error)
    if not generated:
        return block(
            NOT_RUN,
            reasons=reasons + ["catalog_v2_not_generated"],
            detail=detail,
            source=ASSETS / "catalog-v2.json",
        )
    if total and eligible == total:
        return block(PASSED, reasons=reasons, detail=detail, source=V2_REVIEW)
    if isinstance(findings, dict) and findings.get("decision"):
        # A recorded, failed visual check is a result, not a missing measurement.
        reasons.append(str(findings["decision"]))
        state = FAILED
    else:
        reasons.append("未確認: review-findings.json がありません")
        state = NOT_RUN
    missing = None
    if total:
        missing = (total - (eligible or 0)) / total
    return block(
        state,
        reasons=reasons,
        detail=detail,
        missing_rate=missing,
        source=OUTPUTS / "preparation/catalog-v2/review-findings.json",
    )


# --------------------------------------------------------------------- 精度実証


def experiment_summaries():
    """The newest summary.json per phase, keyed by phase name."""
    found = {}
    if not EVALUATION.is_dir():
        return found
    for directory in sorted(EVALUATION.iterdir()):
        if not directory.is_dir() or not EXPERIMENT_DIR.match(directory.name):
            continue
        path = directory / "summary.json"
        summary = read_json(path, None)
        if not isinstance(summary, dict) or summary.get("phase") not in PHASES:
            continue
        phase = summary["phase"]
        previous = found.get(phase)
        if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
            found[phase] = (path, summary)
    return found


def experiment_block(phase, found):
    entry = found.get(phase)
    if entry is None:
        return block(NOT_RUN, reasons=[f"{phase}_not_run"], source=EVALUATION)
    path, summary = entry
    decision = summary.get("decision") or {}
    metrics = summary.get("metrics") or {}
    counts = metrics.get("status_counts") or {}
    total = metrics.get("record_count") or sum(counts.values()) or 0
    measured = counts.get("measured", 0)
    missing = (total - measured) / total if total else None
    detail = {
        "experiment_hash": summary.get("experiment_hash"),
        "directory": relative(path.parent),
        "decision_status": decision.get("status"),
        "decision_hash": decision.get("decision_hash"),
        "status_counts": counts,
        "record_count": total,
        "selected": [
            {
                "policy_id": item.get("policy_id"),
                "policy_hash": item.get("policy_hash"),
                "mean_history_delta_vs_legacy": item.get(
                    "mean_history_delta_vs_legacy"
                ),
                "mean_target_delta_vs_legacy": item.get("mean_target_delta_vs_legacy"),
            }
            for item in decision.get("selected") or []
        ],
        "candidates": [
            {
                "policy_hash": item.get("policy_hash"),
                "status": item.get("status"),
                "reasons": item.get("reasons"),
                "mean_history_delta_vs_legacy": item.get(
                    "mean_history_delta_vs_legacy"
                ),
                "mean_target_delta_vs_legacy": item.get("mean_target_delta_vs_legacy"),
            }
            for item in decision.get("candidates") or []
        ],
    }
    reasons = []
    unmeasured = total - measured
    if unmeasured:
        reasons.append(f"unmeasured_images:{unmeasured}/{total}")
    for item in decision.get("candidates") or []:
        if item.get("status") != "pass":
            reasons.extend(
                f"{item.get('policy_hash', '?')[:12]}:{reason}"
                for reason in item.get("reasons") or [item.get("status")]
            )
    if decision.get("status") == "selected" and not unmeasured:
        state = PASSED
    elif decision.get("status"):
        state = FAILED
    else:
        state = NOT_RUN
        reasons.append("decision_missing")
    return block(
        state, reasons=reasons, detail=detail, missing_rate=missing, source=path
    )


def study_block():
    directory = EVALUATION / "study"
    status = read_json(directory / "status.json", None)
    summary = read_json(directory / "summary.json", None)
    if not isinstance(status, dict) and not isinstance(summary, dict):
        return block(NOT_RUN, reasons=["study_not_built"], source=directory)
    if not isinstance(summary, dict):
        return block(
            NOT_RUN,
            reasons=["study_summary_missing", str(status.get("comparison_images"))],
            detail={"status": status},
            source=directory / "status.json",
        )
    comparisons = summary.get("comparisons") or {}
    detail = {
        "study_id": summary.get("study_id"),
        "study_kind": summary.get("study_kind"),
        "status": summary.get("status"),
        "participants_in_manifest": summary.get("participants_in_manifest"),
        "participants_with_answers": summary.get("participants_with_answers"),
        "participants_required": summary.get("participants_required"),
        "comparisons": {
            name: {
                "conclusion": item.get("conclusion"),
                "mean": item.get("mean"),
                "interval": item.get("interval"),
                "participants_scored": item.get("participants_scored"),
                "answered_pairs": item.get("answered_pairs"),
                "pair_count": item.get("pair_count"),
                "reasons": item.get("reasons"),
            }
            for name, item in comparisons.items()
        },
        "default_policy_change_eligible": (
            summary.get("default_policy_change") or {}
        ).get("eligible"),
    }
    answered = sum(item.get("answered_pairs") or 0 for item in comparisons.values())
    pairs = sum(item.get("pair_count") or 0 for item in comparisons.values())
    missing = (pairs - answered) / pairs if pairs else None
    if not summary.get("participants_with_answers"):
        return block(
            NOT_RUN,
            reasons=["no_answers"] + list(summary.get("reasons") or []),
            detail=detail,
            missing_rate=missing,
            source=directory / "summary.json",
        )
    reasons = [
        f"{name}:{reason}"
        for name, item in comparisons.items()
        for reason in item.get("reasons") or []
    ]
    eligible = (summary.get("default_policy_change") or {}).get("eligible")
    return block(
        PASSED if eligible else FAILED,
        reasons=reasons,
        detail=detail,
        missing_rate=missing,
        source=directory / "summary.json",
    )


# ----------------------------------------------------------------- collection


def collect(report_dir):
    found = experiment_summaries()
    implementation = {
        "自動テスト": tests_block(),
        "Preflight（資産・モデル）": preflight_block(),
        "v1資産と個人化ハッシュの不変": legacy_invariant_block(),
    }
    verification = {
        "encoding 数値検査": encoding_block(),
        "実ブラウザーの一連操作": browser_block(report_dir),
        "連続セッション実測": rehearsal_block(),
        "catalog v2 の確認": catalog_v2_block(),
    }
    accuracy = {
        "screen（8条件の絞り込み）": experiment_block("screen", found),
        "refine（alpha 追加比較）": experiment_block("refine", found),
        "heldout（事前基準の判定）": experiment_block("heldout", found),
        "本人によるブラインド評価": study_block(),
    }
    sections = {
        "実装完了": {"state": combine(implementation), "blocks": implementation},
        "実機検証": {"state": combine(verification), "blocks": verification},
        "精度実証": {"state": combine(accuracy), "blocks": accuracy},
    }
    policy_id = FAN_POLICIES["default_policy_id"]
    configuration = {
        "default_policy_id": policy_id,
        "policies": sorted(FAN_POLICIES.get("policies", {})),
        "effective_default_policy": legacy_policy()
        if policy_id == LEGACY_POLICY_ID
        else None,
        "catalog_id": CONFIG["catalog_id"],
        "sample_manifest_catalog_id": (CONFIG.get("sample_manifest") or {}).get(
            "catalog_id"
        ),
        "selection": CONFIG["selection"],
        "seeds": CONFIG["seeds"],
        "timeout_seconds": CONFIG["timeout_seconds"],
        "fan_pin": CONFIG["fan"]["commit"],
        "settings_source": "exhibit/configs/fan-policies.json",
    }
    evidence = {
        "schema_version": 1,
        "generated_at": datetime.now(ZoneInfo("Asia/Tokyo")).strftime(
            "%Y-%m-%d %H:%M JST"
        ),
        "configuration": configuration,
        "sections": sections,
        "statements": [
            "未実施の評価を成功として扱わない。欠損は欠損として数える。",
            "旧 outputs/fan-probe と docs/reports/fan-demo の数値は現行設定の結果として引用しない。",
            "論文の定量結果をこの展示の性能として記載しない。",
            "人の回答は収集していない。代答もしない。",
        ],
    }
    evidence["inputs_hash"] = digest(
        {"configuration": configuration, "sections": sections}
    )
    return evidence


# ----------------------------------------------------------------------- HTML

STYLE = """
:root{color-scheme:dark;--bg:#100d0c;--panel:#191514;--raise:#221c1a;--line:#30282a;
--ink:#f8f1eb;--sub:#ab9f97;--mute:#75695f;--a1:#ff9a6b;--a2:#ff6f93;--on:#1a0e0a;
--ok:#8fd6a4;--warn:#ffd089;--bad:#ff8c8c;
--sans:"Noto Sans CJK JP","Noto Sans JP","Hiragino Kaku Gothic ProN","Yu Gothic",system-ui,sans-serif;
--mono:ui-monospace,"SFMono-Regular",Consolas,"Liberation Mono",monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.85;font-size:15px}
main{max-width:1060px;margin:auto;padding:56px 24px 96px}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;
border-bottom:1px solid var(--line);padding-bottom:20px;flex-wrap:wrap}
.brand{font-size:26px;font-weight:800;letter-spacing:-.5px;
background:linear-gradient(135deg,var(--a1),var(--a2));-webkit-background-clip:text;
background-clip:text;color:transparent}
.meta{font-family:var(--mono);font-size:11px;color:var(--mute);text-align:right}
h1{font-size:clamp(26px,4vw,40px);font-weight:800;letter-spacing:-1px;margin:40px 0 14px}
h2{font-size:22px;font-weight:700;margin:56px 0 8px}
h3{font-size:15px;font-weight:700;margin:26px 0 6px}
p{color:var(--sub);margin:10px 0}
a{color:var(--a1);text-underline-offset:3px}
code,pre{font-family:var(--mono)}
pre{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;
overflow:auto;font-size:12px;color:var(--ink)}
.chips{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0 30px}
.chip{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);
background:var(--panel);border-radius:999px;padding:9px 16px;font-size:12px;color:var(--sub)}
.chip b{color:var(--ink);font-weight:700}
.badge{display:inline-block;border-radius:999px;padding:3px 11px;font-size:12px;font-weight:700;
border:1px solid transparent}
.s-pass{background:rgba(143,214,164,.14);color:var(--ok);border-color:rgba(143,214,164,.35)}
.s-none{background:rgba(255,208,137,.12);color:var(--warn);border-color:rgba(255,208,137,.32)}
.s-fail{background:rgba(255,140,140,.12);color:var(--bad);border-color:rgba(255,140,140,.32)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px 22px;margin:14px 0}
.panel h3{margin-top:0;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:14px 0 4px}
.kv div{background:var(--raise);border-radius:10px;padding:10px 12px}
.kv span{display:block;font-size:11px;color:var(--mute)}
.kv b{font-size:15px;font-weight:700;overflow-wrap:anywhere}
ul{margin:10px 0;padding-left:20px;color:var(--sub)}
li{margin:3px 0;overflow-wrap:anywhere}
.reasons li{font-family:var(--mono);font-size:12px}
.src{font-family:var(--mono);font-size:11px;color:var(--mute);margin-top:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:12px;margin:14px 0}
figure{margin:0}
figure img{width:100%;border-radius:10px;border:1px solid var(--line);display:block}
figcaption{font-size:10px;font-family:var(--mono);color:var(--mute);margin-top:6px;overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mute);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.scroll{overflow:auto}
footer{margin-top:64px;border-top:1px solid var(--line);padding-top:20px;
font-size:11px;font-family:var(--mono);color:var(--mute)}
@media(max-width:640px){main{padding:32px 16px 64px}.meta{text-align:left}}
"""

BADGE = {PASSED: "s-pass", NOT_RUN: "s-none", FAILED: "s-fail"}


def badge(state):
    return f'<span class="badge {BADGE[state]}">{E(state)}</span>'


def thumb(path, size=200, quality=70):
    try:
        image = Image.open(path).convert("RGB")
    except (OSError, ValueError):
        return None
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def figures(items):
    if not items:
        return "<p>未準備。</p>"
    return '<div class="grid">' + "".join(items) + "</div>"


def gallery():
    """Only images that exist: reviewed v1 cards, generic shots, v1 samples."""
    parts = {}
    try:
        catalog = load_catalog(CONFIG["catalog_id"], reviewed_only=False)
    except (OSError, TypeError, ValueError):
        catalog = {"cards": [], "all_cards": []}
    reviewed = {
        card["id"]
        for card in load_catalog(CONFIG["catalog_id"], reviewed_only=True)["cards"]
    }
    cards = []
    for card in catalog["all_cards"]:
        src = thumb(ASSETS / card["path"], size=150)
        if not src:
            continue
        mark = "" if card["id"] in reviewed else " · 未確認"
        cards.append(
            f'<figure><img alt="{E(card["id"])}" loading="lazy" src="{src}">'
            f"<figcaption>{E(card['id'])}{mark}</figcaption></figure>"
        )
    parts["cards"] = cards
    generic = []
    for topic in CONFIG["topics"]:
        src = thumb(ASSETS / "generic" / f"{topic['id']}-0.png", size=150)
        if src:
            generic.append(
                f'<figure><img alt="{E(topic["id"])}" loading="lazy" src="{src}">'
                f"<figcaption>{E(topic['id'])} · seed {CONFIG['seeds'][0]}</figcaption></figure>"
            )
    parts["generic"] = generic
    samples = []
    for sample in read_json(ASSETS / "samples.json", []) or []:
        images = sample.get("images") or []
        if not images:
            continue
        src = thumb(ASSETS / images[0]["path"], size=150)
        if src:
            samples.append(
                f'<figure><img alt="{E(sample["id"])}" loading="lazy" src="{src}">'
                f"<figcaption>{E(sample['id'])}</figcaption></figure>"
            )
    parts["samples"] = samples
    v2 = []
    catalog_v2 = read_json(ASSETS / "catalog-v2.json", None)
    for key, image in sorted(((catalog_v2 or {}).get("images") or {}).items()):
        if not key.startswith("girl-"):
            continue
        src = thumb(ASSETS / image["path"], size=150)
        if src:
            v2.append(
                f'<figure><img alt="{E(key)}" loading="lazy" src="{src}">'
                f"<figcaption>{E(key)} · 未確認</figcaption></figure>"
            )
    parts["catalog_v2"] = v2
    return parts


def block_html(name, item):
    reasons = (
        '<ul class="reasons">'
        + "".join(f"<li>{E(str(reason))}</li>" for reason in item["reasons"])
        + "</ul>"
        if item["reasons"]
        else "<p>指摘なし。</p>"
    )
    missing = (
        f"{item['missing_rate'] * 100:.1f}%"
        if isinstance(item["missing_rate"], (int, float))
        else "—"
    )
    cells = "".join(
        f"<div><span>{E(str(key))}</span><b>{E(str(value))}</b></div>"
        for key, value in item["detail"].items()
        if not isinstance(value, (dict, list)) and value is not None
    )
    source = (
        f'<p class="src">artifact: {E(relative(item["source"]))}</p>'
        if item["source"]
        else ""
    )
    return f"""<div class="panel"><h3>{E(name)} {badge(item["state"])}
<span class="src">欠損率 {E(missing)}</span></h3>
<div class="kv">{cells}</div>{reasons}{source}</div>"""


def section_html(title, section, lead):
    blocks = "".join(block_html(name, item) for name, item in section["blocks"].items())
    return f"""<h2>{E(title)} {badge(section["state"])}</h2>
<p>{lead}</p>{blocks}"""


def catalog_v2_html(item):
    findings = item["detail"].get("visual_findings") or {}
    probe = item["detail"].get("clip_probe") or {}
    if not findings and not probe:
        return ""
    rows = "".join(
        f"<tr><td>{E(axis)}</td><td>{E(str(text))}</td>"
        f"<td>{E(str((probe.get('top1') or {}).get(axis, '—')))}"
        f" / {E(str(probe.get('of', '—')))}</td></tr>"
        for axis, text in findings.items()
    )
    note = f"<p>{E(str(probe['note']))}</p>" if probe.get("note") else ""
    return f"""<h3>catalog v2 の所見</h3>
<div class="scroll"><table><thead><tr><th>軸</th><th>目視の所見</th>
<th>凍結CLIPで水準が1位</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>4軸すべてが1位だったカードは {E(str(probe.get("all_four_axes", "—")))} / {E(str(probe.get("of", "—")))}。
これは診断補助であり、確認（review）そのものではありません。</p>{note}"""


def render(evidence, parts):
    configuration = evidence["configuration"]
    sections = evidence["sections"]
    chips = "".join(
        f'<span class="chip">{E(title)} <b>{E(section["state"])}</b></span>'
        for title, section in sections.items()
    )
    statements = "".join(f"<li>{E(text)}</li>" for text in evidence["statements"])
    config_rows = "".join(
        f"<tr><td>{E(key)}</td><td><code>{E(json.dumps(value, ensure_ascii=False))}</code></td></tr>"
        for key, value in configuration.items()
    )
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FAN 個人化改善 · 実装と検証の状態</title><style>{STYLE}</style></head>
<body><main>
<header><div class="brand">fan personalization</div>
<div class="meta">IMPLEMENTATION / ON-DEVICE / ACCURACY<br>{E(evidence["generated_at"])}<br>
inputs {E(evidence["inputs_hash"][:16])}</div></header>
<h1>実装・実機検証・精度実証を<br>別々に判定した状態。</h1>
<div class="chips">{chips}</div>
<p>このページはディスク上の成果物だけから生成しています。成果物がない項目は
<b>未実施</b>であり、成功とは扱いません。基準を満たさなかった項目は<b>不合格</b>として
理由を残します。過去の <code>outputs/fan-probe</code> や <code>docs/reports/fan-demo</code>
の数値は現行設定の結果として引用していません。</p>
<h2>現在の設定</h2>
<p>エンコーダー設定の正は <code>exhibit/configs/fan-policies.json</code> の一箇所です。
<code>configs/demo.json</code> にはパス・pin・decoder hash だけが残り、alpha や skip_pa は
持ちません。評価が揃うまで既定 policy は <code>legacy_exhibit</code>、展示の catalog は
<code>catalog-v1</code> のままです。</p>
<div class="scroll"><table><thead><tr><th>項目</th><th>値</th></tr></thead>
<tbody>{config_rows}</tbody></table></div>
{section_html("実装完了", sections["実装完了"], "設計 §11 A。コードと資産の整合が取れているか。")}
{section_html("実機検証", sections["実機検証"], "設計 §11 B。実際のGPU・実ブラウザー・実カタログで確かめたか。")}
{catalog_v2_html(sections["実機検証"]["blocks"]["catalog v2 の確認"])}
{section_html("精度実証", sections["精度実証"], "設計 §11 C。事前に固定した基準と本人の回答で、改善を示せたか。")}
<h2>固定資産</h2>
<h3>カード（{E(CONFIG["catalog_id"])}）</h3>
{figures(parts["cards"])}
<h3>お題ごとの通常生成（seed {E(str(CONFIG["seeds"][0]))}）</h3>
{figures(parts["generic"])}
<h3>代表サンプル（v1の選択から事前生成）</h3>
{figures(parts["samples"])}
<h3>catalog v2 の生成画像（girl のみ抜粋・すべて未確認）</h3>
{figures(parts["catalog_v2"])}
<h2>説明で守っていること</h2>
<div class="panel"><ul>{statements}</ul></div>
<h2>再開コマンド</h2>
<pre>PYTHONPATH=exhibit/src exhibit/.venv/bin/python -m pytest exhibit/tests -q
node --test exhibit/tests/test_browser_state.mjs
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/preflight.py --models
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py encoding --config exhibit/configs/fan-evaluation.json
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py screen  --config exhibit/configs/fan-evaluation.json --resume
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py refine  --config exhibit/configs/fan-evaluation.json --resume
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py heldout --config exhibit/configs/fan-evaluation.json --resume
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_preference_study.py --config exhibit/configs/fan-evaluation.json
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py study   --config exhibit/configs/fan-evaluation.json --resume
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/summarize_preference_study.py --study exhibit/outputs/fan-evaluation/study
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/rehearsal.py --sessions 20 --url http://127.0.0.1:7860
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/browser_check.py --url http://127.0.0.1:7860 --report-dir docs/reports/fan-personalization
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_report.py</pre>
<p>GPU を使うコマンドは同時に実行しないでください。再実行するとこのページと
<a href="evidence.json">evidence.json</a> は、そのときのディスクの内容から作り直されます。</p>
<footer>FAN PERSONALIZATION · 実測と未実施を分けて記載しています ·
<a href="README.md">README.md</a> · <a href="evidence.json">evidence.json</a></footer>
</main></body></html>"""


def readme(evidence):
    sections = evidence["sections"]
    configuration = evidence["configuration"]
    lines = [
        "# FAN 個人化改善 · 検証レポート",
        "",
        f"生成日時: {evidence['generated_at']}　入力hash: `{evidence['inputs_hash']}`",
        "",
        "`build_report.py` がディスク上の成果物だけから作ります。成果物がない項目は",
        "**未実施**であり、成功として扱いません。再実行すると同じ入力からは同じ判定になります。",
        "",
        "## 3つの完了状態",
        "",
        "| 状態 | 判定 | 内訳 |",
        "|---|---|---|",
    ]
    for title, section in sections.items():
        inner = " / ".join(
            f"{name}: {item['state']}" for name, item in section["blocks"].items()
        )
        lines.append(f"| {title} | **{section['state']}** | {inner} |")
    lines += [
        "",
        "## 現在の設定",
        "",
        (
            f"- 既定 policy: `{configuration['default_policy_id']}`"
            "（評価が揃うまで変更しない）"
        ),
        (
            f"- 展示の catalog: `{configuration['catalog_id']}`"
            f"（サンプルの catalog: `{configuration['sample_manifest_catalog_id']}`）"
        ),
        (
            "- エンコーダー設定の正: `exhibit/configs/fan-policies.json`。"
            "`configs/demo.json` はパス・pin・decoder hash のみ。"
        ),
        f"- FAN pin: `{configuration['fan_pin']}`",
        "",
        "## 残っている作業",
        "",
    ]
    remaining = [
        f"- {title} / {name}: {item['state']}"
        + (
            f"（{', '.join(str(r) for r in item['reasons'][:3])}）"
            if item["reasons"]
            else ""
        )
        for title, section in sections.items()
        for name, item in section["blocks"].items()
        if item["state"] != PASSED
    ]
    lines += remaining or ["- なし"]
    lines += [
        "",
        "## 再実行",
        "",
        "```bash",
        "PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_report.py",
        "```",
        "",
        "人の回答は収集していません。代わりの回答を作ることはしません。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    parser.add_argument(
        "--no-images", action="store_true", help="skip the embedded thumbnails"
    )
    args = parser.parse_args(argv)

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    evidence = collect(report_dir)
    parts = (
        {"cards": [], "generic": [], "samples": [], "catalog_v2": []}
        if args.no_images
        else gallery()
    )
    write_json(report_dir / "evidence.json", evidence)
    (report_dir / "index.html").write_text(render(evidence, parts))
    (report_dir / "README.md").write_text(readme(evidence))
    print(
        json.dumps(
            {
                "report": str(report_dir / "index.html"),
                "inputs_hash": evidence["inputs_hash"],
                "states": {
                    title: section["state"]
                    for title, section in evidence["sections"].items()
                },
            },
            ensure_ascii=False,
        )
    )
    return evidence


if __name__ == "__main__":
    main()
