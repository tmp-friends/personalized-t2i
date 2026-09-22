"""The strength report: only what ran is reported, and the judge key stays hidden."""

from __future__ import annotations

import copy
import importlib.util
import json
import math

import pytest
from PIL import Image

from exhibit.config import ROOT
from exhibit.evaluation import (
    build_experiment,
    load_strength_config,
    register_experiment,
    score_records,
    select_candidates,
    summarize_records,
)

CONFIG_PATH = ROOT / "configs/fan-strength.json"
EXPERIMENT_ID = "e1-settings"
# role -> (angle in the 2-d embedding plane, fill colour)
ROLES = {
    "plain": (0.0, (200, 200, 200)),
    "legacy": (0.02, (60, 120, 200)),
}
CANDIDATE_ANGLES = (0.10, 0.06)
CANDIDATE_COLOURS = ((220, 120, 60), (120, 200, 120))
PROVENANCE = {
    "fan_pin": "9d0b76843f6437718195accac9cf3f050a25d26b",
    "adapter_hash": "adapter",
    "worker_hash": "worker",
    "environment": {"python": "3.12", "torch": "test"},
}


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script("build_strength_report")


def trimmed_config(seeds=1):
    """The real experiment, cut to 2 topics x 2 histories x N seeds x 2 policies."""
    config = load_strength_config(CONFIG_PATH, EXPERIMENT_ID)
    config["topics"] = config["topics"][:2]
    config["histories"] = config["histories"][:2]
    config["seeds"] = config["seeds"][:seeds]
    config["policies"] = config["policies"][:2]
    return config


def fake_run(tmp_path, seeds=1):
    """A complete strength run built by the real evaluation functions."""
    config = trimmed_config(seeds)
    experiment = build_experiment(config, PROVENANCE)
    outputs = tmp_path / "strength"
    directory, checkpoint = register_experiment(outputs, experiment, resume=False)
    order = [item["policy_hash"] for item in config["policies"]]

    images, texts = {}, {}
    for job in experiment["jobs"]:
        if job["role"] == "candidate":
            index = order.index(job["policy_hash"])
            angle, colour = CANDIDATE_ANGLES[index], CANDIDATE_COLOURS[index]
        else:
            angle, colour = ROLES[job["role"]]
        path = directory / job["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 80), color=colour).save(path)
        images[job["job_id"]] = {
            "embedding": [math.cos(angle), math.sin(angle)],
            "valid": True,
            "sha256": job["job_id"][:8],
            "seconds": 1.0,
        }
        texts[job["target_text_id"]] = [1.0, 0.0]
        for ref in job.get("refs", []):
            texts["ref:" + ref["ref_id"]] = [0.0, 1.0]
        checkpoint["jobs"][job["job_id"]] = {"status": "done", "attempts": 1}
    checkpoint["runs"] = [
        {
            "started_at": "2026-09-22T09:00:00Z",
            "finished_at": "2026-09-22T10:00:00Z",
            "status": "finished",
        }
    ]
    (directory / "checkpoint.json").write_text(json.dumps(checkpoint))

    records = score_records(
        json.loads((directory / "manifest.json").read_text()),
        {"images": images, "texts": texts, "conditioning": {}},
    )
    rules = {
        **copy.deepcopy(config["rules"]),
        "stage": "strength",
        "expected_cases": len(config["topics"])
        * len(config["histories"])
        * len(config["seeds"]),
        "numerical": {
            policy_hash: {"passed": True, "evidence_hash": "evidence"}
            for policy_hash in order
        },
    }
    decision = select_candidates(records, rules)
    metrics = summarize_records(records, "strength", config["rules"])
    for name, value in (
        ("records", records),
        ("metrics", metrics),
        ("decision", decision),
    ):
        (directory / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False))
    return {
        "outputs": outputs,
        "directory": directory,
        "config": config,
        "policy_ids": [item["policy_id"] for item in config["policies"]],
        "records": records,
    }


@pytest.fixture
def run(tmp_path):
    return fake_run(tmp_path)


def build(tmp_path, run, *, answers=(), extra=()):
    report = tmp_path / "report"
    argv = [
        "--config",
        str(CONFIG_PATH),
        "--outputs",
        str(run["outputs"]),
        "--report",
        str(report),
    ]
    if answers:
        argv += ["--judge-answers", *[str(item) for item in answers]]
    for item in extra:
        argv += ["--extra-run", item]
    result = builder.main(argv)
    return report, result


def undeclared_run(run):
    """The same run as a formal-chain one: no display block, so no experiment id."""
    path = run["directory"] / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["display"]
    path.write_text(json.dumps(manifest))
    return run["directory"]


def test_readme_reports_the_run_and_ranks_policies_by_history_delta(tmp_path, run):
    report, result = build(tmp_path, run)
    text = (report / "README.md").read_text()

    assert result["experiments"][EXPERIMENT_ID] == builder.DONE
    assert f"| `{EXPERIMENT_ID}` | {builder.DONE} |" in text
    first, second = run["policy_ids"]
    # The stronger policy (a larger history angle) has to come first.
    assert text.index(f"| `{first}` |") < text.index(f"| `{second}` |")
    assert "Δhistory vs legacy" in text
    assert "履歴別 Δhistory vs legacy" in text


def test_experiments_without_a_run_are_not_run(tmp_path, run):
    report, result = build(tmp_path, run)
    text = (report / "README.md").read_text()

    assert result["experiments"]["e2-adapter"] == builder.NOT_RUN
    assert f"| `e2-adapter` | {builder.NOT_RUN} |" in text
    assert not (report / "e2-adapter").exists()


def test_every_case_gets_a_sheet_one_column_per_policy(tmp_path, run):
    report, _ = build(tmp_path, run)
    config = run["config"]
    expected = len(config["topics"]) * len(config["histories"]) * len(config["seeds"])
    sheets = sorted((report / EXPERIMENT_ID).glob("sheet-*.jpg"))

    assert len(sheets) == expected
    columns = 2 + len(config["policies"])  # plain, legacy, then the candidates
    for sheet in sheets:
        with Image.open(sheet) as image:
            assert image.size[0] == columns * builder.TILE[0]
    for topic in config["topics"]:
        for history in config["histories"]:
            name = f"sheet-{topic['id']}-{history['id']}-seed{config['seeds'][0]}.jpg"
            assert (report / EXPERIMENT_ID / name).is_file()
    assert (report / EXPERIMENT_ID / "overview.jpg").is_file()
    assert (report / EXPERIMENT_ID / "index.md").is_file()


def test_judge_tasks_cover_the_candidates_without_leaking_the_key(tmp_path, run):
    report, _ = build(tmp_path, run)
    judge = report / EXPERIMENT_ID / "judge"
    tasks = json.loads((judge / "tasks.json").read_text())["tasks"]
    key = json.loads((judge / "key.json").read_text())
    candidates = [item for item in run["records"] if item["role"] == "candidate"]

    assert len(tasks) == len(candidates)
    assert {item["pair_id"] for item in tasks} == set(key)
    raw = (judge / "tasks.json").read_text()
    for entry in key.values():
        assert entry["policy_id"] not in raw
        assert entry["policy_hash"] not in raw
        assert entry["job_id"] not in raw
        assert entry["candidate_side"] in {"A", "B"}
    for task in tasks:
        assert (judge / task["image"]).is_file()
        assert task["history_text"] and task["target_text"]


def test_merged_answers_give_the_win_rate_over_decided_pairs(tmp_path, run):
    report, _ = build(tmp_path, run)
    key = json.loads(
        (report / EXPERIMENT_ID / "judge/key.json").read_text()
    )
    policy_id = run["policy_ids"][0]
    pairs = sorted(
        pair for pair, entry in key.items() if entry["policy_id"] == policy_id
    )[:3]
    answers = {}
    for index, pair in enumerate(pairs):
        side = key[pair]["candidate_side"]
        other = "B" if side == "A" else "A"
        # Two pairs go to the candidate, one to plain: 2 / 3 decided pairs.
        answers[pair] = {
            "preference": side if index < 2 else other,
            "target_kept": {"A": "yes", "B": "yes"},
            "note": "",
        }
    path = tmp_path / "judge-answers.json"
    path.write_text(json.dumps({"judge": "gpt-test", "answers": answers}))

    report, _ = build(tmp_path, run, answers=[path])
    summary = json.loads(
        (report / EXPERIMENT_ID / "judge/summary.json").read_text()
    )
    row = next(
        item for item in summary["ai"]["policies"] if item["policy_id"] == policy_id
    )

    assert (row["wins"], row["losses"]) == (2, 1)
    assert row["win_rate"] == pytest.approx(2 / 3)
    assert summary["human"] is None
    assert "0.6667" in (report / "README.md").read_text()


def test_human_answers_stay_未実施_until_a_reviewer_exports(tmp_path, run):
    report, _ = build(tmp_path, run)
    text = (report / "README.md").read_text()

    assert "### 人による評価" in text
    assert "review.html" in text
    assert "未実施。人の回答を代わりに作ることはしません。" in text


def test_review_html_lists_every_pair(tmp_path, run):
    report, result = build(tmp_path, run)
    page = (report / "review.html").read_text()
    key = json.loads((report / EXPERIMENT_ID / "judge/key.json").read_text())

    assert result["pairs"] == len(key)
    for pair_id in key:
        assert pair_id in page
        assert f"{EXPERIMENT_ID}/judge/pairs/{pair_id}.jpg" in page
    assert "http://" not in page and "https://" not in page


def test_extra_run_without_a_display_block_is_rendered_for_reference(tmp_path, run):
    directory = undeclared_run(run)
    config = run["config"]
    expected = len(config["topics"]) * len(config["histories"]) * len(config["seeds"])

    report, result = build(tmp_path, run, extra=[f"{directory}=heldout-ref"])
    text = (report / "README.md").read_text()

    # Without a display block the run belongs to no declared experiment at all.
    assert result["experiments"][EXPERIMENT_ID] == builder.NOT_RUN
    assert result["extra_runs"] == {"heldout-ref": expected}
    assert "## 参考: 正式チェーンの run" in text
    assert "### `heldout-ref`" in text
    assert "[heldout-ref/index.md](heldout-ref/index.md)" in text
    for policy_id in run["policy_ids"]:
        assert f"| `{policy_id}` |" in text

    sheets = sorted((report / "heldout-ref").glob("sheet-*.jpg"))
    assert len(sheets) == expected
    columns = 2 + len(config["policies"])
    for sheet in sheets:
        with Image.open(sheet) as image:
            assert image.size[0] == columns * builder.TILE[0]
    assert (report / "heldout-ref/overview.jpg").is_file()
    assert (report / "heldout-ref/index.md").is_file()

    # Reference only: no judge pairs and nothing to review blind.
    assert not (report / "heldout-ref/judge").exists()
    assert result["pairs"] == 0
    assert "heldout-ref" not in (report / "review.html").read_text()


@pytest.mark.parametrize(
    "spec", ["/nowhere/at/all=missing", "no-separator", "{directory}=e2-adapter"]
)
def test_unusable_extra_runs_are_reported_not_guessed(tmp_path, run, spec):
    directory = undeclared_run(run)

    report, result = build(tmp_path, run, extra=[spec.format(directory=directory)])

    assert result["extra_runs"] == {}
    assert len(result["problems"]) == 1
    assert "## 入力の問題" in (report / "README.md").read_text()


def outside_and_inside_details(text):
    """The gallery folds later seeds away; this separates the two halves."""
    outside, inside, depth = [], [], 0
    for line in text.splitlines():
        if line.startswith("<details>"):
            depth += 1
            continue
        if line.startswith("</details>"):
            depth -= 1
            continue
        (inside if depth else outside).append(line)
    assert depth == 0
    return "\n".join(outside), "\n".join(inside)


def test_gallery_lists_every_condition_with_its_settings(tmp_path, run):
    report, result = build(tmp_path, run)
    text = (report / "GALLERY.md").read_text()

    assert result["gallery"] == str(report / "GALLERY.md")
    assert "[GALLERY.md](GALLERY.md)" in (report / "README.md").read_text()
    assert "## 共通の生成設定" in text
    # Every column of a sheet has a row that says how it was made.
    assert "| plain（個人化なし） |" in text
    assert "| `legacy_exhibit` |" in text
    for policy_id in run["policy_ids"]:
        assert f"| `{policy_id}` |" in text
    for name in ("alpha", "skip_pa", "pooled_mode", "profiling", "reference_unit"):
        assert name in text
    # The references behind each history are printed in full.
    for history in run["config"]["histories"]:
        assert f"### 履歴 `{history['id']}`" in text
        for ref in history["refs"]:
            assert ref["text"] in text
    assert "**未実施** — 画像がないので一覧はありません。" in text
    assert "未実施。`a2-paper-regime/report.json` がありません。" in text


def test_gallery_inlines_the_first_seed_and_folds_the_others(tmp_path):
    run = fake_run(tmp_path, seeds=2)
    report, result = build(tmp_path, run)
    text = (report / "GALLERY.md").read_text()
    outside, inside = outside_and_inside_details(text)
    first, second = run["config"]["seeds"]
    topics = len(run["config"]["topics"])
    histories = len(run["config"]["histories"])

    assert result["gallery_images"][EXPERIMENT_ID] == {
        "inline": topics * histories,
        "details": topics * histories,
    }
    assert f"<details><summary>seed {second} のシート（{topics} 枚）</summary>" in text
    for topic in run["config"]["topics"]:
        for history in run["config"]["histories"]:
            head = f"{EXPERIMENT_ID}/sheet-{topic['id']}-{history['id']}-seed"
            assert f"{head}{first}.jpg" in outside
            assert f"{head}{first}.jpg" not in inside
            assert f"{head}{second}.jpg" in inside
            assert f"{head}{second}.jpg" not in outside
    # A blank line each side of the folded block, or the images will not render.
    for part in text.split("<details>")[1:]:
        body = part.split("</summary>", 1)[1]
        assert body.startswith("\n\n")
        assert body.split("</details>")[0].endswith("\n\n")


def test_gallery_covers_an_extra_run_with_its_own_settings(tmp_path, run):
    directory = undeclared_run(run)

    report, result = build(tmp_path, run, extra=[f"{directory}=heldout-ref"])
    text = (report / "GALLERY.md").read_text()

    assert result["gallery_images"]["heldout-ref"]["inline"] == 4
    assert "## `heldout-ref`（strength）" in text
    assert "heldout-ref/sheet-cat-warm-seed" in text
