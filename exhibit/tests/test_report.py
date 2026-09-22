"""The report's three states: absent is 未実施, never a success."""

import importlib.util
import json
import re

import pytest
from exhibit.config import FAN_POLICIES, ROOT


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load_script("build_report")
NOT_RUN, FAILED, PASSED = report.NOT_RUN, report.FAILED, report.PASSED


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """An empty outputs/assets tree, so every block starts from nothing."""
    outputs = tmp_path / "outputs"
    assets = tmp_path / "assets"
    (outputs / "fan-evaluation").mkdir(parents=True)
    assets.mkdir()
    monkeypatch.setattr(report, "OUTPUTS", outputs)
    monkeypatch.setattr(report, "EVALUATION", outputs / "fan-evaluation")
    monkeypatch.setattr(report, "ASSETS", assets)
    monkeypatch.setattr(report, "V2_REVIEW", tmp_path / "cards-v2-review.json")
    return tmp_path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def summary(*, phase="heldout", status="selected", measured=24, failed=0, unmeasured=0):
    counts = {"measured": measured}
    if failed:
        counts["failed"] = failed
    if unmeasured:
        counts["unmeasured"] = unmeasured
    candidate = {
        "policy_hash": "a" * 64,
        "status": "pass" if status == "selected" else "fail",
        "reasons": [] if status == "selected" else ["history_improvement"],
        "mean_history_delta_vs_legacy": 0.01,
        "mean_target_delta_vs_legacy": 0.0,
    }
    return {
        "phase": phase,
        "experiment_hash": "b" * 64,
        "metrics": {
            "record_count": measured + failed + unmeasured,
            "status_counts": counts,
        },
        "decision": {
            "status": status,
            "decision_hash": "c" * 64,
            "candidates": [candidate],
            "selected": [{**candidate, "policy_id": "official_encoder"}]
            if status == "selected"
            else [],
        },
    }


def test_an_absent_experiment_is_not_run_rather_than_passed(tree):
    item = report.experiment_block("heldout", report.experiment_summaries())
    assert item["state"] == NOT_RUN
    assert item["reasons"] == ["heldout_not_run"]
    assert item["missing_rate"] is None


def test_a_failed_experiment_reports_its_reasons_and_missing_rate(tree):
    write(
        tree / "outputs/fan-evaluation" / ("d" * 64) / "summary.json",
        summary(status="keep_legacy", measured=20, failed=4),
    )
    item = report.experiment_block("heldout", report.experiment_summaries())
    assert item["state"] == FAILED
    assert item["missing_rate"] == pytest.approx(4 / 24)
    assert "unmeasured_images:4/24" in item["reasons"]
    assert any("history_improvement" in reason for reason in item["reasons"])
    assert item["detail"]["selected"] == []


def test_a_passed_experiment_needs_a_decision_and_no_missing_image(tree):
    write(
        tree / "outputs/fan-evaluation" / ("e" * 64) / "summary.json",
        summary(measured=24),
    )
    item = report.experiment_block("heldout", report.experiment_summaries())
    assert item["state"] == PASSED
    assert item["missing_rate"] == 0
    assert item["detail"]["selected"][0]["policy_id"] == "official_encoder"

    # One unmeasured image is enough to stop it being a pass.
    write(
        tree / "outputs/fan-evaluation" / ("e" * 64) / "summary.json",
        summary(measured=23, unmeasured=1),
    )
    assert (
        report.experiment_block("heldout", report.experiment_summaries())["state"]
        == FAILED
    )


def test_a_phase_never_borrows_another_phases_summary(tree):
    write(
        tree / "outputs/fan-evaluation" / ("f" * 64) / "summary.json",
        summary(phase="screen"),
    )
    found = report.experiment_summaries()
    assert report.experiment_block("screen", found)["state"] == PASSED
    assert report.experiment_block("heldout", found)["state"] == NOT_RUN


def study(*, answers, eligible, conclusion="not_confirmed"):
    return {
        "study_id": "encoder-preference-1",
        "study_kind": "encoder",
        "status": "collected" if answers else "not_started",
        "participants_in_manifest": 20,
        "participants_with_answers": answers,
        "comparisons": {
            "candidate_vs_legacy": {
                "conclusion": conclusion,
                "mean": 0.6 if answers else None,
                "interval": None,
                "participants_scored": answers,
                "answered_pairs": 8 * answers,
                "pair_count": 160,
                "reasons": [] if eligible else ["insufficient_participants"],
            }
        },
        "default_policy_change": {"eligible": eligible},
    }


def test_a_study_without_answers_is_not_run(tree):
    directory = tree / "outputs/fan-evaluation/study"
    assert report.study_block()["state"] == NOT_RUN
    write(directory / "status.json", {"comparison_images": "not_generated"})
    assert report.study_block()["reasons"][0] == "study_summary_missing"
    write(directory / "summary.json", study(answers=0, eligible=False))
    item = report.study_block()
    assert item["state"] == NOT_RUN
    assert "no_answers" in item["reasons"]
    assert item["missing_rate"] == 1.0


def test_a_study_with_answers_but_no_verdict_is_a_failure(tree):
    directory = tree / "outputs/fan-evaluation/study"
    write(directory / "summary.json", study(answers=3, eligible=False))
    item = report.study_block()
    assert item["state"] == FAILED
    assert "candidate_vs_legacy:insufficient_participants" in item["reasons"]
    write(
        directory / "summary.json",
        study(answers=20, eligible=True, conclusion="improved"),
    )
    assert report.study_block()["state"] == PASSED


def test_catalog_v2_separates_未確認_from_不合格(tree):
    assert report.catalog_v2_block()["state"] == NOT_RUN
    write(
        tree / "assets/catalog-v2.json",
        {"images": {f"girl-{index}": {"path": "x.png"} for index in range(64)}},
    )
    item = report.catalog_v2_block()
    assert item["state"] == NOT_RUN
    assert any("未確認" in reason for reason in item["reasons"])
    assert item["detail"]["generated"] == 64
    card_ids = [card["id"] for card in report.load_catalog()["all_cards"]]
    # Every card looked at, none accepted: a measured failure, not a gap.
    write(
        tree / "cards-v2-review.json",
        {card_id: {"reviewed": False} for card_id in card_ids},
    )
    item = report.catalog_v2_block()
    assert item["state"] == FAILED
    assert item["reasons"] == ["no_usable_cards"]
    assert item["detail"]["reviewed_entries"] == 0


def test_catalog_v2_passes_with_cards_left_out_on_purpose(tree, monkeypatch):
    write(
        tree / "assets/catalog-v2.json",
        {"images": {f"girl-{index}": {"path": "x.png"} for index in range(64)}},
    )
    card_ids = [f"card-{index}" for index in range(4)]
    cards = [{"id": card_id} for card_id in card_ids]
    monkeypatch.setattr(
        report,
        "load_catalog",
        lambda **_: {"all_cards": cards, "cards": cards[1:]},
    )
    write(
        tree / "cards-v2-review.json",
        {card_id: {"reviewed": card_id != "card-0"} for card_id in card_ids},
    )
    item = report.catalog_v2_block()
    assert item["state"] == PASSED
    assert item["reasons"] == ["excluded:card-0"]
    assert item["detail"]["usable_cards"] == 3


def test_the_encoding_diagnostic_separates_a_detected_failure_from_a_broken_run(tree):
    path = tree / "outputs/fan-evaluation/diagnostics/abc/report.json"

    def diagnostic(legacy_passed):
        return {
            "policies": {
                "a" * 64: {
                    "policy_id": "legacy_exhibit",
                    "eligibility": {
                        "passed": legacy_passed,
                        "failures": {}
                        if legacy_passed
                        else {"single": {"cat": ["alpha_zero_vs_no_reference"]}},
                    },
                },
                "b" * 64: {
                    "policy_id": "screen-skip1-fan-all",
                    "eligibility": {
                        "passed": False,
                        "failures": {"single": {"cat": ["alpha_zero_vs_no_reference"]}},
                    },
                },
            },
            "exception_restoration": {"fan_restored_saved_forward": True},
        }

    assert report.encoding_block()["state"] == NOT_RUN
    write(path, diagnostic(True))
    item = report.encoding_block()
    # A candidate policy failing the gate is a measured result, not a broken run.
    assert item["state"] == PASSED
    assert item["reasons"] == ["detected_numerical_failure:screen-skip1-fan-all"]
    assert item["detail"]["passed_count"] == 1
    write(path, diagnostic(False))
    item = report.encoding_block()
    assert item["state"] == FAILED
    assert item["reasons"][0] == "exhibit_policy_failed:legacy_exhibit"


def test_test_logs_distinguish_a_missing_log_from_a_failing_suite(tree):
    assert report.tests_block()["state"] == NOT_RUN
    (tree / "outputs/all-tests.log").write_text("318 passed, 2 warnings in 4.00s\n")
    assert report.tests_block()["state"] == NOT_RUN  # the node log is still missing
    (tree / "outputs/js-tests.log").write_text("# pass 21\n# fail 0\n")
    assert report.tests_block()["state"] == PASSED
    (tree / "outputs/all-tests.log").write_text("1 failed, 317 passed in 4.00s\n")
    item = report.tests_block()
    assert item["state"] == FAILED and "python_failed:1" in item["reasons"]


def browser(*, mock, errors=(), external=()):
    return {
        "url": "http://127.0.0.1:7860",
        "mock": mock,
        "summary": {
            "generate_wait_seconds": 31.2,
            "checks": ["welcome", "compare"],
            "errors": list(errors),
            "console": [],
            "failed_requests": [],
            "external_requests": list(external),
        },
    }


def test_a_mock_browser_run_does_not_pass_for_the_real_machine(tree, tmp_path):
    assert report.browser_block(tmp_path)["state"] == NOT_RUN
    write(tmp_path / "browser-evidence.json", browser(mock=True))
    item = report.browser_block(tmp_path)
    assert item["state"] == NOT_RUN and item["reasons"] == ["mock_api_only"]
    write(tmp_path / "browser-evidence.json", browser(mock=False))
    assert report.browser_block(tmp_path)["state"] == PASSED
    write(
        tmp_path / "browser-evidence.json",
        browser(mock=False, errors=["TypeError"], external=["https://cdn"]),
    )
    item = report.browser_block(tmp_path)
    assert item["state"] == FAILED
    assert item["reasons"] == ["page_errors:1", "external_requests:1"]


def test_a_rehearsal_with_a_failed_session_is_not_a_pass(tree):
    assert report.rehearsal_block()["state"] == NOT_RUN
    measured = {
        "sessions": 20,
        "successes": 20,
        "runs": 60,
        "runs_measured": 60,
        "median_seconds": 28.0,
        "p95_seconds": 34.0,
        "deadline_seconds": 120,
        "failures": [],
        "runs_over_deadline": [],
    }
    write(tree / "outputs/rehearsal.json", measured)
    assert report.rehearsal_block()["state"] == PASSED
    write(
        tree / "outputs/rehearsal.json",
        {
            **measured,
            "successes": 18,
            "failures": [{"index": 3, "failed_step": "identical"}],
            "runs_over_deadline": [{"index": 7, "step": "initial", "seconds": 130}],
        },
    )
    item = report.rehearsal_block()
    assert item["state"] == FAILED
    assert item["missing_rate"] == pytest.approx(2 / 20)
    assert "failed_sessions:2" in item["reasons"]
    assert "runs_over_deadline:1" in item["reasons"]


def test_a_section_is_as_weak_as_its_weakest_block():
    assert report.combine({}) == NOT_RUN
    assert report.combine({"a": {"state": PASSED}, "b": {"state": PASSED}}) == PASSED
    assert report.combine({"a": {"state": PASSED}, "b": {"state": NOT_RUN}}) == NOT_RUN
    assert report.combine({"a": {"state": NOT_RUN}, "b": {"state": FAILED}}) == FAILED


def test_the_page_is_self_contained_and_states_every_verdict(tree, tmp_path):
    evidence = report.collect(tmp_path)
    page = report.render(
        evidence,
        {"cards": [], "generic": [], "samples": [], "catalog_v2": []},
    )
    for state in (NOT_RUN,):
        assert state in page
    for title in ("実装完了", "実機検証", "精度実証"):
        assert title in page
    # No external request of any kind: no remote src/href, link or @import.
    assert not re.search(r'(src|href)="https?://', page)
    assert "<link" not in page and "@import" not in page
    # The old probe output is named only as something explicitly not quoted.
    assert not re.search(r'href="[^"]*fan-(probe|demo)', page)
    assert "引用していません" in page
    # The default policy and catalog are stated, not implied.
    default_policy_id = FAN_POLICIES["default_policy_id"]
    assert default_policy_id in page and "catalog-v2" in page
    text = report.readme(evidence)
    assert "未実施" in text and default_policy_id in text
    assert "catalog-v2" in text
