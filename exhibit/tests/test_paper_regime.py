"""A2 paper-regime job planning: CPU-only, no torch/GPU touched."""

import importlib.util

from exhibit.config import ROOT


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper_regime = load_script("fan_paper_regime")


def test_default_job_count_is_two_topics_by_plain_plus_history_alpha_grid():
    args = paper_regime.parse_args([])
    _, jobs = paper_regime.build_job_plan(args)
    # 2 topics x (1 plain + 2 histories x 2 non-zero alphas) = 10, for 1 seed.
    assert len(jobs) == 10
    assert len({job["topic"] for job in jobs}) == 2
    assert sum(1 for job in jobs if job["kind"] == "plain") == 2
    assert sum(1 for job in jobs if job["kind"] == "personalized") == 8


def test_histories_loaded_from_histories_strength_fixture_by_id():
    args = paper_regime.parse_args([])
    histories, _ = paper_regime.build_job_plan(args)
    assert [history["id"] for history in histories] == ["warm-sentence", "cool-sentence"]
    for history in histories:
        assert history["refs"], f"{history['id']} has no refs"
        for ref in history["refs"]:
            assert isinstance(ref["text"], str) and ref["text"]
            assert isinstance(ref["weight"], (int, float))


def test_output_filenames_are_unique():
    args = paper_regime.parse_args([])
    _, jobs = paper_regime.build_job_plan(args)
    files = [job["file"] for job in jobs]
    assert len(files) == len(set(files))
    plain_files = {job["file"] for job in jobs if job["kind"] == "plain"}
    assert plain_files == {"cat-plain-alpha0.0-seed42.png", "tokyo-plain-alpha0.0-seed42.png"}


def test_alpha_zero_is_reference_free_and_not_duplicated_per_history():
    args = paper_regime.parse_args([])
    _, jobs = paper_regime.build_job_plan(args)
    plain_jobs = [job for job in jobs if job["kind"] == "plain"]
    assert all(job["history"] is None and job["refs"] is None for job in plain_jobs)
    assert all(job["alpha"] == 0.0 for job in plain_jobs)


def test_multiple_seeds_scale_the_job_count():
    args = paper_regime.parse_args(["--seeds", "1", "2", "3"])
    _, jobs = paper_regime.build_job_plan(args)
    assert len(jobs) == 30
    files = [job["file"] for job in jobs]
    assert len(files) == len(set(files))


def test_format_alpha_matches_filename_convention():
    assert paper_regime.format_alpha(0.0) == "0.0"
    assert paper_regime.format_alpha(0.4) == "0.4"
    assert paper_regime.format_alpha(0.7) == "0.7"
