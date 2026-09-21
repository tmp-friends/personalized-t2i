"""Blind preference study: collection page, merge, manifest, and aggregation."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from exhibit.catalog import build_catalog, load_catalog
from exhibit.config import ROOT, read_json
from exhibit.domain import ASPECTS, digest, file_hash
from exhibit.elicitation import next_round
from exhibit.evaluation import (
    build_study_experiment,
    cluster_bootstrap_interval,
    load_evaluation_config,
    participant_bootstrap_interval,
    study_manifest_hash,
)

from exhibit import study as study_lib

CONFIG_PATH = ROOT / "configs/fan-evaluation.json"
ASSETS = Path(study_lib.__file__).resolve().parent / "assets"
NODE = shutil.which("node")


def _catalog_v2():
    """catalog-v1 is the only reviewed catalog on disk; v2 is built in memory."""
    cards = build_catalog(read_json(ROOT / "configs/catalog-v2.json"))
    identity = {
        "catalog_id": "catalog-v2",
        "all_cards": cards,
        "image_hashes": {card["id"]: "f" * 64 for card in cards},
    }
    return {
        "catalog_id": "catalog-v2",
        "catalog_hash": digest(identity),
        "all_cards": copy.deepcopy(cards),
        "cards": copy.deepcopy(cards),
    }


@pytest.fixture(scope="module")
def catalog_v2():
    return _catalog_v2()


@pytest.fixture(scope="module")
def catalog_v1():
    return load_catalog("catalog-v1", reviewed_only=True)


@pytest.fixture
def catalogs(catalog_v2, catalog_v1):
    return {
        "new_pool": copy.deepcopy(catalog_v2),
        "legacy_pool": copy.deepcopy(catalog_v1),
    }


def _config(tmp_path, kind="encoder", *, minimum=None):
    config = load_evaluation_config(CONFIG_PATH, "study", study_kind=kind)
    config["study"]["study_dir"] = str(tmp_path / kind)
    if minimum:
        config["study"]["participants"] = minimum
    return config


@pytest.fixture
def config(tmp_path):
    return _config(tmp_path, "encoder", minimum={"pilot_minimum": 2, "main_minimum": 2})


def export(participant, catalog, *, offset=0, condition=None, count=3, **extra):
    cards = catalog["cards"][offset : offset + count]
    value = {
        "schema_version": 1,
        "participant_id": participant,
        "catalog_id": catalog["catalog_id"],
        "selection": [
            {"card_id": card["id"], "strength": 1, "aspects": ["color", "texture"]}
            for card in cards
        ],
        "aspect_gains": {aspect: 1 for aspect in ASPECTS},
        "elapsed_ms": 120000 + offset,
        "round_count": 2,
        "shown_ids": [card["id"] for card in catalog["cards"][: offset + 12]],
    }
    if condition:
        value["collection_condition"] = condition
    value.update(extra)
    return value


def participants_for(config, catalogs, count, *, kind="encoder"):
    exports = []
    for index in range(count):
        participant = f"p{index + 1:03d}"
        if kind == "elicitation":
            exports.append(
                (
                    f"{participant}-new.json",
                    export(
                        participant,
                        catalogs["new_pool"],
                        offset=index,
                        condition="new_pool",
                    ),
                )
            )
            exports.append(
                (
                    f"{participant}-legacy.json",
                    export(
                        participant,
                        catalogs["legacy_pool"],
                        offset=index % 4,
                        condition="legacy_pool",
                    ),
                )
            )
        else:
            exports.append(
                (
                    f"{participant}.json",
                    export(participant, catalogs["new_pool"], offset=index),
                )
            )
    document, report = study_lib.merge_participants(config, exports, catalogs=catalogs)
    assert not report["rejected"], report["rejected"]
    return document


def write_study(directory, config, participants, catalogs):
    manifest, keys = study_lib.build_manifest(config, participants, catalogs=catalogs)
    study_lib.write_json(Path(directory) / "manifest.json", manifest)
    study_lib.write_json(Path(directory) / "keys.json", keys)
    return manifest, keys


# ------------------------------------------------------- collection page only


def test_without_participants_only_the_collection_page_and_instructions(
    tmp_path, config, catalogs
):
    page = study_lib.build_collection_page(config, catalogs)
    status = study_lib.status_document(config)
    instructions = study_lib.instructions_markdown(config, status=status)

    assert status["comparison_images"] == "not_generated"
    assert status["comparison_images_ja"] == "比較画像未生成"
    assert status["participants"] == 0
    assert status["study_manifest"] is None and status["planned_images"] is None
    assert "比較画像未生成" in instructions
    assert "incomplete_participants" in instructions
    assert "conflicting duplicate answer" in instructions
    assert "勝率" not in page["html"]
    for banned in ("0.5", "win_rate", "参加者20人の回答"):
        assert banned not in page["html"]
    assert len(page["images"]) == len(catalogs["new_pool"]["cards"])


def test_collection_page_is_self_contained_and_states_what_is_stored(config, catalogs):
    html = study_lib.build_collection_page(config, catalogs)["html"]

    assert "http://" not in html and "https://" not in html
    assert "<script src" not in html and 'rel="stylesheet"' not in html
    for field in study_lib.STORED_FIELDS:
        assert field in html
    assert study_lib.STORAGE_NOTE in html
    assert study_lib.PARTICIPANT_ID.pattern in html


def test_elicitation_collection_page_offers_both_flows(tmp_path, catalogs):
    config = _config(tmp_path, "elicitation")
    html = study_lib.build_collection_page(config, catalogs)["html"]
    data = json.loads(html.split("globalThis.STUDY = ")[1].split(";\n")[0])

    conditions = {item["condition"]: item for item in data["conditions"]}
    assert set(conditions) == {"legacy_pool", "new_pool"}
    assert conditions["legacy_pool"]["flow"] == "single_screen"
    assert conditions["legacy_pool"]["selection"] == {
        "min": 3,
        "max": 5,
        "round_size": 16,
        "max_rounds": 1,
    }
    assert conditions["legacy_pool"]["aspects_default_on"] is True
    assert conditions["new_pool"]["flow"] == "rounds"
    assert conditions["new_pool"]["selection"]["round_size"] == 12
    assert conditions["new_pool"]["selection"]["max_rounds"] == 3
    assert conditions["new_pool"]["strength"] is True


def test_inline_module_drops_only_the_export_keyword():
    source = (
        "export const A = 1;\nexport function f() {}\nexport { f };\nconst b = 2;\n"
    )

    assert study_lib.inline_module(source) == (
        "const A = 1;\nfunction f() {}\nconst b = 2;"
    )


# ------------------------------------------------- python/javascript equality


def _round_scenarios(catalog, limits):
    first = next_round(
        catalog,
        {"selection": []},
        shown_ids=[],
        round_index=0,
        session_seed="s",
        selection=limits,
    )
    shown = list(first["card_ids"])
    liked = [
        {"card_id": shown[0], "strength": 2, "aspects": ["color"]},
        {"card_id": shown[1], "strength": 1, "aspects": ["color", "texture"]},
        {"card_id": shown[2], "strength": 1, "aspects": []},
        {"card_id": shown[3], "strength": 2, "aspects": ["mood", "texture"]},
        {"card_id": shown[4], "strength": 1, "aspects": ["lighting"]},
    ]
    return [
        {"snapshot": {"selection": []}, "shown_ids": [], "round_index": 0},
        {"snapshot": {"selection": liked}, "shown_ids": shown, "round_index": 1},
        {
            "snapshot": {"selection": liked[:1]},
            "shown_ids": shown[:3],
            "round_index": 1,
        },
        {"snapshot": {"selection": liked[:2]}, "shown_ids": shown, "round_index": 1},
        {"snapshot": {"selection": liked}, "shown_ids": shown, "round_index": 2},
    ]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("catalog_id", ["catalog-v1", "catalog-v2"])
def test_round_order_matches_between_python_and_the_page(
    tmp_path, catalog_id, catalog_v1, catalog_v2
):
    catalog = catalog_v1 if catalog_id == "catalog-v1" else catalog_v2
    size = 6 if catalog_id == "catalog-v1" else 12
    limits = {"min": 3, "max": 10, "round_size": size, "max_rounds": 3}
    seed = f"encoder-preference-1:{catalog_id}"
    table = study_lib.key_table(catalog, session_seed=seed, rounds=3)
    cases = []
    for scenario in _round_scenarios(catalog, limits):
        expected = next_round(
            catalog,
            scenario["snapshot"],
            shown_ids=scenario["shown_ids"],
            round_index=scenario["round_index"],
            session_seed=seed,
            selection=limits,
        )
        cases.append({**scenario, "expected": expected["card_ids"]})
    fixture = {
        "catalog": {
            "cards": [
                {
                    "id": card["id"],
                    "subject_id": card["subject_id"],
                    "axis_levels": card["axis_levels"],
                }
                for card in catalog["cards"]
            ]
        },
        "keyTable": table,
        "selection": limits,
        "cases": cases,
    }
    (tmp_path / "fixture.json").write_text(json.dumps(fixture))
    driver = tmp_path / "driver.mjs"
    driver.write_text(
        "import { readFileSync } from 'node:fs';\n"
        f"import {{ nextRound }} from '{(ASSETS / 'rounds.mjs').as_posix()}';\n"
        "const fixture = JSON.parse(readFileSync(process.argv[2], 'utf8'));\n"
        "const catalog = { ...fixture.catalog, all_cards: fixture.catalog.cards };\n"
        "const out = fixture.cases.map((item) =>\n"
        "  nextRound(catalog, item.snapshot, {\n"
        "    shownIds: item.shown_ids,\n"
        "    roundIndex: item.round_index,\n"
        "    keyTable: fixture.keyTable,\n"
        "    selection: fixture.selection,\n"
        "  }).card_ids,\n"
        ");\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    result = subprocess.run(
        [NODE, str(driver), str(tmp_path / "fixture.json")],
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(result.stdout) == [case["expected"] for case in cases]
    assert any(case["expected"] for case in cases)


def test_key_table_covers_every_round_and_card(config, catalogs):
    table = study_lib.key_table(
        catalogs["new_pool"], session_seed="encoder-preference-1:new_pool", rounds=3
    )

    assert set(table) == {"0", "1", "2"}
    for round_index in table:
        assert set(table[round_index]) == {
            card["id"] for card in catalogs["new_pool"]["cards"]
        }
    assert len({value for row in table.values() for value in row.values()}) == 3 * len(
        catalogs["new_pool"]["cards"]
    )


# ------------------------------------------------------------ merge validation


def test_merge_rejects_identifiable_ids_unknown_catalogs_and_bad_selections(
    config, catalogs
):
    good = export("p001", catalogs["new_pool"])
    bad_id = export("tomoya@example.com", catalogs["new_pool"])
    unknown_catalog = {
        **export("p002", catalogs["new_pool"]),
        "catalog_id": "catalog-v9",
    }
    too_few = export("p003", catalogs["new_pool"], count=1)
    unanswered = export("p004", catalogs["new_pool"])
    unanswered["selection"][0]["aspects"] = []
    other_study = {**export("p005", catalogs["new_pool"]), "study_id": "another-study"}
    missing_field = {
        key: value
        for key, value in export("p006", catalogs["new_pool"]).items()
        if key != "aspect_gains"
    }

    document, report = study_lib.merge_participants(
        config,
        [
            ("good.json", good),
            ("bad-id.json", bad_id),
            ("catalog.json", unknown_catalog),
            ("few.json", too_few),
            ("unanswered.json", unanswered),
            ("study.json", other_study),
            ("missing.json", missing_field),
        ],
        catalogs=catalogs,
    )

    assert [item["participant_id"] for item in document["records"]] == ["p001"]
    reasons = {item["source"]: item["reason"] for item in report["rejected"]}
    assert set(reasons) == {
        "bad-id.json",
        "catalog.json",
        "few.json",
        "unanswered.json",
        "study.json",
        "missing.json",
    }
    assert "participant_id must match" in reasons["bad-id.json"]
    assert "unknown catalog" in reasons["catalog.json"]
    assert "invalid selection" in reasons["few.json"]
    assert "invalid selection" in reasons["unanswered.json"]
    assert "another study_id" in reasons["study.json"]
    assert "aspect_gains" in reasons["missing.json"]


def test_merge_attaches_the_server_catalog_hash_and_ignores_the_client_one(
    config, catalogs
):
    record = export("p001", catalogs["new_pool"])
    record["catalog_hash"] = "0" * 64

    document, report = study_lib.merge_participants(
        config, [("a.json", record)], catalogs=catalogs
    )

    stored = document["records"][0]
    assert stored["catalog_hash"] == catalogs["new_pool"]["catalog_hash"]
    assert stored["snapshot"]["catalog_hash"] == catalogs["new_pool"]["catalog_hash"]
    assert report["accepted"] == [
        {
            "source": "a.json",
            "participant_id": "p001",
            "collection_condition": "new_pool",
        }
    ]


def test_merge_refuses_a_second_differing_record_for_one_participant(config, catalogs):
    first, _ = study_lib.merge_participants(
        config, [("a.json", export("p001", catalogs["new_pool"]))], catalogs=catalogs
    )
    document, report = study_lib.merge_participants(
        config,
        [
            ("again.json", export("p001", catalogs["new_pool"])),
            ("changed.json", export("p001", catalogs["new_pool"], offset=5)),
        ],
        existing=first,
        catalogs=catalogs,
    )

    assert len(document["records"]) == 1
    assert [item["source"] for item in report["unchanged"]] == ["again.json"]
    assert report["rejected"][0]["source"] == "changed.json"
    assert "conflicting" in report["rejected"][0]["reason"]


def test_an_encoder_study_refuses_a_legacy_pool_record(config, catalogs):
    _, report = study_lib.merge_participants(
        config,
        [("a.json", export("p001", catalogs["legacy_pool"], condition="legacy_pool"))],
        catalogs=catalogs,
    )

    assert "only the new_pool condition" in report["rejected"][0]["reason"]


def test_an_elicitation_participant_needs_both_conditions(tmp_path, catalogs):
    config = _config(
        tmp_path, "elicitation", minimum={"pilot_minimum": 2, "main_minimum": 2}
    )
    document, report = study_lib.merge_participants(
        config,
        [
            ("a.json", export("p001", catalogs["new_pool"], condition="new_pool")),
            ("b.json", export("p002", catalogs["new_pool"], condition="new_pool")),
            (
                "c.json",
                export("p002", catalogs["legacy_pool"], condition="legacy_pool"),
            ),
            ("d.json", export("p003", catalogs["new_pool"])),
        ],
        catalogs=catalogs,
    )

    assert report["incomplete"] == [
        {"participant_id": "p001", "missing": ["legacy_pool"]}
    ]
    assert "legacy_pool or new_pool" in report["rejected"][0]["reason"]
    assert list(study_lib.complete_participants(config, document)) == ["p002"]


# ----------------------------------------------------------------- manifest


def test_manifest_fails_before_any_job_when_participants_are_missing(
    tmp_path, catalogs
):
    config = _config(tmp_path, "encoder")
    participants = participants_for(config, catalogs, 3)

    with pytest.raises(study_lib.StudyError) as error:
        study_lib.build_manifest(config, participants, catalogs=catalogs)

    assert "no image job was produced" in str(error.value)
    assert "5 participants" in str(error.value)
    assert "20" in str(error.value)
    assert not (tmp_path / "encoder").exists()


def test_manifest_refuses_a_single_participant(config, catalogs):
    participants = participants_for(config, catalogs, 1)

    with pytest.raises(study_lib.StudyError) as error:
        study_lib.build_manifest(config, participants, catalogs=catalogs)

    assert "at least 2 participants" in str(error.value)


def test_manifest_assigns_a_deterministic_derangement(config, catalogs):
    participants = participants_for(config, catalogs, 6)

    manifest, _ = study_lib.build_manifest(config, participants, catalogs=catalogs)
    again, _ = study_lib.build_manifest(config, participants, catalogs=catalogs)

    assignment = manifest["assignment"]
    assert set(assignment) == set(manifest["participants"])
    assert sorted(assignment.values()) == sorted(assignment)
    assert all(left != right for left, right in assignment.items())
    assert again["assignment"] == assignment
    assert again["study_hash"] == manifest["study_hash"]
    assert manifest["study_hash"] == study_manifest_hash(manifest)


def test_manifest_fixes_balanced_sides_and_a_per_participant_order(config, catalogs):
    participants = participants_for(config, catalogs, 4)

    manifest, keys = study_lib.build_manifest(config, participants, catalogs=catalogs)

    assert manifest["image_count"] == 4 * 20
    assert len(manifest["pairs"]) == 4 * (8 + 4)
    for participant in manifest["participants"]:
        order = manifest["presentation_order"][participant]
        mine = [
            pair["pair_id"]
            for pair in manifest["pairs"]
            if pair["participant_id"] == participant
        ]
        assert sorted(order) == sorted(mine)
        for comparison, expected in (("candidate_vs_legacy", 8), ("own_vs_other", 4)):
            rows = [
                keys["pairs"][pair_id]
                for pair_id in mine
                if keys["pairs"][pair_id]["comparison"] == comparison
            ]
            assert len(rows) == expected
            sides = [row["subject_side"] for row in rows]
            assert sides.count("A") == sides.count("B") == expected // 2
    assert len({pair["pair_id"] for pair in manifest["pairs"]}) == len(
        manifest["pairs"]
    )
    # The other participant's profile is never the participant's own.
    for row in keys["pairs"].values():
        if row["comparison"] != "own_vs_other":
            continue
        variants = {side["variant_id"] for side in row["sides"].values()}
        assert len(variants) == 2


def test_the_two_kinds_never_share_a_directory_a_manifest_or_a_summary(
    tmp_path, catalogs
):
    encoder = _config(
        tmp_path, "encoder", minimum={"pilot_minimum": 2, "main_minimum": 2}
    )
    elicitation = _config(
        tmp_path, "elicitation", minimum={"pilot_minimum": 2, "main_minimum": 2}
    )

    assert encoder["study"]["study_dir"] != elicitation["study"]["study_dir"]
    assert encoder["study"]["study_id"] != elicitation["study"]["study_id"]

    encoder_manifest, _ = study_lib.build_manifest(
        encoder, participants_for(encoder, catalogs, 3), catalogs=catalogs
    )
    elicitation_manifest, _ = study_lib.build_manifest(
        elicitation,
        participants_for(elicitation, catalogs, 3, kind="elicitation"),
        catalogs=catalogs,
    )

    assert encoder_manifest["study_hash"] != elicitation_manifest["study_hash"]
    assert list(encoder_manifest["comparisons"]) == [
        "candidate_vs_legacy",
        "own_vs_other",
    ]
    assert list(elicitation_manifest["comparisons"]) == ["new_pool_vs_legacy_pool"]
    assert elicitation_manifest["assignment"] == {}
    assert elicitation_manifest["image_count"] == 3 * 2 * 2 * 4
    # An elicitation manifest may not be built from encoder-only participants.
    with pytest.raises(study_lib.StudyError):
        study_lib.build_manifest(
            elicitation, participants_for(encoder, catalogs, 3), catalogs=catalogs
        )


def test_study_images_are_content_addressed_and_deduplicated(
    tmp_path, config, catalogs
):
    participants = participants_for(config, catalogs, 3)
    manifest, _ = study_lib.build_manifest(config, participants, catalogs=catalogs)
    provenance = {
        "fan_pin": "9d0b76843f6437718195accac9cf3f050a25d26b",
        "adapter_hash": "a" * 64,
        "decoder_hash": "d" * 64,
        "tokenizer_hash": "t" * 64,
        "generation": config["generation"],
        "seeds": [230923],
    }

    experiment = build_study_experiment(
        config,
        manifest,
        provenance,
        catalog_loader=lambda catalog_id, **kwargs: catalogs["new_pool"],
    )

    assert len(experiment["jobs"]) == manifest["image_count"] == 60
    assert len({job["job_id"] for job in experiment["jobs"]}) == 60
    for job in experiment["jobs"]:
        assert job["path"] == f"images/{job['job_id']}.png"
        for hint in ("cat", "tokyo", "legacy", "candidate", "p001"):
            assert hint not in job["path"]
    assert set(experiment["policy_labels"].values()) == {
        "official_encoder",
        "legacy_exhibit",
    }


def test_a_study_over_the_image_limit_fails_before_generation(tmp_path, catalogs):
    config = _config(
        tmp_path, "encoder", minimum={"pilot_minimum": 2, "main_minimum": 2}
    )
    config["limits"] = {"max_images": 40, "max_attempts": 2}
    participants = participants_for(config, catalogs, 4)

    with pytest.raises(study_lib.StudyError) as error:
        study_lib.build_manifest(config, participants, catalogs=catalogs)

    assert "80 images" in str(error.value) and "40 image limit" in str(error.value)


# -------------------------------------------------------------- answer pages


def _fake_images(manifest, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    images = {}
    for job in manifest["jobs"]:
        key = f"{job['variant_id']}:{job['topic_id']}:{job['seed']}"
        path = directory / f"{digest(key)}.png"
        path.write_bytes(b"png:" + key.encode())
        images[key] = str(path)
    return images


def test_the_answer_page_never_names_a_method_or_another_participant(
    tmp_path, config, catalogs
):
    participants = participants_for(config, catalogs, 4)
    directory = tmp_path / "study"
    manifest, keys = write_study(directory, config, participants, catalogs)
    images = _fake_images(manifest, tmp_path / "generated")

    built = study_lib.build_answer_pages(config, manifest, keys, images, directory)

    assert len(built["pages"]) == 4 and not built["missing_images"]
    for participant in manifest["participants"]:
        page = (directory / "answer" / participant / "index.html").read_text()
        payload = json.loads(page.split("globalThis.PAGE = ")[1].split(";\n")[0])
        assert [item["pair_id"] for item in payload["pairs"]] == manifest[
            "presentation_order"
        ][participant]
        for banned in (
            "legacy",
            "candidate",
            "official",
            "policy",
            "alpha",
            "skip_pa",
            "warm color palette",
            "masterpiece",
            "cat",
            "tokyo",
            manifest["assignment"][participant],
        ):
            assert banned not in json.dumps(payload, ensure_ascii=False)
        assert "候補" not in page and "従来" not in page
        for other in manifest["participants"]:
            if other != participant:
                assert other not in page
        names = [
            item.name
            for item in (directory / "answer" / participant / "images").iterdir()
        ]
        assert len(names) == 2 * len(payload["pairs"])
        for name in names:
            assert name.endswith(".png") and len(name) == 36
            for hint in ("cat", "tokyo", "legacy", "candidate", participant):
                assert hint not in name
        assert "http://" not in page and "https://" not in page
    # The same underlying image is copied under a different opaque name per page.
    everything = [
        item.name
        for participant in manifest["participants"]
        for item in (directory / "answer" / participant / "images").iterdir()
    ]
    assert len(everything) == len(set(everything))


def test_missing_images_are_reported_and_never_silently_dropped(
    tmp_path, config, catalogs
):
    participants = participants_for(config, catalogs, 2)
    directory = tmp_path / "study"
    manifest, keys = write_study(directory, config, participants, catalogs)
    images = _fake_images(manifest, tmp_path / "generated")
    images.pop(min(images))

    built = study_lib.build_answer_pages(config, manifest, keys, images, directory)

    assert built["missing_images"]
    pages = {item["participant_id"]: item["pairs"] for item in built["pages"]}
    assert min(pages.values()) < len(manifest["pairs"]) // 2


# ------------------------------------------------------------------ scoring


def answer_file(directory, manifest, keys, participant, chooser, *, study_id=None):
    pairs = [
        pair for pair in manifest["pairs"] if pair["participant_id"] == participant
    ]
    rows = []
    for pair in pairs:
        choice = chooser(pair, keys["pairs"][pair["pair_id"]])
        if choice is not False:
            rows.append({"pair_id": pair["pair_id"], "choice": choice})
    value = {
        "schema_version": 1,
        "study_id": study_id or manifest["study_id"],
        "participant_id": participant,
        "answers": rows,
        "fidelity": [
            {"pair_id": pair["pair_id"], "subject_kept": "both", "breakage": "none"}
            for pair in pairs
        ],
    }
    study_lib.write_json(Path(directory) / "answers" / f"{participant}.json", value)
    return value


def subject_wins(pair, key):
    return key["subject_side"]


def always_tie(pair, key):
    return "tie"


def test_a_study_without_answers_reports_未実施(
    tmp_path, config, catalogs, monkeypatch
):
    # The report prints its own timestamp; a clock reading like `…:50.5…` is not a rate.
    monkeypatch.setattr(study_lib, "_now", lambda: "2026-01-01T00:00:00Z")
    participants = participants_for(config, catalogs, 3)
    directory = tmp_path / "study"
    write_study(directory, config, participants, catalogs)

    summary = study_lib.summarize_study(directory)
    markdown = study_lib.summary_markdown(summary)

    assert summary["status"] == "not_started"
    assert summary["participants_with_answers"] == 0
    for item in summary["comparisons"].values():
        assert item["mean"] is None and item["interval"] is None
        assert item["conclusion"] == "undetermined"
        assert item["reasons"] == ["no_answers"]
    assert summary["default_policy_change"]["eligible"] is False
    assert "**未実施**" in markdown
    assert "0.5" not in markdown.split("## 既定値の変更条件")[0]


def test_a_study_without_a_manifest_summarizes_as_未実施(tmp_path, config):
    directory = tmp_path / "study"
    status = study_lib.status_document(config)
    study_lib.write_json(directory / "status.json", status)

    summary = study_lib.summarize_study(directory)
    markdown = study_lib.summary_markdown(summary)

    assert summary["status"] == "not_started"
    assert summary["reasons"] == ["study_manifest_missing", "not_generated"]
    assert summary["comparisons"] == {}
    assert summary["study_hash"] is None
    assert "**未実施**" in markdown and "未作成" in markdown
    assert summary["default_policy_change"]["eligible"] is False


def test_ties_only_score_exactly_one_half_and_never_read_as_improved(
    tmp_path, config, catalogs
):
    participants = participants_for(config, catalogs, 4)
    directory = tmp_path / "study"
    manifest, keys = write_study(directory, config, participants, catalogs)
    for participant in manifest["participants"]:
        answer_file(directory, manifest, keys, participant, always_tie)

    summary = study_lib.summarize_study(directory)

    for item in summary["comparisons"].values():
        assert item["mean"] == 0.5
        assert item["interval"]["lower"] == item["interval"]["upper"] == 0.5
        assert item["conclusion"] == "not_confirmed"
    assert summary["default_policy_change"]["eligible"] is False
    assert "改善を確認できず" in study_lib.summary_markdown(summary)


def test_two_participants_with_many_seeds_count_as_two_units(
    tmp_path, config, catalogs
):
    participants = participants_for(config, catalogs, 2)
    directory = tmp_path / "study"
    manifest, keys = write_study(directory, config, participants, catalogs)
    first, second = manifest["participants"]
    answer_file(directory, manifest, keys, first, subject_wins)

    def quarter(pair, key):
        if key["comparison"] != "candidate_vs_legacy":
            return key["subject_side"]
        index = sorted(
            row["pair_id"]
            for row in manifest["pairs"]
            if row["participant_id"] == second
            and row["comparison"] == key["comparison"]
        ).index(pair["pair_id"])
        other = "A" if key["subject_side"] == "B" else "B"
        return key["subject_side"] if index < 2 else other

    answer_file(directory, manifest, keys, second, quarter)

    summary = study_lib.summarize_study(directory)
    item = summary["comparisons"]["candidate_vs_legacy"]

    assert item["participants_scored"] == 2
    assert item["participant_means"] == {first: 1.0, second: 0.25}
    assert item["answered_pairs"] == 16
    assert item["mean"] == pytest.approx(0.625)
    assert item["interval"]["unit"] == "participant"
    assert item["interval"]["participants"] == 2
    # Sixteen images pooled as independent units would give a narrow interval;
    # two people can only resample to 0.25, 0.625 or 1.0.
    assert item["interval"]["lower"] == pytest.approx(0.25)
    assert item["interval"]["upper"] == pytest.approx(1.0)


def test_a_participant_who_skipped_pairs_is_reported_not_scored(
    tmp_path, config, catalogs
):
    participants = participants_for(config, catalogs, 3)
    directory = tmp_path / "study"
    manifest, keys = write_study(directory, config, participants, catalogs)
    first, second, third = manifest["participants"]
    answer_file(directory, manifest, keys, first, always_tie)
    answer_file(directory, manifest, keys, second, always_tie)
    mine = sorted(
        row["pair_id"]
        for row in manifest["pairs"]
        if row["participant_id"] == third and row["comparison"] == "candidate_vs_legacy"
    )

    def only_one_win(pair, key):
        if key["comparison"] != "candidate_vs_legacy":
            return key["subject_side"]
        return key["subject_side"] if pair["pair_id"] == mine[0] else None

    answer_file(directory, manifest, keys, third, only_one_win)

    summary = study_lib.summarize_study(directory)
    item = summary["comparisons"]["candidate_vs_legacy"]
    markdown = study_lib.summary_markdown(summary)

    # One favourable answer out of eight must not enter the mean as a full person.
    assert item["participants_scored"] == 2
    assert sorted(item["participant_means"]) == sorted([first, second])
    assert item["mean"] == 0.5
    assert item["incomplete_participants"] == {
        third: {"answered": 1, "expected": 8, "partial_mean": 1.0}
    }
    # The report names the excluded participant instead of quietly dropping them.
    assert "平均・区間から除外" in markdown
    assert f"{third}: 1 / 8 組" in markdown


def test_conflicting_duplicate_answers_void_the_pair(tmp_path, config, catalogs):
    participants = participants_for(config, catalogs, 2)
    directory = tmp_path / "study"
    manifest, keys = write_study(directory, config, participants, catalogs)
    first = manifest["participants"][0]
    value = answer_file(directory, manifest, keys, first, subject_wins)
    row = value["answers"][0]
    flipped = "A" if row["choice"] == "B" else "B"
    value["answers"].append({"pair_id": row["pair_id"], "choice": flipped})
    study_lib.write_json(directory / "answers" / f"{first}.json", value)

    summary = study_lib.summarize_study(directory)

    reasons = [item["reason"] for item in summary["answer_problems"][first]]
    assert reasons == ["conflicting duplicate answer"]
    comparison = keys["pairs"][row["pair_id"]]["comparison"]
    item = summary["comparisons"][comparison]
    assert first in item["incomplete_participants"]
    assert first not in item["participant_means"]
    assert "conflicting duplicate answer" in study_lib.summary_markdown(summary)


def test_an_elicitation_study_never_qualifies_a_policy_change(tmp_path, catalogs):
    config = _config(
        tmp_path, "elicitation", minimum={"pilot_minimum": 2, "main_minimum": 2}
    )
    participants = participants_for(config, catalogs, 3, kind="elicitation")
    directory = tmp_path / "study"
    manifest, keys = write_study(directory, config, participants, catalogs)
    for participant in manifest["participants"]:
        answer_file(directory, manifest, keys, participant, subject_wins)

    summary = study_lib.summarize_study(directory, participants=participants)

    assert all(
        item["conclusion"] == "improved" for item in summary["comparisons"].values()
    )
    assert summary["default_policy_change"]["eligible"] is False


def test_the_two_comparisons_are_aggregated_independently(tmp_path, config, catalogs):
    participants = participants_for(config, catalogs, 4)
    directory = tmp_path / "study"
    manifest, keys = write_study(directory, config, participants, catalogs)
    for participant in manifest["participants"]:
        answer_file(
            directory,
            manifest,
            keys,
            participant,
            lambda pair, key: (
                key["subject_side"]
                if key["comparison"] == "candidate_vs_legacy"
                else "tie"
            ),
        )

    summary = study_lib.summarize_study(directory)
    candidate = summary["comparisons"]["candidate_vs_legacy"]
    own = summary["comparisons"]["own_vs_other"]

    # Everybody answered every pair of both comparisons, so nobody is excluded
    # and one comparison's verdict cannot leak into the other.
    assert candidate["incomplete_participants"] == {}
    assert own["incomplete_participants"] == {}
    assert candidate["participants_scored"] == own["participants_scored"] == 4
    assert candidate["mean"] == 1.0
    assert candidate["conclusion"] == "improved"
    assert own["mean"] == 0.5
    assert own["conclusion"] == "not_confirmed"
    assert summary["default_policy_change"]["eligible"] is False


def test_both_comparisons_improved_reports_eligibility_without_changing_policies(
    tmp_path, config, catalogs
):
    participants = participants_for(config, catalogs, 4)
    directory = tmp_path / "study"
    manifest, keys = write_study(directory, config, participants, catalogs)
    for participant in manifest["participants"]:
        answer_file(directory, manifest, keys, participant, subject_wins)

    summary = study_lib.summarize_study(directory)
    policies = read_json(ROOT / "configs/fan-policies.json")

    assert set(summary["comparisons"]) == {"candidate_vs_legacy", "own_vs_other"}
    assert all(
        item["conclusion"] == "improved" and item["incomplete_participants"] == {}
        for item in summary["comparisons"].values()
    )
    assert summary["default_policy_change"]["eligible"] is True
    # Eligibility is reported; the registry on disk is never touched.
    assert policies["default_policy_id"] == "legacy_exhibit"
    assert "fan-policies.jsonを変更しない" in summary["default_policy_change"]["note"]
    assert "`legacy_exhibit` のまま" in study_lib.summary_markdown(summary)


def test_unknown_pairs_other_studies_and_duplicates_are_reported(
    tmp_path, config, catalogs
):
    participants = participants_for(config, catalogs, 3)
    directory = tmp_path / "study"
    manifest, keys = write_study(directory, config, participants, catalogs)
    first, second, third = manifest["participants"]
    answer_file(directory, manifest, keys, first, subject_wins)
    answer_file(directory, manifest, keys, second, subject_wins, study_id="other-study")
    value = answer_file(directory, manifest, keys, third, subject_wins)
    value["answers"].append(dict(value["answers"][0]))
    value["answers"].append({"pair_id": "0" * 32, "choice": "A"})
    borrowed = next(
        pair["pair_id"] for pair in manifest["pairs"] if pair["participant_id"] == first
    )
    value["answers"].append({"pair_id": borrowed, "choice": "A"})
    study_lib.write_json(directory / "answers" / f"{third}.json", value)
    study_lib.write_json(
        directory / "answers" / "stray.json",
        {"study_id": manifest["study_id"], "participant_id": "p900", "answers": []},
    )
    (directory / "answers" / "broken.json").write_text("{not json")

    summary = study_lib.summarize_study(directory)

    reasons = {item["file"]: item["reason"] for item in summary["excluded_files"]}
    assert reasons[f"{second}.json"] == "another study_id"
    assert reasons["stray.json"] == "unknown participant_id"
    assert "unreadable" in reasons["broken.json"]
    problems = {item["reason"] for item in summary["answer_problems"][third]}
    assert problems == {
        "duplicate answer",
        "unknown pair_id",
        "pair of another participant",
    }
    assert summary["duplicate_answers"] == [
        {"participant_id": third, "pair_id": value["answers"][0]["pair_id"]}
    ]
    item = summary["comparisons"]["candidate_vs_legacy"]
    assert item["participants_scored"] == 2
    assert sorted(item["participant_means"]) == sorted([first, third])
    assert len(item["missing_pairs"]) == 8


def test_a_second_answer_file_for_one_participant_is_excluded(
    tmp_path, config, catalogs
):
    participants = participants_for(config, catalogs, 2)
    directory = tmp_path / "study"
    manifest, keys = write_study(directory, config, participants, catalogs)
    first = manifest["participants"][0]
    value = answer_file(directory, manifest, keys, first, subject_wins)
    study_lib.write_json(directory / "answers" / "zzz-copy.json", value)

    summary = study_lib.summarize_study(directory)

    assert [item["reason"] for item in summary["excluded_files"]] == [
        "a second answer file for this participant"
    ]
    assert summary["comparisons"]["candidate_vs_legacy"]["participants_scored"] == 1


def test_fidelity_is_reported_apart_from_the_preference_score(
    tmp_path, config, catalogs
):
    participants = participants_for(config, catalogs, 2)
    directory = tmp_path / "study"
    manifest, keys = write_study(directory, config, participants, catalogs)
    for participant in manifest["participants"]:
        answer_file(directory, manifest, keys, participant, subject_wins)

    summary = study_lib.summarize_study(directory)

    assert summary["fidelity"]["answered_rows"] == len(manifest["pairs"])
    assert summary["fidelity"]["counts"]["subject_kept"] == {"both": 24}
    assert summary["comparisons"]["candidate_vs_legacy"]["mean"] == 1.0
    assert "好みの勝率には加えない" in summary["fidelity"]["note"]


def test_the_elicitation_summary_labels_the_pool_and_flow_effect(tmp_path, catalogs):
    config = _config(
        tmp_path, "elicitation", minimum={"pilot_minimum": 2, "main_minimum": 2}
    )
    participants = participants_for(config, catalogs, 3, kind="elicitation")
    directory = tmp_path / "study"
    manifest, keys = write_study(directory, config, participants, catalogs)
    for participant in manifest["participants"]:
        answer_file(directory, manifest, keys, participant, subject_wins)

    summary = study_lib.summarize_study(directory, participants=participants)
    markdown = study_lib.summary_markdown(summary)

    effect = summary["collection_effect"]
    assert set(effect["by_condition"]) == {"new_pool", "legacy_pool"}
    assert effect["by_condition"]["new_pool"]["participants"] == 3
    assert effect["by_condition"]["new_pool"]["mean_selection_count"] == 3
    assert effect["by_condition"]["new_pool"]["mean_explicit_aspects"] == 6
    assert effect["by_condition"]["new_pool"]["mean_elapsed_ms"] is not None
    assert effect["preference_rate_new_pool"] == 1.0
    assert effect["participants_scored"] == 3
    assert effect["incomplete_participants"] == []
    assert effect["effect_label"] == "pool + selection flow"
    assert "枚数増加だけの効果とは呼ばない" in markdown
    assert effect["scope_note"] in markdown
    assert "新poolの選好率: 1.000" in markdown
    assert list(summary["comparisons"]) == ["new_pool_vs_legacy_pool"]


# ------------------------------------------------------------- command line


def _script(name, module_name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_builder_stops_at_the_collection_stage_without_participants(tmp_path):
    builder = _script("build_preference_study.py", "task8_build_study")
    directory = tmp_path / "study"

    builder.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--study-dir",
            str(directory),
            "--catalog-id",
            "catalog-v1",
        ]
    )

    assert (directory / "collect/index.html").is_file()
    assert (directory / "README.md").is_file()
    assert len(list((directory / "collect/images").iterdir())) == 16
    assert not (directory / "manifest.json").exists()
    assert not (directory / "keys.json").exists()
    assert not (directory / "participants.json").exists()
    status = json.loads((directory / "status.json").read_text())
    assert status["comparison_images"] == "not_generated"
    assert status["planned_images"] is None


def test_the_builder_merges_then_refuses_to_plan_images(tmp_path, catalog_v1):
    builder = _script("build_preference_study.py", "task8_build_study_merge")
    directory = tmp_path / "study"
    export_path = tmp_path / "p001.json"
    export_path.write_text(json.dumps(export("p001", catalog_v1)))
    arguments = [
        "--config",
        str(CONFIG_PATH),
        "--study-dir",
        str(directory),
        "--catalog-id",
        "catalog-v1",
    ]

    with pytest.raises(SystemExit) as error:
        builder.main([*arguments, "--merge", str(export_path)])

    assert "no image job was produced" in str(error.value)
    assert json.loads((directory / "participants.json").read_text())["records"]
    assert json.loads((directory / "merge-report.json").read_text())["accepted"]
    assert not (directory / "manifest.json").exists()
    assert not (directory / "images.json").exists()


def test_the_builder_writes_answer_pages_once_the_images_exist(tmp_path, catalog_v1):
    builder = _script("build_preference_study.py", "task8_build_study_pages")
    config = _config(tmp_path, "encoder")
    config["study"]["catalog_id"] = "catalog-v1"
    catalogs = {"new_pool": catalog_v1}
    directory = Path(config["study"]["study_dir"])
    participants = participants_for(config, catalogs, 5)
    study_lib.write_json(directory / "participants.json", participants)
    manifest, _ = write_study(directory, config, participants, catalogs)
    study_lib.write_json(
        directory / "images.json",
        {
            "schema_version": 1,
            "study_id": manifest["study_id"],
            "study_hash": manifest["study_hash"],
            "images": _fake_images(manifest, tmp_path / "generated"),
        },
    )
    arguments = [
        "--config",
        str(CONFIG_PATH),
        "--study-dir",
        str(directory),
        "--catalog-id",
        "catalog-v1",
    ]

    builder.main(arguments)

    status = json.loads((directory / "status.json").read_text())
    assert status["comparison_images"] == "generated"
    assert status["answer_pages"] == 5
    for participant in manifest["participants"]:
        assert (directory / "answer" / participant / "index.html").is_file()
    # A stale index for another manifest hash is ignored instead of trusted.
    study_lib.write_json(
        directory / "images.json",
        {"study_hash": "0" * 64, "images": {"x": "/nowhere.png"}},
    )
    builder.main(arguments)
    assert (
        json.loads((directory / "status.json").read_text())["comparison_images"]
        == "not_generated"
    )


def test_a_study_experiment_registers_like_any_other_matrix(tmp_path, config, catalogs):
    from exhibit.evaluation import register_experiment

    participants = participants_for(config, catalogs, 2)
    manifest, _ = study_lib.build_manifest(config, participants, catalogs=catalogs)
    experiment = build_study_experiment(
        config,
        manifest,
        {
            "fan_pin": "9d0b76843f6437718195accac9cf3f050a25d26b",
            "adapter_hash": "a" * 64,
            "decoder_hash": "d" * 64,
            "tokenizer_hash": "t" * 64,
            "generation": config["generation"],
            "seeds": [230923],
        },
        catalog_loader=lambda catalog_id, **kwargs: catalogs["new_pool"],
    )

    directory, checkpoint = register_experiment(
        tmp_path / "out", experiment, resume=False
    )

    assert directory.name == experiment["experiment_hash"]
    assert len(checkpoint["jobs"]) == len(experiment["jobs"]) == 40
    assert {item["status"] for item in checkpoint["jobs"].values()} == {"not_run"}
    assert (
        json.loads((directory / "manifest.json").read_text())["identity"]["phase"]
        == "study"
    )


def test_the_study_command_refuses_before_touching_the_gpu(
    tmp_path, monkeypatch, catalogs
):
    controller = _script("evaluate_fan.py", "task8_evaluate_fan")
    directory = tmp_path / "study"
    real = controller.load_evaluation_config

    def loader(path, phase, **kwargs):
        config = real(path, phase, **kwargs)
        if phase == "study":
            config["study"]["study_dir"] = str(directory)
        return config

    monkeypatch.setattr(controller, "load_evaluation_config", loader)
    monkeypatch.setattr(
        controller,
        "validate_evaluator_preparation",
        lambda *args, **kwargs: pytest.fail("the evaluator was opened before refusing"),
    )
    arguments = {
        "study_kind": "encoder",
        "resume": True,
        "cancel": controller.threading.Event(),
        "deadline": controller.time.monotonic() + 10,
        "on_event": lambda event: None,
    }

    with pytest.raises(SystemExit) as absent:
        controller.run_study(str(CONFIG_PATH), **arguments)

    config = _config(
        tmp_path, "encoder", minimum={"pilot_minimum": 2, "main_minimum": 2}
    )
    config["study"]["study_dir"] = str(directory)
    manifest, _ = study_lib.build_manifest(
        config, participants_for(config, catalogs, 2), catalogs=catalogs
    )
    manifest["participants"] = manifest["participants"][:1]
    manifest["participants_required"] = {"pilot_minimum": 5, "main_minimum": 20}
    study_lib.write_json(directory / "manifest.json", manifest)

    with pytest.raises(SystemExit) as too_few:
        controller.run_study(str(CONFIG_PATH), **arguments)

    assert "study manifest is missing" in str(absent.value)
    assert "no GPU work was started" in str(absent.value)
    assert "1 participants and needs 5" in str(too_few.value)
    assert "no GPU work was started" in str(too_few.value)


def test_the_study_command_generates_once_and_indexes_every_image(
    tmp_path, monkeypatch, catalog_v1
):
    from PIL import Image

    controller = _script("evaluate_fan.py", "task8_evaluate_fan_run")
    config = _config(
        tmp_path, "encoder", minimum={"pilot_minimum": 2, "main_minimum": 2}
    )
    config["study"]["catalog_id"] = "catalog-v1"
    directory = Path(config["study"]["study_dir"])
    catalogs = {"new_pool": catalog_v1}
    manifest, _ = write_study(
        directory, config, participants_for(config, catalogs, 2), catalogs
    )
    real = controller.load_evaluation_config

    def loader(path, phase, **kwargs):
        value = real(path, phase, **kwargs)
        if phase == "study":
            value["study"]["study_dir"] = str(directory)
            value["study"]["catalog_id"] = "catalog-v1"
        return value

    monkeypatch.setattr(controller, "load_evaluation_config", loader)
    monkeypatch.setattr(
        controller, "validate_evaluator_preparation", lambda *a, **k: {}
    )
    monkeypatch.setattr(controller, "ensure_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(
        controller,
        "experiment_provenance",
        lambda *a, **k: {
            "fan_pin": "9d0b76843f6437718195accac9cf3f050a25d26b",
            "adapter_hash": "a" * 64,
            "decoder_hash": "d" * 64,
            "tokenizer_hash": "t" * 64,
            "generation": read_json(ROOT / "configs/demo.json")["generation"],
            "seeds": [230923],
        },
    )
    calls = []

    def runner(request, out, cancel, deadline, on_event):
        calls.append(request)
        for item in request["items"]:
            on_event({"type": "image_started", "id": item["id"]})
            path = Path(item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(1, 2, 3)).save(path)
            on_event(
                {
                    "type": "image",
                    "id": item["id"],
                    "path": str(path),
                    "sha256": file_hash(path),
                    "seed": item["seed"],
                    "prompt": item["prompt"],
                    "settings": request["settings"],
                    "effective_policy": item["effective_policy"],
                    "policy_hash": item["policy_hash"],
                    "personalization_hash": item["personalization"][
                        "personalization_hash"
                    ],
                    "seconds": 0.1,
                }
            )
        Path(request["embeddings_output"]).write_text(
            json.dumps({"images": {}, "texts": {}, "conditioning": {}})
        )
        return {"returncode": 0, "wall_seconds": 0.1}

    events = []
    index = controller.run_study(
        str(CONFIG_PATH),
        study_kind="encoder",
        resume=False,
        cancel=controller.threading.Event(),
        deadline=controller.time.monotonic() + 30,
        on_event=events.append,
        runner=runner,
    )

    assert len(calls) == 1 and len(calls[0]["items"]) == 40
    assert index["study_hash"] == manifest["study_hash"]
    assert len(index["images"]) == 40 and not index["missing"]
    plan = next(item for item in events if item["type"] == "study_plan")
    assert plan == {
        "type": "study_plan",
        "study_id": manifest["study_id"],
        "study_kind": "encoder",
        "participants": 2,
        "total_images": 40,
    }
    for key, path in index["images"].items():
        assert Path(path).is_file()
        assert Path(path).parent.name == "images"
        assert key.split(":")[0] not in Path(path).name
    assert json.loads((directory / "images.json").read_text()) == index


def test_the_participant_bootstrap_matches_the_cluster_bootstrap_maths():
    means = {"p001": 0.25, "p002": 1.0, "p003": 0.5}

    participant = participant_bootstrap_interval(means, draws=2000, seed=0)
    cluster = cluster_bootstrap_interval(
        {key: [value] for key, value in means.items()}, draws=2000, seed=0
    )

    assert participant["unit"] == "participant"
    assert participant["participants"] == 3
    assert "diagnostic_only" not in participant
    for key in ("mean", "lower", "upper", "draws", "seed", "confidence"):
        assert participant[key] == cluster[key]
