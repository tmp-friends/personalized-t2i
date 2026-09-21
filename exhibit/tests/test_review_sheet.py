import copy
import json
import os
import subprocess
import sys

import pytest
from exhibit.catalog import build_catalog, card_settings, description_hash, load_catalog
from exhibit.config import ROOT, read_json
from exhibit.domain import ASPECTS, file_hash

from exhibit import catalog as catalog_module

sys.path.insert(0, str(ROOT / "scripts"))
import build_review_sheet as brs


def definition():
    return copy.deepcopy(read_json(ROOT / "configs/catalog-v2.json"))


def fake_png(path, seed):
    """A tiny but real PNG (Pillow must be able to open it for thumbnailing)."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    color = (seed % 256, (seed * 7) % 256, (seed * 13) % 256)
    Image.new("RGB", (8, 8), color).save(path, format="PNG")


def write_fake_v2_assets(root, *, skip_image_for=(), defn=None):
    """A full 64-card catalog-v2 bundle with tiny real PNGs, self-contained."""
    defn = defn if defn is not None else definition()
    cards = build_catalog(defn)
    settings = card_settings("catalog-v2", defn)
    images = {}
    for index, card in enumerate(cards):
        path = root / card["path"]
        if card["id"] in skip_image_for:
            continue
        fake_png(path, card["seed"] + index)
        images[card["id"]] = {
            "path": card["path"],
            "sha256": file_hash(path),
            "seed": card["seed"],
            "prompt": card["prompt"],
            "ref_en": card["ref_en"],
            "aspects": card["aspects"],
            "aspects_ja": card["aspects_ja"],
            "label": card["label"],
            "profile_label": card["profile_label"],
            "settings": settings,
        }
    (root / "catalog-v2.json").write_text(
        json.dumps(
            {
                "version": 2,
                "catalog_id": "catalog-v2",
                "generation": settings,
                "token_validation": {},
                "images": images,
            }
        )
    )
    return cards


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "assets"
    cards = write_fake_v2_assets(root)
    return {"root": root, "cards": cards}


def extract_payload(html):
    marker = '<script id="review-data" type="application/json">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    return json.loads(html[start:end])


# ------------------------------------------------------------------- build()


def test_build_writes_html_with_every_card_id_and_correct_hashes(bundle, tmp_path):
    out = tmp_path / "out" / "cards-v2-review.html"
    report = brs.build(assets=bundle["root"], out=out, review_path=None)
    assert report["cards"] == 64
    assert report["missing_images"] == []
    assert out.is_file()

    html = out.read_text()
    payload = extract_payload(html)
    assert len(payload["cards"]) == 64
    by_id = payload["cardsById"]
    assert set(by_id) == {card["id"] for card in bundle["cards"]}
    for card in bundle["cards"]:
        assert card["id"] in html
        entry = by_id[card["id"]]
        expected_sha = file_hash(bundle["root"] / card["path"])
        assert entry["image_sha256"] == expected_sha
        assert entry["description_hash"] == description_hash(card)
        assert entry["aspects"] == card["aspects"]
        assert entry["aspects_ja"] == card["aspects_ja"]


def test_build_has_no_external_resource_references(bundle, tmp_path):
    out = tmp_path / "out" / "cards-v2-review.html"
    brs.build(assets=bundle["root"], out=out, review_path=None)
    html = out.read_text()
    assert "http://" not in html
    assert "https://" not in html


def test_build_relative_image_paths_resolve_to_the_real_files(bundle, tmp_path):
    # Nest the output the same two levels deep as the real default
    # (exhibit/outputs/review/cards-v2-review.html vs exhibit/assets).
    out = tmp_path / "outputs" / "review" / "cards-v2-review.html"
    brs.build(assets=bundle["root"], out=out, review_path=None)
    payload = extract_payload(out.read_text())
    for card in bundle["cards"]:
        entry = payload["cardsById"][card["id"]]
        assert entry["image_full"] is not None
        assert not entry["image_full"].startswith("/")
        resolved = (out.parent / entry["image_full"]).resolve()
        assert resolved == (bundle["root"] / card["path"]).resolve()
        assert resolved.is_file()
        # Default (no --embed-thumbnails): the grid image is the same relative path.
        assert entry["image_src"] == entry["image_full"]


def test_build_reports_and_marks_missing_images(tmp_path):
    root = tmp_path / "assets"
    missing_id = build_catalog(definition())[0]["id"]
    cards = write_fake_v2_assets(root, skip_image_for={missing_id})
    out = tmp_path / "out" / "cards-v2-review.html"
    report = brs.build(assets=root, out=out, review_path=None)
    assert report["missing_images"] == [missing_id]
    payload = extract_payload(out.read_text())
    entry = payload["cardsById"][missing_id]
    assert entry["image_exists"] is False
    assert entry["image_sha256"] is None
    assert entry["image_src"] is None
    assert len(payload["cards"]) == len(cards) == 64


def test_build_refuses_clearly_when_manifest_generation_is_stale(tmp_path):
    """A manifest built under an older `card_settings` (e.g. an old negative
    prompt) must not crash with a traceback or silently produce an empty
    page -- `load_catalog` rejects the whole manifest in that case, exactly
    like the currently-checked-in old catalog-v2 assets do."""
    root = tmp_path / "assets"
    write_fake_v2_assets(root)
    manifest_path = root / "catalog-v2.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["generation"] = {**manifest["generation"], "negative_prompt": "stale"}
    manifest_path.write_text(json.dumps(manifest))

    out = tmp_path / "out" / "cards-v2-review.html"
    with pytest.raises(SystemExit) as excinfo:
        brs.build(assets=root, out=out, review_path=None)
    message = str(excinfo.value)
    assert "does not match the current catalog-v2 definition" in message
    assert not out.exists()


def test_cli_refuses_clearly_without_a_python_traceback_on_stale_manifest(tmp_path):
    root = tmp_path / "assets"
    write_fake_v2_assets(root)
    manifest_path = root / "catalog-v2.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["generation"] = {**manifest["generation"], "negative_prompt": "stale"}
    manifest_path.write_text(json.dumps(manifest))

    out = tmp_path / "cli-out" / "cards-v2-review.html"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_review_sheet.py"),
            "--assets",
            str(root),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode != 0
    assert "Traceback (most recent call last)" not in result.stderr
    assert "does not match the current catalog-v2 definition" in result.stderr
    assert not out.exists()


def test_embed_thumbnails_flag_produces_data_uris_when_pillow_available(
    bundle, tmp_path
):
    if not brs.have_pillow():
        pytest.skip("Pillow not available in this environment")
    out = tmp_path / "out" / "cards-v2-review.html"
    brs.build(assets=bundle["root"], out=out, review_path=None, embed_thumbnails=True)
    payload = extract_payload(out.read_text())
    card_id = bundle["cards"][0]["id"]
    entry = payload["cardsById"][card_id]
    assert entry["image_src"].startswith("data:image/jpeg;base64,")
    # The full-resolution enlarge target still points at the real file.
    assert not entry["image_full"].startswith("data:")


def test_embed_thumbnails_refuses_without_pillow(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(brs, "have_pillow", lambda: False)
    out = tmp_path / "out" / "cards-v2-review.html"
    with pytest.raises(SystemExit):
        brs.build(
            assets=bundle["root"], out=out, review_path=None, embed_thumbnails=True
        )


# --------------------------------------------------------- export shaping


def test_shape_review_entry_fully_confirmed_is_reviewed_true():
    entry = brs.shape_review_entry(
        image_sha256="a" * 64,
        description_hash="b" * 64,
        aspects={a: True for a in ASPECTS},
        subject_ok=True,
        note="looks good",
    )
    assert entry == {
        "reviewed": True,
        "image_sha256": "a" * 64,
        "description_hash": "b" * 64,
        "aspects": {a: True for a in ASPECTS},
        "subject_ok": True,
        "note": "looks good",
    }


@pytest.mark.parametrize(
    "aspects,subject_ok",
    [
        ({a: True for a in ASPECTS}, False),
        ({**{a: True for a in ASPECTS}, "mood": False}, True),
        ({"color": True}, True),
        ({}, True),
    ],
)
def test_shape_review_entry_partial_confirmation_is_reviewed_false(aspects, subject_ok):
    entry = brs.shape_review_entry(
        image_sha256="a" * 64,
        description_hash="b" * 64,
        aspects=aspects,
        subject_ok=subject_ok,
        note="",
    )
    assert entry["reviewed"] is False
    # Every axis is still reported, defaulting missing ones to False.
    assert set(entry["aspects"]) == set(ASPECTS)


def test_build_export_includes_every_card_and_top_level_metadata(bundle):
    cards = brs.gather_cards(assets=bundle["root"], review_path=None)
    states = {
        cards[0]["id"]: {"aspects": {a: True for a in ASPECTS}, "subject_ok": True}
    }
    review = brs.build_export(cards, states, reviewer="tester", date="2026-09-21")
    assert set(review) == {card["id"] for card in cards} | {"_metadata"}
    assert review[cards[0]["id"]]["reviewed"] is True
    assert review[cards[1]["id"]]["reviewed"] is False
    assert review["_metadata"] == {
        "reviewer": "tester",
        "date": "2026-09-21",
        "generated_by": "build_review_sheet.py",
    }


# ---------------------------------------- load_catalog accepts/rejects this shape


def test_load_catalog_accepts_a_fully_confirmed_export_entry_and_excludes_partial(
    bundle,
):
    cards = brs.gather_cards(assets=bundle["root"], review_path=None)
    full_id, partial_id = cards[0]["id"], cards[1]["id"]
    states = {
        full_id: {
            "aspects": {a: True for a in ASPECTS},
            "subject_ok": True,
            "note": "confirmed by hand",
        },
        partial_id: {
            "aspects": {a: True for a in ASPECTS},
            "subject_ok": False,  # subject box unchecked -> not reviewed
            "note": "face too small",
        },
    }
    review = brs.build_export(cards, states, reviewer="owner", date="2026-09-21")
    review_path = bundle["root"].parent / "cards-v2-review.json"
    review_path.write_text(json.dumps(review, ensure_ascii=False))

    loaded = load_catalog("catalog-v2", assets=bundle["root"], review_path=review_path)
    loaded_ids = {card["id"] for card in loaded["cards"]}
    assert full_id in loaded_ids
    assert partial_id not in loaded_ids
    # The exported entries for the other 62 unreviewed cards, and the extra
    # top-level "_metadata" key, do not break the loader either.
    assert len(loaded["all_cards"]) == 64
    assert loaded["catalog_hash"]


def test_load_catalog_tolerates_reviewed_false_entries_for_untouched_cards(bundle):
    """catalog.py must not error on the export's untouched, reviewed:false rows."""
    cards = brs.gather_cards(assets=bundle["root"], review_path=None)
    review = brs.build_export(cards, {})  # nothing confirmed at all
    assert all(
        entry["reviewed"] is False
        for cid, entry in review.items()
        if cid != "_metadata"
    )
    review_path = bundle["root"].parent / "cards-v2-review.json"
    review_path.write_text(json.dumps(review, ensure_ascii=False))
    loaded = load_catalog("catalog-v2", assets=bundle["root"], review_path=review_path)
    assert loaded["cards"] == []
    assert len(loaded["all_cards"]) == 64


# -------------------------------------------------------------------- import


def test_import_prefills_only_hash_matching_entries(bundle, tmp_path):
    cards = brs.gather_cards(assets=bundle["root"], review_path=None)
    matching, stale = cards[0], cards[1]
    review = {
        matching["id"]: {
            "reviewed": True,
            "image_sha256": matching["image_sha256"],
            "description_hash": matching["description_hash"],
            "aspects": {a: True for a in ASPECTS},
            "subject_ok": True,
            "note": "carried over",
        },
        stale["id"]: {
            "reviewed": True,
            "image_sha256": "0" * 64,  # no longer matches the built image
            "description_hash": stale["description_hash"],
            "aspects": {a: True for a in ASPECTS},
            "subject_ok": True,
            "note": "should not be imported",
        },
    }
    review_path = tmp_path / "existing-review.json"
    review_path.write_text(json.dumps(review, ensure_ascii=False))

    out = tmp_path / "out" / "cards-v2-review.html"
    brs.build(assets=bundle["root"], out=out, review_path=review_path)
    payload = extract_payload(out.read_text())

    assert payload["imported"][matching["id"]]["subject_ok"] is True
    assert payload["imported"][matching["id"]]["aspects"] == {a: True for a in ASPECTS}
    assert stale["id"] not in payload["imported"]


def test_description_change_invalidates_import_like_the_loader(
    bundle, tmp_path, monkeypatch
):
    """A definition edit changes description_hash; catalog.py invalidates the same way."""
    # Compute the original hashes *before* the definition changes, so the
    # review file below is bound to the pre-edit description, like a real
    # earlier export would be.
    original_cards = brs.gather_cards(assets=bundle["root"], review_path=None)
    card = original_cards[0]

    changed = definition()
    changed["axes"]["color"][0] += ", edited"
    changed_path = tmp_path / "catalog-definition.json"
    changed_path.write_text(json.dumps(changed))
    monkeypatch.setattr(catalog_module, "V2", changed_path)

    review = {
        card["id"]: {
            "reviewed": True,
            "image_sha256": card["image_sha256"],
            "description_hash": card["description_hash"],
            "aspects": {a: True for a in ASPECTS},
            "subject_ok": True,
            "note": "",
        }
    }
    review_path = tmp_path / "existing-review.json"
    review_path.write_text(json.dumps(review, ensure_ascii=False))

    out = tmp_path / "out" / "cards-v2-review.html"
    brs.build(assets=bundle["root"], out=out, review_path=review_path)
    payload = extract_payload(out.read_text())
    assert card["id"] not in payload["imported"]


# ------------------------------------------------------------------------ CLI


def test_cli_main_builds_the_page(bundle, tmp_path):
    out = tmp_path / "cli-out" / "cards-v2-review.html"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_review_sheet.py"),
            "--assets",
            str(bundle["root"]),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["cards"] == 64
    assert out.is_file()
