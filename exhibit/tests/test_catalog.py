import copy

from exhibit.catalog import build_catalog, load_catalog
from exhibit.config import ROOT, read_json


def test_v2_build_has_orthogonal_64_card_layout():
    cards = build_catalog(read_json(ROOT / "configs/catalog-v2.json"))
    assert len(cards) == 64
    assert len({c["profile_id"] for c in cards}) == 16
    for subject in {c["subject_id"] for c in cards}:
        rows = [c for c in cards if c["subject_id"] == subject]
        assert len({c["seed"] for c in rows}) == 1
        for left in ("color", "lighting", "texture", "mood"):
            for right in ("color", "lighting", "texture", "mood"):
                if left < right:
                    assert len({(c["axis_levels"][left], c["axis_levels"][right]) for c in rows}) == 16


def test_v2_loader_excludes_unreviewed_cards():
    catalog = load_catalog("catalog-v2")
    assert len(catalog["all_cards"]) == 64
    assert catalog["cards"] == []
    assert catalog["catalog_hash"] == load_catalog("catalog-v2", reviewed_only=False)["catalog_hash"]
