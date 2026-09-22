"""Fixed-asset trees built from the real contract, without touching the GPU."""

import json
from pathlib import Path

import pytest
from exhibit.catalog import build_catalog, card_settings, description_hash, load_catalog
from exhibit.config import CONFIG, ROOT, read_json, write_json
from exhibit.domain import (
    ASPECTS,
    build_personalization,
    digest,
    file_hash,
    target_prompt,
)
from exhibit.elicitation import normalize_preferences
from exhibit.service import default_policy

# Injected so a CPU test never needs the Hub cache; the shape is the real one.
# Only the two source hashes are fabricated: the rest is the shipped contract.
PROVENANCE = {
    "fan_pin": CONFIG["fan"]["commit"],
    "adapter_hash": "a" * 64,
    "decoder_hash": digest(CONFIG["fan"]["decoders"]),
    "tokenizer_hash": "t" * 64,
    "generation": CONFIG["generation"],
    "seeds": CONFIG["seeds"],
}


def fake_png(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.encode())
    return file_hash(path)


def generic_images(root):
    images = {}
    prompts = {}
    for topic in CONFIG["topics"]:
        prompt = target_prompt(topic)
        prompts[topic["id"]] = {"prompt": prompt, "topic_id": topic["id"]}
        for index, seed in enumerate(CONFIG["seeds"]):
            relative = f"generic/{topic['id']}-{index}.png"
            images[f"{topic['id']}-{index}"] = {
                "path": relative,
                "sha256": fake_png(root / relative, f"generic {topic['id']} {index}"),
                "seed": seed,
                "prompt": prompt,
                "settings": CONFIG["generation"],
                "personalization_hash": None,
            }
    return images, prompts


def token_validation(cards, settings, *, overflow=False):
    """A report shaped exactly like check_card_tokens.py's, with stub token ids."""
    rows = [
        {
            "card_id": card["id"],
            "prompt_hash": digest(card["prompt"]),
            "tokenizer": tokenizer,
            "token_ids": list(range(78))
            if overflow and card is cards[0]
            else [1, 2, 3],
            "tokens": 78 if overflow and card is cards[0] else 3,
            "limit": 77,
            "overflow": overflow and card is cards[0],
            "special_tokens": True,
        }
        for card in cards
        for tokenizer in ("tokenizer", "tokenizer_2")
    ]
    negative = [
        {
            "prompt_hash": digest(settings["negative_prompt"]),
            "tokenizer": tokenizer,
            "token_ids": [1, 2, 3],
            "tokens": 3,
            "limit": 77,
            "overflow": False,
            "special_tokens": True,
        }
        for tokenizer in ("tokenizer", "tokenizer_2")
    ]
    return {
        "schema_version": 3,
        "catalog_id": "catalog-v2",
        "generation": settings,
        "prompt_set_hash": digest(
            [{"id": card["id"], "prompt": card["prompt"]} for card in cards]
        ),
        "tokenizers": {
            "repo_id": CONFIG["generation"]["pipeline_config"]["model"],
            "revision": CONFIG["generation"]["pipeline_config"]["revision"],
            "files": {
                f"{name}/{filename}": "1" * 64
                for name in ("tokenizer", "tokenizer_2")
                for filename in (
                    "vocab.json",
                    "merges.txt",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                )
            },
        },
        "results": rows,
        "max_tokens": max(row["tokens"] for row in rows),
        "negative_validation": {
            "prompt_hash": digest(settings["negative_prompt"]),
            "results": negative,
            "max_tokens": 3,
        },
    }


def write_catalog_bundle(root, review_path, *, reviewed=True):
    """The generated catalog, its images and a matching human review."""
    definition = read_json(ROOT / "configs/catalog-v2.json")
    cards = build_catalog(definition)
    settings = card_settings(definition)
    images, reviews = {}, {}
    for card in cards:
        images[card["id"]] = {
            "path": card["path"],
            "sha256": fake_png(root / card["path"], "image:" + card["id"]),
            "seed": card["seed"],
            "prompt": card["prompt"],
            "ref_en": card["ref_en"],
            "aspects": card["aspects"],
            "aspects_ja": card["aspects_ja"],
            "label": card["label"],
            "profile_label": card["profile_label"],
            "settings": settings,
        }
        reviews[card["id"]] = {
            # `reviewed` is a flag for every card, or the set that passed.
            "reviewed": reviewed
            if isinstance(reviewed, bool)
            else card["id"] in reviewed,
            "image_sha256": images[card["id"]]["sha256"],
            "description_hash": description_hash(card),
            "aspects": {aspect: True for aspect in ASPECTS},
            "note": "human note",
        }
    write_json(
        root / "catalog-v2.json",
        {
            "version": 2,
            "catalog_id": "catalog-v2",
            "generation": settings,
            "token_validation": token_validation(cards, settings),
            "images": images,
        },
    )
    review_path.write_text(json.dumps(reviews))
    return cards, images, reviews


def sample_records(root, review_path):
    """Three selections x two topics, built exactly as prepare.py builds them."""
    catalog = load_catalog(reviewed_only=True, assets=root, review_path=review_path)
    policy_id, policy = default_policy()
    ids = [card["id"] for card in catalog["cards"]]
    samples = []
    for number in range(3):
        snapshot = {
            "revision": 0,
            **normalize_preferences(
                {
                    "cards": [
                        {
                            "card_id": ids[number * 3 + offset],
                            "strength": 1,
                            "aspects": ["color", "texture"],
                        }
                        for offset in range(3)
                    ],
                    "aspect_gains": {aspect: 1 for aspect in ASPECTS},
                },
                catalog,
                commit=True,
                selection=CONFIG["selection"],
            ),
        }
        for topic in CONFIG["topics"][:2]:
            sample_id = f"s{number + 1}-{topic['id']}"
            personalization = build_personalization(
                snapshot,
                prompt=target_prompt(topic),
                policy=policy,
                provenance=PROVENANCE,
                catalog=catalog,
            )
            personalization["policy_id"] = policy_id
            images = []
            for index, seed in enumerate(CONFIG["seeds"]):
                relative = f"samples/{sample_id}-{index}.png"
                images.append(
                    {
                        "id": f"{sample_id}-{index}",
                        "path": relative,
                        "sha256": fake_png(
                            root / relative, f"sample {sample_id}{index}"
                        ),
                        "seed": seed,
                        "prompt": target_prompt(topic),
                        "settings": CONFIG["generation"],
                        "personalization_hash": personalization["hash"],
                    }
                )
            samples.append(
                {
                    "id": sample_id,
                    "topic_id": topic["id"],
                    "label": f"サンプル{number + 1} · {topic['label']}",
                    "preference": snapshot,
                    "personalization": personalization,
                    "mode": "sample",
                    "images": images,
                }
            )
    return samples


@pytest.fixture
def asset_tree(tmp_path):
    """A complete, self-consistent bundle: cards, generic images and samples."""
    root = tmp_path / "assets"
    review = tmp_path / "cards-v2-review.json"
    cards, card_map, reviews = write_catalog_bundle(root, review)
    images, prompts = generic_images(root)
    write_json(
        root / "manifest.json",
        {"version": 5, "generation": CONFIG["generation"], "images": images},
    )
    write_json(root / "generic-prompts.json", prompts)
    write_json(root / "samples.json", sample_records(root, review))
    return {
        "root": root,
        "review": review,
        "cards": cards,
        "images": card_map,
        "reviews": reviews,
    }


@pytest.fixture
def assets(asset_tree, monkeypatch):
    """The service and app read the bundle through their own module globals."""
    from exhibit import app as app_module
    from exhibit import catalog as catalog_module
    from exhibit import preflight as preflight_module
    from exhibit import service as service_module

    root = asset_tree["root"]
    for module in (service_module, app_module, preflight_module):
        monkeypatch.setattr(module, "ASSETS", root, raising=False)
    # The catalog is loaded through the service, so one review path covers both.
    monkeypatch.setattr(catalog_module, "REVIEW", asset_tree["review"])
    monkeypatch.setattr(catalog_module, "ASSETS", root)
    return root


@pytest.fixture
def stage_stub(monkeypatch):
    """Replace only the GPU subprocess; publication and caching stay real."""
    from exhibit import service as service_module

    calls = []

    def stage(request, directory, cancel, deadline, on_event):
        calls.append(request)
        for item in request["items"]:
            path = Path(item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item["id"].encode())
            personalization = item["personalization"]
            # The same identity fields the real worker emits with every image.
            on_event(
                {
                    "type": "image",
                    "id": item["id"],
                    "path": item["path"],
                    "sha256": file_hash(path),
                    "seed": item["seed"],
                    "prompt": item["prompt"],
                    "policy_id": personalization["policy_id"],
                    "policy_hash": personalization["policy_hash"],
                    "effective_policy": personalization["effective_policy"],
                    "personalization_hash": personalization["hash"],
                    "seconds": 0.01,
                }
            )
        return {"returncode": 0, "wall_seconds": 0.02}

    monkeypatch.setattr(service_module, "run_stage", stage)
    return calls
