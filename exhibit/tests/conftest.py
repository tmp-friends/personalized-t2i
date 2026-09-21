"""Fixed-asset trees built from the real contract, without touching the GPU."""

import json
from pathlib import Path

import pytest
from exhibit.config import CONFIG, write_json
from exhibit.domain import CARDS, build_personalization, file_hash, target_prompt


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


def card_images(root):
    images = {}
    for card in CARDS.values():
        relative = card["path"]
        images[card["id"]] = {
            "path": relative,
            "sha256": fake_png(root / relative, f"card {card['id']}"),
            "seed": card["seed"],
            "prompt": card["prompt"],
            "ref_en": card["ref_en"],
            "aspects": card["aspects"],
            "settings": CONFIG["generation"],
        }
    return images


def sample_records(root):
    ids = list(CARDS)
    samples = []
    for number in range(3):
        selection = [
            {"card_id": ids[number * 3 + offset], "aspects_off": []}
            for offset in range(3)
        ]
        personalization = build_personalization(
            selection, {entry["card_id"]: "normal" for entry in selection}, "mid"
        )
        for topic in CONFIG["topics"][:2]:
            sample_id = f"s{number + 1}-{topic['id']}"
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
                    "alpha_key": "mid",
                    "selection": selection,
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
    images, prompts = generic_images(root)
    images.update(card_images(root))
    write_json(
        root / "manifest.json",
        {"version": 5, "generation": CONFIG["generation"], "images": images},
    )
    write_json(root / "generic-prompts.json", prompts)
    write_json(root / "samples.json", sample_records(root))
    review = tmp_path / "cards-review.json"
    review.write_text(
        json.dumps({card_id: {"reviewed": True, "note": ""} for card_id in CARDS})
    )
    return {"root": root, "review": review}


@pytest.fixture
def assets(asset_tree, monkeypatch):
    """The service and app read the bundle through their own module globals."""
    from exhibit import app as app_module
    from exhibit import preflight as preflight_module
    from exhibit import service as service_module

    root = asset_tree["root"]
    for module in (service_module, app_module, preflight_module):
        monkeypatch.setattr(module, "ASSETS", root, raising=False)
    monkeypatch.setattr(app_module, "reviewed_ids", lambda: set(CARDS), raising=False)
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
            on_event(
                {
                    "type": "image",
                    "id": item["id"],
                    "path": item["path"],
                    "sha256": file_hash(path),
                    "seed": item["seed"],
                    "prompt": item["prompt"],
                    "personalization_hash": item["personalization"]["hash"],
                    "seconds": 0.01,
                }
            )
        return {"returncode": 0, "wall_seconds": 0.02}

    monkeypatch.setattr(service_module, "run_stage", stage)
    return calls
