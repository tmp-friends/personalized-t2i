#!/usr/bin/env python3
"""Validate exact card prompts before generation; exits nonzero on truncation."""

import argparse
import json
from pathlib import Path

from exhibit.catalog import (
    LEGACY_OVERFLOW_IDS,
    load_catalog,
    prompt_set_hash,
    validate_card_tokens,
)
from exhibit.config import CONFIG, ROOT
from exhibit.domain import file_hash
from exhibit.evaluation import runtime_tokenizer_provenance


def build_token_report(catalog_id, cards, tokenizers, provenance, legacy_path):
    """Create an auditable report bound to prompts, special tokens, and files."""
    rows = validate_card_tokens(cards, tokenizers)
    legacy_path = Path(legacy_path)
    try:
        legacy = json.loads(legacy_path.read_text())
        overflow_sets = [
            [item["id"] for item in tokenizer["over_limit"]] for tokenizer in legacy
        ]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("Legacy overflow evidence is invalid") from exc
    expected = list(LEGACY_OVERFLOW_IDS)
    if not overflow_sets or any(items != expected for items in overflow_sets):
        raise ValueError("Legacy overflow evidence changed")
    try:
        relative = legacy_path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        relative = legacy_path
    return {
        "schema_version": 1,
        "catalog_id": catalog_id,
        "generation": CONFIG["generation"],
        "prompt_set_hash": prompt_set_hash(cards),
        "tokenizers": provenance,
        "results": rows,
        "max_tokens": max((row["tokens"] for row in rows), default=0),
        "legacy_overflow_evidence": {
            "path": str(relative),
            "sha256": file_hash(legacy_path),
            "over_limit_ids": expected,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", choices=("v1", "v2"), default="v2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    pipeline = CONFIG["generation"]["pipeline_config"]
    tokenizers = {
        name: AutoTokenizer.from_pretrained(
            pipeline["model"],
            revision=pipeline["revision"],
            subfolder=name,
            local_files_only=True,
        )
        for name in ("tokenizer", "tokenizer_2")
    }
    catalog_id = "catalog-" + args.catalog
    cards = load_catalog(catalog_id, reviewed_only=False)["all_cards"]
    report = build_token_report(
        catalog_id,
        cards,
        tokenizers,
        runtime_tokenizer_provenance(CONFIG["generation"]),
        ROOT / "configs/legacy-card-token-overflow.json",
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if any(row["overflow"] for row in report["results"]):
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
