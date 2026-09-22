#!/usr/bin/env python3
"""Validate exact card prompts before generation; exits nonzero on truncation."""

import argparse
import json
from pathlib import Path

from exhibit.catalog import (
    CATALOG_ID,
    build_catalog,
    card_settings,
    prompt_set_hash,
    validate_card_tokens,
    validate_negative_tokens,
)
from exhibit.config import CONFIG, ROOT, read_json
from exhibit.evaluation import runtime_tokenizer_provenance


def build_token_report(cards, tokenizers, provenance, generation=None):
    """Create an auditable report bound to prompts, special tokens, and files."""
    generation = generation if generation is not None else card_settings()
    rows = validate_card_tokens(cards, tokenizers)
    return {
        "schema_version": 3,
        "catalog_id": CATALOG_ID,
        "generation": generation,
        "prompt_set_hash": prompt_set_hash(cards),
        "tokenizers": provenance,
        "results": rows,
        "max_tokens": max((row["tokens"] for row in rows), default=0),
        "negative_validation": validate_negative_tokens(
            generation["negative_prompt"], tokenizers
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
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
    # This gate runs before generation, so it never reads the assets it gates.
    definition = read_json(ROOT / "configs/catalog-v2.json")
    cards = build_catalog(definition)
    generation = card_settings(definition)
    report = build_token_report(
        cards,
        tokenizers,
        runtime_tokenizer_provenance(CONFIG["generation"]),
        generation,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    checked = report["results"] + report["negative_validation"]["results"]
    if any(row["overflow"] for row in checked):
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
