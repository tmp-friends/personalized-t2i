"""Validate exact card prompts before generation; exits nonzero on truncation."""
import argparse
import hashlib
import json
from pathlib import Path

from exhibit.catalog import load_catalog, validate_card_tokens
from exhibit.config import CONFIG


def main():
 p=argparse.ArgumentParser(); p.add_argument("--catalog", choices=("v1","v2"), default="v2"); p.add_argument("--output", type=Path); a=p.parse_args()
 from transformers import AutoTokenizer
 pipeline=CONFIG["generation"]["pipeline_config"]
 tokenizers={}
 for name in ("tokenizer", "tokenizer_2"):
  tok=AutoTokenizer.from_pretrained(pipeline["model"], revision=pipeline["revision"], subfolder=name, local_files_only=True)
  tokenizers[name] = tok
 rows=validate_card_tokens(load_catalog("catalog-"+a.catalog, reviewed_only=False)["all_cards"], tokenizers)
 value = {"catalog": "catalog-" + a.catalog, "pipeline": pipeline, "tokenizers": {name: {"name_or_path": str(tok.name_or_path), "vocab_hash": hashlib.sha256(json.dumps(tok.get_vocab(), sort_keys=True).encode()).hexdigest()} for name, tok in tokenizers.items()}, "results": rows, "max_tokens": max(row["tokens"] for row in rows)}
 text=json.dumps(value,ensure_ascii=False,indent=2)+"\n"
 if a.output:
  a.output.parent.mkdir(parents=True, exist_ok=True)
  a.output.write_text(text)
 print(text,end="")
 if any(row["overflow"] for row in rows): raise SystemExit(1)
if __name__ == "__main__": main()
