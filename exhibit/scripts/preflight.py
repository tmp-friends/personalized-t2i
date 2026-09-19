#!/usr/bin/env python3
import argparse
import json

from exhibit.preflight import check_assets, check_models

p = argparse.ArgumentParser()
p.add_argument("--models", action="store_true")
a = p.parse_args()
r = check_assets()
if a.models:
    r["models"] = check_models()
    r["ready"] = r["ready"] and r["models"]["ready"]
print(json.dumps(r, ensure_ascii=False, indent=2))
raise SystemExit(0 if r["ready"] else 1)
