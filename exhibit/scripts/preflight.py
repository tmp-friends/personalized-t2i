#!/usr/bin/env python3
import argparse
import json

from exhibit.config import OUTPUTS
from exhibit.preflight import write_preflight

p = argparse.ArgumentParser()
p.add_argument("--models", action="store_true")
a = p.parse_args()
r = write_preflight(OUTPUTS / "preflight.json", include_models=a.models)
print(json.dumps(r, ensure_ascii=False, indent=2))
raise SystemExit(0 if r["ready"] else 1)
