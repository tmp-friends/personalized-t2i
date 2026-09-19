#!/usr/bin/env bash
# Full reproduction, start to finish. Roughly 3-4 hours on one RTX 4090.
#
#   ./run_all.sh                    # everything
#   ./run_all.sh --skip-ablations   # Table 2 + image metrics only (~1.5h)
#
# Every stage is resumable: rerunning skips work whose output already exists,
# so an interrupted run picks up where it stopped. Pass --overwrite to a stage
# to force it.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
N_EVAL_USERS=${N_EVAL_USERS:-500}     # users scored by the image metrics
SKIP_ABLATIONS=0
[[ "${1:-}" == "--skip-ablations" ]] && SKIP_ABLATIONS=1

step() { printf '\n=== %s ===\n' "$1"; }

step "0/8  dataset"
$PY scripts/00_download_data.py

step "1/8  dedup histories + CLIP embeddings"
$PY scripts/01_prepare.py

step "2/8  retrieve + rewrite (Table 2, full test set)"
$PY scripts/02_rewrite.py --methods table2 --batch-size 16

step "3/8  leakage analysis"
# Run after stage 2 so it can also check whether retrieval surfaces the leak.
$PY scripts/07_leakage.py

step "4/8  evaluation subset"
$PY scripts/03_subset.py --n-users "$N_EVAL_USERS"

step "5/8  user preference summaries (for PMS)"
$PY scripts/04_preferences.py --users-from outputs/eval_subset.json

step "6/8  generate images + CLIP embeddings"
$PY scripts/05_generate.py --methods table2 --save-images 8

if [[ $SKIP_ABLATIONS -eq 0 ]]; then
  # Ablations run on the same users the image metrics score, so Tables 4 and 5
  # get PMS and Image-Align too rather than a ROUGE-L-only column.
  step "7/8  ablations (Tables 4 & 5)"
  $PY scripts/02_rewrite.py --methods ablation_topk,ablation_icl \
      --users-from outputs/eval_subset.json --batch-size 16
  $PY scripts/05_generate.py --methods all
fi

step "8/8  metrics + qualitative examples"
$PY scripts/06_evaluate.py
$PY scripts/08_examples.py --n 12

printf '\nDone. See outputs/RESULTS.md, outputs/EXAMPLES.md and docs/DEVIATIONS.md\n'
