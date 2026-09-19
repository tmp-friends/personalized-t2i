# Layout migration manifest

Captured before directory moves on 2026-09-19. The recoverable patch files live
under the ignored `.migration-backup/` directory in the implementation
worktree. They are intentionally retained until the completed migration has
passed fresh-clone verification and their removal is approved explicitly.

## Repository states

| Repository | Base/HEAD | Remote at capture | Dirty state | Final classification |
|---|---|---|---|---|
| Parent | `288994b560cf3180d37550769d54da2f4a73bf08` plus the paths listed below | `ssh://git@github.com/tmp-friends/personalized-t2i.git` | FAN and Tailored Visions edits plus one deleted FAN sample | Preserve in moved README/source/patches |
| FAN | `9d0b76843f6437718195accac9cf3f050a25d26b` | `ssh://git@github.com/Burf/FAN.git` | Modified `inference.py`, deleted sample, and untracked integration files that the parent repository tracks | CLI/docs outside upstream; source delta as a patch only if still required |
| Premier | `42473476a189b6b0127890a98e92b7edf49c0d59` | `ssh://git@github.com/120L020904/Premier.git` | Single-block checkpoint fix plus ignored Python caches | `patches/0001-fix-single-block-checkpoint-kwarg.patch`; caches are discarded |
| Tailored Visions | upstream `d0f4454ca08c68c5d30f08a01ff4a23a8b33b610`; local HEAD `f53e7481f60954e30e37275791034e4201bbc6bd` plus working changes | `ssh://git@github.com/zzjchen/Tailored-Visions.git` | Four local commits and five modified files | Numbered official patches plus outer wrappers/docs |

The final `.gitmodules` URLs use HTTPS even though the captured nested clones
used SSH.

## Parent dirty paths at capture

```text
 M FAN/README_ja.md
 D FAN/image/00000.png
 M tailored-visions-repro/README.md
 M tailored-visions-repro/docs/DEVIATIONS.md
 M tailored-visions-repro/official/SD.py
 M tailored-visions-repro/official/SETUP.md
 M tailored-visions-repro/official/demo.py
 M tailored-visions-repro/official/run.sh
 M tailored-visions-repro/scripts/05_generate.py
 M tailored-visions-repro/tv/generate.py
```

## Nested dirty paths at capture

### FAN

```text
 D image/00000.png
 M inference.py
?? .gitignore
?? FAN_summary.html
?? README_ja.md
?? pyproject.toml
?? smoke_clip.py
?? uv.lock
```

### Premier

```text
 M scripts/pipeline/flux_adapter.py
?? scripts/__pycache__/
?? scripts/pipeline/__pycache__/
?? scripts/train_flux/__pycache__/
?? scripts/utils/__pycache__/
```

### Tailored Visions

```text
 M SD.py
 M SETUP.md
 M demo.py
 M main.py
 M run.sh
```

The four local Tailored Visions commits are also exported separately:

```text
0001-Make-the-CVPR-2024-release-run-in-2026.patch
0002-Add-run.sh-wrapper-so-the-repo-runs-from-one-termina.patch
0003-SETUP.md-complete-the-patch-table.patch
0004-SETUP.md-note-that-run.sh-leaves-the-LLM-server-resi.patch
```

## Backup patch checksums

```text
a28fa0dc683727d1b1678c52073a84a8dcf3a7b4e00c22a4c1cbb07c5652a838  FAN-working-tree.patch
9361809f93211002e3fd511954b0d9be192c10a3fa7527d6047541b0a0dd2877  Premier-working-tree.patch
d11a96921679439ea0495d9769e258fbe46e41f6faf4014cc09f14f031f6dfdf  Tailored-Visions-working-tree.patch
7d16f077f36ad62a0d3031d353940ca016b9be40a591a12be94787237c958724  root-working-tree.patch
```

## Final destination mapping

Paths ending in `/` describe the whole subtree. Ignored local artifacts stay out of Git;
where the isolated worktree did not contain them, the original checkout remains untouched
and the move is a one-time post-integration operation.

| Method | Old path | Final path / classification |
|---|---|---|
| FAN | `FAN/` | `fan-repro/` |
| FAN | `FAN/README_ja.md` | `fan-repro/README.md` |
| FAN | `FAN/FAN_summary.html` | `fan-repro/docs/summary.html` |
| FAN | `FAN/smoke_clip.py` | `fan-repro/scripts/smoke_clip.py` |
| FAN | `FAN/inference.py`, `FAN/fan/`, official assets | `fan-repro/upstream/`; local CLI behavior is in `fan-repro/scripts/generate.py` |
| FAN | `FAN/image/` | `fan-repro/outputs/` (ignored; generated sample deletion preserved) |
| Premier | `premier-repro/official/Premier/` | `premier-repro/upstream/` |
| Premier | `premier-repro/official/patches/` | `premier-repro/patches/` |
| Premier | `premier-repro/premier/` | `premier-repro/src/premier_repro/` |
| Premier | `premier-repro/official/premier_local.py` | `premier-repro/src/premier_repro/official.py` |
| Premier | `premier-repro/official/run_premier.py` | `premier-repro/scripts/run_official.py` |
| Premier | `premier-repro/official/train_new_user.py` | `premier-repro/scripts/train_official_user.py` |
| Premier | `premier-repro/official/README_ja.md` | `premier-repro/docs/OFFICIAL_SETUP.md` |
| Premier | `premier-repro/docs/premier_summary.html` | `premier-repro/docs/summary.html` |
| Premier | `premier-repro/official/requirements-local.txt` | `premier-repro/requirements-official.txt` |
| Premier | `premier-repro/official/weights/` | `premier-repro/artifacts/weights/` (ignored) |
| Premier | `premier-repro/official/test_data/` | `premier-repro/data/examples/` |
| Premier | `premier-repro/official/outputs/` | `premier-repro/outputs/official/` (ignored) |
| Tailored Visions | `tailored-visions-repro/official/` official source | `tailored-visions-repro/upstream/` + `tailored-visions-repro/patches/` |
| Tailored Visions | `tailored-visions-repro/official/serve_local_llm.py` | `tailored-visions-repro/scripts/serve_local_llm.py` |
| Tailored Visions | `tailored-visions-repro/official/run.sh` | `tailored-visions-repro/scripts/run_official.sh` |
| Tailored Visions | `tailored-visions-repro/official/SETUP.md` | `tailored-visions-repro/docs/OFFICIAL_SETUP.md` |
| Tailored Visions | `tailored-visions-repro/tv/` | `tailored-visions-repro/src/tailored_visions_repro/` |
| Tailored Visions | `tailored-visions-repro/docs/tailored-visions.html` | `tailored-visions-repro/docs/summary.html` |
| Tailored Visions | `tailored-visions-repro/data/user_data/` | `tailored-visions-repro/data/raw/user_data/` (ignored) |
| Tailored Visions | `tailored-visions-repro/results/` | `tailored-visions-repro/outputs/` (ignored) |
| Tailored Visions | official generated images | `tailored-visions-repro/outputs/official/` (ignored; none existed in the isolated worktree) |
| Tailored Visions | `official/demo_user.jsonl` | generated on demand from `data/raw/user_data/`; no stable fixture was present to migrate |

## Final submodules

| Path | HTTPS URL | Pinned commit |
|---|---|---|
| `fan-repro/upstream` | `https://github.com/Burf/FAN.git` | `9d0b76843f6437718195accac9cf3f050a25d26b` |
| `premier-repro/upstream` | `https://github.com/120L020904/Premier.git` | `42473476a189b6b0127890a98e92b7edf49c0d59` |
| `tailored-visions-repro/upstream` | `https://github.com/zzjchen/Tailored-Visions.git` | `d0f4454ca08c68c5d30f08a01ff4a23a8b33b610` |

## Migration commits

| Task | Commit | Purpose |
|---|---|---|
| 1 | `7b147a7` | Atomic upstream patch materializer |
| 2 | `bad69a7` | Migration inputs and recovery patches |
| 3 | `3847ec2` | Premier normalized layout and submodule |
| 4 | `6de8df6` | FAN normalized layout and submodule |
| 5 | `430d8ff` | Tailored Visions compatibility patch series |
| 6 | `0ecd9c0` | Tailored Visions local package and output layout |
| 7 | `963bd4f` | Tailored Visions submodule and outer integration |
| 8 | recorded by Task 9 after Task 8 exists | Root documentation and repository contract; a commit cannot contain its own SHA |

## Verification on the Task 8 working tree

| Check | Result |
|---|---|
| `python3 -m unittest discover -s tests -v` | 8 tests passed |
| `python3 -m unittest discover -s fan-repro/tests -v` | 3 tests passed |
| `python3 -m unittest discover -s premier-repro/tests -v` | 4 tests passed |
| `python3 -m unittest discover -s tailored-visions-repro/tests -v` | 7 tests passed |
| all three `scripts/prepare_upstream.py` commands | passed; materialization is idempotent |
| `git submodule status --recursive` | all three expected pinned commits present |
| `git submodule foreach --recursive git status --short` | all three submodules clean |
| obsolete-path audit | only the explicit historical migration note and the regression-test pattern remain |
| tracked-artifact audit | no `.venv`, `.work`, `artifacts`, `outputs`, or `data/raw` paths tracked |
| `git diff --check` | clean |

## Recovery backup retention

`.migration-backup/` remains ignored and retained in the implementation worktree.
It must not be deleted until Task 9 fresh-clone verification passes **and** a human gives
explicit approval. Fresh-clone success alone is not authorization to remove it.
