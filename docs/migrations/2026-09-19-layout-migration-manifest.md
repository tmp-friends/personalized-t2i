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

## Planned destination mapping

| Captured content | Destination |
|---|---|
| FAN local README and summary | `fan-repro/README.md`, `fan-repro/docs/summary.html` |
| FAN local CLI and smoke script | `fan-repro/scripts/` |
| FAN official files | `fan-repro/upstream` submodule |
| Premier official checkpoint fix | `premier-repro/patches/` |
| Premier official wrappers | `premier-repro/src/premier_repro/` and `premier-repro/scripts/` |
| Tailored Visions official source changes | `tailored-visions-repro/patches/` |
| Tailored Visions local server, runner, and setup guide | `tailored-visions-repro/scripts/` and `tailored-visions-repro/docs/` |
| Tailored Visions local package changes | `tailored-visions-repro/src/tailored_visions_repro/` |

This table is extended with actual migration commits and verification results
after all three methods have moved.
