# Repository Layout and Submodules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize FAN, Premier, and Tailored Visions around the same repository contract while preserving every current local change and tracking each official implementation as a clean, pinned Git submodule.

**Architecture:** A shared standard-library tool materializes a patched copy from each clean `upstream/` submodule into an ignored `.work/upstream/` directory. Method-local shims expose the same preparation command, while local packages, scripts, documentation, data, artifacts, and outputs remain outside the submodule. The migration proceeds Premier → FAN → Tailored Visions so the simplest repository establishes the pattern before the repositories with mixed local changes.

**Tech Stack:** Git submodules, Python 3.10–3.12 standard library, `uv`, `unittest`, existing PyTorch/diffusers environments

**Spec:** `docs/superpowers/specs/2026-09-19-repository-layout-and-submodules-design.md`

## Global Constraints

- Preserve all parent-repository and nested-repository working-tree changes visible at the start of migration.
- Keep the three Python environments independent; do not create a shared runtime environment.
- Use HTTPS for all three URLs in `.gitmodules`.
- Pin FAN to `9d0b76843f6437718195accac9cf3f050a25d26b`, Premier to `42473476a189b6b0127890a98e92b7edf49c0d59`, and Tailored Visions to `d0f4454ca08c68c5d30f08a01ff4a23a8b33b610` for the initial migration.
- Never edit a file inside a checked-out `upstream/`; patches are applied only to `.work/upstream/`.
- Keep `.venv/`, `.work/`, `data/raw/`, `data/cache/`, `artifacts/`, and `outputs/` outside Git.
- Keep only small deterministic fixtures under `data/examples/`.
- Do not delete the migration backup until fresh-clone verification passes and the human partner explicitly approves its removal.
- Use one commit per independently reviewable task; do not mix unrelated existing changes into a task commit.

## Review Focus

- A missing or uninitialized submodule must fail with an actionable `git submodule update --init --recursive` message; Task 1 tests this.
- A submodule checked out at an unexpected commit must fail before any patched tree is replaced; Task 1 tests this.
- A malformed patch must leave the previous `.work/upstream/` intact; Task 1 tests this.
- Re-running preparation with unchanged SHA and patches must be idempotent and preserve the materialized tree; Task 1 tests this.
- Every dirty or local-only nested-repository change must be recoverable from `.migration-backup/` and represented in its final outer-tree location or patch series; Tasks 2, 4, and 6 verify this with binary diffs and manifests.

---

### Task 1: Shared Upstream Materializer and Contract Tests

**Files:**
- Create: `tools/upstream.py`
- Create: `tests/test_upstream.py`
- Create: `fan-repro/scripts/prepare_upstream.py` later in Task 4
- Create: `premier-repro/scripts/prepare_upstream.py` later in Task 3
- Create: `tailored-visions-repro/scripts/prepare_upstream.py` later in Task 7
- Modify: `.gitignore`

**Interfaces:**
- Consumes: a method root containing `upstream/`, optional `patches/series`, and the expected SHA supplied by the method-local shim.
- Produces: `tools.upstream.prepare(method_root: Path, expected_sha: str) -> Path`, returning `method_root/.work/upstream` only after a complete atomic materialization.
- Produces: `tools.upstream.main(argv: Sequence[str] | None = None) -> int` for method-local shims.

- [ ] **Step 1: Add ignored migration and materialization directories**

Append these repository-wide rules to `.gitignore`, retaining the existing credential and editor rules:

```gitignore
**/.work/
**/data/raw/
**/data/cache/
**/artifacts/
**/outputs/
.migration-backup/
```

Run: `git check-ignore -v premier-repro/.work/upstream premier-repro/artifacts/model.bin .migration-backup/root.patch`

Expected: all three paths are matched by the new rules.

- [ ] **Step 2: Write failing tests for missing submodules and wrong SHAs**

Create `tests/test_upstream.py` with a temporary Git repository helper and these tests:

```python
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.upstream import PreparationError, prepare


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


class PrepareUpstreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.method = self.root / "method"
        self.method.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_upstream(self) -> str:
        upstream = self.method / "upstream"
        upstream.mkdir()
        git(upstream, "init")
        git(upstream, "config", "user.email", "test@example.com")
        git(upstream, "config", "user.name", "Test")
        (upstream / "value.txt").write_text("upstream\n")
        git(upstream, "add", "value.txt")
        git(upstream, "commit", "-m", "initial")
        return git(upstream, "rev-parse", "HEAD")

    def test_missing_submodule_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(PreparationError, "git submodule update --init --recursive"):
            prepare(self.method, "0" * 40)

    def test_wrong_sha_fails_before_creating_output(self) -> None:
        self.make_upstream()
        with self.assertRaisesRegex(PreparationError, "expected commit"):
            prepare(self.method, "0" * 40)
        self.assertFalse((self.method / ".work/upstream").exists())
```

- [ ] **Step 3: Run the tests and confirm the module is missing**

Run: `python -m unittest tests.test_upstream -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.upstream'`.

- [ ] **Step 4: Implement validation and archive extraction**

Create `tools/upstream.py` with the following public API and helpers:

```python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Sequence


class PreparationError(RuntimeError):
    pass


def _run(*args: str, cwd: Path | None = None, stdout=None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=stdout is None,
        stdout=stdout or subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _head(upstream: Path) -> str:
    if not (upstream / ".git").exists():
        raise PreparationError(
            f"{upstream} is not initialized; run git submodule update --init --recursive"
        )
    return _run("git", "-C", str(upstream), "rev-parse", "HEAD").stdout.strip()


def _patches(method_root: Path) -> list[Path]:
    series = method_root / "patches/series"
    if not series.exists():
        return []
    names = [line.strip() for line in series.read_text().splitlines()]
    return [series.parent / name for name in names if name and not name.startswith("#")]


def _fingerprint(expected_sha: str, patches: list[Path]) -> str:
    digest = hashlib.sha256(expected_sha.encode())
    for patch in patches:
        digest.update(patch.name.encode())
        digest.update(patch.read_bytes())
    return digest.hexdigest()


def _extract(upstream: Path, sha: str, destination: Path) -> None:
    archive_path = destination.parent / "upstream.tar"
    with archive_path.open("wb") as archive:
        _run("git", "-C", str(upstream), "archive", sha, stdout=archive)
    with tarfile.open(archive_path) as tar:
        base = destination.resolve()
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if target != base and base not in target.parents:
                raise PreparationError(f"unsafe archive path: {member.name}")
        tar.extractall(destination)
    archive_path.unlink()


def _apply(destination: Path, patches: list[Path]) -> None:
    _run("git", "init", "-q", cwd=destination)
    try:
        for patch in patches:
            _run("git", "apply", "--check", str(patch), cwd=destination)
            _run("git", "apply", str(patch), cwd=destination)
    finally:
        shutil.rmtree(destination / ".git", ignore_errors=True)


def prepare(method_root: Path, expected_sha: str) -> Path:
    method_root = method_root.resolve()
    upstream = method_root / "upstream"
    actual_sha = _head(upstream)
    if actual_sha != expected_sha:
        raise PreparationError(f"expected commit {expected_sha}, found {actual_sha}")

    patches = _patches(method_root)
    for patch in patches:
        if not patch.is_file():
            raise PreparationError(f"missing patch listed in series: {patch}")

    fingerprint = _fingerprint(expected_sha, patches)
    work = method_root / ".work"
    current = work / "upstream"
    stamp = current / ".prepared.json"
    if stamp.is_file() and json.loads(stamp.read_text()).get("fingerprint") == fingerprint:
        return current

    work.mkdir(parents=True, exist_ok=True)
    next_dir = Path(tempfile.mkdtemp(prefix="upstream-next-", dir=work))
    try:
        _extract(upstream, expected_sha, next_dir)
        _apply(next_dir, patches)
        (next_dir / ".prepared.json").write_text(
            json.dumps({"commit": expected_sha, "fingerprint": fingerprint}, indent=2) + "\n"
        )
        previous = work / "upstream.previous"
        if previous.exists():
            shutil.rmtree(previous)
        if current.exists():
            os.replace(current, previous)
        os.replace(next_dir, current)
        shutil.rmtree(previous, ignore_errors=True)
    except Exception:
        shutil.rmtree(next_dir, ignore_errors=True)
        raise
    return current


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("method_root", type=Path)
    parser.add_argument("expected_sha")
    args = parser.parse_args(argv)
    print(prepare(args.method_root, args.expected_sha))
    return 0
```

- [ ] **Step 5: Add patching, atomicity, and idempotency tests**

Append these methods to `PrepareUpstreamTests`:

```python
    def write_patch(self, body: str) -> None:
        patches = self.method / "patches"
        patches.mkdir()
        (patches / "series").write_text("0001-change.patch\n")
        (patches / "0001-change.patch").write_text(body)

    def test_patch_is_applied_and_second_run_is_idempotent(self) -> None:
        sha = self.make_upstream()
        self.write_patch(
            "diff --git a/value.txt b/value.txt\n"
            "--- a/value.txt\n+++ b/value.txt\n"
            "@@ -1 +1 @@\n-upstream\n+patched\n"
        )
        first = prepare(self.method, sha)
        first_stat = first.stat().st_mtime_ns
        self.assertEqual((first / "value.txt").read_text(), "patched\n")
        second = prepare(self.method, sha)
        self.assertEqual(second, first)
        self.assertEqual(second.stat().st_mtime_ns, first_stat)

    def test_bad_patch_keeps_previous_materialization(self) -> None:
        sha = self.make_upstream()
        current = prepare(self.method, sha)
        (current / "sentinel.txt").write_text("keep\n")
        self.write_patch("not a patch\n")
        with self.assertRaises(subprocess.CalledProcessError):
            prepare(self.method, sha)
        self.assertEqual((current / "sentinel.txt").read_text(), "keep\n")
```

- [ ] **Step 6: Run the shared tests**

Run: `python -m unittest tests.test_upstream -v`

Expected: 4 tests PASS.

- [ ] **Step 7: Commit the common contract**

```bash
git add .gitignore tools/upstream.py tests/test_upstream.py
git commit -m "build: add atomic upstream patch materializer"
```

### Task 2: Capture a Recoverable Migration Snapshot

**Files:**
- Create, ignored: `.migration-backup/root-working-tree.patch`
- Create, ignored: `.migration-backup/FAN-working-tree.patch`
- Create, ignored: `.migration-backup/Premier-working-tree.patch`
- Create, ignored: `.migration-backup/Tailored-Visions-working-tree.patch`
- Create, ignored: `.migration-backup/Tailored-Visions-local-commits/`
- Create: `docs/migrations/2026-09-19-layout-migration-manifest.md`

**Interfaces:**
- Consumes: current parent and nested Git states before any directory move.
- Produces: a human-readable manifest of every captured HEAD, remote, status, patch checksum, and destination classification.

- [ ] **Step 1: Capture the parent working tree without changing the index**

Run:

```bash
mkdir -p .migration-backup/Tailored-Visions-local-commits
git diff --binary --output=.migration-backup/root-working-tree.patch
```

Expected: `root-working-tree.patch` contains the current FAN and Tailored Visions edits, including the deleted `FAN/image/00000.png` marker. Record `git status --short` directly in the tracked manifest rather than creating another generated status file.

- [ ] **Step 2: Capture each nested working tree and local-only history**

Run:

```bash
git -C FAN diff --binary --output=../.migration-backup/FAN-working-tree.patch
git -C premier-repro/official/Premier diff --binary --output=../../../.migration-backup/Premier-working-tree.patch
git -C tailored-visions-repro/official diff --binary --output=../../.migration-backup/Tailored-Visions-working-tree.patch
git -C tailored-visions-repro/official format-patch d0f4454..HEAD -o ../../.migration-backup/Tailored-Visions-local-commits
```

Expected: four Tailored Visions commit patches are written; none of the source working trees changes.

- [ ] **Step 3: Write the migration manifest**

Create `docs/migrations/2026-09-19-layout-migration-manifest.md` with this table populated from the captured commands:

```markdown
# Layout migration manifest

| Repository | Base/HEAD | Dirty state | Final classification |
|---|---|---|---|
| Parent | `9a77749` plus listed working changes | See backup checksum | Preserve in moved README/source/patches |
| FAN | `9d0b76843f6437718195accac9cf3f050a25d26b` | `inference.py`, deleted sample, local integration files | CLI/docs outside upstream; source delta as patch only if still required |
| Premier | `42473476a189b6b0127890a98e92b7edf49c0d59` | single-block checkpoint fix | `patches/0001-fix-single-block-checkpoint-kwarg.patch` |
| Tailored Visions | upstream `d0f4454ca08c68c5d30f08a01ff4a23a8b33b610`; local HEAD `f53e7481f60954e30e37275791034e4201bbc6bd` plus working changes | local commits and five modified files | numbered patches plus outer wrappers/docs |
```

Add the output of `sha256sum .migration-backup/*.patch` and list the existing parent dirty paths exactly as reported by `git status --short` before migration.

- [ ] **Step 4: Verify every backup patch is readable**

Run:

```bash
git apply --numstat .migration-backup/root-working-tree.patch
git -C FAN apply --numstat ../.migration-backup/FAN-working-tree.patch
git -C premier-repro/official/Premier apply --numstat ../../../.migration-backup/Premier-working-tree.patch
git -C tailored-visions-repro/official apply --numstat ../../.migration-backup/Tailored-Visions-working-tree.patch
```

Expected: each command prints the files represented by its patch and exits 0.

- [ ] **Step 5: Commit only the tracked manifest**

```bash
git add docs/migrations/2026-09-19-layout-migration-manifest.md
git commit -m "docs: record repository migration inputs"
```

### Task 3: Migrate Premier as the Reference Layout

**Files:**
- Create: `.gitmodules`
- Create: `premier-repro/upstream` as submodule
- Create: `premier-repro/patches/series`
- Move: `premier-repro/official/patches/0001-fix-single-block-checkpoint-kwarg.patch` → `premier-repro/patches/0001-fix-single-block-checkpoint-kwarg.patch`
- Move: `premier-repro/premier/` → `premier-repro/src/premier_repro/`
- Move: `premier-repro/official/premier_local.py` → `premier-repro/src/premier_repro/official.py`
- Move: `premier-repro/official/run_premier.py` → `premier-repro/scripts/run_official.py`
- Move: `premier-repro/official/train_new_user.py` → `premier-repro/scripts/train_official_user.py`
- Move: `premier-repro/official/README_ja.md` → `premier-repro/docs/OFFICIAL_SETUP.md`
- Move local directories: `premier-repro/official/weights/` → `premier-repro/artifacts/weights/`, `premier-repro/official/test_data/` → `premier-repro/data/examples/`, `premier-repro/official/outputs/` → `premier-repro/outputs/official/`
- Create: `premier-repro/scripts/prepare_upstream.py`
- Create: `premier-repro/tests/test_layout.py`
- Modify: `premier-repro/pyproject.toml`
- Modify: all `premier-repro/scripts/*.py` imports from `premier` to `premier_repro`
- Modify: `premier-repro/README.md`

**Interfaces:**
- Consumes: `tools.upstream.prepare()` from Task 1 and the Premier official repository at the pinned SHA.
- Produces: importable package `premier_repro`; `scripts/prepare_upstream.py`; official wrappers that import from `.work/upstream` through `premier_repro.official`.

- [ ] **Step 1: Write the layout and import tests before moving files**

Create `premier-repro/tests/test_layout.py`:

```python
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PremierLayoutTests(unittest.TestCase):
    def test_common_paths_exist(self) -> None:
        for relative in ["upstream", "patches/series", "src/premier_repro", "docs/summary.html"]:
            self.assertTrue((ROOT / relative).exists(), relative)

    def test_upstream_is_clean(self) -> None:
        status = subprocess.run(
            ["git", "-C", str(ROOT / "upstream"), "status", "--short"],
            check=True, text=True, stdout=subprocess.PIPE,
        ).stdout
        self.assertEqual(status, "")

    def test_package_imports(self) -> None:
        sys.path.insert(0, str(ROOT / "src"))
        import premier_repro
        self.assertIsNotNone(premier_repro)
```

Run: `python -m unittest discover -s premier-repro/tests -v`

Expected: FAIL because the common paths and package do not yet exist.

- [ ] **Step 2: Move local code, documentation, fixtures, artifacts, and outputs**

Use `git mv` for tracked paths and ordinary `mv` for ignored large paths. Preserve file content exactly during this step. Do not change imports yet.

After moving, run:

```bash
git diff --summary -- premier-repro
du -sh premier-repro/artifacts/weights premier-repro/data/examples premier-repro/outputs/official
```

Expected: Git reports renames for tracked files; the weights, fixture set, and output sizes match their pre-migration sizes.

- [ ] **Step 3: Replace the embedded official repository with the pinned submodule**

Remove only the parent Git tracking for `premier-repro/official/Premier`, preserve its nested repository in `.migration-backup/Premier`, and add:

```bash
git submodule add https://github.com/120L020904/Premier.git premier-repro/upstream
git -C premier-repro/upstream checkout --detach 42473476a189b6b0127890a98e92b7edf49c0d59
```

Expected: `.gitmodules` contains the HTTPS URL and `git diff --cached --summary` reports a mode `160000` entry for `premier-repro/upstream`.

- [ ] **Step 4: Add the patch series and preparation shim**

Create `premier-repro/patches/series`:

```text
0001-fix-single-block-checkpoint-kwarg.patch
```

Create `premier-repro/scripts/prepare_upstream.py`:

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from tools.upstream import prepare

EXPECTED_SHA = "42473476a189b6b0127890a98e92b7edf49c0d59"

if __name__ == "__main__":
    print(prepare(ROOT, EXPECTED_SHA))
```

Run: `python premier-repro/scripts/prepare_upstream.py`

Expected: prints `premier-repro/.work/upstream`; the patched `scripts/pipeline/flux_adapter.py` omits the invalid single-block `use_img_mod` argument.

- [ ] **Step 5: Update the local package name and build configuration**

In `premier-repro/pyproject.toml`, set:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/premier_repro"]
```

Replace imports and local `sys.path` setup in `premier-repro/scripts/*.py` so they add `ROOT / "src"` and import `premier_repro.*` rather than `premier.*`. Change internal package imports if any absolute `premier.*` imports exist.

Run: `rg -n "from premier\b|import premier\b|packages = \[\"premier\"\]" premier-repro -g '*.py' -g '*.toml'`

Expected: no matches.

- [ ] **Step 6: Adapt the official integration wrapper**

In `src/premier_repro/official.py`, define and use:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATCHED_UPSTREAM = PROJECT_ROOT / ".work/upstream"
ARTIFACTS = PROJECT_ROOT / "artifacts/weights/pino10010_Premier"


def require_patched_upstream() -> Path:
    if not (PATCHED_UPSTREAM / "scripts").is_dir():
        raise RuntimeError("run `python scripts/prepare_upstream.py` first")
    path = str(PATCHED_UPSTREAM)
    if path not in sys.path:
        sys.path.insert(0, path)
    return PATCHED_UPSTREAM
```

Update `scripts/run_official.py` and `scripts/train_official_user.py` to add `ROOT / "src"`, import from `premier_repro.official`, call `require_patched_upstream()` before official imports, use `ARTIFACTS`, and default to `outputs/official/...`.

- [ ] **Step 7: Normalize Premier documentation**

Update `premier-repro/README.md` to the ten-section order in the spec. Point official setup instructions to `docs/OFFICIAL_SETUP.md`, official source to `upstream/`, patches to `patches/`, weights to `artifacts/`, examples to `data/examples/`, and results to `outputs/`.

Move `premier-repro/docs/premier_summary.html` to `premier-repro/docs/summary.html` and update root links.

- [ ] **Step 8: Run Premier checks**

Run:

```bash
python -m unittest discover -s premier-repro/tests -v
python premier-repro/scripts/prepare_upstream.py
premier-repro/.venv/bin/python -m compileall -q premier-repro/src premier-repro/scripts
premier-repro/.venv/bin/python premier-repro/scripts/run_official.py --help
git -C premier-repro/upstream status --short
```

Expected: tests pass, compilation succeeds, help exits 0 without loading model weights, and submodule status is empty.

- [ ] **Step 9: Commit Premier migration**

Stage only `.gitmodules`, the Premier paths, root link changes, and the necessary `.gitignore` state. Verify `git diff --cached --stat` contains no FAN or Tailored Visions content.

```bash
git commit -m "refactor: normalize Premier and track upstream"
```

### Task 4: Migrate FAN Without Editing Upstream

**Files:**
- Rename: `FAN/` → `fan-repro/`
- Create: `fan-repro/upstream` as submodule
- Create: `fan-repro/patches/series`
- Move: `FAN/README_ja.md` → `fan-repro/README.md`
- Move: `FAN/FAN_summary.html` → `fan-repro/docs/summary.html`
- Move: `FAN/smoke_clip.py` → `fan-repro/scripts/smoke_clip.py`
- Create: `fan-repro/scripts/generate.py`
- Create: `fan-repro/scripts/prepare_upstream.py`
- Create: `fan-repro/tests/test_layout.py`
- Modify: `fan-repro/pyproject.toml`
- Modify: root `README.md`

**Interfaces:**
- Consumes: clean FAN submodule at the pinned SHA and `tools.upstream.prepare()`.
- Produces: a no-patch materialization unless the local `inference.py` behavior cannot be represented by the outer generation CLI; `scripts/generate.py` imports `fan` from `.work/upstream`.

- [ ] **Step 1: Write failing FAN layout tests**

Create the equivalent of the Premier layout test at `fan-repro/tests/test_layout.py`, asserting these paths: `upstream`, `patches/series`, `scripts/generate.py`, `scripts/smoke_clip.py`, and `docs/summary.html`. Also assert a clean upstream and that `scripts/generate.py --help` exits 0.

Run: `python -m unittest discover -s fan-repro/tests -v`

Expected: FAIL because `fan-repro/` does not exist yet.

- [ ] **Step 2: Separate FAN local files from official files**

Create `fan-repro/` and move the tracked local integration files to their target paths. Preserve the current modified `README_ja.md` content as `README.md`. Preserve the deletion of `image/00000.png` by not migrating the generated `image/` directory.

Move the existing `.venv/` to `fan-repro/.venv/` without copying it. Move any remaining generated `image/` content to `fan-repro/outputs/`.

- [ ] **Step 3: Add the FAN submodule**

Preserve the original nested repository under `.migration-backup/FAN`, remove the parent-tracked official snapshot, and run:

```bash
git submodule add https://github.com/Burf/FAN.git fan-repro/upstream
git -C fan-repro/upstream checkout --detach 9d0b76843f6437718195accac9cf3f050a25d26b
```

Create an empty `fan-repro/patches/series` containing only this comment:

```text
# FAN is currently integrated without source patches.
```

- [ ] **Step 4: Implement the FAN preparation shim and outer generation CLI**

Create `fan-repro/scripts/prepare_upstream.py` with the same shim as Premier and FAN's pinned SHA.

Create `fan-repro/scripts/generate.py` by retaining the current local CLI arguments `--variant`, `--dtype`, and `--ref_weight_dir`, then:

```python
ROOT = Path(__file__).resolve().parents[1]
PATCHED_UPSTREAM = ROOT / ".work/upstream"
if not (PATCHED_UPSTREAM / "fan").is_dir():
    raise SystemExit("run `python scripts/prepare_upstream.py` first")
sys.path.insert(0, str(PATCHED_UPSTREAM))
```

Import `FAN` and `personalized_t2i_encoder` from the materialized official package. Resolve official decoder weights relative to `PATCHED_UPSTREAM / "weight"`; default image output under `ROOT / "outputs"`. Do not modify `upstream/inference.py`.

- [ ] **Step 5: Update the FAN smoke script and environment metadata**

Update `scripts/smoke_clip.py` to import from `.work/upstream` and resolve `L.pth` under `.work/upstream/weight/`.

In `pyproject.toml`, change the project name from `fan-official` to `fan-repro` and keep `[tool.uv] package = false` because the local project exposes scripts rather than a package.

- [ ] **Step 6: Normalize FAN documentation**

Use the standard README order. State the official URL and pinned SHA, no-patch status, preparation command, output directory, and distinction between official source and local wrappers. Update all root README links from `FAN/` and `FAN/FAN_summary.html` to `fan-repro/` and `fan-repro/docs/summary.html`.

- [ ] **Step 7: Run FAN checks**

Run:

```bash
python fan-repro/scripts/prepare_upstream.py
python -m unittest discover -s fan-repro/tests -v
fan-repro/.venv/bin/python -m py_compile fan-repro/scripts/generate.py fan-repro/scripts/smoke_clip.py
fan-repro/.venv/bin/python fan-repro/scripts/generate.py --help
fan-repro/.venv/bin/python fan-repro/scripts/smoke_clip.py
git -C fan-repro/upstream status --short
```

Expected: layout tests, compilation, CLI help, and existing CLIP smoke test pass; upstream status is empty.

- [ ] **Step 8: Commit FAN migration**

Stage only FAN/fan-repro paths, `.gitmodules`, and root link changes. Confirm the cached diff does not include Tailored Visions.

```bash
git commit -m "refactor: normalize FAN and track upstream"
```

### Task 5: Extract and Verify Tailored Visions Patch Series

**Files:**
- Create: `tailored-visions-repro/patches/series`
- Create: `tailored-visions-repro/patches/0001-fix-official-runtime-bugs.patch`
- Create: `tailored-visions-repro/patches/0002-support-current-diffusers-and-sdxl.patch`
- Create: `tailored-visions-repro/patches/0003-add-bounded-experiment-cli.patch`
- Create: `tailored-visions-repro/tests/test_patches.py`

**Interfaces:**
- Consumes: upstream base `d0f4454`, the local commits through `f53e748`, the current nested working diff, and the Task 2 backups.
- Produces: three non-overlapping patches which reproduce only changes that must live in the official working copy; wrappers and local documentation are excluded.

- [ ] **Step 1: Write a patch-series reproduction test**

Create `tailored-visions-repro/tests/test_patches.py`:

```python
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "d0f4454ca08c68c5d30f08a01ff4a23a8b33b610"


class TailoredPatchTests(unittest.TestCase):
    def test_series_applies_to_expected_upstream(self) -> None:
        head = subprocess.run(
            ["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"],
            check=True, text=True, stdout=subprocess.PIPE,
        ).stdout.strip()
        self.assertEqual(head, EXPECTED)
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["git", "clone", "--quiet", "--no-hardlinks", str(ROOT / "upstream"), tmp],
                check=True,
            )
            subprocess.run(["git", "-C", tmp, "checkout", "--detach", EXPECTED], check=True)
            names = [
                line.strip() for line in (ROOT / "patches/series").read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
            for name in names:
                patch = ROOT / "patches" / name
                subprocess.run(["git", "-C", tmp, "apply", "--check", str(patch)], check=True)
                subprocess.run(["git", "-C", tmp, "apply", str(patch)], check=True)
```

Run: `python -m unittest discover -s tailored-visions-repro/tests -p 'test_patches.py' -v`

Expected: FAIL because `upstream/` and the patch series do not yet exist.

- [ ] **Step 2: Create a clean temporary comparison tree at `d0f4454`**

Clone the existing nested Tailored repository with `--no-hardlinks` into `.migration-backup/Tailored-Visions-base`, check out `d0f4454`, and keep the current `official/` tree untouched. Use the Task 2 backups as the recovery source throughout this task.

- [ ] **Step 3: Build patch 0001 from official runtime fixes**

Include only these changes relative to `d0f4454`:

- `apiuse.py`: environment-based credentials/endpoint and bounded retry
- `download.py`: timeout callback fix
- `language.py`: BM25 tokenization and stable scoring fixes
- `prompts.py`: cached CLIP model for example ranking
- `main.py`: missing `MY_BASE`, valid retrieval branch, `None` handling, per-query ICL construction, and safe output directory creation
- `demo.py`: accepted input option and disable-T2I option

Exclude SDXL/model-loader changes, `--data_folder`, `--limit_users`, wrapper scripts, local server, setup prose, and dependency files from this patch. Apply it to the temporary base and verify `git diff --check` is clean.

- [ ] **Step 4: Build patch 0002 from current dependency and model support**

Against the result of patch 0001, include:

- `SD.py`: `AutoPipelineForText2Image`, SDXL default, fp16 fallback, VAE slicing, memory policy, and common `load_pipeline`
- `demo.py` and `main.py`: use `load_pipeline` and preserve CPU offload
- `requirements-2026.txt`: Python 3.12/current environment dependency set

Do not include `run.sh`, `serve_local_llm.py`, or `SETUP.md`. Apply it after patch 0001 and run `git diff --check`.

- [ ] **Step 5: Build patch 0003 from bounded experiment CLI changes**

Include only `main.py` changes for `--data_folder` and `--limit_users`, plus any direct plumbing needed for those two options. Apply it after patches 0001 and 0002.

- [ ] **Step 6: Define series order and verify reconstruction**

Create `patches/series`:

```text
0001-fix-official-runtime-bugs.patch
0002-support-current-diffusers-and-sdxl.patch
0003-add-bounded-experiment-cli.patch
```

Compare the patched official files with the current nested working tree:

```bash
diff -ru --exclude=.git --exclude=SETUP.md --exclude=run.sh --exclude=serve_local_llm.py --exclude=.gitignore \
  .migration-backup/Tailored-Visions-patched tailored-visions-repro/official
```

Expected: no differences for files represented by the patch series. Differences are limited to excluded outer integration files.

- [ ] **Step 7: Commit the reviewed patch series**

```bash
git add tailored-visions-repro/patches tailored-visions-repro/tests/test_patches.py
git commit -m "build: preserve Tailored Visions compatibility patches"
```

### Task 6: Normalize Tailored Visions Local Package and Outputs

**Files:**
- Move: `tailored-visions-repro/tv/` → `tailored-visions-repro/src/tailored_visions_repro/`
- Modify: `tailored-visions-repro/pyproject.toml`
- Modify: `tailored-visions-repro/scripts/*.py`
- Modify: `tailored-visions-repro/run_all.sh`
- Move: `tailored-visions-repro/results/` → `tailored-visions-repro/outputs/`
- Move: `tailored-visions-repro/data/user_data/` → `tailored-visions-repro/data/raw/user_data/`
- Move: `tailored-visions-repro/docs/tailored-visions.html` → `tailored-visions-repro/docs/summary.html`
- Create: `tailored-visions-repro/tests/test_layout.py`

**Interfaces:**
- Consumes: the existing local `tv` package, including current uncommitted changes in `tv/generate.py` and `scripts/05_generate.py`.
- Produces: importable `tailored_visions_repro` package and output defaults rooted at `outputs/`.

- [ ] **Step 1: Write failing package and default-path tests**

Create `tests/test_layout.py` that adds `ROOT / "src"`, imports `tailored_visions_repro`, and inspects each script plus `run_all.sh` to assert that the literals `from tv`, `import tv`, and `results/` no longer occur outside historical documentation.

Run: `python -m unittest discover -s tailored-visions-repro/tests -p 'test_layout.py' -v`

Expected: FAIL on the old package and output paths.

- [ ] **Step 2: Move the package while preserving the current working changes**

Use `git mv tailored-visions-repro/tv tailored-visions-repro/src/tailored_visions_repro`. Confirm that the content hash of the modified `generate.py` before and after the move is identical to the checksum recorded in the migration manifest.

In `pyproject.toml`, set:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/tailored_visions_repro"]
```

- [ ] **Step 3: Update imports and local source paths**

For every `scripts/*.py`, change its source path insertion to `ROOT / "src"` and imports from `tv.*` to `tailored_visions_repro.*`. Update internal absolute imports the same way.

Run:

```bash
rg -n "from tv\b|import tv\b|parents\[1\]\)\)" tailored-visions-repro/src tailored-visions-repro/scripts -g '*.py'
```

Expected: no obsolete imports or root-only source path insertions.

- [ ] **Step 4: Normalize runtime data and output defaults**

Move the existing directories without copying. Change all active CLI defaults and `run_all.sh` references from `results/` to `outputs/`. Change the default user dataset path from `data/user_data` to `data/raw/user_data`. Retain historical result path mentions only where explicitly labeled as historical in documentation.

Run:

```bash
rg -n "results/|data/user_data" tailored-visions-repro/scripts tailored-visions-repro/src tailored-visions-repro/run_all.sh
```

Expected: no matches.

- [ ] **Step 5: Run local package checks**

Run:

```bash
python -m unittest discover -s tailored-visions-repro/tests -p 'test_layout.py' -v
tailored-visions-repro/.venv/bin/python -m compileall -q tailored-visions-repro/src tailored-visions-repro/scripts
tailored-visions-repro/.venv/bin/python tailored-visions-repro/scripts/05_generate.py --help
tailored-visions-repro/.venv/bin/python tailored-visions-repro/scripts/06_evaluate.py --help
```

Expected: tests, compilation, and both help commands pass without loading models.

- [ ] **Step 6: Commit local package normalization**

```bash
git add tailored-visions-repro/src tailored-visions-repro/scripts tailored-visions-repro/run_all.sh \
  tailored-visions-repro/pyproject.toml tailored-visions-repro/docs/summary.html tailored-visions-repro/tests
git commit -m "refactor: normalize Tailored Visions local pipeline"
```

Do not stage ignored `data/raw/` or `outputs/` content.

### Task 7: Add Tailored Visions Submodule and Outer Integration

**Files:**
- Create: `tailored-visions-repro/upstream` as submodule
- Move: `tailored-visions-repro/official/serve_local_llm.py` → `tailored-visions-repro/scripts/serve_local_llm.py`
- Move/adapt: `tailored-visions-repro/official/run.sh` → `tailored-visions-repro/scripts/run_official.sh`
- Move: `tailored-visions-repro/official/SETUP.md` → `tailored-visions-repro/docs/OFFICIAL_SETUP.md`
- Create: `tailored-visions-repro/scripts/prepare_upstream.py`
- Create: `tailored-visions-repro/scripts/official_paths.py`
- Modify: `tailored-visions-repro/README.md`
- Modify: `tailored-visions-repro/docs/DEVIATIONS.md`

**Interfaces:**
- Consumes: Tailored patch series from Task 5, normalized local package from Task 6, and shared materializer from Task 1.
- Produces: clean pinned submodule, patched `.work/upstream`, and outer scripts for the local LLM and official demo/main entry points.

- [ ] **Step 1: Preserve outer integration files and replace the official snapshot**

Move `serve_local_llm.py`, `run.sh`, and `SETUP.md` to their outer destinations with their current uncommitted content preserved. Move any official generated images to `outputs/official/` and the generated `demo_user.jsonl` to `data/examples/` only if it is needed as a stable fixture.

Preserve the original nested repository under `.migration-backup/Tailored-Visions`, remove the parent-tracked official snapshot, and add:

```bash
git submodule add https://github.com/zzjchen/Tailored-Visions.git tailored-visions-repro/upstream
git -C tailored-visions-repro/upstream checkout --detach d0f4454ca08c68c5d30f08a01ff4a23a8b33b610
```

- [ ] **Step 2: Add preparation and path helpers**

Create `scripts/prepare_upstream.py` using Tailored Visions' pinned SHA.

Create `scripts/official_paths.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHED_UPSTREAM = ROOT / ".work/upstream"
DATA = ROOT / "data/raw/user_data"
OUTPUTS = ROOT / "outputs/official"


def require_upstream() -> Path:
    if not (PATCHED_UPSTREAM / "main.py").is_file():
        raise RuntimeError("run `python scripts/prepare_upstream.py` first")
    return PATCHED_UPSTREAM
```

- [ ] **Step 3: Adapt the official runner without editing materialized source**

Update `scripts/run_official.sh` so it:

- resolves the repository root from its own path;
- uses `ROOT/.venv/bin/python`;
- starts `scripts/serve_local_llm.py` from the outer tree;
- sets `PYTHONPATH` to `ROOT/.work/upstream`;
- passes `--data_folder ROOT/data/raw/user_data` and writes under `ROOT/outputs/official`;
- invokes `.work/upstream/demo.py` or `.work/upstream/main.py`;
- keeps the existing `demo`, `main`, `serve`, and `stop` commands and CUDA allocator setting.

Run: `bash tailored-visions-repro/scripts/run_official.sh --help`

Expected: usage text exits 2 without starting the local model server.

- [ ] **Step 4: Rewrite official setup and deviations documents**

In `docs/OFFICIAL_SETUP.md`, replace direct `official/` execution with `scripts/prepare_upstream.py` and `scripts/run_official.sh`. State the pinned SHA and explain `.work/upstream`.

In `docs/DEVIATIONS.md`, add a patch table with the exact three patch filenames, affected files, reasons, and validation commands. Preserve all current research findings and the current uncommitted additions.

Normalize `README.md` to the ten-section standard and update paths to `upstream/`, `patches/`, `src/tailored_visions_repro/`, `data/raw/`, and `outputs/`.

- [ ] **Step 5: Run Tailored integration checks**

Run:

```bash
python tailored-visions-repro/scripts/prepare_upstream.py
python -m unittest discover -s tailored-visions-repro/tests -v
tailored-visions-repro/.venv/bin/python -m py_compile tailored-visions-repro/.work/upstream/*.py
tailored-visions-repro/.venv/bin/python tailored-visions-repro/.work/upstream/demo.py --help
tailored-visions-repro/.venv/bin/python tailored-visions-repro/.work/upstream/main.py --help
git -C tailored-visions-repro/upstream status --short
```

Expected: patch and layout tests pass, official files compile, both help commands exit 0, and upstream is clean.

- [ ] **Step 6: Commit Tailored submodule integration**

```bash
git add .gitmodules tailored-visions-repro/upstream tailored-visions-repro/scripts \
  tailored-visions-repro/docs tailored-visions-repro/README.md
git commit -m "refactor: track Tailored Visions upstream separately"
```

### Task 8: Repository-Wide Documentation and Structural Verification

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Create: `tests/test_repository_layout.py`
- Modify: `docs/migrations/2026-09-19-layout-migration-manifest.md`

**Interfaces:**
- Consumes: all three normalized method layouts.
- Produces: one root entry point and automated checks for the cross-method contract.

- [ ] **Step 1: Write the repository contract test**

Create `tests/test_repository_layout.py`:

```python
import configparser
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "fan-repro": "https://github.com/Burf/FAN.git",
    "premier-repro": "https://github.com/120L020904/Premier.git",
    "tailored-visions-repro": "https://github.com/zzjchen/Tailored-Visions.git",
}


class RepositoryLayoutTests(unittest.TestCase):
    def test_method_contract(self) -> None:
        for method in METHODS:
            base = ROOT / method
            for relative in ["README.md", "pyproject.toml", "uv.lock", "upstream", "patches/series", "scripts", "docs/summary.html"]:
                self.assertTrue((base / relative).exists(), f"{method}/{relative}")

    def test_submodule_urls_are_https(self) -> None:
        parser = configparser.ConfigParser()
        parser.read(ROOT / ".gitmodules")
        urls = {section.split('"')[1]: parser[section]["url"] for section in parser.sections()}
        for path, url in METHODS.items():
            self.assertEqual(urls[f"{path}/upstream"], url)

    def test_submodules_are_clean(self) -> None:
        for method in METHODS:
            status = subprocess.run(
                ["git", "-C", str(ROOT / method / "upstream"), "status", "--short"],
                check=True, text=True, stdout=subprocess.PIPE,
            ).stdout
            self.assertEqual(status, "", method)
```

Run: `python -m unittest tests.test_repository_layout -v`

Expected: PASS after Tasks 3–7.

- [ ] **Step 2: Rewrite the root README as the common entry point**

Keep the scientific comparison, but update the table and links to the three normalized directories. Add this clone/setup block:

```bash
git clone --recurse-submodules https://github.com/tmp-friends/personalized-t2i.git
cd personalized-t2i
# Existing clone only:
git submodule update --init --recursive
```

Document the shared directory meanings and link to each method's Quick start rather than duplicating it. Update the exhibition design link without changing that design.

- [ ] **Step 3: Audit obsolete paths and tracked artifacts**

Run:

```bash
rg -n "\bFAN/|official/Premier|tailored-visions-repro/official|FAN_summary\.html|premier_summary\.html|tailored-visions\.html|image_result/|results/" \
  README.md fan-repro premier-repro tailored-visions-repro \
  -g '*.md' -g '*.py' -g '*.sh' -g '*.toml' -g '*.yaml'
git ls-files | rg '(^|/)(\.venv|\.work|artifacts|outputs|data/raw)/'
```

Expected: the first command reports only explicitly labeled historical paths or patch content; the second reports no files.

- [ ] **Step 4: Run all lightweight checks**

Run:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s fan-repro/tests -v
python -m unittest discover -s premier-repro/tests -v
python -m unittest discover -s tailored-visions-repro/tests -v
python fan-repro/scripts/prepare_upstream.py
python premier-repro/scripts/prepare_upstream.py
python tailored-visions-repro/scripts/prepare_upstream.py
git submodule status --recursive
git submodule foreach --recursive git status --short
git diff --check
```

Expected: all tests pass, all three preparation commands are idempotent, three pinned submodules are listed, every submodule status is empty, and `git diff --check` prints nothing.

- [ ] **Step 5: Complete the migration manifest**

Append a final mapping table from every old path to its new path, the final submodule SHAs, the commit IDs produced by Tasks 1–8, and verification outputs. Mark `.migration-backup/` as retained pending explicit human approval; do not remove it.

- [ ] **Step 6: Commit repository-wide documentation and tests**

```bash
git add README.md .gitignore tests/test_repository_layout.py docs/migrations/2026-09-19-layout-migration-manifest.md
git commit -m "docs: document normalized reproduction layout"
```

### Task 9: Fresh-Clone Acceptance Test

**Files:**
- Modify only if verification reveals a defect: files owned by the failing prior task
- Retain, ignored: `.migration-backup/`

**Interfaces:**
- Consumes: the complete committed repository.
- Produces: evidence that a new consumer can obtain official sources and prepare all patched trees without local nested-repository state.

- [ ] **Step 1: Create a temporary clone with submodules**

Run from the repository parent, using an explicit temporary path:

```bash
tmpdir=$(mktemp -d /tmp/personalized-t2i-verify.XXXXXX)
git clone --recurse-submodules /home/tomoya/stable-diffusion/personalized-t2i "$tmpdir/repo"
```

Expected: clone succeeds and all three submodules are checked out. Because `.gitmodules` uses HTTPS, run `git -C "$tmpdir/repo" submodule sync --recursive` followed by an online `git submodule update --init --recursive` when validating remote accessibility rather than only local object reuse.

- [ ] **Step 2: Run structural and preparation checks in the clone**

Run:

```bash
python -m unittest discover -s "$tmpdir/repo/tests" -v
python "$tmpdir/repo/fan-repro/scripts/prepare_upstream.py"
python "$tmpdir/repo/premier-repro/scripts/prepare_upstream.py"
python "$tmpdir/repo/tailored-visions-repro/scripts/prepare_upstream.py"
git -C "$tmpdir/repo" submodule foreach --recursive git status --short
```

Expected: tests and preparation pass; submodules remain clean.

- [ ] **Step 3: Verify method-local dependency setup without GPU execution**

Run `uv sync --locked` in each method directory. Then run each method's CLI `--help` and Python compilation checks documented in Tasks 3, 4, and 7. Reuse global package caches; do not copy `.venv` from the original checkout.

Expected: dependency sync, compilation, and help commands succeed without model downloads or GPU allocation.

- [ ] **Step 4: Run available lightweight behavioral smoke tests**

Run FAN's CLIP smoke test if its Hugging Face assets are cached. Run Premier's configuration/import smoke test and Tailored Visions' no-T2I import/help test. Record any skipped GPU/model test with its exact missing artifact; do not report it as passing.

- [ ] **Step 5: Apply the verification-before-completion gate**

Invoke `superpowers:verification-before-completion`. Re-run the final commands it requires, inspect `git status --short`, and confirm that only the user's intentionally retained pre-existing changes or no changes remain.

- [ ] **Step 6: Request code review**

Invoke `superpowers:requesting-code-review` for the complete migration. Address all blocking findings and re-run the affected acceptance checks before reporting completion.

- [ ] **Step 7: Report the retained recovery backup**

Tell the human partner that `.migration-backup/` still contains recoverable pre-migration states and is ignored by Git. Ask separately before deleting it; deletion is not part of this plan.
