# ZIPP-style Exhibition Demo Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Track verification here. The user has authorized implementation and localhost verification.

**Goal:** Run the five-choice, editable-persona, matched-seed image-generation demonstration locally and deliver an evidence-based HTML report.

**Architecture:** A local FastAPI application serves a dependency-free Japanese browser UI. Pure Python context and cache contracts are separated from a single-session coordinator. GPU stages run in isolated subprocesses under one filesystem lease, with offline pinned model loading, cancellation and bounded execution. A separate PIGReward adapter parses real model output conservatively; adoption requires measured G0 evidence.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, vanilla JS/CSS, pytest, Playwright; existing local PyTorch, Transformers, Diffusers GPU environment.

**Spec:** ../specs/2026-09-19-zipp-pigreward-exhibition-demo-design.md

## Global Constraints

- Five fixed pairs, random left/right order, skip is not a preference; require three valid choices.
- Six topics, four matched seeds, SDXL base fp16, initial 1024 square / 20 steps / CFG 7 / no negative prompt.
- A single EffectivePreferenceContext hash is shared by generation and recommendation; remove entire multi-axis evidence fragments when one axis is disabled or corrected.
- All-off means generic cache only. Stale job/session/context results never reach a new session.
- One GPU stage at a time, process exit before lease transfer, 120-second live deadline, 90-second idle reset outside jobs.
- Retain completed images when a later stage fails. Never invent recommendations, scores, evidence, live status or latency results.
- Keep original user changes. Work on branch codex/zipp-exhibition-demo; existing model environments remain unmodified.
- Model integration is independent research code; no fictitious official upstream. Raw session data are temporary.

## Review Focus

1. Reset during GPU inference, old browser requests and stale artifact access: tests in task 3.
2. Corrections to multi-axis evidence, all skipped and all-off: tests in task 1.
3. PIGReward partial output, mismatched totals, ties and opposite image order: tests in task 2.
4. Cache invalidation on model/settings/context, corrupt files and offline missing assets: tests in tasks 1 and 4.
5. Browser reload, duplicate generation and second session competing for GPU: tests in task 3 and real browser in task 5.

## Task 1: Context, assets, prompt and cache contracts

Files: exhibit/src/exhibit/{domain.py,config.py}, exhibit/configs/demo.json, exhibit/tests/test_domain.py.
Interfaces: build_persona(choices, evidence); effective_context(persona, edits); digest(value); cache_key(prompt, settings, seed, context=None); validate_prompt(prompt, topic, tokenizers).

- [x] Write tests that off/corrected dimensions remove old fragments, skips never contribute, conflict remains unknown, hashes change with edits/settings.
- [x] Run `uv run --project exhibit pytest exhibit/tests/test_domain.py` and observe the missing contracts fail.
- [x] Implement immutable JSON contracts and controlled aesthetic values. Explicit sources and revisions travel with assets and evidence. Preserve the full basic prompt as the immutable prompt prefix; tokenizer bounds are measured by both SDXL tokenizers.
- [x] Re-run the tests and record results below.

Example contract:
```python
context = effective_context(persona, {"lighting": None})
assert all("lighting" not in f["dimensions"] for f in context["evidence"])
assert context["hash"] != effective_context(persona, {})["hash"]
```

## Task 2: PIGReward adapter and sequential GPU workers

Files: pigreward-repro/src/pigreward_repro/{adapter.py,worker.py}, configs/model.json, docs/DEVIATIONS.md, tests/test_adapter.py; exhibit/src/exhibit/{workers.py,gpu.py}.
Interfaces: parse_judgment(raw, candidate_ids, finished); tournament(ids, compare, seed); run_stage(request, directory, cancel, deadline, on_event).

- [x] Write parser/tournament tests for valid card-format output, tie, score disagreement, truncation and invalid candidate IDs.
- [x] Observe failures; implement strict card-format parsing with per-match audit, no fabricated numeric fallback, at most three matches.
- [x] Implement pinned offline worker stages for local LLM rewrite, SDXL and evaluator, JSONL progress, process group termination and exclusive flock.
- [x] Run adapter tests and a real GPU smoke. Record observed load/generate/release wall time and peak allocated VRAM.

Example contract:
```python
assert parse_judgment("Image 1 is better", ["a", "b"], finished=False)["winner_id"] is None
```

## Task 3: Session coordinator, API and Japanese browser UI

Files: exhibit/src/exhibit/{service.py,app.py}, static/{index.html,app.js,style.css}; exhibit/tests/test_service.py.
Interfaces: create_session(); answer(session_id,pair_id,chosen_id); start_run(session_id,topic,edits,request_id); snapshot(session_id); reset(session_id).

- [x] Write tests for duplicate requests, insufficient choices, reset invalidation, artifact containment and single-session locking.
- [x] Observe failures; implement guarded state transitions and temporary session directories. Serialize all mutations, reject invalid enum edits and IDs.
- [x] Implement UI: five choices, retry skips, evidence thumbnails, per-axis off/correction, six topics, neutral/personalized 4-column comparison, progress, manual final choice, reset and idle timer.
- [x] Run all fast tests; exercise actual HTTP endpoints, cancellation and all-off.

## Task 4: Reproducible preparation and offline fallback

Files: exhibit/scripts/{prepare.py,preflight.py,rehearsal.py}, exhibit/assets/manifest.json, static fallback HTML, README files.
Interfaces: preparation writes immutable assets/evidence/cache records; preflight validates hashes, revisions and adoption mode.

- [x] Prepare actual SDXL fixed pairs and 24 generic candidates. Record source/model, prompts, hashes and matching settings.
- [x] Run generic VLM evidence preparation for all ten directed comparisons; inspect outputs and curate only supported fragments. Mark context_source=generic_vlm.
- [x] Prepare three representative histories across two topics (six experiences), recording actual rewrite and generation. Without G0 approval these remain ZIPP-style samples without recommendations.
- [x] Test offline preflight with missing/corrupt asset fixtures. Generate a standalone sample page that opens without the server.

## Task 5: Localhost rehearsal and report

Files: exhibit/scripts/browser_check.py, docs/reports/zipp-demo/{index.html,evidence.json,screenshots/}, README.md.

- [x] Run a real server on localhost:7860. Drive Chromium through selection → edit → topic → live result → final choice → reset, recording screenshots and console errors.
- [x] Verify all-off, skipped choices, reload, responsive layout, stale requests, exact-cache classification and offline asset serving.
- [x] Run relevant project and root repository tests. Obtain independent code review under executing-plans and fix important issues with regression tests.
- [x] Write HTML with actual screenshots, measured timings, commands, mode limitations and remaining G0/G2 work; distinguish verified functionality from exhibition-readiness gates.

## Execution record

- Initial inspection: no application existed; GPU idle, SDXL and Qwen3.5-4B cached. PIGReward not cached. User's design file already modified and preserved.
- Ruling: operate in the existing checkout on a new feature branch so the user can open the requested app and HTML at their workspace paths; changes are additive and the uncommitted spec is not staged.

- Task 1: complete — context/cache/tokenizer contracts, OFF/correction/skip tests. Additional raw-output exclusion regression failed before fix, passed after.
- Task 2: complete — GPU workers and strict PIG adapter. Actual SDXL/VLM/rewrite run successfully. Evaluator loads but G0 yields 0/20 accepted forward outputs, 0/20 valid reverse matches; recommendation disabled.
- Task 3: complete — API and Japanese UI. Real Chromium full flow passed. GPU processes terminate before lease release; cancellation and stale publication tests pass.
- Task 4: complete — 34 fixed images, 10 VLM directions, 6 samples (24 personalized images), checked hashes and standalone embedded-image fallback HTML.
- Review: independent read-only reviewer found four Important issues; all received regression coverage and fixes: old poll response / old 404 identity; reload after skips; sample seed/settings/rewriter contracts; parser ignored contradictory lines. Minor correction persistence concern addressed using per-session sessionStorage and browser reload assertions.
- Ruling: persona is the explicitly labeled checked-evidence summary; free-form online persona generation is not adopted in this reduced mode. Cost: less flexible persona language; no ungrounded inference.
- Ruling: rewriter preserves the original prompt verbatim and orders approved aesthetic phrases using Qwen. Cost: less expressive than unrestricted ZIPP-style rewriting, with stronger invariants.
- Ruling: PIGReward remains disabled because the actual custom adapter failed G0. Cost: the demo has no automatic recommendation; visitors choose manually as designed for fallback.
- Ruling: G2 third-party usability and two-hour endurance require separate exhibition acceptance; this delivery reports them as untested and does not claim full exhibition readiness.

- Task 5: complete — localhost Chromium full journey verified after fixes (20.9 s, no JS exceptions, no external requests), mobile 390px verified, offline file fallback loaded 48 embedded images. 20 real sequential sessions: 20/20 successful, median 20.652 s, p95 20.861 s (manual mode only). Rewrite sweep: 18/18 prefix/dual-tokenizer checks passed. Actual GPU reset after first image: released in 0.159 s, VRAM 1 MiB, old URL 404, next session accepted.
- Final verification: 38 Python tests including 9 existing repository tests, 4 Node browser-state tests; ruff clean; two upstream deprecation warnings retained in report. All reviewed Important issues fixed; no deferred review minors.
- Delivery: docs/reports/zipp-demo/index.html with actual screenshots and JSON evidence. Branch codex/zipp-exhibition-demo kept in the user's workspace; no merge or remote publication requested.

## Recovery verification — 2026-09-20

- Resumed from the completed implementation and preserved existing uncommitted work. No model/settings changes were needed for the successful replay.
- OOM cause remains unconfirmed: retained inference logs contain no OOM trace; kernel logs were unavailable. During the first replay the previous localhost server disappeared, and the browser timed out at final selection after four images had been generated. Its exit cause is unknown.
- Restarted localhost server and reran the complete browser journey successfully: 20.88 seconds, no JS exceptions or external requests, matched seeds/context, manual choice, reload, exact cache, all-off, skips, sample and reset all checked. Measured peak allocated VRAM: rewrite 8127.6 MiB, SDXL 10736.3 MiB; idle driver usage after completion: 1 MiB.
- Added two process-boundary integration tests (`test_worker_recovery.py`): child-only RLIMIT_AS produces MemoryError without exhausting host RAM; SIGKILL models abrupt worker loss. Actual subprocess/lease/coordinator paths retain the first image and allow selection, reap the worker, invalidate artifacts on reset, and complete four images in the next session. Model computation alone is replaced by a small fixture process. Both tests passed against the existing implementation; no production inference change or OOM root-cause fix is claimed.
- Existing 20-session timings remain historical measurements. PIGReward remains disabled under the existing failed G0 decision. Third-party usability and two-hour endurance remain exhibition acceptance work.
- Final recovery checks: 40 Python tests and 4 Node tests passed; ruff clean; offline model/asset preflight ready. Updated HTML report served HTTP 200, and the standalone fallback displayed all 48 embedded images with browser networking disabled and no JS errors. The original two upstream deprecation warnings remain. Evidence: `docs/reports/zipp-demo/recovery-evidence.json` and `recovery-display.json`.
