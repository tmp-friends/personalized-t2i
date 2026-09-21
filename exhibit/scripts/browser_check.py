#!/usr/bin/env python3
"""Real Chromium rehearsal of the multi-round FAN flow.

Drives: welcome -> round 1 (pick 3, leave one card unanswered so the commit is
refused) -> explicit aspects incl. 「全部好き」 -> round 2 (double click asks for
one round) -> reload -> commit -> topic -> compare -> optional labelled answer ->
adjust an aspect gain -> redraw -> back to identical content (exact cache) ->
another topic -> cancel mid-generation -> finish -> sample (which must not fill
the visitor's draft) -> reset. Desktop and ~390px viewports are both asserted to
be free of horizontal overflow, console errors, failed and external requests.

By default the script starts ``exhibit/tests/mock_api.mjs`` itself, so it needs
no GPU. Point it at a running server with ``--url http://127.0.0.1:7860`` to
rehearse against real generation; two checks then read the GPU worker's own
event log from disk (``OUTPUTS/sessions/...``) because the public API never
republishes ``settings``/``pooled`` per image -- only ``exhibit.workers.generate``
emits them.
"""

import argparse
import json
import os
import socket
import subprocess
import time
from contextlib import closing
from pathlib import Path

from exhibit.config import CONFIG, OUTPUTS
from playwright.sync_api import expect, sync_playwright

REPO = Path(__file__).resolve().parents[2]
MOCK = REPO / "exhibit/tests/mock_api.mjs"
DEFAULT_CHROMIUM = (
    "/home/tomoya/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome"
)


def free_port():
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(url, process, seconds=20):
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process and process.poll() is not None:
            raise RuntimeError(f"mock API exited with {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/api/config", timeout=1):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    raise RuntimeError(f"{url} never became ready")


def worker_image_events(sid, run_id):
    """Parse the GPU worker's ``events.jsonl`` for one run, if it ran locally."""
    base = OUTPUTS / "sessions" / sid / run_id / "personal"
    events = []
    for path in sorted(base.glob("generate-*/events.jsonl")):
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "image":
                events.append(event)
    return events


def image_records(run):
    """Every image dict the snapshot exposes, for a generic personalization-hash check."""
    for image in run.get("plain") or []:
        yield None, image
    hash_ = (run.get("personalization") or {}).get("hash")
    for image in run.get("personal") or []:
        yield hash_, image


class Drive:
    """The page plus the little bit of bookkeeping every step needs."""

    def __init__(self, page, base_url, checks):
        self.page = page
        self.base = base_url
        self.checks = checks

    def sid(self):
        return self.page.evaluate('sessionStorage.getItem("fan-session")')

    def snapshot(self):
        return self.page.request.get(f"{self.base}/api/sessions/{self.sid()}").json()

    def cards(self):
        return self.page.eval_on_selector_all(
            ".card", "els => els.map(e => e.dataset.card)"
        )

    def settled(self, ready, seconds=15):
        """Wait until the debounced, possibly retried draft save reached the server."""
        deadline = time.monotonic() + seconds
        while True:
            state = self.snapshot()
            if ready(state):
                return state
            if time.monotonic() > deadline:
                raise AssertionError(f"draft never reached the server: {state}")
            self.page.wait_for_timeout(100)

    def columns(self):
        return len(
            self.page.eval_on_selector(
                ".card-grid", "el => getComputedStyle(el).gridTemplateColumns"
            ).split()
        )

    def no_overflow(self, where):
        overflow = self.page.evaluate(
            "document.documentElement.scrollWidth - innerWidth"
        )
        assert overflow <= 0, f"{where}: {overflow}px of horizontal overflow"

    def done(self, note):
        self.checks.append(note)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--url",
        default=None,
        help="drive this already running server instead of the bundled mock API",
    )
    p.add_argument(
        "--report-dir", type=Path, default=REPO / "docs/reports/fan-personalization"
    )
    p.add_argument("--pair-ms", default="700", help="mock API: milliseconds per image")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    process = None
    if args.url:
        base_url = args.url.rstrip("/")
        mock = False
    else:
        port = free_port()
        base_url = f"http://127.0.0.1:{port}"
        process = subprocess.Popen(
            ["node", str(MOCK), "--port", str(port), "--pair-ms", str(args.pair_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        mock = True
    try:
        wait_for(base_url, process)
        report = run_checks(base_url, args, mock)
    finally:
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    print(json.dumps(report["summary"], ensure_ascii=False))


def run_checks(base_url, args, mock):
    report_dir = args.report_dir
    shots = report_dir / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    errors, console, failed, external, checks = [], [], [], [], []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=os.environ.get("EXHIBIT_CHROMIUM", DEFAULT_CHROMIUM),
            headless=True,
        )
        page = browser.new_page(
            viewport={"width": 1440, "height": 1100}, device_scale_factor=1
        )
        # A busy machine (a GPU run, a parallel test suite) must not read as a bug.
        page.set_default_timeout(60000)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on(
            "console",
            lambda m: (
                console.append(f"{m.type}: {m.text}")
                if m.type in ("error", "warning")
                else None
            ),
        )
        page.on("requestfailed", lambda r: failed.append(r.url))
        rounds_requested = []
        page.on(
            "request",
            lambda req: (
                external.append(req.url)
                if not req.url.startswith(base_url)
                else rounds_requested.append(req.url)
                if req.url.endswith("/rounds")
                else None
            ),
        )
        d = Drive(page, base_url, checks)

        page.goto(base_url)
        page.get_by_role("button", name="体験をはじめる").wait_for()
        page.screenshot(path=shots / "01-welcome.png", full_page=True)
        d.done("welcome renders and offers to start")
        if args.quick:
            browser.close()
            return {
                "summary": {"checks": checks, "errors": errors, "console": console},
            }

        cfg = page.request.get(f"{base_url}/api/config").json()
        assert cfg["schema_version"] == CONFIG["schema_version"]
        limits = cfg["selection"]
        by_id = {card["id"]: card for card in cfg["cards"]}

        # ------------------------------------------------------- 01 round one
        page.get_by_role("button", name="体験をはじめる").click()
        page.wait_for_selector(".card")
        first_round = d.cards()
        assert 0 < len(first_round) <= limits["round_size"], first_round
        assert (
            len(first_round) < len(cfg["cards"])
            or len(cfg["cards"]) <= limits["round_size"]
        ), "the whole catalog must never be one grid"
        picks = first_round[:3]
        for card_id in picks:
            page.locator(f'.card[data-card="{card_id}"]').click()
        page.wait_for_function(
            "(n) => document.querySelectorAll('.ref-edit').length === n", arg=len(picks)
        )
        d.done(
            f"round 1 shows {len(first_round)} of {len(cfg['cards'])} cards and 3 are chosen"
        )

        # -------------------------------------------- 02 explicit aspects
        assert page.locator("#commit").is_disabled(), "commit must wait for aspects"
        assert page.locator(".ref-edit.needs").count() == len(picks)
        for card_id in picks:
            row = page.locator(f"#pick-{card_id}")
            for aspect in cfg["aspects"]:
                expect(row.locator(f'.chip[data-aspect="{aspect}"]')).to_have_attribute(
                    "aria-pressed", "false"
                )
        page.screenshot(path=shots / "02-cards-unanswered.png", full_page=True)
        d.done("a newly selected card answers no aspect and blocks 決定")

        chip = page.locator(f'.chip[data-aspect="color"][data-target="{picks[0]}"]')
        chip.click()
        expect(chip).to_have_attribute("aria-pressed", "true")
        assert page.locator("#commit").is_disabled(), "one answered card is not enough"
        reason = page.locator(".action-row .note").first.inner_text()
        assert "1つ以上" in reason, reason
        # The remaining two cards: one by 「全部好き」, one aspect plus とても好き.
        page.locator(f'[data-likeall="{picks[1]}"]').click()
        page.wait_for_function(
            "(id) => [...document.querySelectorAll(`#pick-${id} .chip[data-aspect]`)]"
            ".every(el => el.getAttribute('aria-pressed') === 'true')",
            arg=picks[1],
        )
        page.locator(f'.chip[data-aspect="texture"][data-target="{picks[2]}"]').click()
        strong = page.locator(f'.step[data-strength="2"][data-target="{picks[2]}"]')
        strong.click()
        expect(strong).to_have_attribute("aria-pressed", "true")
        page.wait_for_function("() => !document.querySelector('#commit').disabled")
        state = d.settled(
            lambda value: any(
                row["card_id"] == picks[2] and row["strength"] == 2
                for row in value["selection"]
            )
        )
        assert state["revision"] >= 1 and state["committed"] is False
        assert len(state["selection"]) == 3
        answers = {row["card_id"]: row for row in state["selection"]}
        assert answers[picks[0]]["aspects"] == ["color"], answers[picks[0]]
        assert sorted(answers[picks[1]]["aspects"]) == sorted(cfg["aspects"])
        assert answers[picks[2]]["strength"] == 2
        page.screenshot(path=shots / "03-cards-answered.png", full_page=True)
        d.done("aspects are explicit: one chip, 「全部好き」, and a strength of 2")

        # ---------------------------------------------------- 03 round two
        before = len(rounds_requested)
        # Two clicks in the same task: exactly one round may be requested.
        page.evaluate(
            "() => { const b = document.querySelector('#more'); b.click(); b.click(); }"
        )
        page.wait_for_function(
            "() => document.querySelector('.round-tag')?.textContent.startsWith('2')"
        )
        page.wait_for_timeout(400)
        assert len(rounds_requested) - before == 1, rounds_requested[before:]
        state = d.snapshot()
        assert len(state["rounds"]) == 2
        second_round = d.cards()
        assert set(second_round).isdisjoint(first_round), "a card is never shown twice"
        assert len(second_round) == len(state["rounds"][1]["card_ids"])
        assert page.locator(".ref-edit").count() == 3, "earlier picks stay editable"
        extra = second_round[0]
        page.locator(f'.card[data-card="{extra}"]').click()
        page.locator(f'[data-likeall="{extra}"]').click()
        d.settled(
            lambda value: any(
                row["card_id"] == extra and row["aspects"] for row in value["selection"]
            )
        )
        assert len(d.snapshot()["selection"]) == 4
        page.screenshot(path=shots / "04-round-two.png", full_page=True)
        d.done(
            "double clicking 「ほかの候補も見る」 asks for one round; earlier picks stay editable"
        )

        # ------------------------------------------------- 04 mobile, cards
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(200)
        assert 2 <= d.columns() <= 3, f"{d.columns()} columns at 390px"
        d.no_overflow("cards at 390px")
        page.screenshot(path=shots / "05-mobile-cards.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1100})
        d.done("the round grid is 2-3 columns at 390px with no horizontal scroll")

        # ---------------------------------------------------- 05 reload
        page.reload()
        page.wait_for_selector(".card")
        assert d.cards() == second_round, "the reload resumes on the current round"
        assert page.locator(".ref-edit").count() == 4
        assert page.locator(".card.selected").count() == 1, (
            "only round 2's pick is here"
        )
        d.done("a reload restores the round, the draft and the commit state")

        # -------------------------------------------- 06 commit and generate
        with page.expect_response(lambda r: "/selection" in r.url):
            page.locator("#commit").click()
        page.wait_for_selector(".topic")
        state = d.snapshot()
        assert state["committed"] is True
        revision = state["revision"]
        topic_ids = [t["id"] for t in cfg["topics"]]
        topic_id = topic_ids[0]
        page.locator(f'.topic[data-topic="{topic_id}"]').click()
        page.screenshot(path=shots / "06-topics.png", full_page=True)
        started = time.monotonic()
        with page.expect_response(lambda r: r.url.endswith("/runs")):
            page.locator("#generate").click()
        page.wait_for_selector(".image-grid")
        page.screenshot(path=shots / "07-compare-generating.png", full_page=True)
        page.wait_for_function(
            "() => document.querySelectorAll('.shot').length === 8 &&"
            " document.querySelectorAll('.image-grid .placeholder').length === 0",
            timeout=150000,
        )
        generate_wait_seconds = round(time.monotonic() - started, 2)
        page.screenshot(path=shots / "08-compare.png", full_page=True)
        run = d.snapshot()["run"]
        run_id = run["id"]
        sid = d.sid()
        assert run["preference_revision"] == revision
        assert run["mode"] == "live" and not run["error"]
        assert len(run["plain"]) == 4 and len(run["personal"]) == 4
        assert [x["seed"] for x in run["plain"]] == [x["seed"] for x in run["personal"]]
        for expected_hash, image in image_records(run):
            if "personalization_hash" in image:
                assert image["personalization_hash"] == expected_hash
        refs = run["personalization"]["refs"]
        contributing = {cid for ref in refs for cid in ref.get("card_ids", [])}
        assert contributing == set(picks) | {extra}, contributing
        # The card that answered only 「色」 contributes nothing else.
        for ref in refs:
            if picks[0] in ref["card_ids"]:
                assert ref["text"] == by_id[picks[0]]["aspects"]["color"], ref
        assert not page.locator(".prompt-details[open]").count(), (
            "reference texts and policy details stay folded away"
        )
        if not mock:
            events = worker_image_events(sid, run_id)
            assert events, "no worker image events found (real generation only)"
            assert all(e["settings"] == CONFIG["generation"] for e in events)
        d.done(
            f"4 plain + 4 personalized within 150s ({generate_wait_seconds}s), refs match the answers"
            + ("" if mock else ", worker settings verified")
        )

        # ------------------------------------------------- 07 optional answer
        # A real worker exits a moment after its last image; the panel follows `done`.
        feedback = page.locator(".feedback")
        feedback.wait_for()
        assert feedback.count() == 1
        body = feedback.inner_text()
        for banned in ("ブラインド", "評価実験", "当てて"):
            assert banned not in body, body
        assert "任意" in body, body
        page.locator('[data-pref="personal"]').click()
        page.wait_for_function(
            "() => document.querySelector('[data-pref=\"personal\"]')"
            ".getAttribute('aria-pressed') === 'true'"
        )
        assert d.snapshot()["run"]["feedback"]["preference"] == "personal"
        page.locator('[data-pref="tie"]').click()
        page.wait_for_function(
            "() => document.querySelector('[data-pref=\"tie\"]')"
            ".getAttribute('aria-pressed') === 'true'"
        )
        assert d.snapshot()["run"]["feedback"]["preference"] == "tie"
        d.done("the optional labelled answer is stored and can be replaced")

        # --------------------------------------------- 08 adjust and redraw
        strengthen = page.locator('[data-gain="texture"][data-level="2"]')
        strengthen.click()
        expect(strengthen).to_have_attribute("aria-pressed", "true")
        assert d.snapshot()["aspect_gains"]["texture"] == 1, (
            "result-screen edits only apply on 描き直す"
        )
        page.screenshot(path=shots / "09-adjust.png", full_page=True)
        with page.expect_response(lambda r: r.url.endswith("/runs")):
            page.locator("#redraw").click()
        page.wait_for_function(
            "() => document.querySelectorAll('.shot').length === 8 &&"
            " document.querySelectorAll('.image-grid .placeholder').length === 0",
            timeout=150000,
        )
        state = d.snapshot()
        assert state["aspect_gains"]["texture"] == 2, state["aspect_gains"]
        assert state["revision"] > revision
        assert state["run"]["id"] != run_id and state["run"]["topic_id"] == topic_id
        assert state["run"]["mode"] == "live" and state["run"]["feedback"] is None
        redrawn = state["revision"]
        d.done("「強める」 commits a new revision and redraws the same topic with it")

        # ------------------------------------------ 09 identical content again
        page.locator('[data-gain="texture"][data-level="1"]').click()
        with page.expect_response(lambda r: r.url.endswith("/runs")):
            page.locator("#redraw").click()
        page.wait_for_function(
            "() => document.querySelectorAll('.shot').length === 8 &&"
            " document.querySelectorAll('.image-grid .placeholder').length === 0",
            timeout=150000,
        )
        state = d.snapshot()
        assert state["aspect_gains"]["texture"] == 1
        assert state["revision"] > redrawn
        cached = state["run"]
        assert cached["mode"] == "exact-cache", cached["mode"]
        assert cached["message"] in page.locator(".statusbox").inner_text()
        page.screenshot(path=shots / "10-exact-cache.png", full_page=True)
        d.done("returning to identical content reuses the images and says so")

        # ------------------------------------------------ 10 mobile, compare
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(200)
        d.no_overflow("compare at 390px")
        page.screenshot(path=shots / "11-mobile-compare.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1100})
        d.done("the comparison and the adjust panel fit 390px")

        # ------------------------------------------- 11 cancel mid-generation
        page.locator("#another").click()
        page.wait_for_selector(".topic")
        other_topic = next((t for t in topic_ids if t != topic_id), topic_id)
        page.locator(f'.topic[data-topic="{other_topic}"]').click()
        with page.expect_response(lambda r: r.url.endswith("/runs")):
            page.locator("#generate").click()
        page.wait_for_selector("#cancel")
        with page.expect_response(lambda r: "/cancel" in r.url):
            page.locator("#cancel").click()
        page.wait_for_selector(".topic")
        assert d.snapshot()["run"] is None
        page.screenshot(path=shots / "12-after-cancel.png", full_page=True)
        # The selection survives a cancel and can still be edited right away.
        page.locator("#back").click()
        page.wait_for_selector(".card")
        page.locator(f'[data-aspect="mood"][data-target="{picks[0]}"]').click()
        d.settled(
            lambda value: any(
                row["card_id"] == picks[0] and "mood" in row["aspects"]
                for row in value["selection"]
            )
        )
        assert "mood" in {
            aspect
            for row in d.snapshot()["selection"]
            if row["card_id"] == picks[0]
            for aspect in row["aspects"]
        }, "editing right after a cancel must not break"
        d.done("cancel drops the run, keeps the selection and still accepts edits")

        # -------------------------------------------------------- 12 finish
        personal_image_url = run["personal"][0]["url"]
        with page.expect_response(lambda r: r.url.endswith(f"/sessions/{sid}")):
            page.locator("#reset").click()
        page.get_by_role("button", name="体験をはじめる").wait_for()
        assert page.request.get(f"{base_url}/api/sessions/{sid}").status == 404
        assert page.request.get(base_url + personal_image_url).status == 404
        d.done("finish deletes the session and the old personal image URL 404s")

        # -------------------------------------------------------- 13 sample
        if page.locator("#sample").count():
            with page.expect_response(lambda r: "/sample" in r.url):
                page.locator("#sample").click()
            page.wait_for_selector(".shot")
            assert page.locator(".shot").count() == 8
            assert page.locator(".feedback").count() == 0, "a sample is never rated"
            assert page.locator(".adjust").count() == 0
            sample_state = d.snapshot()
            assert sample_state["run"]["mode"] == "sample"
            assert sample_state["selection"] == [], "a sample never fills the draft"
            page.screenshot(path=shots / "13-sample.png", full_page=True)
            page.locator("#reselect").click()
            page.wait_for_selector(".card")
            assert page.locator(".card.selected").count() == 0
            assert page.locator(".ref-edit").count() == 0
            d.done("a sample is labelled, unrated, and leaves the draft empty")
            page.locator("#reset").click()
            page.get_by_role("button", name="体験をはじめる").wait_for()

        d.no_overflow("welcome at 1440px")
        checks.append("reset")

        summary = {
            "generate_wait_seconds": generate_wait_seconds,
            "checks": checks,
            "errors": errors,
            "console": console,
            "failed_requests": failed,
            "external_requests": external,
        }
        report = {
            "browser": "Chromium headless / actual localhost",
            "url": base_url,
            "mock": mock,
            "run": run,
            "summary": summary,
        }
        (report_dir / "browser-evidence.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        assert not errors, errors
        assert not console, console
        assert not failed, failed
        assert not external, external
        browser.close()
        return report


if __name__ == "__main__":
    main()
