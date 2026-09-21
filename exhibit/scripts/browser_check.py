#!/usr/bin/env python3
"""Real Chromium rehearsal of the FAN comparison flow; never mocks the API.

Drives: welcome -> pick 3 cards (one aspect toggled off) -> topics -> compare
(4 plain + 4 personalized) -> mobile check -> reload -> same topic again ->
another topic -> cancel mid-generation -> finish -> sample -> reset.

Against the real server (``--url http://127.0.0.1:7860``, no ``--mock``) two
checks read the GPU worker's own event log from disk (``OUTPUTS/sessions/...``)
because the public API never republishes ``settings``/``pooled`` per image --
only ``exhibit.workers.generate`` emits them. ``--mock`` skips those two checks
since ``exhibit/tests/mock_api.mjs`` fakes images and never writes that log.
"""

import argparse
import json
import os
import time
from pathlib import Path

from exhibit.config import CONFIG, OUTPUTS
from playwright.sync_api import expect, sync_playwright

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CHROMIUM = (
    "/home/tomoya/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome"
)


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:7860")
    p.add_argument("--report-dir", type=Path, default=REPO / "docs/reports/fan-demo")
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--mock",
        action="store_true",
        help="skip assertions that only hold for real generated assets",
    )
    args = p.parse_args()
    base_url = args.url.rstrip("/")
    report_dir = args.report_dir
    shots = report_dir / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    errors = []
    external = []
    checks = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=os.environ.get("EXHIBIT_CHROMIUM", DEFAULT_CHROMIUM),
            headless=True,
        )
        page = browser.new_page(
            viewport={"width": 1440, "height": 1100}, device_scale_factor=1
        )
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on(
            "request",
            lambda req: (
                external.append(req.url) if not req.url.startswith(base_url) else None
            ),
        )

        page.goto(base_url)
        page.get_by_role("button", name="体験をはじめる").wait_for()
        page.screenshot(path=shots / "01-welcome.png", full_page=True)
        checks.append("welcome renders and offers to start")
        if args.quick:
            print(json.dumps({"errors": errors, "external_requests": external}))
            browser.close()
            return

        cfg = page.request.get(f"{base_url}/api/config").json()

        # ---------------------------------------------------------- 01 cards
        page.get_by_role("button", name="体験をはじめる").click()
        page.wait_for_selector(".card")
        sid = page.evaluate('sessionStorage.getItem("fan-session")')
        session = page.request.get(f"{base_url}/api/sessions/{sid}").json()
        card_by_id = {c["id"]: c for c in cfg["cards"]}
        # card_order lists all 16 subject x profile ids; only reviewed ones (in
        # cfg["cards"]) actually render as clickable cards on a real deployment.
        card_ids = [cid for cid in session["card_order"] if cid in card_by_id][:3]
        for card_id in card_ids:
            page.locator(f'.card[data-card="{card_id}"]').click()
        page.wait_for_function(
            "(n) => document.querySelectorAll('.card.selected').length === n",
            arg=len(card_ids),
        )
        toggled_aspect = "mood"
        toggled_phrase = card_by_id[card_ids[0]]["aspects"][toggled_aspect]
        chip = page.locator(
            f'.chip[data-aspect="{toggled_aspect}"][data-target="{card_ids[0]}"]'
        )
        chip.click()
        expect(chip).to_have_attribute("aria-pressed", "false")
        page.screenshot(path=shots / "02-cards.png", full_page=True)
        checks.append("select 3 cards, toggle one aspect off on the first")

        # --------------------------------------------------------- 02 topics
        with page.expect_response(lambda r: "/selection" in r.url):
            page.locator("#to-topics").click()
        page.wait_for_selector(".topic")
        topic_ids = [t["id"] for t in cfg["topics"]]
        topic_id = topic_ids[1] if len(topic_ids) > 1 else topic_ids[0]
        topic_button = page.locator(f'.topic[data-topic="{topic_id}"]')
        topic_button.click()
        expect(topic_button).to_have_attribute("aria-pressed", "true")
        page.screenshot(path=shots / "03-topics.png", full_page=True)
        checks.append("proceed to topics and pick one")

        # --------------------------------------------------------- 03 compare
        started = time.monotonic()
        with page.expect_response(lambda r: "/runs" in r.url):
            page.locator("#generate").click()
        page.wait_for_selector(".image-grid")
        page.screenshot(path=shots / "04-compare-generating.png", full_page=True)
        page.wait_for_function(
            "() => document.querySelectorAll('.shot').length === 8 &&"
            " document.querySelectorAll('.image-grid .placeholder').length === 0",
            timeout=150000,
        )
        generate_wait_seconds = round(time.monotonic() - started, 2)
        checks.append(
            f"4 plain + 4 personalized images shown within 150s ({generate_wait_seconds}s)"
        )
        page.screenshot(path=shots / "05-compare.png", full_page=True)

        sid = page.evaluate('sessionStorage.getItem("fan-session")')
        data = page.request.get(f"{base_url}/api/sessions/{sid}").json()
        run = data["run"]
        run_id = run["id"]
        assert "blind" not in run, "the blind comparison is gone"
        assert len(run["plain"]) == 4
        assert "variants" not in run
        assert len(run["personal"]) == 4
        assert [x["seed"] for x in run["plain"]] == [x["seed"] for x in run["personal"]]
        assert [x["seed"] for x in run["plain"]] == CONFIG["seeds"]
        for expected_hash, image in image_records(run):
            if "personalization_hash" in image:
                assert image["personalization_hash"] == expected_hash
        # One reference per distinct aspect phrase, merged across the 3 selected
        # cards (see exhibit.domain.build_personalization); not one ref per card.
        refs = run["personalization"]["refs"]
        assert refs, "no references at all"
        contributing = {cid for ref in refs for cid in ref.get("card_ids", [])}
        assert set(card_ids) <= contributing, (card_ids, contributing)
        toggled_ref = next((r for r in refs if r["text"] == toggled_phrase), None)
        if toggled_ref:
            assert card_ids[0] not in toggled_ref["card_ids"], (
                "the toggled-off card must not contribute its dropped aspect"
            )
        assert all(r["weight"] == len(r["card_ids"]) for r in refs), refs
        if not args.mock:
            events = worker_image_events(sid, run_id)
            assert events, "no worker image events found (real generation only)"
            assert all(e["settings"] == CONFIG["generation"] for e in events)
            assert all(e["pooled"] == "plain" for e in events)
        personal_image_url = run["personal"][0]["url"]
        checks.append(
            "compare: matched seeds, references merged by phrase with aspect-off honored"
            + ("" if args.mock else ", worker settings/pooled")
        )

        # ------------------------------------------------------- mobile shot
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=shots / "06-mobile.png", full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.set_viewport_size({"width": 1440, "height": 1100})
        checks.append("mobile 390x844 has no horizontal scroll")

        # --------------------------------------------------------- reload
        page.reload()
        page.get_by_text("SAME PROMPT, SAME SEEDS").wait_for()
        assert page.locator(".shot").count() == 8
        checks.append("reload keeps the comparison")

        # ------------------------------------------- same topic again: no rerun
        page.locator("#another").click()
        page.wait_for_selector(".topic")
        page.locator(f'.topic[data-topic="{topic_id}"]').click()
        page.locator("#generate").click()
        page.wait_for_selector(".image-grid")
        again = page.request.get(f"{base_url}/api/sessions/{sid}").json()["run"]
        assert again["id"] == run_id
        checks.append("the same topic again shows the finished comparison as is")

        # ---------------------------------------------------- another topic
        page.locator("#another").click()
        page.wait_for_selector(".topic")
        other_topic = next((t for t in topic_ids if t != topic_id), topic_ids[0])
        page.locator(f'.topic[data-topic="{other_topic}"]').click()
        with page.expect_response(lambda r: "/runs" in r.url):
            page.locator("#generate").click()
        page.wait_for_selector("#cancel")
        checks.append("another topic starts a fresh comparison")

        # ---------------------------------------------- cancel mid-generation
        with page.expect_response(lambda r: "/cancel" in r.url):
            page.locator("#cancel").click()
        page.wait_for_selector(".topic")
        page.screenshot(path=shots / "07-topics-after-cancel.png", full_page=True)
        assert page.request.get(f"{base_url}/api/sessions/{sid}").json()["run"] is None
        checks.append("cancel mid-generation drops the run back to the topic picker")

        # -------------------------------------------------------------- finish
        with page.expect_response(lambda r: r.url.endswith(f"/sessions/{sid}")):
            page.locator("#reset").click()
        page.get_by_role("button", name="体験をはじめる").wait_for()
        assert page.request.get(f"{base_url}/api/sessions/{sid}").status == 404
        assert page.request.get(base_url + personal_image_url).status == 404
        checks.append("finish deletes the session and the old personal image URL 404s")

        # -------------------------------------------------------------- sample
        if page.locator("#sample").count():
            with page.expect_response(lambda r: "/sample" in r.url):
                page.locator("#sample").click()
            page.wait_for_selector(".shot")
            assert page.locator(".shot").count() == 8, "expected 4 plain + 4 personal"
            page.screenshot(path=shots / "08-sample.png", full_page=True)
            checks.append("sample picker shows 4 plain + 4 personalized images")
            page.locator("#reset").click()
            page.get_by_role("button", name="体験をはじめる").wait_for()

        checks.append("reset")

        report = {
            "browser": "Chromium headless / actual localhost",
            "url": base_url,
            "mock": args.mock,
            "generate_wait_seconds": generate_wait_seconds,
            "run": run,
            "checks": checks,
            "page_errors": errors,
            "external_requests": external,
        }
        (report_dir / "browser-evidence.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        assert not errors, errors
        assert not external, external
        print(
            json.dumps(
                {
                    "generate_wait_seconds": generate_wait_seconds,
                    "checks": checks,
                    "errors": errors,
                    "external_requests": external,
                },
                ensure_ascii=False,
            )
        )
        browser.close()


if __name__ == "__main__":
    main()
