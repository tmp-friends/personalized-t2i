#!/usr/bin/env python3
"""Real Chromium rehearsal of the FAN blind-comparison flow; never mocks the API.

Drives: welcome -> pick 3 cards (one aspect toggled off) -> topics -> blind (4
pairs) -> answer 3 of 4 -> reveal -> mobile check -> adjust alpha/refs and
regenerate -> repeat the same setting (exact-cache) -> reload -> another topic
-> cancel mid-blind -> finish -> sample -> reset.

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


def worker_image_events(sid, run_id, variant_id):
    """Parse the GPU worker's ``events.jsonl`` for one variant, if it ran locally."""
    base = OUTPUTS / "sessions" / sid / run_id / variant_id
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
    for variant in run.get("variants") or []:
        hash_ = (variant.get("personalization") or {}).get("hash")
        for image in variant.get("images") or []:
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

        # ------------------------------------------------- 03 blind (4 pairs)
        started = time.monotonic()
        with page.expect_response(lambda r: "/runs" in r.url):
            page.locator("#generate").click()
        page.wait_for_selector(".pair-row")
        page.screenshot(path=shots / "04-blind-generating.png", full_page=True)
        page.wait_for_function(
            "() => document.querySelectorAll('.pair-row').length === 4 &&"
            " document.querySelectorAll('.pair-grid .placeholder').length === 0",
            timeout=150000,
        )
        blind_wait_seconds = round(time.monotonic() - started, 2)
        checks.append(f"all 4 blind pairs ready within 150s ({blind_wait_seconds}s)")

        sid = page.evaluate('sessionStorage.getItem("fan-session")')
        pre_reveal = page.request.get(f"{base_url}/api/sessions/{sid}").json()
        pre_run = pre_reveal["run"]
        run_id = pre_run["id"]
        assert pre_run["blind"]["mapping"] is None, "mapping leaked before reveal"
        assert pre_run["plain"] == [], "plain images leaked before reveal"
        assert pre_run["variants"][0]["images"] == [], (
            "variant images leaked before reveal"
        )
        direct = page.request.get(
            f"{base_url}/api/sessions/{sid}/images/{run_id}/plain-0.png"
        )
        assert direct.status == 404, "direct plain image path must 404 before reveal"
        checks.append(
            "pre-reveal snapshot hides mapping/plain/variant images and 404s direct path"
        )

        page.locator(".pair-row").nth(0).locator(".pair").nth(0).click()
        page.locator(".pair-row").nth(1).locator(".pair").nth(0).click()
        page.wait_for_function(
            "() => document.querySelectorAll('.pair-row.answered').length === 2"
        )
        page.locator(".pair-row").nth(2).locator(".tie").click()
        page.wait_for_function(
            "() => document.querySelectorAll('.pair-row.answered').length === 3"
        )
        checks.append("pick A on pairs 0-1, tie on pair 2, leave pair 3 unanswered")

        # -------------------------------------------------------- 04 reveal
        with page.expect_response(lambda r: "/reveal" in r.url):
            page.locator("#reveal").click()
        page.wait_for_selector(".adjust, .error-message")
        page.screenshot(path=shots / "05-result.png", full_page=True)

        data = page.request.get(f"{base_url}/api/sessions/{sid}").json()
        run = data["run"]
        assert run["blind"]["revealed"] is True
        score = run["blind"]["score"]
        assert score["answered"] == 3, score
        assert score["tie"] == 1, score
        assert len(run["plain"]) == 4
        v0 = run["variants"][0]
        assert len(v0["images"]) == 4
        assert [x["seed"] for x in run["plain"]] == [x["seed"] for x in v0["images"]]
        assert [x["seed"] for x in run["plain"]] == CONFIG["seeds"]
        for expected_hash, image in image_records(run):
            if "personalization_hash" in image:
                assert image["personalization_hash"] == expected_hash
        # One reference per distinct aspect phrase, merged across the 3 selected
        # cards (see exhibit.domain.build_personalization); not one ref per card.
        refs = v0["personalization"]["refs"]
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
            events = worker_image_events(sid, run_id, "v0")
            assert events, "no worker image events found for v0 (real generation only)"
            assert all(e["settings"] == CONFIG["generation"] for e in events)
            assert all(e["pooled"] == "plain" for e in events)
        personal_image_url = v0["images"][0]["url"]
        checks.append(
            "reveal: score, matched seeds, references merged by phrase with aspect-off honored"
            + ("" if args.mock else ", worker settings/pooled")
        )

        # ------------------------------------------------------- mobile shot
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=shots / "06-mobile.png", full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.set_viewport_size({"width": 1440, "height": 1100})
        checks.append("mobile 390x844 has no horizontal scroll")

        # ------------------------------------------------------------ adjust
        page.locator('[data-alpha="strong"]').click()
        page.locator(f'[data-weight="emphasis"][data-card="{card_ids[0]}"]').click()
        page.locator(f'[data-weight="exclude"][data-card="{card_ids[1]}"]').click()
        with page.expect_response(lambda r: "/runs" in r.url):
            page.locator("#regen").click()
        page.wait_for_function(
            "(n) => document.querySelectorAll('.variant').length === n &&"
            " document.querySelectorAll('.variant')[0].querySelectorAll('.placeholder').length === 0",
            arg=2,
            timeout=150000,
        )
        page.screenshot(path=shots / "07-adjusted.png", full_page=True)
        data = page.request.get(f"{base_url}/api/sessions/{sid}").json()
        run = data["run"]
        assert len(run["variants"]) == 2, run["variants"]
        v1 = run["variants"][1]
        v1_refs = v1["personalization"]["refs"]
        assert v1_refs, "no references after adjusting weights"
        weight_of = {card_ids[0]: 2.0, card_ids[2]: 1.0}  # card_ids[1] excluded
        for ref in v1_refs:
            assert card_ids[1] not in ref["card_ids"], (
                "the excluded card must not contribute any reference",
                ref,
            )
            expected = sum(weight_of[cid] for cid in ref["card_ids"])
            assert ref["weight"] == expected, (ref, expected)
        contributing = {cid for ref in v1_refs for cid in ref["card_ids"]}
        assert contributing == {card_ids[0], card_ids[2]}, contributing
        checks.append(
            "adjust alpha=strong, emphasis+exclude: excluded card drops out, "
            "remaining weights match emphasis/normal exactly"
        )

        # ------------------------------------------------- same setting again
        with page.expect_response(lambda r: "/runs" in r.url):
            page.locator("#regen").click()
        page.wait_for_function(
            "(n) => document.querySelectorAll('.variant').length === n",
            arg=3,
            timeout=150000,
        )
        page.get_by_text("同一条件のキャッシュ", exact=True).wait_for(timeout=15000)
        page.screenshot(path=shots / "08-cache.png", full_page=True)
        checks.append("same setting again reuses the exact-cache badge")

        # --------------------------------------------------------- reload
        page.reload()
        page.get_by_text("THE ANSWER").wait_for()
        page.wait_for_selector(".adjust, .error-message")
        checks.append("reload keeps the result screen")

        # ---------------------------------------------------- another topic
        page.locator("#another").click()
        page.wait_for_selector(".topic")
        other_topic = next((t for t in topic_ids if t != topic_id), topic_ids[0])
        page.locator(f'.topic[data-topic="{other_topic}"]').click()
        with page.expect_response(lambda r: "/runs" in r.url):
            page.locator("#generate").click()
        page.wait_for_selector(".pair-row")
        checks.append("another topic starts a fresh comparison")

        # ------------------------------------------------- cancel mid-blind
        with page.expect_response(lambda r: "/cancel" in r.url):
            page.locator("#cancel").click()
        page.wait_for_function(
            "(n) => document.querySelectorAll('.card.selected').length === n",
            arg=len(card_ids),
        )
        page.screenshot(path=shots / "09-cards-preselected.png", full_page=True)
        assert page.locator(".card.selected").count() == len(card_ids)
        checks.append("cancel mid-blind drops the run back to preselected cards")

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
            assert page.locator(".adjust").count() == 0, "sample must not offer adjust"
            page.screenshot(path=shots / "10-sample.png", full_page=True)
            checks.append("sample picker shows 8 images with no adjust panel")
            page.locator("#reset").click()
            page.get_by_role("button", name="体験をはじめる").wait_for()

        checks.append("reset")

        report = {
            "browser": "Chromium headless / actual localhost",
            "url": base_url,
            "mock": args.mock,
            "blind_wait_seconds": blind_wait_seconds,
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
                    "blind_wait_seconds": blind_wait_seconds,
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
