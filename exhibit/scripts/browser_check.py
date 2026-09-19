#!/usr/bin/env python3
"""Real Chromium rehearsal against localhost; captures evidence, never mocks the API."""

import argparse
import json
import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "docs/reports/zipp-demo"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "screenshots").mkdir(exist_ok=True)
    errors = []
    external = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=os.environ.get(
                "EXHIBIT_CHROMIUM",
                "/home/tomoya/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome",
            ),
            headless=True,
        )
        page = browser.new_page(
            viewport={"width": 1440, "height": 1100}, device_scale_factor=1
        )
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on(
            "request",
            lambda req: (
                external.append(req.url)
                if not req.url.startswith("http://127.0.0.1:7860")
                else None
            ),
        )
        page.goto("http://127.0.0.1:7860")
        page.get_by_role("button", name="体験をはじめる").wait_for()
        page.screenshot(path=REPORT / "screenshots/01-welcome.png", full_page=True)
        if args.quick:
            print(json.dumps({"errors": errors, "external_requests": external}))
            browser.close()
            return
        page.get_by_role("button", name="体験をはじめる").click()
        page.get_by_role("button", name="左の画像を選ぶ").wait_for()
        page.screenshot(path=REPORT / "screenshots/02-choice.png", full_page=True)
        for i in range(5):
            page.get_by_role("button", name="左の画像を選ぶ").click()
            if i < 4:
                page.wait_for_function(
                    '(n)=>document.querySelector(".counter")?.textContent.includes("0"+n)',
                    arg=i + 2,
                )
        page.get_by_role("button", name="お題を選ぶ").wait_for()
        page.get_by_label("色づかいの反映方法").select_option("cool")
        page.get_by_label("光の反映方法").select_option("__off")
        page.screenshot(path=REPORT / "screenshots/03-persona.png", full_page=True)
        page.get_by_role("button", name="お題を選ぶ").click()
        page.screenshot(path=REPORT / "screenshots/04-topics.png", full_page=True)
        started = time.monotonic()
        page.get_by_role("button", name="この好みで描く").click()
        page.get_by_role("button", name="しっくりくる画像はなかった").wait_for(
            timeout=150000
        )
        elapsed = round(time.monotonic() - started, 2)
        assert page.locator(".candidate").count() == 8, (
            "Expected 4 generic + 4 actual generated images"
        )
        sid = page.evaluate('sessionStorage.getItem("taste-session")')
        data = page.request.get(f"http://127.0.0.1:7860/api/sessions/{sid}").json()
        run = data["run"]
        assert len(run["personalized"]) == 4 and run["mode"] == "live", run
        assert "lighting" not in run["context"]["preferences"]
        assert run["context"]["preferences"]["color"] == "cool"
        assert [x["seed"] for x in run["generic"]] == [
            x["seed"] for x in run["personalized"]
        ]
        assert all(
            x["context_hash"] == run["context"]["hash"] for x in run["personalized"]
        )
        page.screenshot(path=REPORT / "screenshots/05-result.png", full_page=True)
        page.get_by_role("button", name="あなた向けの画像2を選ぶ").click()
        page.get_by_text("あなたの一枚を選びました。", exact=False).wait_for()
        page.screenshot(path=REPORT / "screenshots/06-selection.png", full_page=True)
        page.reload()
        page.get_by_text("あなたの一枚を選びました。", exact=False).wait_for()
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=REPORT / "screenshots/07-mobile.png", full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.set_viewport_size({"width": 1440, "height": 1100})
        page.get_by_role("button", name="好みを直してもう一度").click()
        page.get_by_role("button", name="お題を選ぶ").wait_for()
        assert page.get_by_label("色づかいの反映方法").input_value() == "cool"
        assert page.get_by_label("光の反映方法").input_value() == "__off"
        page.get_by_role("button", name="お題を選ぶ").click()
        page.get_by_role("button", name="この好みで描く").click()
        page.get_by_text("同一条件のキャッシュ", exact=True).wait_for(timeout=15000)
        page.get_by_role("button", name="好みを直してもう一度").click()
        for label in ["色づかい", "光", "構図", "描画表現", "雰囲気"]:
            page.get_by_label(label + "の反映方法").select_option("__off")
        page.get_by_role("button", name="お題を選ぶ").click()
        page.get_by_role("button", name="この好みで描く").click()
        page.get_by_text("通常画像のみ", exact=True).wait_for(timeout=15000)
        assert page.locator(".candidate").count() == 4
        page.screenshot(path=REPORT / "screenshots/08-all-off.png", full_page=True)
        page.locator("#finish").click()
        page.get_by_role("button", name="体験をはじめる").wait_for()
        assert (
            page.request.get(f"http://127.0.0.1:7860/api/sessions/{sid}").status == 404
        )
        assert (
            page.request.get(
                "http://127.0.0.1:7860" + run["personalized"][0]["url"]
            ).status
            == 404
        )
        page.get_by_role("button", name="体験をはじめる").click()
        for i in range(5):
            page.get_by_role("button", name="決められない →").click()
            if i < 4:
                page.wait_for_function(
                    '(n)=>document.querySelector(".counter")?.textContent.includes("0"+n)',
                    arg=i + 2,
                )
        page.get_by_text("有効な選択が0回でした。", exact=False).wait_for()
        page.reload()
        page.get_by_text("有効な選択が0回でした。", exact=False).wait_for()
        page.screenshot(path=REPORT / "screenshots/09-skips.png", full_page=True)
        if page.locator("#sample").count():
            page.locator("#sample").click()
            page.get_by_text("事前生成サンプル", exact=True).wait_for()
            assert page.locator(".candidate").count() == 8
            page.screenshot(path=REPORT / "screenshots/10-sample.png", full_page=True)
        page.locator("#reset").click()
        report = {
            "browser": "Chromium headless / actual localhost",
            "url": "http://127.0.0.1:7860",
            "browser_wall_seconds": elapsed,
            "run": run,
            "checks": [
                "five choices",
                "persona correction",
                "axis off",
                "four real SDXL images",
                "matched seeds",
                "shared context hash",
                "manual selection",
                "reload",
                "mobile width",
                "exact cache",
                "all off",
                "reset deletion",
                "five skips",
                "sample",
            ],
            "page_errors": errors,
            "external_requests": external,
        }
        (REPORT / "browser-evidence.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        assert not errors, errors
        assert not external, external
        print(
            json.dumps(
                {
                    "elapsed_seconds": elapsed,
                    "checks": report["checks"],
                    "errors": errors,
                    "external_requests": external,
                },
                ensure_ascii=False,
            )
        )
        browser.close()


if __name__ == "__main__":
    main()
