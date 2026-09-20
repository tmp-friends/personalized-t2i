#!/usr/bin/env python3
"""Embed real prepared images in a file://-compatible, server-free sample page."""

import base64
import html
import io

from exhibit.config import ASSETS, CONFIG, read_json
from exhibit.domain import CARDS
from exhibit.preflight import sample_errors
from PIL import Image

STYLE = (
    "body{margin:0;color-scheme:dark;background:#0b0d0c;color:#f3f5ed;"
    'font-family:"Noto Sans CJK JP","Hiragino Kaku Gothic ProN",system-ui,sans-serif;line-height:1.9}'
    "main{max-width:1100px;margin:auto;padding:40px 24px}h1{font-size:40px;font-weight:900;letter-spacing:-1px}"
    "h2{font-weight:900;margin-top:65px;border-top:1px solid #343c35;padding-top:24px}"
    ".badge{background:#1c2816;color:#c5f55a;border:1px solid #425a2c;padding:10px 18px;display:inline-block;border-radius:2px}"
    ".grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}"
    ".cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}"
    "img{width:100%;border-radius:2px}figure{margin:0}"
    "figcaption{font-size:11px;font-family:monospace;color:#a2aba5}p{color:#a2aba5}"
    "details{border:1px solid #343c35;padding:14px;margin-top:20px}summary{cursor:pointer;color:#c5f55a}"
    "a{color:inherit}@media(max-width:600px){.grid,.cards{grid-template-columns:1fr 1fr}h1{font-size:28px}}"
)


def data(path):
    image = Image.open(path)
    image.thumbnail((360, 360))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def figure(path, alt, caption):
    return (
        f'<figure><img alt="{html.escape(alt)}" src="{data(path)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def main():
    # Only samples that still satisfy the contract are embedded.
    samples = [
        sample
        for sample in (read_json(ASSETS / "samples.json", []) or [])
        if not sample_errors(sample, ASSETS)
    ]
    manifest = read_json(ASSETS / "manifest.json", {}) or {}
    images = manifest.get("images", {})
    parts = [
        (
            '<!doctype html><html lang="ja"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>FAN · 事前生成サンプル</title>"
            f"<style>{STYLE}</style><main>"
            '<span class="badge">事前生成サンプル · オフライン表示</span>'
            "<h1>同じ一文から、あなたの一枚を。</h1>"
            "<p>このページは代表的な選択から事前に生成したサンプルです。いま選んだ内容を反映した結果ではありません。"
            "画像はすべてローカルの Illustrious XL v2.0 で生成しています。</p>"
            "<p>好きな画像を3〜5枚選ぶ → 選んだ画像に付けた確認済みの説明文を参照にする → "
            "同じお題・同じ seed・同じ生成設定で、参照なしの通常生成と FAN 個人化生成を並べる。"
            "来場者ごとの追加学習はありません。ただし FAN 公式実装の ClassTokenDecoder"
            "（<code>weight/L.pth</code> / <code>weight/bigG.pth</code>）を使います。</p>"
            "<p>変わるのは参照の内容・重み <code>weight</code>・反映の強さ <code>alpha</code> だけで、"
            "お題の文も負のプロンプトも生成モデルも同じです。強く反映するほど良いとは限りません。</p>"
        )
    ]
    prepared = [card for card in CARDS.values() if card["id"] in images]
    if prepared:
        parts.append("<h2>選べるカード（4被写体 × 4表現）</h2>")
        parts.append('<div class="cards">')
        for card in prepared:
            parts.append(
                figure(
                    ASSETS / images[card["id"]]["path"], card["label"], card["label"]
                )
            )
        parts.append("</div>")
        parts.append(
            "<details><summary>カードに付いている参照の説明文</summary><p>"
            + "<br>".join(
                f"{html.escape(card['label'])}: {html.escape(card['ref_en'])}"
                for card in prepared
            )
            + "</p></details>"
        )
    for sample in samples:
        topic = next(t for t in CONFIG["topics"] if t["id"] == sample["topic_id"])
        refs = sample["personalization"]["refs"]
        parts.append(
            f"<h2>{html.escape(sample.get('label', sample['id']))}</h2>"
            f"<p>お題：{html.escape(topic['label'])} / "
            f"alpha {sample['personalization']['alpha']} / 参照 {len(refs)}件</p>"
        )
        for label, entries in [
            (
                "参照なしの通常生成",
                [images.get(f"{sample['topic_id']}-{i}") for i in range(4)],
            ),
            ("参照ありの個人化生成", sample["images"]),
        ]:
            parts.append(f'<h3>{label}</h3><div class="grid">')
            for entry in entries:
                if entry:
                    parts.append(
                        figure(
                            ASSETS / entry["path"],
                            topic["label"],
                            f"seed {entry['seed']}",
                        )
                    )
            parts.append("</div>")
        reference_lines = "<br>".join(
            f"{html.escape(ref['text'])}"
            f"（{html.escape(', '.join(ref['card_ids']))} / weight {ref['weight']}）"
            for ref in refs
        )
        prompt = sample["images"][0]["prompt"] if sample["images"] else ""
        parts.append(
            "<details><summary>使った参照の説明文とお題の文</summary>"
            f"<p>{reference_lines}</p><p>{html.escape(prompt)}</p></details>"
        )
    parts.append(
        '<p>モデル・出典：<a href="https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0">'
        "Illustrious XL v2.0</a> / FAN (Foundation Encoders Are All You Need for "
        "Preference-Aware Personalization, CVPR 2026) 公式実装。"
        "論文の定量結果はこの展示の性能ではありません。</p></main></html>"
    )
    (ASSETS / "fallback.html").write_text("".join(parts))


if __name__ == "__main__":
    main()
