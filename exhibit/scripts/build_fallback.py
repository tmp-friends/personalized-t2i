#!/usr/bin/env python3
"""Embed real prepared images in a file://-compatible, server-free sample page."""

import base64
import html
import io

from build_tech import FILE_LINKS, ICON, MARK, font_face, link
from build_tech import STYLE as TECH_STYLE
from exhibit.catalog import load_catalog
from exhibit.config import ASSETS, CONFIG, read_json
from exhibit.preflight import sample_errors
from PIL import Image

# The tech page's look (tokens, fonts, header, headings), plus the image grids.
STYLE = (
    TECH_STYLE
    + """
main{max-width:1100px}
.grid,.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:12px}
figure{margin:0}
figure img{display:block;width:100%;height:auto;border-radius:12px;border:1px solid var(--line)}
figcaption{margin-top:4px;font-family:var(--mono);font-size:11.5px;color:var(--mute)}
.meta{font-family:var(--mono);font-size:13px;color:var(--mute)}
details{margin-top:18px;border:1px solid var(--line);border-radius:14px;background:var(--panel);padding:12px 16px}
summary{cursor:pointer;color:var(--a1);font-size:14px}
details p{margin-top:10px;font-size:13px;overflow-wrap:anywhere}
@media(max-width:640px){.grid,.cards{grid-template-columns:1fr 1fr}}
"""
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
    topic_images = (read_json(ASSETS / "manifest.json", {}) or {}).get("images", {})
    card_images = (read_json(ASSETS / "catalog-v2.json", {}) or {}).get("images", {})
    catalog = load_catalog(reviewed_only=True)
    parts = [
        (
            '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="theme-color" content="#100d0c">'
            "<title>サンプル · パーソナライズ画像生成</title>"
            f'<link rel="icon" href="{ICON}">'
            "<style>"
            + font_face("Geist", "Geist-Variable.woff2")
            + font_face("Geist Mono", "GeistMono-Variable.woff2")
            + " ".join(STYLE.split("\n"))
            + "</style></head><body>"
            f'<header><span class="brand">{MARK}パーソナライズ画像生成</span>'
            '<span class="powered">powered by FAN</span><nav>'
            + link("home", "体験に戻る")
            + link("tech", "技術解説")
            + "</nav></header><main>"
            '<span class="pill">サンプル</span>'
            '<h1>代表的な好みで描いた、<span class="grad">サンプル</span></h1>'
            '<p class="lead">好みを選ぶだけで、あなた向けの画像を生成する展示です。'
            "このページは代表的な選択から事前に生成したサンプルです。"
            "いま選んだ内容を反映した結果ではありません。"
            "画像はすべてローカルの Illustrious XL v2.0 で生成しています。</p>"
        )
    ]
    number = 0

    def heading(title):
        nonlocal number
        number += 1
        return f'<section><h2><span class="n">{number}</span>{title}</h2>'

    prepared = [card for card in catalog["cards"] if card["id"] in card_images]
    if prepared:
        parts.append(
            heading("選べるカード") + f'<p class="meta">確認済み {len(prepared)}枚 / '
            f"{html.escape(catalog['catalog_id'])}</p>"
        )
        parts.append('<div class="cards">')
        for card in prepared:
            parts.append(
                figure(
                    ASSETS / card_images[card["id"]]["path"],
                    card["label"],
                    card["label"],
                )
            )
        parts.append("</div>")
        parts.append(
            "<details><summary>カードに付いている Reference Prompt（説明文）</summary><p>"
            + "<br>".join(
                f"{html.escape(card['label'])}: {html.escape(card['ref_en'])}"
                for card in prepared
            )
            + "</p></details></section>"
        )
    for sample in samples:
        topic = next(t for t in CONFIG["topics"] if t["id"] == sample["topic_id"])
        refs = sample["personalization"]["refs"]
        parts.append(
            heading(html.escape(sample.get("label", sample["id"])))
            + f'<p class="meta">お題：{html.escape(topic["label"])} / '
            f"policy {html.escape(sample['personalization']['policy_id'])} / "
            f"alpha {sample['personalization']['effective_policy']['alpha']} / "
            f"Reference Prompt {len(refs)}件</p>"
        )
        for label, entries in [
            (
                "パーソナライズなし",
                [topic_images.get(f"{sample['topic_id']}-{i}") for i in range(4)],
            ),
            ("パーソナライズあり", sample["images"]),
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
            "<details><summary>使った Reference Prompt と Target Prompt</summary>"
            f"<p>{reference_lines}</p><p>{html.escape(prompt)}</p></details></section>"
        )
    parts.append(
        '<p class="note" style="margin-top:56px">モデル・出典：'
        '<a href="https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0">'
        "Illustrious XL v2.0</a> / FAN (Foundation Encoders Are All You Need for "
        "Preference-Aware Personalization, CVPR 2026) 公式実装。"
        "論文の定量結果はこの展示の性能ではありません。</p>"
        "</main><footer><span>このページは <code>exhibit/scripts/build_fallback.py</code> が"
        '準備済みの画像から生成しています。</span><span class="links">'
        + link("home", "体験に戻る")
        + link("tech", "技術解説")
        + "</span></footer>"
        + FILE_LINKS
        + "</body></html>\n"
    )
    (ASSETS / "fallback.html").write_text("".join(parts))


if __name__ == "__main__":
    main()
