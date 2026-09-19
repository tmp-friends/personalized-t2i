#!/usr/bin/env python3
"""Embed real prepared images in a file://-compatible, server-free sample page."""

import base64
import html
import io

from exhibit.config import ASSETS, CONFIG, read_json
from PIL import Image


def data(path):
    image = Image.open(path)
    image.thumbnail((360, 360))
    b = io.BytesIO()
    image.save(b, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    samples = read_json(ASSETS / "samples.json", [])
    manifest = read_json(ASSETS / "manifest.json")
    parts = [
        """<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Taste · 事前生成サンプル</title><style>body{margin:0;background:#f4f2eb;color:#24392d;font-family:system-ui,sans-serif;line-height:1.9}main{max-width:1100px;margin:auto;padding:40px 24px}h1{font-size:40px;font-weight:500}h2{font-weight:500;margin-top:65px}.badge{background:#eddbcd;padding:10px 18px;display:inline-block;border-radius:5px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}img{width:100%;border-radius:5px}figure{margin:0}figcaption{font-size:11px}p{color:#657263}a{color:inherit}@media(max-width:600px){.grid{grid-template-columns:1fr 1fr}h1{font-size:28px}}</style><main><span class="badge">事前生成サンプル · オフライン表示</span><h1>あなたの「好き」を、一枚の絵へ。</h1><p>このページは代表的な選択履歴の結果を紹介するサンプルです。今のあなたの選択を反映した結果ではありません。画像は実際にローカルのSDXLで生成しました。</p><p>5回の画像選択 → 好みを確かめて訂正 → 通常と個人化の4枚を比較 → 最後は本人が選択。ZIPPのpersonaを用いたプロンプト個人化を応用した独自実装です。画像特徴はQwen3.5-4Bで事前解析し、確認した根拠を集計しています。個人別の追加学習はありません。</p><p>PIGRewardは公開モデルの統合検証中です。このサンプルでは自動推薦を表示しません。</p>"""
    ]
    for s in samples:
        topic = next(t for t in CONFIG["topics"] if t["id"] == s["topic_id"])
        parts.append(
            f"<h2>{html.escape(s['id'])} · {html.escape(topic['label'])}</h2><p>好みの根拠：{html.escape(s['context']['text'])}</p>"
        )
        for label, images in [
            (
                "好みを使わずに描く",
                [manifest["images"][f"{s['topic_id']}-{i}"] for i in range(4)],
            ),
            ("代表的な好みで描く", s["images"]),
        ]:
            parts.append(f'<h3>{label}</h3><div class="grid">')
            for im in images:
                parts.append(
                    f'<figure><img alt="{html.escape(topic["label"])}" src="{data(ASSETS / im["path"])}"><figcaption>seed {im["seed"]}</figcaption></figure>'
                )
            parts.append("</div>")
        parts.append(
            f"<details><summary>実際の個人化プロンプト</summary><p>{html.escape(s['rewrite']['prompt'])}</p></details>"
        )
    parts.append(
        '<p>モデル・出典：<a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0">SDXL base</a> / <a href="https://huggingface.co/Qwen/Qwen3.5-4B">Qwen3.5-4B</a>。研究の報告精度はこの展示の性能を表すものではありません。</p></main></html>'
    )
    (ASSETS / "fallback.html").write_text("".join(parts))


if __name__ == "__main__":
    main()
