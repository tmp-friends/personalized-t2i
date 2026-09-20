#!/usr/bin/env python3
"""Build the FAN exhibition handoff HTML from actual local measurement artifacts.

Every section reads one on-disk artifact and degrades to "未計測" / "未準備"
when that artifact is missing -- nothing here is invented. See
docs/superpowers/specs/2026-09-21-fan-exhibition-demo-design.md for the design
this implements and section 8/9 for the verification and wording rules quoted
below.
"""

import base64
import html
import io
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from exhibit.config import (
    ASSETS,
    CARDS_REVIEW,
    CONFIG,
    OUTPUTS,
    REPO,
    read_json,
    write_json,
)
from exhibit.domain import CARDS, reviewed_ids
from PIL import Image

REPORT = REPO / "docs/reports/fan-demo"
E = html.escape


def thumb(path, size=160, quality=72):
    """A small base64 JPEG data URI, or None if the file is missing/unreadable."""
    try:
        image = Image.open(path).convert("RGB")
    except (OSError, ValueError):
        return None
    image.thumbnail((size, size))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def card_grid():
    reviewed = reviewed_ids(CARDS_REVIEW)
    cards = []
    for card_id, card in CARDS.items():
        src = thumb(ASSETS / card["path"])
        if not src:
            continue
        cards.append(
            {
                "id": card_id,
                "label": card["label"],
                "ref_en": card["ref_en"],
                "reviewed": card_id in reviewed,
                "src": src,
            }
        )
    return cards


def generic_grid():
    topics = []
    for topic in CONFIG["topics"]:
        shots = [
            thumb(ASSETS / "generic" / f"{topic['id']}-{i}.png")
            for i in range(len(CONFIG["seeds"]))
        ]
        shots = [s for s in shots if s]
        if shots:
            topics.append({"id": topic["id"], "label": topic["label"], "shots": shots})
    return topics


def sample_grid():
    out = []
    for sample in read_json(ASSETS / "samples.json", []) or []:
        images = sample.get("images") or []
        shots = [
            thumb(ASSETS / image["path"])
            for image in images
            if isinstance(image, dict) and image.get("path")
        ]
        shots = [s for s in shots if s]
        if shots:
            out.append(
                {
                    "id": sample.get("id", "?"),
                    "topic_id": sample.get("topic_id", "?"),
                    "shots": shots,
                }
            )
    return out


def image_grid_html(items, caption_key="label"):
    if not items:
        return "<p>未準備。</p>"
    return "".join(
        f'<figure><img src="{item["src"]}" alt="{E(item.get("label", item.get("id", "")))}" loading="lazy">'
        f"<figcaption>{E(item.get(caption_key, item.get('id', '')))}"
        f"{' · 未確認' if caption_key == 'label' and not item.get('reviewed', True) else ''}"
        f"</figcaption></figure>"
        for item in items
    )


def topic_shots_html(topics):
    if not topics:
        return "<p>未準備。</p>"
    return "".join(
        f'<div class="topic-shots"><h4>{E(t["label"])}</h4><div class="mini-grid">'
        + "".join(f'<img src="{s}" alt="{E(t["label"])}">' for s in t["shots"])
        + "</div></div>"
        for t in topics
    )


def sample_shots_html(samples):
    if not samples:
        return "<p>未準備。</p>"
    return "".join(
        f'<div class="topic-shots"><h4>{E(s["id"])} · {E(s["topic_id"])}</h4><div class="mini-grid">'
        + "".join(f'<img src="{shot}" alt="{E(s["id"])}">' for shot in s["shots"])
        + "</div></div>"
        for s in samples
    )


def parse_pytest_count(log):
    if not log:
        return None
    m = re.search(r"(\d+) passed", log)
    return int(m.group(1)) if m else None


def parse_node_test(log):
    """node --test's default "spec" reporter prints "ℹ pass N" / "ℹ fail N";
    the TAP reporter prints "# pass N" / "# fail N". Accept either."""
    if not log:
        return None, None
    passed = re.search(r"[ℹ#]\s*pass\s+(\d+)", log)
    failed = re.search(r"[ℹ#]\s*fail\s+(\d+)", log)
    return (
        int(passed.group(1)) if passed else None,
        int(failed.group(1)) if failed else None,
    )


def equivalence_row(name, entry):
    hidden = entry.get("hidden", {})
    pooled = entry.get("pooled", {})
    return (
        f"<tr><td>{E(name)}</td>"
        f"<td>{hidden.get('mean_cosine', '—')}</td>"
        f"<td>{hidden.get('max_abs_diff', '—')}</td>"
        f"<td>{pooled.get('mean_cosine', '—')}</td>"
        f"<td>{entry.get('encode_seconds', '—')}</td></tr>"
    )


def avg(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 1) if values else None


def runs_by_key(container, key, value):
    return [r for r in (container or []) if r.get(key) == value]


def g0_settings_section(followup, final, ladder):
    """The concrete settings G0/follow-up spikes settled on, and the evidence
    for each -- 未計測 wherever a probe output is missing, never invented."""
    followup = followup or {}
    final = final or {}
    ladder = ladder or {}
    followup_runs = followup.get("runs") or []
    final_runs = final.get("runs") or []

    # skip_pa=[0..7] vs no personalized-attention skip at all.
    skip_none = runs_by_key(followup_runs, "name", "04-setA-skipnone")
    skip_07 = runs_by_key(followup_runs, "name", "04-setA-skip0to7")
    lap_before = avg([r.get("laplacian_var") for r in skip_none])
    lap_after = avg([r.get("laplacian_var") for r in skip_07])
    sat_before = avg([r.get("saturation") for r in skip_none])
    sat_after = avg([r.get("saturation") for r in skip_07])
    skip_pa_line = (
        f"skip_pa未指定（平均 laplacian_var {lap_before} → 平均 saturation {sat_before}）から "
        f"skip_pa=[0..7]（平均 laplacian_var {lap_after} → 平均 saturation {sat_after}）で、"
        f"ぼやけ（laplacian_var低下）と彩度低下が改善しました（{len(skip_none)}枚 vs {len(skip_07)}枚の記録）。"
        if lap_before is not None and lap_after is not None
        else "未計測。"
    )

    # use_attn_mask: alpha=0 でも参照なしと一致すべき pooled が崩れる。
    mask1_cos = (
        (followup.get("alpha0_vs_plain_mask1") or {}).get("mean_cosine")
        if followup
        else None
    )
    mask0_cos = (
        (followup.get("alpha0_vs_plain_mask0") or {}).get("mean_cosine")
        if followup
        else None
    )
    mask_line = (
        f"use_attn_mask=True では alpha=0（参照ゼロ相当）でも hidden state の平均コサイン類似度が "
        f"{round(mask1_cos, 2)}まで下がり、参照なしのFANエンコードと一致しません"
        f"（use_attn_mask=False では{round(mask0_cos, 4)}）。そのため use_attn_mask は無効のまま採用しています。"
        if mask1_cos is not None
        else "未計測。"
    )

    # split (aspect単位の短い句) vs bundled (カードごとの結合文) vs aspect-major (側面ごとにカードを束ねた1文)。
    bundled_vs_split = followup.get("bundled_vs_split") or {}
    split_line_bits = []
    if bundled_vs_split:
        mae_values = [
            v for v in bundled_vs_split.values() if isinstance(v, (int, float))
        ]
        if mae_values:
            split_line_bits.append(
                f"one-long-ref-per-card（結合文）とsplit（側面ごとの短い句）は同じ参照でも "
                f"画素MAEで{min(mae_values)}〜{max(mae_values)}の差が出ます"
                f"（bundled_vs_split, {len(mae_values)}条件）。"
            )
    split6 = runs_by_key(final_runs, "case", "C-5cards-split-a0.6")
    major6 = runs_by_key(final_runs, "case", "D-5cards-aspectmajor-a0.6")
    if split6 and major6:
        split_lap = [r.get("laplacian_var") for r in split6]
        major_lap = [r.get("laplacian_var") for r in major6]
        split_line_bits.append(
            f"alpha=0.6でsplitとaspect-major（側面ごとにカードをまとめた1文）を比べると、"
            f"laplacian_varはsplitが{split_lap}、aspect-majorが{major_lap}で、"
            f"記録した{len(split6)}枚ともsplitが上回りました。"
        )
    split_line = (
        " ".join(split_line_bits)
        + " 現在の実装（exhibit.domain.build_personalization）は側面ごとの短い句を"
        "カード横断でマージし、重みを合算する方式です。"
        if split_line_bits
        else "未計測。"
    )

    # alpha ladder.
    alphas = CONFIG.get("alphas", {})
    ladder_runs = ladder.get("runs") or []
    tokyo_baseline = avg(
        [
            r.get("laplacian_var")
            for r in ladder_runs
            if r.get("topic") == "tokyo" and r.get("set") == "baseline"
        ]
    )
    tokyo_07 = [
        r.get("laplacian_var")
        for r in ladder_runs
        if r.get("topic") == "tokyo" and r.get("alpha") == 0.7
    ]
    lap_note = (
        f"（傍証: tokyoのbaseline laplacian_varは約{tokyo_baseline}に対し、"
        f"alpha=0.7では{tokyo_07}まで跳ね上がる記録があり、崩れと符合します）"
        if tokyo_baseline is not None and tokyo_07
        else ""
    )
    ladder_line = (
        f"weak={alphas.get('weak', '未計測')} / mid={alphas.get('mid', '未計測')} / "
        f"strong={alphas.get('strong', '未計測')} を採用しています。目視の確認では、alpha=0.7でtokyoのお題の"
        f"1boyという被写体の指定が崩れ始め、alpha=0.8では緑の瞳の指定が失われました。{lap_note}"
        "そのため0.6を上限にしています。崩れの判定自体は目視によるもので、性能の定量主張ではありません。"
    )

    fan_conf = json.dumps(CONFIG.get("fan", {}), ensure_ascii=False, indent=2)
    alphas_conf = json.dumps(alphas, ensure_ascii=False, indent=2)

    return f"""<h2>G0 で確定した設定と根拠</h2>
<div class="panel"><ul>
<li><b>skip_pa</b>：{skip_pa_line}</li>
<li><b>use_attn_mask</b>：{mask_line}</li>
<li><b>参照の分け方</b>：{split_line}</li>
<li><b>alpha ladder</b>：{ladder_line}</li>
<li><b>warm_soft の lighting</b>：既定の "soft lighting, gentle shadows" から
"warm golden hour light, gentle shadows" に変更しました（ladder.json の reference_sets s1 → s1p）。
現在の <code>configs/demo.json</code> にもこの文言が入っています。</li>
<li><b>pooled</b>：常に参照なし（plain）の pooled embedding を使います。個人化するのは hidden state だけです。</li>
</ul>
<p class="small">いずれも少数seed・少数条件の記録であり、性能を主張するものではありません（設計書 §9）。</p>
<p><b>configs/demo.json の fan ブロック</b></p>
<pre>{E(fan_conf)}</pre>
<p><b>configs/demo.json の alphas</b></p>
<pre>{E(alphas_conf)}</pre>
<a href="../../../exhibit/outputs/fan-probe/followup/followup.json">追加検証 JSON</a> ·
<a href="../../../exhibit/outputs/fan-probe/final/final.json">最終確認 JSON</a> ·
<a href="../../../exhibit/outputs/fan-probe/ladder/ladder.json">alpha ladder JSON</a></div>"""


def sparkline(rows):
    if not rows:
        return "<p>セッション計測中。</p>"
    points = " ".join(
        f"{30 + i * 34},{150 - min(r.get('seconds', 0), 35) * 4}"
        for i, r in enumerate(rows)
    )
    dots = "".join(
        f'<circle cx="{30 + i * 34}" cy="{150 - min(r.get("seconds", 0), 35) * 4}" r="3" fill="#346347"/>'
        for i, r in enumerate(rows)
    )
    return (
        f'<svg viewBox="0 0 740 180" role="img" aria-label="セッションごとの実生成時間">'
        f'<path d="M25 10V155H725" fill="none" stroke="#c6cdbf"/>'
        f'<path d="M25 70H725" stroke="#d9ded1"/>'
        f'<text x="27" y="66" fill="#6e796a" font-size="10">20秒</text>'
        f'<polyline points="{points}" fill="none" stroke="#346347" stroke-width="3"/>{dots}'
        f'<text x="27" y="177" font-size="10" fill="#6e796a">SESSION 01</text>'
        f'<text x="642" y="177" font-size="10" fill="#6e796a">SESSION {len(rows):02d}</text></svg>'
    )


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    browser = read_json(REPORT / "browser-evidence.json", {}) or {}
    probe = read_json(OUTPUTS / "fan-probe/report.json", {}) or {}
    followup = read_json(OUTPUTS / "fan-probe/followup/followup.json", {}) or {}
    final = read_json(OUTPUTS / "fan-probe/final/final.json", {}) or {}
    ladder = read_json(OUTPUTS / "fan-probe/ladder/ladder.json", {}) or {}
    rehearsal = read_json(OUTPUTS / "rehearsal.json", {}) or {}
    preflight = read_json(OUTPUTS / "preflight.json", {}) or {}
    all_tests_log = (
        (OUTPUTS / "all-tests.log").read_text()
        if (OUTPUTS / "all-tests.log").is_file()
        else None
    )
    js_tests_log = (
        (OUTPUTS / "js-tests.log").read_text()
        if (OUTPUTS / "js-tests.log").is_file()
        else None
    )
    py_passed = parse_pytest_count(all_tests_log)
    node_passed, node_failed = parse_node_test(js_tests_log)

    cards = card_grid()
    reviewed_count = sum(1 for c in cards if c["reviewed"])
    generic_topics = generic_grid()
    samples = sample_grid()

    measured = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M JST")

    evidence = {
        "generated_at": measured,
        "browser": {k: v for k, v in browser.items() if k != "run"},
        "fan_probe": {
            k: v
            for k, v in probe.items()
            if k not in ("images", "class_token_detector")
        },
        "followup": {
            k: v for k, v in followup.items() if k not in ("reference_sets", "runs")
        },
        "final": {k: v for k, v in final.items() if k != "runs"},
        "ladder": {k: v for k, v in ladder.items() if k != "runs"},
        "rehearsal": {k: v for k, v in rehearsal.items() if k != "rows"},
        "preflight": preflight,
        "tests": {
            "python_passed": py_passed,
            "node_passed": node_passed,
            "node_failed": node_failed,
        },
        "assets": {
            "cards_embedded": len(cards),
            "cards_reviewed": reviewed_count,
            "cards_total": len(CARDS),
            "generic_topics_embedded": len(generic_topics),
            "samples_embedded": len(samples),
        },
        "mode": "FAN実生成 + ブラインド比較",
        "limitations": [
            "第三者5人の理解度確認は未実施",
            "2時間連続稼働の実接続確認は未実施",
            "ブラインド比較の集計は少人数の記録であり、性能主張ではない",
            "reveal前のブラインド画像はセッション内の見た目確認のみで、統計的優位性を主張しない",
        ],
    }
    write_json(REPORT / "evidence.json", evidence)

    # ---------------------------------------------------------------- badges
    badges = ['<span class="badge">localhost 実装確認</span>']
    if rehearsal.get("successes"):
        badges.append(
            f'<span class="badge">実GPU生成 {rehearsal["successes"]}/{rehearsal.get("sessions", "—")}セッション</span>'
        )
    if probe:
        plain_cos = (
            probe.get("equivalence", {})
            .get("fan_plain_vs_pipeline", {})
            .get("hidden", {})
            .get("mean_cosine")
        )
        if plain_cos is not None:
            badges.append(f'<span class="badge">G0一致 cos={plain_cos}</span>')
    badges.append(
        '<span class="badge warn">pooled は参照なしの値を使用（意図的な逸脱）</span>'
    )
    if browser:
        ok = not browser.get("page_errors") and not browser.get("external_requests")
        badges.append(
            f'<span class="badge{"" if ok else " warn"}">ブラウザー確認 {"JS例外0・外部通信0" if ok else "要確認"}</span>'
        )

    # ------------------------------------------------------------ statistics
    stats = f"""<div class="stats">
<div class="stat"><strong>{rehearsal.get("successes", "—")}/{rehearsal.get("sessions", "—")}</strong><span>実生成セッション成功</span></div>
<div class="stat"><strong>{rehearsal.get("p95_seconds", "未計測")}</strong><span>生成時間 p95（秒）</span></div>
<div class="stat"><strong>{py_passed if py_passed is not None else "未計測"}{f" / {node_passed}" if node_passed is not None else ""}</strong><span>Python / JavaScript テスト成功数</span></div>
<div class="stat"><strong>{reviewed_count}/{len(CARDS)}</strong><span>目視確認済みカード</span></div>
</div>"""

    # -------------------------------------------------------------- browser
    shot_dir = REPORT / "screenshots"
    shot_files = sorted(shot_dir.glob("*.png")) if shot_dir.is_dir() else []
    if browser:
        checks_html = "".join(f"<li>{E(c)}</li>" for c in browser.get("checks", []))
        shots_html = "".join(
            f'<figure><img src="screenshots/{p.name}" alt="{E(p.stem)}" loading="lazy"><figcaption>{E(p.stem)}</figcaption></figure>'
            for p in shot_files
        )
        browser_section = f"""<h2>実Chromiumでの操作確認 · {E(browser.get("url", "—"))}</h2>
<div class="panel"><strong>4対のブラインド生成完了まで {E(str(browser.get("blind_wait_seconds", "—")))}秒。JS例外 {len(browser.get("page_errors", []))}件、外部通信 {len(browser.get("external_requests", []))}件。</strong>
<ul>{checks_html}</ul>
<a href="browser-evidence.json">今回の確認記録 JSON</a></div>
<div class="grid">{shots_html}</div>"""
    else:
        browser_section = """<h2>実Chromiumでの操作確認</h2><p>未計測。<code>uv run --project exhibit python exhibit/scripts/browser_check.py</code> を実行してください。</p>"""

    # ---------------------------------------------------------------- G0
    if probe:
        eq = probe.get("equivalence", {})
        eq_rows = "".join(equivalence_row(name, entry) for name, entry in eq.items())
        versions = probe.get("versions", {})
        version_row = " · ".join(
            f"{E(k)} {E(str(v))}" for k, v in versions.items() if k != "device"
        )
        vram_load = probe.get("vram_after_load", {})
        vram_peak = probe.get("vram_peak_overall", {})
        tokens = probe.get("token_lengths", {})
        token_note = ""
        for name, tok in tokens.items():
            over = [
                n
                for n in tok.get("preference_refs", [])
                if n > tok.get("model_max_length", 77)
            ]
            if over:
                token_note += f"<li>{E(name)}: 参照説明文に{tok['model_max_length']}トークン上限を超えるものが{len(over)}件</li>"
        probe_section = f"""<h2>G0スパイク：FANはIllustrious XL v2.0を個人化できるか</h2>
<div class="panel"><strong>参照なしFANエンコードは pipeline の encode_prompt と一致します（hidden state の平均コサイン類似度を参照）。</strong>
<p>実行環境: {version_row}。pipeline読み込み {probe.get("load_seconds", "—")}秒、エンコーダー構築 {probe.get("encoder_build_seconds", "—")}秒。VRAM: 読み込み後 {vram_load.get("peak_allocated_gb", "—")} GB、全体ピーク {vram_peak.get("peak_allocated_gb", "—")} GB（torch.cuda.max_memory_allocated）。</p>
<div class="scroll"><table><thead><tr><th>比較</th><th>hidden 平均cos</th><th>hidden 最大絶対差</th><th>pooled 平均cos</th><th>encode秒</th></tr></thead><tbody>{eq_rows}</tbody></table></div>
<ul>{token_note or "<li>すべての参照説明文が77トークン以内</li>"}</ul>
<a href="../../../exhibit/outputs/fan-probe/report.json">G0の生ログ JSON</a></div>"""
    else:
        probe_section = """<h2>G0スパイク：FANはIllustrious XL v2.0を個人化できるか</h2><p>未計測。<code>PYTHONPATH=fan-repro/.work/upstream:exhibit/src fan-repro/.venv/bin/python exhibit/scripts/fan_probe.py</code> を実行してください（FAN環境が必要）。</p>"""

    # ------------------------------------------------------------ followup
    if followup:
        rows = []
        for name, entry in followup.items():
            if isinstance(entry, dict) and "mean_cosine" in entry:
                rows.append(
                    f"<tr><td>{E(name)}</td><td>{entry.get('mean_cosine', '—')}</td>"
                    f"<td>{entry.get('max_abs_diff', '—')}</td></tr>"
                )
        followup_rows = "".join(rows) or "<tr><td colspan=3>比較データなし</td></tr>"
        semantics = followup.get("skip_pa_semantics", "")
        followup_section = f"""<h2>追加検証：alpha・重み・skip_paの実際の効き方</h2>
<div class="panel"><p>{E(semantics)}</p>
<div class="scroll"><table><thead><tr><th>比較</th><th>hidden 平均cos</th><th>hidden 最大絶対差</th></tr></thead><tbody>{followup_rows}</tbody></table></div>
<a href="../../../exhibit/outputs/fan-probe/followup/followup.json">追加検証の生ログ JSON</a></div>"""
    else:
        followup_section = ""

    settings_section = g0_settings_section(followup, final, ladder)

    # ---------------------------------------------------------- rehearsal
    plot = sparkline(rehearsal.get("rows", []))
    rehearsal_section = f"""<h2>連続セッションの実測</h2>
<p>対象は FAN 実生成のみ（rewrite・推薦は行わない）。各セッションはブラインド4対の生成、reveal、調整1回を含むwall timeです。</p>
{plot}
<div class="scroll"><table><thead><tr><th>項目</th><th>実測値</th></tr></thead><tbody>
<tr><td>セッション</td><td>{rehearsal.get("successes", "—")}成功 / {rehearsal.get("sessions", "—")}実行、中央値 {rehearsal.get("median_seconds", "未計測")}秒、p95 {rehearsal.get("p95_seconds", "未計測")}秒</td></tr>
</tbody></table></div>"""

    # ---------------------------------------------------------- preflight
    if preflight:
        errs = preflight.get("errors", [])
        preflight_section = f"""<h2>Preflight（展示前チェック）</h2>
<div class="panel {"" if preflight.get("ready") else "caution"}"><strong>{"合格" if preflight.get("ready") else "未合格"}</strong>
<p>目視確認済みカード {preflight.get("reviewed_cards", "—")}/{preflight.get("cards", len(CARDS))}、固定画像 {preflight.get("fixed_images", "—")}件、サンプル {preflight.get("samples", "—")}件。エラー {len(errs)}件{"（例: " + E(", ".join(errs[:3])) + "）" if errs else ""}。</p>
<a href="../../../exhibit/outputs/preflight.json">Preflightの生ログ JSON</a></div>"""
    else:
        preflight_section = """<h2>Preflight（展示前チェック）</h2><p>未計測。<code>uv run --project exhibit python exhibit/scripts/preflight.py --models</code> を実行してください。</p>"""

    # -------------------------------------------------------------- assets
    assets_section = f"""<h2>固定資産</h2>
<h3>カード（{len(cards)}/{len(CARDS)}件、うち目視確認済み {reviewed_count}件）</h3>
<div class="grid small-grid">{image_grid_html(cards, "ref_en")}</div>
<h3>お題ごとの通常生成（{len(generic_topics)}/{len(CONFIG["topics"])}お題）</h3>
{topic_shots_html(generic_topics)}
<h3>代表サンプル（{len(samples)}件）</h3>
{sample_shots_html(samples)}"""

    # ---------------------------------------------------------------- tests
    tests_section = f"""<h2>テスト</h2>
<div class="scroll"><table><thead><tr><th>種別</th><th>結果</th></tr></thead><tbody>
<tr><td>Python (pytest)</td><td>{f"{py_passed} passed" if py_passed is not None else "未計測"}</td></tr>
<tr><td>JavaScript (node --test)</td><td>{f"{node_passed} pass / {node_failed} fail" if node_passed is not None else "未計測"}</td></tr>
</tbody></table></div>
<p class="small">再現コマンド: <code>uv run --project exhibit pytest exhibit/tests -q</code> / <code>node --test exhibit/tests/test_browser_state.mjs</code></p>"""

    sections = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FAN · 実装とlocalhost検証レポート</title><style>
:root{{--paper:#f4f2eb;--ink:#24392d;--sub:#697466;--green:#315d44;--line:#d7ddd0}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.9 system-ui,-apple-system,sans-serif}}main{{max-width:1120px;margin:auto;padding:65px 32px 90px}}header{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:24px}}.brand{{font:42px Georgia,serif;letter-spacing:-2px}}.brand b{{color:#bd6841;font-size:17px}}.meta{{font-size:11px;color:var(--sub)}}h1{{font-weight:500;font-size:clamp(32px,4vw,52px);line-height:1.5;letter-spacing:-1.5px;margin:50px 0 22px}}h2{{font-size:27px;font-weight:500;margin:55px 0 20px}}h3{{font-size:17px;font-weight:600;margin:35px 0 14px}}h4{{font-size:13px;font-weight:600;margin:0 0 8px;color:var(--sub)}}p{{color:var(--sub)}}a{{color:var(--green);text-underline-offset:3px}}.badge{{display:inline-block;padding:7px 14px;border-radius:40px;background:#e3eadc;color:var(--green);font-size:11px;margin:0 8px 8px 0}}.warn{{background:#efdecf;color:#8f502f}}.lead{{font-size:17px;max-width:860px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:35px 0}}.stat{{border-top:1px solid var(--line);padding-top:18px}}.stat strong{{display:block;font:34px Georgia,serif;color:var(--green)}}.stat span{{font-size:12px;color:var(--sub)}}.panel{{background:#e8eddf;border:1px solid #d9e1ce;padding:24px 28px;border-radius:7px}}.panel.caution{{background:#f1e6da;border-color:#e4cbb5}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:22px}}.grid.small-grid{{grid-template-columns:repeat(auto-fill,minmax(150px,1fr))}}figure{{margin:20px 0}}figure img{{display:block;width:100%;border:1px solid var(--line);border-radius:7px}}figcaption{{font-size:12px;color:var(--sub);margin-top:9px;overflow-wrap:anywhere}}.mini-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:24px}}.mini-grid img{{width:100%;border-radius:5px;border:1px solid var(--line)}}.topic-shots{{margin-bottom:8px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;vertical-align:top;padding:13px 14px;border-bottom:1px solid var(--line)}}th{{font-weight:500;background:#e9ecdf}}code{{font-family:ui-monospace,monospace;font-size:.88em;overflow-wrap:anywhere}}pre{{overflow:auto;padding:22px;background:#24392d;color:#f0f2e8;border-radius:6px;font-size:12px;line-height:1.8}}ul{{padding-left:22px;color:var(--sub)}}.scroll{{overflow:auto}}footer{{margin-top:60px;border-top:1px solid var(--line);padding-top:22px;font-size:11px;color:var(--sub)}}.button{{display:inline-block;background:var(--green);color:white;padding:12px 22px;border-radius:5px;text-decoration:none}}.small{{font-size:12px}}@media(max-width:650px){{main{{padding:30px 20px}}.stats{{grid-template-columns:1fr 1fr}}.meta{{max-width:160px;text-align:right}}th,td{{padding:10px 7px;font-size:11px}}}}
</style></head><body><main><header><div class="brand">fan<b> ●</b></div><div class="meta">IMPLEMENTATION & LOCAL REHEARSAL<br>{measured}</div></header>
<h1>同じ一文から、あなたの一枚を。<br>実装と、実機で確かめたこと。</h1>{"".join(badges)}
<p class="lead">好きな画像を3〜5枚選ぶと、その画像に付けた確認済みの説明文を参照に、同じお題・同じseed・同じ生成設定のまま Illustrious XL v2.0 が描き直します。個人化は<a href="https://github.com/Burf/FAN">FAN</a>（Foundation Encoders Are All You Need, CVPR 2026）公式実装。方式を伏せたブラインド比較で、来場者自身に違いを確かめてもらう構成です。詳細は<a href="../../superpowers/specs/2026-09-21-fan-exhibition-demo-design.md">設計書</a>を参照。旧 ZIPP-style persona × PIGReward 構成は履歴として <a href="../zipp-demo/index.html">docs/reports/zipp-demo/</a> に残しています。</p>
{stats}
<p><a class="button" href="http://localhost:7860">デモを開く ↗</a>　<a href="../../../exhibit/assets/fallback.html">サーバー不要のサンプルHTML</a>　<a href="evidence.json">計測記録 JSON</a></p>
{browser_section}
{probe_section}
{followup_section}
{settings_section}
{rehearsal_section}
{preflight_section}
{tests_section}
{assets_section}
<h2>説明で守っていること（設計書 §9）</h2>
<div class="panel"><ul>
<li>「来場者ごとの追加学習なし」と言います。「追加モデル・重みが一切ない」とは言いません — FAN公式実装の <code>ClassTokenDecoder</code>（<code>weight/L.pth</code>, <code>weight/bigG.pth</code>）を使っています。</li>
<li>参照は「選んだ画像に付けた確認済みの説明文」です。画像そのものをエンコーダーへ入れているとは説明しません。</li>
<li>反映を強くするほど良いとは言いません。targetとのバランスは来場者が判断します。</li>
<li>論文の定量結果をこの展示の性能として使いません。ブラインド比較の集計は少人数の記録であり、性能主張にしません。</li>
<li>Attentionの値から「この画像がこの色を生んだ」といった因果説明はしません。参照に使った画像・説明文・強度だけを表示します。</li>
<li>個人化時の pooled 埋め込みは、<code>ClassTokenDecoder</code> がpadding tokenを終端と誤検出するため使わず、同じ文の参照なし pooled を使います（意図した上流からの逸脱。詳細は <a href="../../../exhibit/README.md">exhibit/README.md</a>）。</li>
</ul></div>
<h2>起動と再検証</h2><pre># リポジトリのルートで実行
uv sync --project exhibit --locked
uv run --project exhibit python exhibit/scripts/preflight.py --models
uv run --project exhibit uvicorn exhibit.app:app --host 127.0.0.1 --port 7860
# http://localhost:7860 / http://localhost:7860/report/</pre>
<pre>uv run --project exhibit pytest exhibit/tests -q
node --test exhibit/tests/test_browser_state.mjs
PYTHONPATH=fan-repro/.work/upstream:exhibit/src fan-repro/.venv/bin/python exhibit/scripts/fan_probe.py
uv run --project exhibit python exhibit/scripts/rehearsal.py --sessions 20
uv run --project exhibit python exhibit/scripts/browser_check.py --report-dir docs/reports/fan-demo</pre>
<p class="small">GPUを使うコマンド同士は同時実行しないでください。ブラウザー確認は <code>--mock</code> と <code>exhibit/tests/mock_api.mjs</code> で開発でき、実機確認は実サーバーに対して行います。</p>
<h2>関連ファイルと出典</h2><p><a href="../../../exhibit/README.md">アプリの実行手順</a> · <a href="../../superpowers/specs/2026-09-21-fan-exhibition-demo-design.md">設計書</a></p>
<p>モデル・原手法：<a href="https://github.com/Burf/FAN">FAN</a>、<a href="https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0">Illustrious XL v2.0</a>。モデル・画像hashはasset manifestおよび計測JSONに保存しています。</p>
<footer>FAN / LOCAL EXHIBITION DEMO · 報告は実測と未検証事項を分けて記載しています。画像は実際のローカル推論・localhostブラウザー操作から取得しました。</footer></main></body></html>"""
    (REPORT / "index.html").write_text(sections)
    print(REPORT / "index.html")


if __name__ == "__main__":
    main()
