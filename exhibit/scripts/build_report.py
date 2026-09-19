#!/usr/bin/env python3
"""Build the handoff HTML from actual local measurement artifacts."""

import html
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from exhibit.config import OUTPUTS, REPO, read_json, write_json

REPORT = REPO / "docs/reports/zipp-demo"
E = html.escape


def main():
    browser = read_json(REPORT / "browser-evidence.json", {})
    bench = read_json(OUTPUTS / "rehearsal.json", {})
    pig = read_json(OUTPUTS / "pigreward-g0.json", {})
    preflight = read_json(OUTPUTS / "preflight.json", {})
    tests = (OUTPUTS / "all-tests.log").read_text()
    count = re.search(r"(\d+) passed", tests).group(1)
    p95 = bench.get("p95_seconds", "計測中")
    rows = bench.get("rows", [])
    metrics = [m for r in rows for m in r.get("metrics", [])]
    gpu = {
        stage: max(
            (m["peak_vram_mib"] for m in metrics if m["stage"] == stage), default=0
        )
        for stage in ["rewrite", "generate"]
    }
    recovery = read_json(REPORT / "recovery-evidence.json", {})
    cancel_check = read_json(OUTPUTS / "live-cancel.json", {})
    rewrite_check = read_json(OUTPUTS / "rewrite-sweep.json", {})
    measured = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M JST")
    evidence = {
        "generated_at": measured,
        "browser": {k: v for k, v in browser.items() if k != "run"},
        "rehearsal": bench,
        "pigreward": {k: v for k, v in pig.items() if k != "records"},
        "preflight": preflight,
        "recovery_verification": recovery,
        "live_gpu_cancel": cancel_check,
        "rewrite_sweep": {k: v for k, v in rewrite_check.items() if k != "events"},
        "tests": {
            "python_passed": int(count),
            "node_passed": 4,
            "warnings": "Starlette/httpx and anyio deprecation warnings (2)",
        },
        "mode": "ZIPP-style generation + manual selection",
        "limitations": [
            "PIGReward G0 not passed",
            "persona uses checked-evidence summary",
            "no third-party usability test",
            "no two-hour soak",
        ],
    }
    write_json(REPORT / "evidence.json", evidence)
    write_json(REPORT / "pigreward-g0.json", pig)
    recovery_section = ""
    if recovery:
        recovery_section = f"""<h2>中断後の再確認 · {E(recovery['verified_at'])}</h2>
<div class="panel"><strong>起動し直したサーバーで、実生成から終了まで確認しました。</strong>
<p>今回のブラウザー確認は {recovery['browser_seconds']}秒。5回答・訂正・4枚生成・本人の選択・再読込・全OFF・リセットを通過し、JS例外と外部通信は0件でした。終了後のGPU使用量は {recovery['gpu_idle_mib']} MiBです。</p>
<p>実際の子プロセスに限定したメモリ上限によるMemoryErrorとSIGKILLの2条件を追加検証しました。どちらも完成画像を保持し、誤った推薦を出さず、子プロセス終了・リセット後に次の体験で4枚生成できました。モデル実行部は小さなテスト用プロセスに置き換え、排他制御・JSON入出力・セッション処理は実装本体を使っています。GPUやホスト全体のメモリを使い切る試験ではありません。</p>
<p>申告されたOOMの原因は未特定です。保存済み推論ログにはOOMの記録がなく、カーネルログは取得できませんでした。初回の再確認中には以前のサーバーが終了し、最終選択の検査がタイムアウトしました。再起動後の同じ検査は成功しています。以前の20セッション計測と、今回の1回の実ブラウザー確認は別の記録です。</p>
<a href="recovery-evidence.json">今回の確認記録 JSON</a></div>"""
    if rows:
        points = " ".join(
            f"{30 + i * 34},{150 - r['seconds'] * 4}" for i, r in enumerate(rows)
        )
        plot = (
            f'<svg viewBox="0 0 740 180" role="img" aria-label="20セッションの実生成時間"><path d="M25 10V155H725" fill="none" stroke="#c6cdbf"/><path d="M25 70H725" stroke="#d9ded1"/><text x="27" y="66" fill="#6e796a" font-size="10">20秒</text><polyline points="{points}" fill="none" stroke="#346347" stroke-width="3"/>'
            + "".join(
                f'<circle cx="{30 + i * 34}" cy="{150 - r["seconds"] * 4}" r="3" fill="#346347"/>'
                for i, r in enumerate(rows)
            )
            + '<text x="27" y="177" font-size="10" fill="#6e796a">SESSION 01</text><text x="642" y="177" font-size="10" fill="#6e796a">SESSION 20</text></svg>'
        )
    else:
        plot = "<p>セッション計測中。</p>"
    sections = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Taste · 実装とlocalhost検証レポート</title><style>
:root{{--paper:#f4f2eb;--ink:#24392d;--sub:#697466;--green:#315d44;--line:#d7ddd0}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.9 system-ui,-apple-system,sans-serif}}main{{max-width:1120px;margin:auto;padding:65px 32px 90px}}header{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:24px}}.brand{{font:42px Georgia,serif;letter-spacing:-2px}}.brand b{{color:#bd6841;font-size:17px}}.meta{{font-size:11px;color:var(--sub)}}h1{{font-weight:500;font-size:clamp(32px,4vw,52px);line-height:1.5;letter-spacing:-1.5px;margin:50px 0 22px}}h2{{font-size:27px;font-weight:500;margin:55px 0 20px}}h3{{font-size:17px;font-weight:600}}p{{color:var(--sub)}}a{{color:var(--green);text-underline-offset:3px}}.badge{{display:inline-block;padding:7px 14px;border-radius:40px;background:#e3eadc;color:var(--green);font-size:11px;margin:0 8px 8px 0}}.warn{{background:#efdecf;color:#8f502f}}.lead{{font-size:17px;max-width:860px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:35px 0}}.stat{{border-top:1px solid var(--line);padding-top:18px}}.stat strong{{display:block;font:42px Georgia,serif;color:var(--green)}}.stat span{{font-size:12px;color:var(--sub)}}.panel{{background:#e8eddf;border:1px solid #d9e1ce;padding:24px 28px;border-radius:7px}}.caution{{background:#f1e6da;border-color:#e4cbb5}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}figure{{margin:20px 0}}figure img{{display:block;width:100%;border:1px solid var(--line);border-radius:7px}}figcaption{{font-size:12px;color:var(--sub);margin-top:9px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;vertical-align:top;padding:13px 14px;border-bottom:1px solid var(--line)}}th{{font-weight:500;background:#e9ecdf}}code{{font-family:ui-monospace,monospace;font-size:.88em;overflow-wrap:anywhere}}pre{{overflow:auto;padding:22px;background:#24392d;color:#f0f2e8;border-radius:6px;font-size:12px;line-height:1.8}}.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:25px 0}}.flow div{{padding:17px 13px;border:1px solid var(--line);border-radius:5px;font-size:12px}}.flow b{{display:block;font:24px Georgia,serif;color:#8b9c7e}}details{{border-bottom:1px solid var(--line);padding:16px 0}}summary{{cursor:pointer;color:var(--green)}}.small{{font-size:12px}}ul{{padding-left:22px;color:var(--sub)}}.scroll{{overflow:auto}}footer{{margin-top:60px;border-top:1px solid var(--line);padding-top:22px;font-size:11px;color:var(--sub)}}.button{{display:inline-block;background:var(--green);color:white;padding:12px 22px;border-radius:5px;text-decoration:none}}@media(max-width:650px){{main{{padding:30px 20px}}.stats{{grid-template-columns:1fr 1fr}}.grid{{grid-template-columns:1fr}}.flow{{grid-template-columns:1fr}}.meta{{max-width:160px;text-align:right}}th,td{{padding:10px 7px;font-size:11px}}}}
</style></head><body><main><header><div class="brand">taste<b> ●</b></div><div class="meta">IMPLEMENTATION & LOCAL REHEARSAL<br>{measured}</div></header>
<h1>あなたの「好き」を描く。<br>実装と、実機で確かめたこと。</h1><span class="badge">localhost 実動作確認済み</span><span class="badge">RTX 4090 / ローカル実生成</span><span class="badge warn">PIGReward 推薦は無効</span>
<p class="lead">5回の画像選択から好みを確認・訂正し、通常4枚と個人化4枚を同じseedで比較する展示デモを実装しました。最後の一枚は本人が選べます。Chromiumで実際の画面を操作し、SDXLの実生成、再読込、キャッシュ、リセットまで検証しています。</p>
<div class="stats"><div class="stat"><strong>{bench.get("successes", "—")}/{bench.get("sessions", "—")}</strong><span>実生成セッション成功</span></div><div class="stat"><strong>{p95}<small style="font-size:18px"> s</small></strong><span>生成時間 p95 / 推薦なし</span></div><div class="stat"><strong>{count} + 4</strong><span>Python + JavaScript テスト</span></div><div class="stat"><strong>58</strong><span>固定画像・サンプル画像</span></div></div>
<p><a class="button" href="http://localhost:7860">デモを開く ↗</a>　<a data-server="/fallback" href="../../../exhibit/assets/fallback.html">サーバー不要のサンプルHTML</a>　<a href="evidence.json">計測記録 JSON</a></p>
<figure><img src="screenshots/05-result.png" alt="localhostで実生成した通常4枚と個人化4枚の比較画面"><figcaption>実操作時の比較画面。上段が通常、下段が訂正した好みによる実生成。上下の同じ位置は同じseedです。これはスタッフ検証用の選択履歴であり、一般の来場者データではありません。</figcaption></figure>
{recovery_section}
<h2>できるようになったこと</h2><div class="flow"><div><b>01</b>左右をランダム化した5対<br>決められない場合はskip</div><div><b>02</b>根拠付きの好み<br>項目OFF・候補への訂正</div><div><b>03</b>6つのお題<br>元のお題を保った書き換え</div><div><b>04</b>通常4枚＋個人化4枚<br>同じseed・同じ生成条件</div><div><b>05</b>本人による最終選択<br>終了で一時データ削除</div></div>
<div class="grid"><figure><img src="screenshots/02-choice.png" alt="二択の画面"><figcaption>選ばなかった画像を「嫌い」と扱いません。有効選択が3回に達しなければ未選択対を再提示します。</figcaption></figure><figure><img src="screenshots/03-persona.png" alt="好みを訂正する画面"><figcaption>美的な好みのみを扱います。根拠の画像を開いて確認でき、性格・年齢・職業を推定しません。</figcaption></figure></div>
<div class="panel"><strong>OFF・訂正は、生成と推薦の共通入力に反映。</strong><p>EffectivePreferenceContextを確定し、同じhashを使います。OFFされた軸の根拠、訂正前の根拠、未加工VLM出力を下流へ渡しません。すべてOFFなら通常画像だけを表示し、GPU推論を省略します。</p></div>
<h2>localhostで確認した操作</h2><div class="scroll"><table><thead><tr><th>検証</th><th>結果</th><th>確認したこと</th></tr></thead><tbody>
<tr><td>一連の体験</td><td>成功</td><td>5回答 → 色を訂正・光をOFF → 窓辺の猫 → 実生成4枚 → 本人の選択</td></tr>
<tr><td>6お題×3代表履歴</td><td>18/18成功</td><td>実LLM書き換えで元のお題の固定とSDXL両tokenizer上限を確認。全18条件の画像品質評価ではありません</td></tr><tr><td>画像と条件</td><td>成功</td><td>8枚が表示され、上下4組のseed一致、個人化画像のcontext hash一致</td></tr>
<tr><td>再読込</td><td>成功</td><td>進行中sessionと最終選択を復元。訂正項目も保持。全skip後は未回答画面へ復帰</td></tr>
<tr><td>キャッシュ / 全OFF</td><td>成功</td><td>同じsession・条件ではexact-cache表示。全OFFは通常4枚のみ</td></tr>
<tr><td>終了とアクセス</td><td>成功</td><td>旧sessionと旧生成画像URLが404。一時ディレクトリを削除</td></tr>
<tr><td>実GPUでの途中リセット</td><td>成功</td><td>1枚完成後にSDXLを停止。GPU解放まで0.159秒、残留VRAM 1 MiB。旧画像404、次session開始可能</td></tr><tr><td>異常時</td><td>テスト成功</td><td>子プロセスtimeout・cancel・GPU lease、古い応答、誤った画像ID、推薦解析失敗</td></tr>
<tr><td>表示と通信</td><td>成功</td><td>1440px / 390pxで確認。横はみ出しなし。JS例外0件、外部リクエスト0件</td></tr>
<tr><td>サンプル</td><td>成功</td><td>代表履歴であることを明示。単独HTMLは画像を埋め込み、サーバーが不要</td></tr>
</tbody></table></div>
<h2>実測した時間とメモリ</h2><p>対象は <b>ZIPP-style書き換え＋SDXLの4枚生成</b>。各stageのモデル読込・IPC・保存・終了を含むwall timeです。PIGRewardによる推薦時間、来場者の操作時間は含みません。1回のブラウザー操作では {browser.get("browser_wall_seconds", "—")}秒でした。</p>{plot}
<div class="scroll"><table><thead><tr><th>項目</th><th>実測値 / 設定</th></tr></thead><tbody><tr><td>20セッション</td><td>{bench.get("successes", "—")}成功 / {bench.get("sessions", "—")}実行、中央値 {bench.get("median_seconds", "—")}秒、nearest-rank p95 {p95}秒</td></tr><tr><td>LLM最大割当VRAM</td><td>{gpu["rewrite"]:.1f} MiB</td></tr><tr><td>SDXL最大割当VRAM</td><td>{gpu["generate"]:.1f} MiB</td></tr><tr><td>生成条件</td><td>SDXL base / EulerDiscreteScheduler / 1024×1024 / 20 steps / CFG 7.0 / fp16 / negative promptなし / batch 1</td></tr><tr><td>プロセス切替</td><td>同時常駐なし。flockで排他し、子プロセスをwaitしてから次のstageを開始</td></tr></tbody></table></div><p class="small">VRAMはtorch.cuda.max_memory_allocatedの最大値。ドライバー全体の占有量とは異なります。準備時にSDXL34枚の連続生成は約100.9秒、Qwen画像解析10方向は約12.7秒でした。</p>
<h2>PIGRewardの採用判定</h2><div class="panel caution"><strong>実モデルは動作しましたが、今回のadapterではライブ推薦を採用できません。</strong><p>公開重みを固定revisionで取得し、6つの代表体験から20組を作って左右反転を含む40入力で検証しました。正方向で有効に解析できた比較は <b>{pig.get("forward_parseable", "—")}/20</b>、有効な左右一致は <b>{pig.get("reverse_consistent", "—")}/20</b>。設計書の各18/20以上を満たさないため、推薦を無効化しています。</p><p>観察された出力には同じ点数の反復、軸別理由の欠落、独自形式、打切りがありました。これをPIGReward一般の精度とは扱いません。専用タスク指示の不確実性を含む、この実装の接続試験結果です。</p><a href="pigreward-g0.json">実入力の識別子・モデル版・全生出力・判定を確認する</a></div>
<h2>実装上の判断と残っている検証</h2><ul><li>既存workspaceの新規feature branchで実装し、ユーザーが編集中だった設計書を保持しました。</li><li>Personaは、実際のVLM解析から確認した根拠節を集計する簡易方式です。オンラインLLMによる自由なpersona文章生成は未採用です。</li><li>プロンプトは被写体等を守るため、元のお題を固定し、美的表現を制約付きで補足します。原ZIPPのReddit・GATを再現するものではありません。</li><li>PIGReward bootstrapは未使用。Qwen3.5-4Bで両方向を事前解析し、context_source=generic_vlmとして記録しています。</li><li>独立レビューの4件（古いブラウザー応答、skip後の再読込、サンプル設定照合、推薦の矛盾検出）を回帰テストで修正しました。未加工VLM出力をcontextに含めない追加検証も行いました。</li><li>第三者5人の理解度確認、2時間連続稼働、画像・根拠の展示担当者による最終確認は未実施です。</li><li>今回のp95はこの実機・この縮小モードの値です。PIGReward統合版や別GPUの待ち時間を保証しません。</li></ul>
<h2>起動と再検証</h2><pre>cd /home/tomoya/stable-diffusion/personalized-t2i
uv sync --project exhibit --locked
uv run --project exhibit python exhibit/scripts/preflight.py --models
uv run --project exhibit uvicorn exhibit.app:app --host 127.0.0.1 --port 7860
# http://localhost:7860 / http://localhost:7860/report/</pre>
<pre>uv run --project exhibit pytest exhibit/tests pigreward-repro/tests tests -q
node --test exhibit/tests/test_browser_state.mjs
uv run --project exhibit python exhibit/scripts/browser_check.py
uv run --project exhibit python exhibit/scripts/rehearsal.py --sessions 20</pre>
<p class="small">PythonテストにはStarlette/httpx・anyioの非推奨警告が2件あります。失敗はありません。GPU環境は既存Tailored Visionsの.venvを使用し、EXHIBIT_GPU_PYTHONで変更できます。アプリの依存関係はuv.lockで固定しました。</p>
<details><summary>画面キャプチャをさらに見る</summary><div class="grid">{"".join(f'<figure><img src="screenshots/{name}" alt="{label}" loading="lazy"><figcaption>{label}</figcaption></figure>' for name, label in [("01-welcome.png", "開始画面"), ("04-topics.png", "6つのお題"), ("06-selection.png", "本人の最終選択"), ("07-mobile.png", "390pxの表示"), ("08-all-off.png", "全項目OFF"), ("09-skips.png", "全skipからの再選択"), ("10-sample.png", "サンプル表示")])}</div></details>
<h2>関連ファイルと出典</h2><p><a data-server="/reference/readme" href="../../../exhibit/README.md">アプリの実行手順</a> · <a data-server="/reference/pigreward" href="../../../pigreward-repro/docs/DEVIATIONS.md">PIGRewardとの差分</a> · <a data-server="/reference/spec" href="../../superpowers/specs/2026-09-19-zipp-pigreward-exhibition-demo-design.md">設計書</a> · <a data-server="/reference/plan" href="../../superpowers/plans/2026-09-19-zipp-exhibition-demo.md">実装計画と記録</a></p><p>モデル・原手法：<a href="https://behavior-in-the-wild.github.io/zipp.html">ZIPP</a>、<a href="https://huggingface.co/jeongeunnn/pigreward">PIGReward evaluator</a>、<a href="https://huggingface.co/Qwen/Qwen3.5-4B">Qwen3.5-4B</a>、<a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0">SDXL base</a>。モデル・template・画像hashはasset manifestおよび計測JSONに保存しています。</p><footer>TASTE / LOCAL EXHIBITION DEMO · 報告は実測と未検証事項を分けて記載しています。画像は実際のローカル推論・localhostブラウザー操作から取得しました。</footer></main><script>if(location.protocol!=="file:")for(const a of document.querySelectorAll("a[data-server]"))a.href=a.dataset.server;</script></body></html>"""
    (REPORT / "index.html").write_text(sections)
    print(REPORT / "index.html")


if __name__ == "__main__":
    main()
