# Taste — あなたの「好き」を描く

固定5対の選択から好みを確かめ、訂正した好みで画像を生成するローカル展示デモ。
設計は [ZIPP-style persona × PIGReward](../docs/superpowers/specs/2026-09-19-zipp-pigreward-exhibition-demo-design.md)。

## 起動

リポジトリのルートから実行します。

```bash
uv sync --project exhibit --locked
uv run --project exhibit python exhibit/scripts/preflight.py --models
uv run --project exhibit uvicorn exhibit.app:app --host 127.0.0.1 --port 7860
```

ブラウザーで **http://localhost:7860** を開きます。作業結果は [HTML report](../docs/reports/zipp-demo/index.html)、サーバー経由では http://localhost:7860/report/ 。

GPU推論は既存の `tailored-visions-repro/.venv/bin/python` を別プロセスで使います。別環境を使う場合は `EXHIBIT_GPU_PYTHON=/absolute/path/to/python` を設定してください。参照環境は PyTorch 2.14.0 / Transformers 5.16.1 / Diffusers 0.40.0 / Accelerate 1.14.0 / Pillow 12.3.0。既存環境は変更していません。

アプリは1プロセスで起動してください。`--workers` は指定しません。モデル・初期画像が準備済みならインターネット不要です。推論時はHF/Transformersのoffline modeを強制します。

## 操作

1. 5対を選択。「決められない」は勝敗に変換しません。3回未満の選択なら未選択対を再表示します。
2. 根拠を確認し、各項目をOFF／候補値へ訂正します。
3. 6お題の1つを選択。通常4枚を表示し、個人化4枚を1枚ずつ実生成します。
4. 好きな画像、または「しっくりくる画像はなかった」を選べます。
5. 終了／90秒の無操作で一時データを削除します。実行中は無操作リセットを停止し、120秒の処理期限を適用します。

「中止して好みを直す」はワーカーを終了させて編集へ戻ります。終了確認前の新規GPU処理は受け付けません。同じsession・context・モデル・生成条件は「同一条件のキャッシュ」として再利用します。キャッシュも体験終了時に消えます。

## 現在の採用モード

**ZIPP-style実生成 + 本人の手動選択**。原ZIPPのReddit/GATを再現していません。

- Qwen3.5-4Bによる固定10方向のVLM解析を実施。確認した軸別の根拠を集計するpersona簡易表示を採用しています。自由文personaのオンライン生成は未採用です。
- ローカルLLMが承認済みの美的表現を並べる制約付き書き換え。元のお題は文字列として固定し、SDXLの両tokenizer上限を検査。書き換え失敗時は通常画像だけを表示。
- 通常も個人化も同じローカルLLM・同じtemplate・同じ長さ上限で書き換え。個人化により語数が変わる点は制約として残ります。
- PIGRewardは実checkpoint用adapterを実装済みですが、G0の採用条件を満たすまでは推薦無効です。理由やwinnerの欠損を補って推薦を作りません。
- 学習済みモデルの公式スコアをこの展示の満足度と扱いません。

## 固定assetの準備

重みは `configs/demo.json` / `pigreward-repro/configs/model.json` の固定revisionを準備時に取得します。起動時downloadは行いません。画像・根拠は `assets/manifest.json` に出典、hash、生成条件を保持します。

```bash
uv run --project exhibit python exhibit/scripts/prepare.py base
uv run --project exhibit python exhibit/scripts/prepare.py analyze
# 再生成時は実画像とraw VLM出力を目視し、CLAUSESの対応を確認してから実行
uv run --project exhibit python exhibit/scripts/curate_evidence.py
uv run --project exhibit python exhibit/scripts/prepare.py samples
uv run --project exhibit python exhibit/scripts/build_fallback.py
uv run --project exhibit python exhibit/scripts/preflight.py --models
```

`assets/fallback.html` は実画像を埋め込んだ単独HTMLです。サーバーが停止していてもファイルを開けます。6体験は代表履歴のサンプルで、今の回答を反映した画像とは表示しません。

固定展示画像と6サンプルは配布用assetとしてGit対象。モデルとセッション一時生成物は `artifacts/` / `outputs/` 配下でGit対象外です。リハーサルではスタッフの操作結果のみを明示的にレポートへ保存します。

## 検証

```bash
uv run --project exhibit pytest exhibit/tests pigreward-repro/tests tests -q
uv run --project exhibit python exhibit/scripts/browser_check.py
uv run --project exhibit python exhibit/scripts/pigreward_probe.py --smoke
uv run --project exhibit python exhibit/scripts/pigreward_probe.py
uv run --project exhibit python exhibit/scripts/rehearsal.py --sessions 20
```

ブラウザー試験は既存Chromiumを `EXHIBIT_CHROMIUM` で指定できます。初期値はこの実機のキャッシュパス。GPUを使うコマンド同士は同時実行しないでください。

第三者5人での理解度確認と2時間連続稼働は別の展示受入作業です。実施済みの内容・計測範囲・残項目はHTMLレポートに記載します。

2026-09-20に中断後の再確認を実施。実ブラウザーで4枚生成から最終選択・リセットまで20.88秒で成功し、子プロセスのMemoryError／SIGKILLでも完成画像の保持と次sessionへの復帰を確認しました。申告されたOOMの原因自体は未特定です。詳細は[再確認記録](../docs/reports/zipp-demo/recovery-evidence.json)を参照してください。
