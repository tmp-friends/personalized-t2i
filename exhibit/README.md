# Taste — あなたの「好き」を描く

固定5対のキャラクターイラストから好みを確かめ、訂正した好みでIllustrious XL v2.0のキャラクターイラストを生成するローカル展示デモ。
設計は [ZIPP-style persona × PIGReward](../docs/superpowers/specs/2026-09-19-zipp-pigreward-exhibition-demo-design.md)。

## 起動

リポジトリのルートから実行します。

```bash
uv sync --project exhibit --locked
uv run --project exhibit python exhibit/scripts/preflight.py --models
uv run --project exhibit uvicorn exhibit.app:app --host 127.0.0.1 --port 7860
```

ブラウザーで **http://localhost:7860** を開きます。キャラクター版の検証記録は [HTML report](../docs/reports/zipp-demo/characters/index.html)、サーバー経由では http://localhost:7860/report/characters/ 。[初期Illustrious版の記録](../docs/reports/zipp-demo/illustrious/index.html)も保存しています。[旧SDXL構成の検証記録](../docs/reports/zipp-demo/index.html)も保存しています。

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
- ローカルLLMが承認済みの美的表現を並べる制約付き書き換え。元のお題とキャラクターの描画指定を文字列として固定し、Illustrious XLの両tokenizer上限を検査。書き換え失敗時は通常画像だけを表示。
- 通常も個人化も同じローカルLLM・同じtemplate・同じ長さ上限で書き換え。個人化により語数が変わる点は制約として残ります。
- PIGRewardは実checkpoint用adapterを実装済みですが、G0の採用条件を満たすまでは推薦無効です。理由やwinnerの欠損を補って推薦を作りません。
- 学習済みモデルの公式スコアをこの展示の満足度と扱いません。

## 生成モデルとモチーフ

[Illustrious XL v2.0-STABLE](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0)（revision `69459c1fe6f46db41ab31e6114f05acc0e06bcaa`）を使用します。1024×1024・28 steps・Euler ancestral・CFG 5.0・fp16で生成します。単一safetensorsを `from_single_file` で読み込み、構成ファイルとtokenizerだけを初期Illustriousの固定revisionから読みます。推論時は全てローカルファイルを使用します。コミュニティガイドの「Clip skip 2」はWebUI側の表記であり、使用中のDiffusers SDXLは既定でpenultimate hidden stateを使います。`clip_skip=1`/`2`の直指定は実画像比較で出力を破綻させたため、追加の`clip_skip`引数は渡しません。

お題は「窓辺で猫と過ごす少女」「東京の夜景と青年」「森を旅する魔法使い」「海辺の灯台と船乗り」「雨の街角の少女」「カフェで迎える店員」の6件です。人物・衣装・場面を基本プロンプトに固定し、色・光・構図・描画表現・雰囲気を好みに合わせます。髪型や顔立ちの好み推定と、画像間で同一キャラクターを厳密に維持する機能はありません。

公式モデルカードのタグ・生成設定を出発点に、実画像を見てプロンプトを調整しています。[プロンプトの出典と設計根拠](../docs/reports/zipp-demo/characters/prompt-notes.md)を参照してください。通常側と個人化側のモデル・設定・seedは一致させます。選択対もキャラクター中心で、外見や小物の変化は残るため各軸が完全に独立した比較とは扱いません。

モデル・お題・生成設定を変えた際は、選択10枚・通常24枚・サンプル24枚、VLM根拠、オフラインHTMLをすべて再生成します。旧モデルの測定結果はv2.0の性能値として扱いません。

## 固定assetの準備

重みは `configs/demo.json` / `pigreward-repro/configs/model.json` の固定revisionを準備時に取得します。起動時downloadは行いません。画像・根拠は `assets/manifest.json` に出典、hash、生成条件を保持します。Illustriousの取得例（リポジトリルートで一度だけ実行）:

```bash
tailored-visions-repro/.venv/bin/python - <<'PY'
import json
from huggingface_hub import hf_hub_download, snapshot_download
settings = json.load(open("exhibit/configs/demo.json"))["generation"]
hf_hub_download(
    settings["model"], filename=settings["checkpoint"], revision=settings["revision"]
)
config = settings["pipeline_config"]
snapshot_download(
    config["model"], revision=config["revision"],
    allow_patterns=["model_index.json", "scheduler/*.json", "unet/config.json",
                    "vae/config.json", "text_encoder*/config.json", "tokenizer/*", "tokenizer_2/*"],
)
PY
```

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
uv run --project exhibit python exhibit/scripts/browser_check.py --report-dir docs/reports/zipp-demo/characters
uv run --project exhibit python exhibit/scripts/pigreward_probe.py --smoke
uv run --project exhibit python exhibit/scripts/pigreward_probe.py
uv run --project exhibit python exhibit/scripts/rehearsal.py --sessions 20
```

ブラウザー試験は既存Chromiumを `EXHIBIT_CHROMIUM` で指定できます。初期値はこの実機のキャッシュパス。GPUを使うコマンド同士は同時実行しないでください。

第三者5人での理解度確認と2時間連続稼働は別の展示受入作業です。実施済みの内容・計測範囲・残項目はHTMLレポートに記載します。

旧SDXL構成について、2026-09-20に中断後の再確認を実施。実ブラウザーで4枚生成から最終選択・リセットまで20.88秒で成功し、子プロセスのMemoryError／SIGKILLでも完成画像の保持と次sessionへの復帰を確認しました。申告されたOOMの原因自体は未特定です。詳細は[再確認記録](../docs/reports/zipp-demo/recovery-evidence.json)を参照してください。

旧Illustrious初期版（v0）の記録：2026-09-20にSDXLから切替。固定画像58枚と根拠を更新し、Python 40件・JavaScript 4件、18条件の実LLM書き換え、起動前検査が成功しました。実ブラウザーで4枚生成を20.87秒で確認し、訂正・キャッシュ・全OFF・リセット・サンプル表示も通過しました。画像内容の残る制約を含め、[Illustrious検証記録](../docs/reports/zipp-demo/illustrious/index.html)を参照してください。

2026-09-20にキャラクター版へ更新し、Illustrious XL v2.0-STABLEを採用。全58枚を再生成・目視確認し、VLM根拠とオフラインHTMLを更新しました。Python 46件・JavaScript 4件、18条件の実LLM書き換え、起動前検査が成功しました。実ブラウザーの4枚生成は24.4秒で、訂正・キャッシュ・全OFF・リセット・サンプル表示も確認しました。検証範囲と全画像は[キャラクター版の記録](../docs/reports/zipp-demo/characters/index.html)を参照してください。

2026-09-21にIllustrious v2.0向け生成プロファイルをv4へ更新。品質タグを先頭、`absurdres, highres`を末尾へ統一し、negative promptを整理、CFGを5.0へ変更して全58枚を再生成しました。WebUIの「Clip skip 2」をDiffusersへ直訳すると出力が破綻することを同一prompt・seedで確認し、SDXL既定のpenultimate hidden stateを採用しています。Python 52件・JavaScript 4件、実LLM書き換え18/18件、preflight、実ブラウザー全14項目が成功し、4枚生成は24.4秒、外部通信とJS例外は0件でした。
