# PIGReward evaluator adapter

公開 checkpoint `jeongeunnn/pigreward` を呼び出す独自 adapter です。公式コードの再現を称するものではありません。`configs/model.json` に固定revisionと実行設定を記録しています。

- 画像2枚と元のお題、有効な選好contextだけを渡します。
- 公開モデルカード形式の軸別理由・点数・合計・明示的winner・EOSを検査します。
- 同点、数値矛盾、切断、解析不能は推薦なし。4枚のトーナメントは最大3比較で勝者だけを返します。
- 当面は `live_approved=false`。20組の解析率と左右反転一致率を確認するまでライブ展示の推薦は無効です。
- 原bootstrapは使用せず、固定画像の事前解析は別のローカルVLMを使用します。

## 実行とテスト

アプリ側の軽量環境は `uv sync --project exhibit --locked`。GPU側は既存の `tailored-visions-repro/.venv/bin/python` を共用し、プロセスを分離します。

```bash
uv run --project exhibit pytest pigreward-repro/tests -q
uv run --project exhibit python exhibit/scripts/pigreward_probe.py
```

重みの取得は準備時だけ実施します。`artifacts/` はGit対象外です。CLIは取得済みのローカル重みしか読みません。

出典: [公開モデル](https://huggingface.co/jeongeunnn/pigreward)、[モデルカード](https://huggingface.co/jeongeunnn/pigreward/blob/main/README.md)。配布カードはApache-2.0を表示しています。独自実装との差分は [DEVIATIONS](docs/DEVIATIONS.md) を参照。
