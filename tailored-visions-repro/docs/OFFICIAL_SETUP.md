# Tailored Visions 公式実装のローカルセットアップ

- 公式 repository: <https://github.com/zzjchen/Tailored-Visions>
- 固定 commit: `d0f4454ca08c68c5d30f08a01ff4a23a8b33b610`
- 論文: <https://arxiv.org/abs/2310.08129>

公式 source は `upstream/` submodule に固定し、直接変更しない。互換性修正は `patches/series`、
実行用 source は `scripts/prepare_upstream.py` が生成する `.work/upstream/` に置く。

## セットアップ

```bash
cd ~/stable-diffusion/personalized-t2i
git submodule update --init --recursive
cd tailored-visions-repro
uv sync --locked
uv run python scripts/prepare_upstream.py
```

PIP dataset は `data/raw/user_data/` に置く。未取得なら次を実行する。

```bash
uv run python scripts/00_download_data.py
```

## Local LLM で実行

`scripts/run_official.sh` は local OpenAI-compatible server と公式 entry point を外側から管理する。

```bash
scripts/run_official.sh demo 'a cat'              # rewrite + images
scripts/run_official.sh demo 'a cat' --no-t2i     # rewrite only
scripts/run_official.sh main --limit_users=5      # bounded PIP run
scripts/run_official.sh main --limit_users=5 --t2i
scripts/run_official.sh serve                      # foreground server
scripts/run_official.sh stop                       # resident server を停止
```

出力は `outputs/official/` に集約される。demo では dataset への ignored symlink を同ディレクトリに作り、
公式 source の相対パス前提を保つ。main には `--data_folder` の絶対パスを渡す。

local server は `Qwen/Qwen3.5-4B` を既定とし、`127.0.0.1` のみで待受する。認証はないため
同一マシンの他ユーザーからはアクセス可能である。background 起動後は約 9 GB の VRAM を保持するので、
使用後は必ず `scripts/run_official.sh stop` を実行する。

## OpenAI API で実行

論文の設定へ近づける場合は local server を使わず、patch 適用済み source を直接実行する。

```bash
export TV_OPENAI_KEY=sk-...
export TV_OPENAI_MODEL=gpt-4o-mini
unset TV_OPENAI_BASE
export PYTHONPATH="$PWD/.work/upstream"

cd outputs/official
../../.venv/bin/python ../../.work/upstream/demo.py --input_prompt='a cat' --no-t2i
```

paper の 2023 年版 `gpt-3.5-turbo` と現在利用できる model は同一ではない。rewriter は手法の一部なので、
model が変われば評価値も変わる。

## Patch と 24 GB GPU 対応

3 patches の正確な対象ファイル、理由、検証コマンドは [DEVIATIONS.md](DEVIATIONS.md) に記載する。
主な内容は API の bounded retry、retrieval / BM25 / ICL の runtime fixes、current diffusers / SDXL loader、
`--data_folder` / `--limit_users` である。

SDXL と local LLM を同時に載せるため、loader は空き VRAM に応じて attention slicing または
CPU offload を使う。`run_official.sh` は expandable CUDA allocator を有効にする。

## 制約

- PIP dataset の元画像 URL は失効している。
- 公式 repository に論文評価を一括再現する script は含まれない。
- local LLM を使う結果は paper の ChatGPT result と直接比較できない。
- `.work/`、`data/raw/`、`outputs/` は再生成・再取得可能なため Git 管理外。
