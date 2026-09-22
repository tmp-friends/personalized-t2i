# パーソナライズ × 画像生成

数枚の評価や prompt 履歴からユーザーの美的嗜好を推定し、画像生成へ反映する
preference personalization の公式実装・再現・比較をまとめた repository。
人物や物体そのものを再現する subject personalization（DreamBooth / LoRA 系）とは異なり、
3 手法とも「好みを推定して反映する」ことを対象とする。

## 手法一覧

| フォルダ | 論文 | 介入する層 | 公式実装 | サマリー |
|---|---|---|---|---|
| [`tailored-visions-repro/`](tailored-visions-repro/) | Tailored Visions（CVPR 2024）<br>[arXiv:2310.08129](https://arxiv.org/abs/2310.08129) | **prompt 層**。履歴を検索して LLM に渡し、prompt を書き換える。生成モデルは変更しない | [`upstream/`](tailored-visions-repro/upstream) + [`patches/`](tailored-visions-repro/patches) | [`docs/summary.html`](tailored-visions-repro/docs/summary.html) |
| [`premier-repro/`](premier-repro/) | Premier（CVPR 2026 Highlight）<br>[arXiv:2603.20725](https://arxiv.org/abs/2603.20725) | **条件付け層**。学習可能な user embedding を Preference Adapter で text 条件と融合する | [`upstream/`](premier-repro/upstream) + [`patches/`](premier-repro/patches) | [`docs/summary.html`](premier-repro/docs/summary.html) |
| [`fan-repro/`](fan-repro/) | FAN（CVPR 2026）<br>Kim, Ahn, Seo | **encoder 層**。self-attention を Personalized Attention に差し替え、参照 prompt 群の嗜好を強度 α で混ぜる | [`upstream/`](fan-repro/upstream) + [`patches/`](fan-repro/patches) | [`docs/summary.html`](fan-repro/docs/summary.html) |

## Clone と submodule

```bash
git clone --recurse-submodules https://github.com/tmp-friends/personalized-t2i.git
cd personalized-t2i

# 既存 clone のみ:
git submodule update --init --recursive
```

公式 repository は各 `upstream/` submodule に HTTPS URL と固定 commit で保持する。
`upstream/` は直接編集せず、互換性修正は `patches/series`、実行用の適用済み tree は
`scripts/prepare_upstream.py` が生成する `.work/upstream/` に置く。

## 共通ディレクトリ contract

3 手法のフォルダは同じ役割名に揃えている。

| パス | 役割 | Git 管理 |
|---|---|---|
| `README.md` | 手法概要、再現状況、Quick start、差分 | する |
| `upstream/` | clean な公式実装（submodule） | gitlink のみ |
| `patches/` | 公式実装へ順に適用する patch と `series` | する |
| `src/<package>/`（該当手法） | repository 独自の再現・統合 code | する |
| `scripts/` | setup、生成、評価、公式 runner | する |
| `docs/` | setup、deviations、HTML summary | する |
| `tests/` | layout、patch、smoke test | する |
| `data/raw/` | download dataset | しない |
| `artifacts/` | model weights / checkpoint | しない |
| `outputs/` | 生成物と評価結果 | しない |
| `.work/` | patch 適用済み一時 tree | しない |

環境は手法ごとに `uv sync --locked` で作る。具体的な command と必要な model / data は、
各 README の Quick start を参照する。

- [Tailored Visions Quick start](tailored-visions-repro/README.md#3-quick-start)
- [Premier Quick start](premier-repro/README.md#3-quick-start)
- [FAN Quick start](fan-repro/README.md#3-quick-start)

GPU は 1 枚（RTX 4090 / 24 GB）を前提とする。複数手法を同時実行しないこと。
Tailored Visions の local LLM server は VRAM を保持するため、使用後に
`tailored-visions-repro/scripts/run_official.sh stop` で停止する。

## 手法の対比

```text
prompt 層      Tailored Visions   履歴を検索 → LLM が prompt を書き換え
                                  学習不要・軽量・説明可能

条件付け層    Premier            user embedding を adapter で text 条件へ融合
                                  user embedding の学習が必要

encoder 層    FAN                text encoder の self-attention を差し替え
                                  参照 prompt の嗜好を混ぜ、追加学習は不要
```

## 展示デモ（`exhibit/`）

2026-09-23 の展示は FAN を主役にしたローカルデモ。来場者が好きな画像を選び、
その画像の**どこが好きか**（色・光・描画・雰囲気）を指定すると、同じお題・同じ seed・同じ生成設定のまま
パーソナライズなし 4 枚とパーソナライズあり 4 枚を並べて比較できる。結果の下で好みを調整し、同じお題で描き直せる。

- 生成モデル: Illustrious XL v2.0（1024×1280、DPM++ 2M SDE Karras、fp16-fix VAE）
- 個人化: FAN 公式実装（`ClassTokenDecoder` を含む）。**来場者ごとの追加学習なし**
- 参照: 選んだカードに付けた確認済みの説明文（`catalog-v2`、4 被写体 × 16 表現）
- 既定 encoder policy: `legacy_exhibit`（`configs/fan-policies.json` が唯一の正）

```bash
uv sync --project exhibit --locked
(cd fan-repro && uv sync --locked && uv run python scripts/prepare_upstream.py)
uv run --project exhibit uvicorn exhibit.app:app --host 127.0.0.1 --port 7860
```

http://localhost:7860 が展示画面、`/tech` が技術解説、`/fallback` が事前生成サンプル（サーバー停止時も開ける単独 HTML）。
準備・評価・API・説明上の注意は [`exhibit/README.md`](exhibit/README.md)、
評価結果は [`docs/reports/fan-personalization/`](docs/reports/fan-personalization/) を参照。
LLM による prompt 書き換え、VLM 解析、PIGReward による推薦は展示の主経路から外している
（[`pigreward-repro/`](pigreward-repro/) は評価用 adapter として残置、ライブ推薦は無効）。

## ドキュメント

| パス | 内容 |
|---|---|
| [`docs/superpowers/specs/`](docs/superpowers/specs/) | 展示・FAN 個人化改善の設計書 |
| [`docs/superpowers/plans/`](docs/superpowers/plans/) | 実装・強度実験の計画 |
| [`docs/reports/fan-personalization/`](docs/reports/fan-personalization/) | 現行設定の評価・強度実験レポート |
| [`docs/reports/fan-demo/`](docs/reports/fan-demo/)、[`docs/reports/zipp-demo/`](docs/reports/zipp-demo/) | 旧構成の記録（現行結果としては引用しない） |
| [`docs/migrations/`](docs/migrations/) | layout 移行の記録 |

## Repository 方針

- 公式 source と独自実装を混ぜず、上流更新時に patch が適用できるかで差分を検証する。
- dataset、weights、生成物、virtual environment は commit しない。
- layout と submodule URL は [`tests/test_repository_layout.py`](tests/test_repository_layout.py) で検証する。
- 移行判断と旧 path 対応は
  [`layout migration manifest`](docs/migrations/2026-09-19-layout-migration-manifest.md) に残す。
