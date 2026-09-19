# personalized-t2i — 嗜好パーソナライズ × 画像生成

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

9/23の展示は「5回の画像選択 → 好みの言語化と訂正 → 個人化生成 → PIGRewardによる推薦」
を中心とする設計案へ更新した。通常生成との比較と、AIの解釈を本人が修正できる体験を見せる。
詳細は [ZIPP-style persona × PIGReward 展示設計](docs/superpowers/specs/2026-09-19-zipp-pigreward-exhibition-demo-design.md)
を参照。モデル統合と展示アプリは実装前。

## Repository 方針

- 公式 source と独自実装を混ぜず、上流更新時に patch が適用できるかで差分を検証する。
- dataset、weights、生成物、virtual environment は commit しない。
- layout と submodule URL は [`tests/test_repository_layout.py`](tests/test_repository_layout.py) で検証する。
- 移行判断と旧 path 対応は
  [`layout migration manifest`](docs/migrations/2026-09-19-layout-migration-manifest.md) に残す。
