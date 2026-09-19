# 3手法のリポジトリ構成統一と公式実装管理の設計

## 1. 目的

FAN、Premier、Tailored Visions の3手法を、初見の利用者が同じ要領でセットアップ・実行・比較できる構成へ揃える。
同時に、公式実装とこのリポジトリ固有のラッパ、互換性修正、再現実装、データ、生成物の責任範囲を分離する。

成功条件は次のとおり。

- 3手法のディレクトリ名、主要ディレクトリ、READMEの章立て、生成物の置き場所が揃っている。
- 各公式実装の取得元と固定コミットをGitが記録し、新規cloneでも同じ状態を取得できる。
- 公式実装へのローカル修正が、未コミット差分ではなく追跡可能なパッチとして残る。
- モデル重み、取得済みデータ、仮想環境、生成物を親リポジトリで追跡しない。
- 既存のローカル修正、検証記録、独自再現実装を失わない。
- `git clone --recurse-submodules` した別環境で、各手法の軽量な検証コマンドまで再現できる。

## 2. 現状

3手法はいずれも公式リポジトリの `.git` を内部に残している一方、親リポジトリではその内容を通常ファイルとして追跡している。
公式コードとローカル変更の境界は手法ごとに異なる。

| 手法 | 現在の公式コード | ローカル変更の状態 | 主な不統一 |
|---|---|---|---|
| FAN | `FAN/` 全体 | 公式ファイルの未コミット変更と、ローカル用ファイルが同居 | フォルダ名、サマリー、生成物、公式境界 |
| Premier | `premier-repro/official/Premier/` | 公式コードへの1行修正。ラッパとパッチは概ね外側 | `official/` の一部だけが公式で、重みやラッパも同階層 |
| Tailored Visions | `tailored-visions-repro/official/` | 上流より4コミット先のローカル履歴に加え、未コミット変更あり | 公式コード、互換性修正、ローカルLLM、実行ラッパが混在 |

親リポジトリにも未コミット変更があるため、移行は一括置換せず、既存差分を退避・分類してから手法ごとに行う。

## 3. 対象範囲

### 含めるもの

- `FAN/` の `fan-repro/` への名称統一
- 3つの公式実装のGit submodule化
- 公式実装への変更を `patches/` へ分離
- README、サマリー、差分文書、スクリプト、設定、データ、生成物の配置統一
- `.gitignore` とルートREADMEの更新
- submodule取得とパッチ適用を含むセットアップ手順
- 軽量な構造検査とsmoke test

### 含めないもの

- 各手法のアルゴリズム変更
- 既存の実験結果の再計算
- 3手法のPython依存関係を単一環境へ統合すること
- 公式リポジトリへの変更のupstream提出
- 大規模データやモデル重みのGit LFS移行
- 展示アプリケーションの実装

## 4. 共通ディレクトリ契約

トップレベルに余分な `methods/` 階層は追加せず、次の3ディレクトリを置く。

```text
fan-repro/
premier-repro/
tailored-visions-repro/
```

各手法は必要な範囲で次の構成に合わせる。利用しない空ディレクトリは作らない。

```text
<method>-repro/
├── README.md
├── pyproject.toml
├── uv.lock
├── upstream/                 # 公式Git submodule。直接編集しない
├── patches/                  # upstreamへ適用する番号付きパッチ
├── src/<method>_repro/       # このリポジトリ固有の再利用可能コード
├── scripts/                  # セットアップ、生成、学習、評価のCLI
├── configs/                  # 追跡対象の設定
├── tests/                    # 軽量な構造・単体検査
├── docs/
│   ├── summary.html
│   └── DEVIATIONS.md
├── data/
│   ├── examples/             # 小さな追跡対象fixture
│   └── raw/                  # 取得データ。Git対象外
├── artifacts/                # モデル重み。Git対象外
└── outputs/                  # 生成物とログ。Git対象外
```

`src/` はローカルPythonパッケージがある場合に使う。既存の `premier/` と `tv/` は、それぞれ `src/premier_repro/` と `src/tailored_visions_repro/` へ移し、importとパッケージ設定を更新する。FANで再利用可能コードを追加しない場合、`src/` は作らず、薄いCLIを `scripts/` に置く。

## 5. 公式実装の管理

### 5.1 submodule

公式実装はすべて `upstream/` に配置し、HTTPS URLと取得可能なコミットを親リポジトリで固定する。

| submodule path | URL | 移行開始時の基準コミット |
|---|---|---|
| `fan-repro/upstream` | `https://github.com/Burf/FAN.git` | `9d0b76843f6437718195accac9cf3f050a25d26b` |
| `premier-repro/upstream` | `https://github.com/120L020904/Premier.git` | `42473476a189b6b0127890a98e92b7edf49c0d59` |
| `tailored-visions-repro/upstream` | `https://github.com/zzjchen/Tailored-Visions.git` | `d0f4454ca08c68c5d30f08a01ff4a23a8b33b610` |

submodule内へ直接変更を加えない。通常の作業完了時に `git submodule foreach --recursive git status --short` が空であることを必須とする。

### 5.2 パッチ

公式コードの修正が必要な場合は、手法ディレクトリの `patches/` に番号付きパッチとして保存する。

```text
patches/
├── series
├── 0001-<reason>.patch
└── 0002-<reason>.patch
```

`series` は適用順を1行1ファイルで記録する。各パッチの目的、根拠、影響、検証方法は `docs/DEVIATIONS.md` に記載する。

submoduleをdirtyにしないため、セットアップスクリプトは `upstream/` を直接変更せず、Git対象外の `.work/upstream/` に作業コピーを作ってパッチを適用する。実行コードはパッチが必要な場合は `.work/upstream/`、不要な場合は `upstream/` を参照する。`.work/` は全手法でGit対象外とする。

パッチ適用は次の条件を満たす。

- 途中で失敗した場合は非ゼロ終了し、半端な作業コピーを実行に使わない。
- 基準submodule SHAがREADMEまたは検証済みSHAと異なる場合は警告する。
- 同じSHAとパッチ列に対して繰り返し実行できる。
- 適用後のソースSHAまたはstampを `.work/` に記録し、不要な再作成を避ける。

### 5.3 上流更新

submodule更新は通常の依存更新と分けた独立変更として行う。

1. submoduleの候補コミットを更新する。
2. 全パッチを新しい作業コピーへ適用する。
3. smoke testを実行する。
4. `DEVIATIONS.md` と固定コミット表を更新する。
5. submodule pointer、必要なパッチ変更、文書変更を同じコミットまたはPRにまとめる。

パッチが適用不能な場合は自動解決せず、公式変更を確認してパッチの削除・更新を判断する。

## 6. 手法別の移行

### 6.1 Premier

Premierを共通構成の先行実装とする。既存の境界が最も明確で、公式コードへの修正が1行に限定されているためである。

- `official/Premier/` を `upstream/` submoduleへ置換する。
- `official/patches/0001-fix-single-block-checkpoint-kwarg.patch` を `patches/` へ移す。
- `premier_local.py` の再利用可能部分を `src/premier_repro/`、CLIを `scripts/` へ移す。
- `run_premier.py` と `train_new_user.py` を番号付きまたは用途名付きの `scripts/` へ移す。
- `official/weights/` を `artifacts/weights/` へ移し、再取得手順を文書化する。
- `official/test_data/watercolor/` を `data/examples/watercolor/` へ移す。
- `official/outputs/` を `outputs/` に統一する。
- 独自再現パッケージ `premier/` は `src/premier_repro/` に統合する。

公式コードをimportする箇所は、セットアップ済みの `.work/upstream/` を明示的に解決する共通ヘルパーを通し、相対的な `sys.path` 操作を各CLIへ重複させない。

### 6.2 FAN

- `FAN/` を `fan-repro/` へ改名する。
- 公式由来の `fan/`, `readme.md`, `LICENSE`, `asset/`, `weight/`, `inference.ipynb`, `inference.py` を `upstream/` submoduleへ置換する。
- 現在の `inference.py` の変更は、可能な部分を `scripts/generate.py` の引数処理へ移す。公式ファイル修正が不可避な部分だけパッチ化する。
- `README_ja.md` を手法ディレクトリの `README.md` にする。
- `FAN_summary.html` を `docs/summary.html` へ移す。
- `smoke_clip.py` を `scripts/smoke_clip.py` へ移す。
- `image/` は `outputs/` へ統一する。
- 公式配布の小さなdecoder重みはsubmodule内の公式資産として扱う。追加ダウンロードした生成モデルは `artifacts/` または外部キャッシュを使用する。

### 6.3 Tailored Visions

Tailored Visionsは最後に移行する。上流 `d0f4454` より先のローカル4コミットと未コミット変更を、内容別のパッチへ再構成する。

- `official/` の公式由来ファイルを `upstream/` submoduleへ置換する。
- 公式バグ修正とPython 3.12／2026年依存互換性修正を `patches/` へ分離する。
- `serve_local_llm.py`、`run.sh`、ローカル環境向け設定など、公式実装ではない追加機能を `scripts/` または `src/tailored_visions_repro/` へ移す。
- `tv/` を `src/tailored_visions_repro/` へ移し、CLI importを更新する。
- `tailored-visions.html` を `docs/summary.html` に改名する。
- 取得済みPIPデータは `data/raw/` とし、取得スクリプトから再生成できるようにする。
- `results/` と公式コードの `image_result/` を `outputs/` 配下へ統一する。
- `demo_user.jsonl` のような小さな入力例は、生成可能なら生成手順を優先し、固定fixtureが必要な場合だけ `data/examples/` で追跡する。

パッチは既存コミット境界を機械的に保存するだけでなく、レビュー可能な目的単位へ整理する。少なくとも「公式バグ修正」「現行依存・モデル対応」「実験用CLI拡張」を分離する。

## 7. READMEと文書

3手法のREADMEは次の章立てに揃える。

1. 手法概要
2. 現在の再現状況
3. Quick start
4. ディレクトリ構成
5. 公式実装と固定コミット
6. 生成・学習・評価手順
7. 動作確認結果
8. 論文・公式実装との差分
9. データ、重み、生成物
10. 引用

ルートREADMEは3手法の比較、各READMEへの入口、submoduleを含むclone方法、共通運用だけを扱う。手法固有の長い手順を重複させない。

各 `DEVIATIONS.md` は最低限、次を含む。

- 固定した公式URLとコミット
- パッチ一覧と理由
- 論文との差分
- 公式実装との差分
- ローカル代替モデルやデータを使う場合の再現性への影響
- 検証済み環境とコマンド

## 8. データ、重み、生成物

共通の追跡規則は次のとおり。

| 種類 | パス | Git |
|---|---|---|
| 小さなテスト入力 | `data/examples/` | 追跡する |
| ダウンロードデータ | `data/raw/` | 追跡しない |
| 中間キャッシュ | `data/cache/` | 追跡しない |
| モデル重み | `artifacts/` | 追跡しない |
| 生成画像・評価結果・ログ | `outputs/` | 追跡しない |
| パッチ済み公式作業コピー | `.work/` | 追跡しない |

ルート `.gitignore` は手法名に依存する個別規則を減らし、次の共通規則を持つ。

```gitignore
**/.venv/
**/.work/
**/data/raw/
**/data/cache/
**/artifacts/
**/outputs/
**/__pycache__/
```

既に親Gitで追跡されているデータは、再取得手段と必要な小規模fixtureを確認してから追跡解除する。作業中のファイルを先に削除しない。

## 9. 共通操作

各手法はREADME冒頭で、同じ形の操作を案内する。

```bash
git submodule update --init --recursive
cd <method>-repro
uv sync
uv run python scripts/prepare_upstream.py
uv run python scripts/smoke.py
```

手法固有の生成・学習・評価は、その後に番号付きまたは用途名付きスクリプトとして提示する。GPUを必要としない構造検査を `smoke.py` の最初に置き、重い検証は明示的なオプションへ分離する。

## 10. 移行手順と安全性

移行は Premier、FAN、Tailored Visions の順に行う。

各手法で次を完了してから次へ進む。

1. 親リポジトリと内部cloneのstatus、HEAD、remote、差分を記録する。
2. 未コミット変更をパッチまたは親リポジトリ側のファイルとして退避する。
3. 公式由来、ローカルラッパ、データ、重み、生成物を分類する。
4. 公式由来ファイルの通常追跡を解除し、同じ位置関係の `upstream/` submoduleを追加する。
5. ローカルコードと文書を共通構成へ移す。
6. パッチ準備スクリプトとsmoke testを通す。
7. 親Gitと全submoduleが意図したstatusであることを確認する。

ファイル移動前後でハッシュまたは差分を確認し、未コミット変更が消えていないことを検証する。既存の仮想環境、取得データ、モデル重み、生成物は、Git追跡対象から外してもローカルでは保持する。

移行コミットは手法単位に分ける。ディレクトリ全体の機械的移動と、コード修正を可能な限り別コミットにし、レビュー時に意味のある差分を読めるようにする。

## 11. 検証

### 構造検査

- `.gitmodules` に3つのHTTPS URLが登録されている。
- `git submodule status --recursive` が3手法の固定コミットを返す。
- `git submodule foreach --recursive git status --short` が空である。
- 公式由来コードが親リポジトリの通常ファイルとして重複追跡されていない。
- READMEと実際のパスが一致する。
- `.venv/`, `.work/`, `data/raw/`, `artifacts/`, `outputs/` が無視される。

### 再現性検査

一時ディレクトリへの新規cloneで次を確認する。

1. `git clone --recurse-submodules` が成功する。
2. 各手法で `uv sync` が成功する。
3. パッチ準備スクリプトが空の `.work/` から成功する。
4. 同じ準備スクリプトの再実行が成功する。
5. GPU不要のsmoke testが成功する。
6. 利用可能な手法では、既存の最小GPU smoke testが成功する。

### 既存機能の回帰検査

- FANのCLIP単体smoke test
- Premierの公式モジュールimportと設定読み込み
- Tailored Visionsのパッチ済み公式モジュールimport、`--help`、書き換えのみの最小実行
- 独自再現パイプラインのCLI `--help` と既存設定の読み込み

## 12. 採用しない案

### 公式コードのsnapshotを親Gitへ直接保存し続ける

cloneは簡単だが、上流コミット、ローカル差分、更新履歴が曖昧になるため採用しない。

### パッチ済みforkだけをsubmoduleにする

実行時は単純だが、「公式実装」とローカル修正の境界が弱くなり、forkの公開・保守も必要になるため既定案にはしない。将来パッチ量が増え、外部共有する価値が出た場合に再検討する。

### 3手法を単一Python環境へ統合する

依存バージョンとGPU要件が異なり、今回の目的に対してリスクが大きい。手法ごとの `.venv` を維持する。

## 13. 完了条件

以下をすべて満たした時点で移行完了とする。

- 3手法が共通ディレクトリ契約に沿っている。
- 公式コードが3つともcleanなsubmoduleとして固定されている。
- 既存の公式コード変更がすべてパッチ、ラッパ、またはローカルパッケージへ分類されている。
- 既存の未コミット変更、検証記録、独自再現コードが保持されている。
- データ、重み、仮想環境、生成物が意図どおりGit対象外である。
- ルートおよび各手法のREADMEから、セットアップと主要コマンドへ迷わず到達できる。
- 新規cloneで構造検査、パッチ準備、軽量smoke testが成功する。
