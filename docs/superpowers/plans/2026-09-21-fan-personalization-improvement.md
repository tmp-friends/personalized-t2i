# FAN Personalization Improvement Implementation Plan

> **For agentic workers:** 利用可能な場合は `superpowers:executing-plans` を使い、Taskごとに実装する。実行方法は通常の単一agentによる順次実装を既定とする。チェックボックスで進捗を記録する。

**Goal:** FAN の公式設定を比較できる検証基盤と、64枚の画像pool・複数回の選択・生成後の修正を実装し、個人化の改善を評価可能にする。

**Architecture:** 既存のFastAPI/単一GPU workerを維持し、FANのpolicy/trace、catalog、候補選択、評価を小さなモジュールへ分ける。生成時のsnapshotと実効設定をhashで固定する。研究評価は展示UIと分離し、未検証のpolicyを既定へ切り替えない。

**Tech Stack:** Python 3.12のWeb環境、既存FAN環境、FastAPI、PyTorch、diffusers、transformers 4.57系、Pillow、NumPy、vanilla JS、pytest、Node test、Playwright。

**Spec:** [FAN個人化改善設計書](../specs/2026-09-21-fan-personalization-improvement-design.md)。必ず先に全文を読む。本書は設計書とセットでcoding agentに渡す。

## Global Constraints

- 公式 pin は `9d0b76843f6437718195accac9cf3f050a25d26b` を維持する。
- `fan-repro/upstream/` と `.work/upstream/` を直接編集しない。
- Web は `exhibit/.venv`、GPU/FAN は `fan-repro/.venv` を使用し、`transformers>=4.57,<5` を維持する。
- GPU は RTX 4090 24 GB ×1。生成・準備・評価は既存 GPU lease を共有し、並列実行しない。
- 通常と個人化で target prompt、negative prompt、モデル、VAE、サンプラー、steps、CFG、解像度、seed をそろえる。
- 本書の比較中は `generation` を現行の値で固定する。変更した場合は別 experiment とし、過去結果に追記しない。
- サーバー起動・通常の推論・ブラウザー利用中にモデルをダウンロードしない。追加重みは明示的な準備コマンドで取得する。
- 通常セッションの入力と生成画像は終了時に削除する。研究用評価の保存先とは分離する。
- 評価が揃うまで展示の既定 encoder policy は `legacy_exhibit` とする。

## Review Focus

1. 同じ割合の重みを持つ重複参照を統合しても、profiling無効時の意味が変わらないこと。Task 2で数値検証。
2. 参照が少ないときのratio/count、bool、NaN、無効な側面を黙って別の意味にしないこと。Task 1/5で検証。
3. cancel直後・revision変更・再送・遅延イベントで、別の好みの画像が表示されないこと。Task 6/7で検証。
4. 別policy、変更済みカード、異なる実験設定のcache/評価画像を再利用しないこと。Task 1/3/4/6で検証。
5. 人の回答がない状態、欠損画像、同点、同一参加者の複数seedを、精度向上・独立した人数として誤集計しないこと。Task 8で検証。

## 実行時の約束

- Task順序は1→2→3、4→5、1/2/4/5→6→7、3/6→8、全Task→9。依存が揃っていない機能を動作済みと記録しない。
- 各Taskで振る舞いを確認するテストを先に追加し、必要な実装を行う。定数を写しただけのテストは増やさない。
- CPUの単体/統合はWeb環境、実際のtorch/encoderを使う検証はFAN環境へ分離する。通常pytestで重みをダウンロードしない。
- Task完了ごとに差分を確認する。コミットする運用なら当該Taskのファイルだけをstageし、ユーザーの未追跡データを含めない。
- 実測の待ち時間には進捗を報告する。GPUジョブの起動・再試行・総枚数をログへ残す。
- 人による回答待ちでも、コード・画面・集計器・再開手順は完成させる。回答待ちを理由に設計をやり直さない。

## Task 1: policy、参照snapshot、結果の同一性

**Files:** 新規 `exhibit/configs/fan-policies.json`、`exhibit/src/exhibit/fan_adapter.py`、`exhibit/tests/test_fan_policy.py`。変更 `domain.py`、`config.py`、`test_domain.py`。

**Interfaces:**

- `resolve_policy(policy_id: str, policies: dict) -> dict`: immutableな実効policy。
- `profiling_argument(policy: dict) -> int | float`: 公式FANへ渡す値。
- `build_personalization(snapshot: dict, *, prompt: str, policy: dict, provenance: dict) -> dict`: refs、実効policy、内容hash。旧テスト/評価fixtureのlist入力は明示的な変換helperで吸収し、本番APIで暗黙変換しない。

- [ ] `all/ratio/count`、`1`と`1.0`、bool/NaN/負値、未知pooled_modeのテストを追加する。
- [ ] 同一内容の順序正規化、profile revisionだけの変更、pooled/skip/profiling/decoder hash変更のcache同一性をテストする。
- [ ] aspect_phraseのstrength×gainと重複加算、card_descriptionで非一様gainを拒否するテストを追加する。
- [ ] 設計§5/7のpolicy、snapshot、hashを実装する。fan_adapterのimportだけでtorchをimportしない。
- [ ] `PYTHONPATH=exhibit/src exhibit/.venv/bin/python -m pytest exhibit/tests/test_fan_policy.py exhibit/tests/test_domain.py -q` を実行する。

**完了条件:** policy名を変えただけの処理ではなく、実効値とprovenanceで生成結果の同一性を判断できる。

## Task 2: 公式encoder経路、pooled切替、profiling trace

**Files:** 変更 `fan_adapter.py`、`workers.py`、`gpu.py`、`test_worker_generate.py`、`test_gpu.py`。新規 `exhibit/tests/test_fan_adapter.py`、`exhibit/src/exhibit/evaluation_worker.py`、`exhibit/scripts/evaluate_fan.py`。

**Interfaces:**

- `encode_conditioning(encoder, prompt: str, refs: list | None, policy: dict, *, collect_trace: bool=False) -> dict`。返り値は`hidden, pooled, trace, effective_policy`。
- `gpu_lease(cancel, deadline)`: 既存のleaseと同じロックを使うcontext manager。
- `evaluate_fan.py encoding --config <path>`: Web controllerからFAN環境のevaluation_workerを起動する。

- [ ] 疑似encoderを使い、fan pooledが捨てられないこと、plain pooled時のpositive/negative参照なしencoding、実効policyのイベント出力を検証する。
- [ ] profiling traceがL/G hidden/G poolの呼び出しと対応し、例外時にhookが戻るテストを追加する。
- [ ] 既存`workers.build_encoder`を利用して公式SDXL wrapperを構築する。参照選別は公式`sample_reference`へ委譲する。
- [ ] `workers.generate`内の固定plain pooledをpolicyへ置き換える。通常/負のpromptは常に参照なしとする。
- [ ] leaseを共通化し、生成・評価の競合、cancelと子プロセス終了の順序を既存テストと合わせて確認する。
- [ ] encodingサブコマンドに設計§9.3の実torch診断を実装する。max/mean差、RMSE、cosine、token/detector情報をJSONへ保存する。
- [ ] `PYTHONPATH=exhibit/src exhibit/.venv/bin/python -m pytest exhibit/tests/test_fan_adapter.py exhibit/tests/test_worker_generate.py exhibit/tests/test_gpu.py -q` を実行する。
- [ ] GPUが使える場合はencoding診断を実行する。公式wrapperとの一致、重複参照、共通倍率、α=0、例外後復元を確認する。公式pooledの既知の不整合は「失敗を検出した結果」として保存する。

**完了条件:** legacyと公式処理が選択可能で、どの参照を使い、どのpooledを生成器へ渡したか追跡できる。診断の失敗を回避するために公式経路を黙って修正しない。

## Task 3: 再現可能な比較生成・自動評価

**Files:** 新規 `exhibit/src/exhibit/evaluation.py`、`exhibit/configs/fan-evaluation.json`、`exhibit/configs/evaluation-cases/`、`exhibit/scripts/prepare_evaluation.py`、`exhibit/tests/test_evaluation.py`。変更 `evaluation_worker.py`、`evaluate_fan.py`。

**Interfaces:**

- `build_experiment(config: dict, provenance: dict) -> dict`: immutable manifestと画像job一覧。
- `score_records(manifest: dict, embeddings: dict) -> list[dict]`: 同一条件に対応したtarget/history/conditioning指標。
- `select_candidates(records: list[dict], rules: dict) -> dict`: 採用不可理由を含む候補選定。
- CLIサブコマンド`encoding/screen/refine/heldout/study`。`--resume`はmanifest hash一致を必須にする。

- [ ] 欠損画像を黙って除外しないこと、別policy/設定のresume拒否、上限512枚、同じ画像の二重計上防止をテストする。
- [ ] history_scoreがprofiling前の全有効参照から計算されることを、選別後だけなら結果が逆転するfixtureでテストする。
- [ ] screenの192+6+24枚、refineの96枚、heldoutの156枚を生成前のjob数テストで検証する。
- [ ] evaluatorの準備CLIを実装し、実在するHub commitとファイルhashをmanifestへ固定する。未準備なら評価側で明示的に失敗する。
- [ ] 既存workersのpipeline/adapterを再利用して画像を生成する。manifestの実効設定が画像イベントと異なれば中断する。
- [ ] 数値基準の通過/失敗、target非劣化条件、history閾値、tie-break、既定policyを自動変更しないことをテストする。
- [x] heldout の catalog 条件を「参照カードの確認済み・各水準 `min_reviewed_per_level` 枚以上・合計 `min_reviewed_cards` 枚以上」に緩め、
      どの条件で落ちたかを名指しでエラーにし、確認済み枚数と id 集合の hash を heldout 出力へ残すことをテストする（2026-09-22、既定 2 / 32）。
- [ ] `PYTHONPATH=exhibit/src exhibit/.venv/bin/python -m pytest exhibit/tests/test_evaluation.py -q` を実行する。

**完了条件:** 未測定と不合格を区別した、再開可能な実験と集計ができる。大規模GPU実験の本実行はTask 9へまとめる。

## Task 4: 64枚のcatalogとtoken/説明の検証

**Files:** 新規 `exhibit/configs/catalog-v2.json`、`exhibit/configs/cards-v2-review.json`、`exhibit/src/exhibit/catalog.py`、`exhibit/scripts/check_card_tokens.py`、`exhibit/tests/test_catalog.py`。変更 `domain.py`、`scripts/prepare.py`、`preflight.py`、`test_preflight.py`、`test_illustrious_prompting.py`。

**Interfaces:**

- `build_catalog(definition: dict) -> list[dict]`: 設計§6の64カード。
- `load_catalog(*, reviewed_only: bool=True) -> dict`: catalog hash付き。catalogは`catalog-v2`だけ。
- `validate_card_tokens(cards, tokenizers) -> list[dict]`: 超過とtoken列を返し、CLIは超過時非ゼロ終了。
- `prepare.py cards`。catalog は1つなので `--catalog` は無い（所有者判断 2026-09-22）。

- [x] 64枚、profile16種類、明示列挙した16組が各軸で均等（水準ごとに4回）かつ禁止水準対を含まないこと、軸ペアの被覆が85/96で最大重複2であること、IDの安定性をテストする（2026-09-21 に XOR 導出から変更、2026-09-22 に文言・禁止8対・16組を改訂。いずれも所有者承認済み）。
- [x] `seed_overrides` で1枚だけ seed を差し替えられること、`prepare.py cards --catalog v2 --only <card_id>` が指定カードだけを生成して manifest を整合して書き直すことをテストする（2026-09-22）。
- [ ] 画像/説明hash変更でreviewが無効になること、未確認画像の除外をテストする。
- [ ] v2生成文を作り、品質語を調整して両tokenizerで77以下に収める。カード専用 negative prompt も同じ検査に含める。旧3カードの超過を診断fixtureとして残す。
- [ ] v1資産を上書きせず、v2パスへ生成する準備処理とmanifestを実装する。
- [ ] preflightの固定16枚/6sample等の前提を、catalogとサンプルmanifestから検証する形へ更新する。countだけの検査に弱めない。
- [ ] `PYTHONPATH=exhibit/src exhibit/.venv/bin/python -m pytest exhibit/tests/test_catalog.py exhibit/tests/test_preflight.py exhibit/tests/test_illustrious_prompting.py -q` を実行する。
- [x] 両実tokenizerによるtoken検査を実行する。64枚の実生成・画像確認はTask 9で順次行う。（2026-09-22 改訂後の最大77/77、negative 75/77。トークン超過は profile 探索の制約として扱い、超過する16組は候補から外す。）

**完了条件:** 画像枚数を増やすだけでなく、軸を組み替えたpoolを生成・確認・利用できる。

## Task 5: 複数ラウンドの候補選択と好みの正規化

**Files:** 新規 `exhibit/src/exhibit/elicitation.py`、`exhibit/tests/test_elicitation.py`。変更 `domain.py`、`test_domain.py`。

**Interfaces:**

- `next_round(catalog: dict, snapshot: dict, *, shown_ids: list[str], round_index: int, session_seed: str) -> dict`。
- `normalize_preferences(payload: dict, catalog: dict, *, commit: bool) -> dict`。
- snapshotの構造とstrength/gainは設計§7、候補順位は設計§7.2に一致させる。

- [ ] 同じseed・履歴で同じ候補、既出cardなし、探索4枚、subject上限、候補不足時の短いroundをテストする。
- [ ] 未選択が負例にならないこと、選択0件時の全探索、未確認カードの除外をテストする。
- [ ] draft0〜10枚/commit3〜10枚、全側面未回答、重複ID、不正strength/gainをテストする。
- [ ] greedyのcost/類似度とtie-breakを実装する。fixtureで暖色だけを好きと指定したとき、textureまで既知の好みにしていないことを確認する。
- [ ] `PYTHONPATH=exhibit/src exhibit/.venv/bin/python -m pytest exhibit/tests/test_elicitation.py exhibit/tests/test_domain.py -q` を実行する。

**完了条件:** poolの提示順・ユーザーの明示的な好みが、再現できるデータとして保存される。

## Task 6: revision付きAPIと再生成・cache

**Files:** 変更 `service.py`、`app.py`、`preflight.py`、`test_service.py`、`test_api.py`、`test_worker_recovery.py`、`tests/conftest.py`。

**Interfaces:** 設計§8.1の各API。serviceのrun snapshotに`preference_revision`と解決済みpolicyを追加し、`publish`もrevisionを照合する。

- [ ] 選択の全置換、同内容PUTの冪等性、round/runのrequest_id再送・不正再利用、revision不一致をAPIテストに追加する。
- [ ] 生成中・cancel終了待ちの変更拒否、完了後の再選択、新runへ旧eventをpublishできないことをテストする。
- [ ] policy変更・catalog変更でcache miss、同一内容へ戻した場合cache hit、旧形式cacheは拒否をテストする。
- [ ] sampleへのfeedback拒否、完了後のfeedback置換、異なるrevisionへの回答拒否をテストする。
- [ ] round履歴・draft/commit状態・request台帳をsessionに実装し、終了時に削除する。
- [ ] `PUBLIC_RUN`、画像イベント、sample検証へ新しい内容hashを渡す。sampleを本人向け結果として扱わない。
- [ ] `PYTHONPATH=exhibit/src exhibit/.venv/bin/python -m pytest exhibit/tests/test_service.py exhibit/tests/test_api.py exhibit/tests/test_worker_recovery.py -q` を実行する。

**完了条件:** 再選択しても画像・参照・回答が混ざらず、既存のGPU終了とsession cleanupが保たれる。

## Task 7: 選択・側面確認・修正のUI

**Files:** 変更 `static/app.js`、`static/session.mjs`、`static/style.css`、`tests/mock_api.mjs`、`tests/test_browser_state.mjs`、`scripts/browser_check.py`。

**Interfaces:** Task 6のsnapshotが唯一の正とする。browser poll identityにはsession/run/revisionを含める。

- [ ] 再読込でround/draft/commitを復元、古いrevisionのpollを破棄、次round連打の重複提示防止をNodeテストへ追加する。
- [ ] 12枚表示、最大3round、途中終了、上限10枚、好きな側面の明示指定、strengthの操作を実装する。
- [ ] 新規カードの側面は未回答にし、「全部好き」を明示操作にする。未回答カードを含むcommitは通さない。
- [ ] 結果から選択を編集し同一お題で再生成できるようにする。gainは0.5/1/2への絶対値変更とする。
- [ ] ラベル付きの任意回答を実装する。ブラインド評価と誤解させる文言を使わない。
- [ ] `node --test exhibit/tests/test_browser_state.mjs` を実行する。
- [ ] mock APIの実ブラウザーで、開始→2round→側面確認→生成→修正→再生成→cache→cancel→終了を確認する。モバイル幅も確認する。

**完了条件:** 表示数の増加が操作不能な長いgridにならず、サーバーのrevisionと画面が一致する。

## Task 8: ブラインド評価ページと集計

**Files:** 新規 `exhibit/scripts/build_preference_study.py`、`exhibit/scripts/summarize_preference_study.py`、`exhibit/tests/test_preference_study.py`。変更 `evaluation.py`、`evaluate_fan.py`。

**Interfaces:** study manifest、匿名participants.json、opaqueなpair_id、A/B/tie回答。方式対応表は集計器だけが読む別ファイルにする。

- [ ] 回答なし、participant不足、同点のみ、未回答、重複回答、未知pair、別studyの回答をテストする。
- [ ] 画像seed数を人数に加算しないことを、2人×多数seedのfixtureで検証する。
- [ ] 参加者単位平均とbootstrapを実装する。candidate/legacyと本人/別人の2比較を独立集計する。
- [ ] 別人profile割当は自己割当なしの固定置換にし、manifestへ保存する。人数不足時は研究画像の生成前に失敗する。
- [ ] 回答前のclient payload、画面、画像名から方式labelが漏れないことをテストする。
- [ ] participants.jsonがなければ入力収集ページと実施説明だけを作る。架空回答や仮の勝率で埋めない。
- [ ] 収集ページのJSON export、匿名participantデータの検証・merge、設計§9.7のpool/選択フロー比較を実装する。encoderとelicitationのstudyを同じ集計へ混ぜない。
- [ ] `PYTHONPATH=exhibit/src exhibit/.venv/bin/python -m pytest exhibit/tests/test_preference_study.py -q` を実行する。

**完了条件:** 別の人が生成済み画像で評価でき、未実施を含む正しいレポートを出せる。

## Task 9: 実測、資産、ドキュメントの統合

**Files:** 変更 `configs/demo.json`、`scripts/prepare.py`、`scripts/build_fallback.py`、`scripts/build_report.py`、`scripts/rehearsal.py`、`exhibit/README.md`。新規 `docs/reports/fan-personalization/`。必要なら `fan-repro/docs/DEVIATIONS.md`、`patches/series`と最小patch。

- [ ] GPU/空き容量/必要重みを確認し、準備manifestを作る。推論時のdownload禁止を確認する。
- [ ] legacyの実効設定・旧カード・既存画像を固定して保存する。新設定の準備で上書きしない。
- [ ] 64枚を逐次生成し、実画像の4側面を確認する。不一致は修正・再生成し、reviewを画像hashと結び付ける。
      所有者承認 2026-09-22: 色3をパステル、描画0をラフスケッチに変更。64枚を厳密に揃えず、確認できたカードで運用する。
      1枚だけの差し替えは `seed_overrides` と `prepare.py cards --catalog v2 --only <card_id>` で行う。
- [ ] encoding→screen→refine→heldoutを実行する。heldout は 64枚すべての確認を待たず、設計§9.2 の3条件
      （参照カードの確認済み・各水準2枚以上・合計32枚以上）を満たした時点で実行できる。満たさない場合はheldoutだけ未実施とし、他の検証を進める。
- [ ] pooledの基準不適合は設計§5.4に沿って診断する。修正を行う場合は再現する数値検査を先に追加し、公式との差分として記録する。
- [ ] 研究用の候補policyを保存する。人の評価が不足する間はdefault_policy_idをlegacyのまま維持する。
- [x] 展示で使うcatalogを`catalog-v2`へ切り替え、catalog-v1を互換なしで削除した
      （**所有者判断 2026-09-22: 展示を catalog-v2 に切り替え、catalog-v1 は互換を残さず削除。heldout と本人評価の完了を待たない。**
      これは設計§6.2・§9.5の「確認完了後に切り替える」ゲートを catalog について上書きする）。
      個人化のサンプルを新snapshot/hashで再生成し、fallback・preflight・レポートを更新した。
      v1 pool を前提にした `elicitation` study（設計§9.7）も同時に削除した。
- [x] 新catalogの代表3選択×2お題の6sampleを作った。旧ラベルは流用せず、実際に有効にした側面で記述している
      （s1: 暖色×強い日差しの color+lighting / s2: 寒色×油彩の color+texture / s3: 逆光×平塗りの lighting+texture, strength 2）。
- [ ] 次の最終チェックを実行する。

```bash
PYTHONPATH=exhibit/src exhibit/.venv/bin/python -m pytest exhibit/tests -q
node --test exhibit/tests/test_browser_state.mjs
uv run --project exhibit ruff check exhibit
uv run --project exhibit ruff format --check exhibit
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/preflight.py --models
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/browser_check.py --report-dir docs/reports/fan-personalization
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/rehearsal.py --sessions 20
```

- [ ] 64枚のreview待ち、人の評価待ちなどがあれば、利用可能なv1/legacyへ設定を保ち、未検証の表示を公開状態にしない。
- [ ] 最終報告を設計§12に従って記載する。実装完了、実機検証、精度実証を別々に判定する。

**完了条件:** 設計§11 Aを満たす。B/Cについては実測・回答が存在する項目のみ完了とし、残る手順を実行可能な状態で引き渡す。
