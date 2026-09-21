# キャラクターイラストのプロンプト設計

2026-09-21。展示の目的は、5回の選択からキャラクターイラストの色・光・構図・描画表現・雰囲気の好みを反映すること。髪型・顔立ち・服装の嗜好推定や、画像間で同一人物を厳密に維持する機能は含まない。

## 参照資料と位置づけ

- [Illustrious XL v2.0-STABLEの公式モデルカード](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0)：今回採用するモデル。安定性を調整したv2のcheckpointとして配布されている。v0より必ず高品質という比較結果はこの記述からは主張しない。
- [Onoma AI公式Text-to-Image APIドキュメント](https://www.illustrious-xl.ai/docs/text-to-image)：品質タグの例とEuler、normal scheduler、28 steps、CFG 7.4のAPI例を確認した。v2.0-STABLEの最適値ではなく、公式サービスの使用例として扱う。
- [GrayMan氏のIllustrious XL v2.0ガイド](https://www.seaart.ai/articleDetail/cvceb6le878c73bckfig)：品質タグの順序、Danbooru tagと自然言語の併用、Euler a、20–28 steps、CFG 4.5–5、WebUI表記のClip skip 2を参照した。Onoma AI公式文書ではなくコミュニティガイドとして扱う。
- [Illustrious XL初期版の公式モデルカード](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0)：人物のタグ形式の生成例、品質タグ、構図タグ、Euler a、20–28 steps、CFG 5–7.5を確認。構図指定の重ねすぎに注意するという説明も採用した。
- [著者の技術報告](https://arxiv.org/abs/2409.19946)：イラスト・アニメ画像向けモデルという位置づけを確認。

他のIllustrious派生モデルの推奨値をそのまま転用せず、ユーザー指定のv2.0のrevisionを固定する。初期版の公式推奨値は試作の出発点とし、v2.0で実画像を確認する。公式作例の文章をそのまま転載したものではなく、展示のお題に合わせてタグを組み立て、同じモデル・seedで試作して確認する。

## プロンプトの構成

1. `masterpiece, best quality, amazing quality, very aesthetic, newest`を先頭に置き、安全な展示内容を示す`safe`を続ける。
2. 人物数・外見・服装・場面をその後に置く。人物抑制の`no humans`とnegative側の`1girl, 1boy, person`は使わない。
3. 通常側には既存の一般的な補足、個人化側には確認・訂正済みの好みだけを追加し、最後を`absurdres, highres`で閉じる。基本プロンプトとseedは両側で維持する。
4. negativeには低品質、手指、JPEG artifact、透かし、署名、構図破綻、複数画面を中心に置き、展示上必要な複数人物とNSFWの抑制を加える。写真・3D・モノクロ・chibi等の画風抑制は外し、個人化表現との衝突を避ける。
5. 展示用の5対は同じ題材・seedで、色／光／構図／塗り／雰囲気を主に変える。画像の外見や小物も変化するため、各軸を完全に独立させた心理測定とは扱わない。

モデルは`OnomaAIResearch/Illustrious-XL-v2.0`、revision `69459c1fe6f46db41ab31e6114f05acc0e06bcaa`。生成設定は1024×1280、DPM++ SDE Karras（`DPMSolverMultistepScheduler`、`sde-dpmsolver++`、Karras sigmas）、30 steps、CFG 5.0、fp16（VAEは`madebyollin/sdxl-vae-fp16-fix`）。参照資料のEuler系・28 stepsは出発点で、実画像確認の結果、現行設定は [demo.json](../../../../exhibit/configs/demo.json) のとおり。通常・個人化・選択画像を同じ設定に揃える。

### Clip skip表記の差

コミュニティガイドの「Clip skip 2」をDiffusersの`clip_skip=2`へ直訳しない。使用中の`StableDiffusionXLPipeline`は引数なしで`hidden_states[-2]`を使い、`clip_skip=1`では`[-3]`、`2`では`[-4]`を使う。同一prompt・seedの実画像比較では、引数なしだけが正常な人物画像となり、`1`は網目状、`2`は灰色のテクスチャ状に破綻した。このため追加の`clip_skip`引数を渡さず、SDXL既定のpenultimate hidden stateを使う。

具体的な最終プロンプトは [demo.json](../../../../exhibit/configs/demo.json)、実際の生成条件・seed・hashは [manifest.json](../../../../exhibit/assets/manifest.json)、個人化例は [samples.json](../../../../exhibit/assets/samples.json) を参照。

## オフラインでの読込

v2.0は単一の `Illustrious-XL-v2.0.safetensors`（6,938,040,674 bytes）で配布される。Diffusersの `from_single_file` にローカルcheckpointを渡す。モデル構成とCLIP tokenizerは既存のSDXL互換構成（初期Illustriousの固定revision）を明示的に渡し、推論時のダウンロードを禁止する。UNet・VAE・text encoderの重みはv2.0のcheckpointから読み込む。checkpointと構成ファイルの欠落は起動前検査で検出する。

## 画像比較の根拠確認

VLMの生出力は保存し、人物画像に即した短い根拠へ編集した。塗り比較のp4-bでは逆方向のVLM説明が目視と食い違ったため、輪郭と色の境界の観察を優先し、その修正理由を `evidence.json` の `review_note` に記録する。AI解析の文を未確認のまま来場者の好みとして使わない。

通常画像の4seedを確認し、猫を描き落とす例は `holding cat`、夜景だけになる例は `male focus` と `portrait` で主役を明示して再生成した。船乗りには帽子と縞のシャツ、カフェ店員にはシャツを明示した。人物を抑制しないことだけでなく、お題ごとの画像確認を行う。

個人化の実LLM試験では、1条件で塗りの表現を落とした。再試行時に必要な全表現を明示し、同じ条件の再試験で確認した。元の表現集合との一致検査は維持し、勝手に不足表現を補った出力をLLMの成功とは扱わない。
