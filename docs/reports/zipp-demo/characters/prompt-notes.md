# キャラクターイラストのプロンプト設計

2026-09-20。展示の目的は、5回の選択からキャラクターイラストの色・光・構図・描画表現・雰囲気の好みを反映すること。髪型・顔立ち・服装の嗜好推定や、画像間で同一人物を厳密に維持する機能は含まない。

## Webで確認した一次資料

- [Illustrious XL v2.0-STABLEの公式モデルカード](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0)：今回採用するモデル。安定性を調整したv2のcheckpointとして配布されている。v0より必ず高品質という比較結果はこの記述からは主張しない。
- [Illustrious XL初期版の公式モデルカード](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0)：人物のタグ形式の生成例、品質タグ、構図タグ、Euler a、20–28 steps、CFG 5–7.5を確認。構図指定の重ねすぎに注意するという説明も採用した。
- [著者の技術報告](https://arxiv.org/abs/2409.19946)：イラスト・アニメ画像向けモデルという位置づけを確認。

他のIllustrious派生モデルの推奨値をそのまま転用せず、ユーザー指定のv2.0のrevisionを固定する。初期版の公式推奨値は試作の出発点とし、v2.0で実画像を確認する。公式作例の文章をそのまま転載したものではなく、展示のお題に合わせてタグを組み立て、同じモデル・seedで試作して確認する。

## プロンプトの構成

1. 人物数・外見・服装・場面を先頭に置く。人物抑制の `no humans` とnegative側の `1girl, 1boy, person` は外す。
2. 公式で扱われる `best quality` と `masterpiece`、目の描写とイラスト指定を加える。品質タグだけで品質を保証するものではない。
3. 通常側には既存の一般的な補足、個人化側には確認・訂正済みの好みだけを追加する。両方で基本プロンプトとseedを維持する。
4. ネガティブには低品質・手指の破綻・複数画面・文字・写真・3D等の抑制を含める。旧版でアニメフレーム抑制も比較したが、最終案では外し、具体的な描画指定と人物数の指定を使う。
5. 展示用の5対は同じ題材・seedで、色／光／構図／塗り／雰囲気を主に変える。画像の外見や小物も変化するため、各軸を完全に独立させた心理測定とは扱わない。

モデルは `OnomaAIResearch/Illustrious-XL-v2.0`、revision `69459c1fe6f46db41ab31e6114f05acc0e06bcaa`。生成設定は1024×1024、Euler ancestral、28 steps、CFG 6.5、fp16。通常・個人化・選択画像を同じ設定に揃える。

具体的な最終プロンプトは [demo.json](../../../../exhibit/configs/demo.json)、実際の生成条件・seed・hashは [manifest.json](../../../../exhibit/assets/manifest.json)、個人化例は [samples.json](../../../../exhibit/assets/samples.json) を参照。

## オフラインでの読込

v2.0は単一の `Illustrious-XL-v2.0.safetensors`（6,938,040,674 bytes）で配布される。Diffusersの `from_single_file` にローカルcheckpointを渡す。モデル構成とCLIP tokenizerは既存のSDXL互換構成（初期Illustriousの固定revision）を明示的に渡し、推論時のダウンロードを禁止する。UNet・VAE・text encoderの重みはv2.0のcheckpointから読み込む。checkpointと構成ファイルの欠落は起動前検査で検出する。

## 画像比較の根拠確認

VLMの生出力は保存し、人物画像に即した短い根拠へ編集した。塗り比較のp4-bでは逆方向のVLM説明が目視と食い違ったため、輪郭と色の境界の観察を優先し、その修正理由を `evidence.json` の `review_note` に記録する。AI解析の文を未確認のまま来場者の好みとして使わない。

通常画像の4seedを確認し、猫を描き落とす例は `holding cat`、夜景だけになる例は `male focus` と `portrait` で主役を明示して再生成した。船乗りには帽子と縞のシャツ、カフェ店員にはシャツを明示した。人物を抑制しないことだけでなく、お題ごとの画像確認を行う。

個人化の実LLM試験では、1条件で塗りの表現を落とした。再試行時に必要な全表現を明示し、同じ条件の再試験で確認した。元の表現集合との一致検査は維持し、勝手に不足表現を補った出力をLLMの成功とは扱わない。
