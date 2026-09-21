# FAN: 論文・公式実装との差分

## Upstream boundary

- 公式 repository: <https://github.com/Burf/FAN>
- 固定 commit: `9d0b76843f6437718195accac9cf3f050a25d26b`
- clean source: `upstream/`
- patch series: `patches/series`

現在 source patch はない。移行前の `inference.py` にあった local CLI の振る舞いは、公式 source を
変更せず `scripts/generate.py` で表現できるためである。`scripts/prepare_upstream.py` は固定 SHA を
検証し、公式 tree を `.work/upstream/` に materialize する。

## Local integration differences

- `scripts/generate.py` は出力先を `outputs/` とし、`--variant`、`--dtype`、
  `--ref_weight_dir` を外側から指定できる。
- `scripts/smoke_clip.py` は拡散 model を使わず、CLIP-L encoder と FAN decoder の数値経路を確認する。
- 公式の attention monkey patch と互換性を保つため `transformers>=4.57,<5` に固定する。
- FLUX.1-dev、SD3.5、SDXL、unCLIP、LLaVA の全経路は再検証していない。
- `exhibit/` は encoder の振る舞いを `exhibit/configs/fan-policies.json` の policy で選ぶ。
  `alpha`、`skip`、`skip_pa`、`use_attn_mask`、`pooled_mode`、`profiling`、`reference_unit`
  はこのファイルだけが持ち、公式 wrapper には呼び出し時の引数として渡す。参照の選別は公式
  `sample_reference` に委譲する。upstream の固定 SHA は変えていない。source patch も追加していない。

## Validation

```bash
python scripts/prepare_upstream.py
python -m unittest discover -s tests -v
.venv/bin/python -m py_compile scripts/generate.py scripts/smoke_clip.py
.venv/bin/python scripts/generate.py --help
git -C upstream status --short
```
