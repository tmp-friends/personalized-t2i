# Deviations from the paper and the official release

Read this before comparing any number here to the paper's. Some of what follows
changes the method, some changes what a metric means, and one item is a property
of the dataset that limits what ROUGE-L can tell you at all.

Paper: Chen, Zhang, Weng, Pan, Lan. *Tailored Visions: Enhancing Text-to-Image
Generation with Personalized Prompt Rewriting.* CVPR 2024. arXiv:2310.08129.
Official code: <https://github.com/zzjchen/Tailored-Visions>.

## Official compatibility patch series

The clean `upstream/` submodule is pinned to
`d0f4454ca08c68c5d30f08a01ff4a23a8b33b610`. It is never edited in place.
`scripts/prepare_upstream.py` applies these patches to `.work/upstream/`:

| Patch | Affected files | Reason |
|---|---|---|
| `0001-fix-official-runtime-bugs.patch` | `apiuse.py`, `demo.py`, `download.py`, `language.py`, `main.py`, `prompts.py` | bounded API retry; valid CLI/retrieval branches; BM25 tokenization and stable ordering; per-query ICL; safe paths; cached CLIP ranking |
| `0002-support-current-diffusers-and-sdxl.patch` | `SD.py`, `apiuse.py`, `demo.py`, `main.py`, `requirements-2026.txt` | current diffusers and OpenAI clients, SDXL / SD v1.5 support, fp16 fallback, VAE slicing, 24 GB memory policy, current Python dependencies |
| `0003-add-bounded-experiment-cli.patch` | `main.py` | `--data_folder` and `--limit_users` for bounded, relocatable runs |

Validation:

```bash
python scripts/prepare_upstream.py
python -m unittest discover -s tests -v
.venv/bin/python -m py_compile .work/upstream/*.py
.venv/bin/python .work/upstream/demo.py --help
.venv/bin/python .work/upstream/main.py --help
git -C upstream status --short
```

---

## 1. The rewriter is a different model — this is the biggest one

The paper's rewriter is `gpt-3.5-turbo` as it existed in 2023. That endpoint no
longer serves that model, and no OpenAI key is configured in this environment,
so the default rewriter here is **Qwen3.5-4B**, run locally, greedy-decoded.

The rewriter *is* the method. Absolute numbers will not match the paper's, and a
run that did match would be more suspicious than reassuring. What can be
reproduced is the **relative ordering** of the conditions: personalized > general >
none, EBR ≳ BM25, ICL > context-independent.

`--backend openai --model gpt-3.5-turbo` restores the original setup if you have
a key (`OPENAI_API_KEY`, optionally `OPENAI_BASE_URL`).

No prompt was tuned toward the paper's numbers. The instruction, the five
in-context demonstrations, and the slot layout are copied verbatim from the
official `prompts.py`.

## 2. The ground-truth images no longer exist → Image-Align is a proxy

Every `result_url` in the PIP dataset points at `cdn1.printidea.art`, which has
been offline since 2024. The official README acknowledges this and suggests
generating a stand-in from the ground-truth prompt. That is what
`05_generate.py` does, under the name `gt_proxy`.

**The proxy measures something different from the paper's Image-Align.** The
paper compares a generated image to *an image the user actually saved*. The
proxy compares it to *another SD v1-5 sample*, so:

* it has a degenerate optimum — echo the ground-truth prompt verbatim and score
  1.0 — which the real metric does not have;
* it is sensitive to sampling noise.

The second point is handled: `generate.seed_for(key)` derives one seed per test
sample from the sample key, and every method reuses it, so differences between
methods come from the prompt rather than from the noise draw. The first is not
fixable. The column is labelled `Image-Align (proxy)` throughout and should not
be read as a reproduction of the paper's 0.6796.

## 3. The user preference summaries P_u are not released

PMS scores a generated image against `P_u`, "5 phrases summarized from the
user's history prompts using ChatGPT" (Sec. 3.3). The summaries are not in the
released dataset and the summarization prompt is not published.
`04_preferences.py` regenerates them with the same local LLM, using a
reconstructed instruction (`prompt_templates.PREFERENCE_SUMMARY_TEMPLATE`).
Users with more than 30 distinct history prompts are subsampled evenly to fit
the context window.

Independent of the reconstruction, **PMS is not a neutral metric**: `P_u` is
derived from the same history the rewriter conditions on, so any method that
copies history-derived style tokens into its output scores higher almost by
construction. A PMS gain is evidence that the rewrite carries history tokens,
not that a user would prefer the image. It is reported because the paper reports
it.

## 4. The "General PR" baseline prompt is not published

The paper compares against a general-purpose rewriter that ignores the user, but
does not give its prompt. `prompt_templates.GENERAL_PR_TEMPLATE` is our
reconstruction, deliberately mirroring the personalized instruction's
constraints (retain the primary object, ≤70 words) so that the two conditions
differ only in whether user history is present.

Promptist is not reconstructed — it runs from the authors' released checkpoint,
`microsoft/Promptist`, with their published decoding settings (8 beams,
`length_penalty=-1.0`, 75 new tokens).

## 5. Dataset size differs slightly from the paper

| | Paper | Released zip |
|---|---|---|
| Users | 3,115 | **3,116** |
| Train samples | 294,007 | **294,023** |
| Test samples | 6,230 | **6,232** |
| Total | 300,237 | **300,255** |

One extra user, carrying 16 extra training prompts and 2 extra test samples.
Everything here uses the released files as-is.

## 6. Bugs in the released reference code

Fixed rather than reproduced. Each one is a real defect, not a stylistic choice.

| Location | Defect | What we do |
|---|---|---|
| `main.py:71` | `if self.retrieval=='full'` — but the CLI only accepts `'ebr'` / `'bm25'`, so `--retrieval=ebr` silently runs **BM25**. The released `main.py` cannot produce the paper's EBR rows. | Dispatch on `'ebr'`. |
| `main.py:45` | `openai.api_base = MY_BASE` with `MY_BASE` never imported → `NameError` on every run. | n/a — different API layer. |
| `main.py:138` | `clip_model,=None` → `TypeError` whenever `--retrieval=bm25`. | n/a. |
| `language.py:191` | `bm25.get_documents_score(query)` is handed the **raw query string**, so `for q in query` iterates *characters* and BM25 degenerates to single-character matching. | Tokenize the query like the documents. `BM25Retriever(legacy_char_query=True)` restores the original behaviour. |
| `language.py:192-193` | Sorting via `sentences.index(x)` — wrong whenever two prompts are equal, and O(n²). | `np.lexsort` on the score array. |
| `language.py:117` | `list(set(sentences))` after `checklines` desynchronizes `sentences` from `image_urls`, so a retrieved prompt is paired with another prompt's image. | Order-preserving dedup, and prompts/URLs never split apart. |
| `prompts.py:159-201` | `get_ZS_example` returns `instruction + question_block` with no separator, gluing "…within 70 words." onto "The history prompts are:". Table 1 of the paper shows them on separate lines. | Insert the newline. |
| `demo.py:63,91` | CLI flag is `--prompt` while the README documents `--input_prompt`; `StableDiffusionPipeline` is used but never imported. | Rewritten. |

## 7. Library substitutions

* **CLIP** — the reference uses OpenAI's `clip` package, which does not install
  on Python 3.12. We load the identical weights through `transformers`
  (`openai/clip-vit-large-patch14` for EBR retrieval,
  `openai/clip-vit-base-patch32` for the metrics — the same split as the
  reference, which uses ViT-L/14 in `main.py` and ViT-B/32 in `metrics.py`).
* **Generator: SDXL, not SD v1-5** — the paper uses SD v1-5 at 512×512 with the
  PNDM scheduler (Sec. 5.1). The default here is
  `stabilityai/stable-diffusion-xl-base-1.0` at 1024×1024 with the scheduler the
  repo ships (EulerDiscrete) — PNDM is a v1-5 setting and SDXL is not tuned for
  it. Steps (50) and guidance scale (7.0) still follow Sec. 5.1.

  This changes the image-side numbers: the Image-Align proxy is a CLIP score
  over generated images, so **results produced under SDXL are not comparable
  with results produced under v1-5**, or with the paper's. `gen_meta.json`
  records the model, scheduler and resolution of each run for that reason.
  Text-side metrics (ROUGE-L, PMS) are unaffected — they never touch the
  generator.

  Pass `--model stable-diffusion-v1-5/stable-diffusion-v1-5` to
  `05_generate.py` (or set `TV_SD_MODEL`) for the paper-faithful configuration;
  resolution, scheduler and batch size follow the model automatically. Note that
  `runwayml/stable-diffusion-v1-5` was taken down in August 2024, so the
  identical weights come from the `stable-diffusion-v1-5` org instead.
* **ROUGE-L** — the vendored `src/tailored_visions_repro/rougeL/` from the official repo, because the
  paper needs β=5 (recall-weighted) and Google's `rouge_score` hardcodes β=1.
  Substituting it would shift the entire ROUGE-L column.
* **Demonstration ranking** — the reference ranks the five in-context examples
  with ViT-B/32; we reuse the ViT-L/14 query embeddings already computed for
  retrieval. Ranking five items by text similarity; the encoder choice is
  unlikely to matter, but it is a difference.

## 8. Evaluation scale

ROUGE-L needs no images, so it runs on the **full 6,232-sample test set** for
every method.

PMS and Image-Align need one SD v1-5 sample per method per test sample. All
methods at full scale would be ~44k images (~24 GPU-hours) to resolve gaps of
0.03–0.06 in metrics whose per-sample spread is ~0.1. At 1,000 samples the
standard error is ~0.003, roughly 10–20× smaller than the gaps in question, so
the image metrics run on a **fixed 500-user (1,000-sample) subset** chosen once
by `03_subset.py` and shared by every method. Every table reports its own `n`,
and `outputs/metrics.json` carries a standard error for every mean.

## 9. What the leakage check found

`07_leakage.py`, over all 6,232 test samples:

| | Count | Share |
|---|---|---|
| Ground-truth prompt appears **verbatim** in the same user's history | 2,217 | **35.6 %** |
| Token-F1 ≥ 0.8 against some history prompt | 3,062 | **49.1 %** |
| Shortened query **is** the ground-truth prompt | 1,135 | **18.2 %** |

The last row follows from the dataset construction: `data_process.process_line`
only abbreviates prompts longer than six words, so short prompts become their
own "shortened query" and the no-op baseline scores a perfect ROUGE-L on them.

The first two mean that on roughly a third to a half of the test set, retrieval
can hand the rewriter the answer. **ROUGE-L on the full test set therefore
overstates every history-conditioned method**, and the gap it shows between
personalized and non-personalized rewriting is partly a gap in retrieval's
ability to find a copy of the target. `06_evaluate.py --exclude-leaked` reports
ROUGE-L on the non-leaked remainder; treat that as the honest number and the
full-set figure as comparable-to-the-paper-but-inflated.

This is a property of the released dataset, not of this reproduction.
