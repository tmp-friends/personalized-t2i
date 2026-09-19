"""Baseline prompt rewriters that ignore the user.

Two of the paper's three baselines live here:

* **Shortened Prompt** -- the unmodified query (``rewriter.EchoRewriter``).
* **Promptist** -- Hao et al., "Optimizing Prompts for Text-to-Image
  Generation" (NeurIPS 2023); a GPT-2 policy RL-tuned to expand SD v1-4 prompts.
  Run through the authors' released checkpoint, ``microsoft/Promptist``.

**General PR** is not here: it is the same LLM as the personalized rewriter,
driven by ``prompt_templates.GENERAL_PR_TEMPLATE``, so it goes through
``rewriter.build_rewriter`` like any other LLM condition.
"""

from __future__ import annotations

from typing import Sequence

from .rewriter import BaseRewriter

PROMPTIST_MODEL = "microsoft/Promptist"
PROMPTIST_TOKENIZER = "gpt2"
_SUFFIX = " Rephrase:"


class PromptistRewriter(BaseRewriter):
    """Beam-search decoding with the released Promptist policy.

    Decoding hyperparameters (8 beams, ``length_penalty=-1.0``, 75 new tokens)
    are the authors' published inference settings.
    """

    name = "promptist"

    def __init__(
        self,
        model_name: str = PROMPTIST_MODEL,
        device: str | None = None,
        batch_size: int = 16,
        max_new_tokens: int = 75,
        num_beams: int = 8,
        length_penalty: float = -1.0,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.tokenizer = AutoTokenizer.from_pretrained(PROMPTIST_TOKENIZER)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()

    def rewrite_batch(self, prompts: Sequence[str]) -> list[str]:
        torch = self.torch
        eos = self.tokenizer.eos_token_id
        results: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = [p.strip() + _SUFFIX for p in prompts[start : start + self.batch_size]]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    num_return_sequences=1,
                    eos_token_id=eos,
                    pad_token_id=eos,
                    length_penalty=self.length_penalty,
                )
            decoded = self.tokenizer.batch_decode(
                out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for text in decoded:
                results.append(text.strip())
        return results

    def unload(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()
