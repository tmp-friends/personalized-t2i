"""Prompt rewriters.

The paper uses ChatGPT (gpt-3.5-turbo, 2023) as its rewriter. That endpoint is
gone, so the default backend here is a local instruction-tuned model. The
rewriter is the method, so swapping it is the single largest deviation in this
reproduction -- see ``docs/DEVIATIONS.md``.

Backends:

* ``local``  -- any chat model loadable through ``transformers`` (default).
* ``openai`` -- any OpenAI-compatible chat endpoint, if you have a key.
* ``echo``   -- returns the query unchanged; the "Shortened Prompt" baseline.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Sequence

DEFAULT_LOCAL_MODEL = "Qwen/Qwen3.5-4B"

# The instruction ends with "The rewritten prompt (one sentence less than 70
# words) is :" and models love to echo that stem back before answering.
_ECHO_PREFIXES = [
    r"^\s*the rewritten prompt\s*(\([^)]*\))?\s*is\s*:?\s*",
    r"^\s*rewritten prompt\s*:?\s*",
    r"^\s*here(?:'s| is) the rewritten prompt\s*:?\s*",
    r"^\s*sure[,!]?\s*",
]
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_DANGLING_THINK = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)


def clean_output(text: str, max_words: int | None = None) -> str:
    """Normalize a raw LLM completion into a usable T2I prompt.

    Strips reasoning traces, the echoed instruction stem, wrapping quotes and
    newlines. ``max_words`` truncates; leave it ``None`` to stay faithful to the
    paper, which only *asks* for <= 70 words and never truncates.
    """
    if not text:
        return ""
    text = _THINK_BLOCK.sub(" ", text)
    if "</think>" in text.lower():  # generation hit the token budget mid-trace
        text = _DANGLING_THINK.sub(" ", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    for pattern in _ECHO_PREFIXES:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = text.strip().strip('"').strip("'").strip("“”").strip()
    if max_words is not None:
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words])
    return text


@dataclass
class RewriteRequest:
    key: str
    prompt: str  # the fully assembled rewriter input


class BaseRewriter:
    name = "base"

    def rewrite_batch(self, prompts: Sequence[str]) -> list[str]:
        raise NotImplementedError

    def rewrite(self, prompt: str) -> str:
        return self.rewrite_batch([prompt])[0]


class EchoRewriter(BaseRewriter):
    """No-op rewriter. Used for the 'Shortened Prompt' row of Table 2."""

    name = "echo"

    def rewrite_batch(self, prompts: Sequence[str]) -> list[str]:
        return list(prompts)


class LocalRewriter(BaseRewriter):
    """Batched greedy-ish decoding through ``transformers``.

    ``enable_thinking=False`` is passed to the chat template: Qwen3.5 opens a
    ``<think>`` block by default, and reasoning text leaking into the rewritten
    prompt silently corrupts every downstream metric.
    """

    name = "local"

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        device: str | None = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        batch_size: int = 16,
        enable_thinking: bool = False,
        seed: int = 0,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.enable_thinking = enable_thinking
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=getattr(torch, dtype), device_map=self.device
        ).eval()

    def _apply_template(self, user_prompt: str) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:  # template does not accept enable_thinking
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def rewrite_batch(self, prompts: Sequence[str]) -> list[str]:
        torch = self.torch
        outputs: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = [self._apply_template(p) for p in prompts[start : start + self.batch_size]]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=8192
            ).to(self.model.device)
            gen_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if self.temperature and self.temperature > 0:
                torch.manual_seed(self.seed + start)
                gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.95)
            else:
                gen_kwargs.update(do_sample=False)
            with torch.no_grad():
                out = self.model.generate(**enc, **gen_kwargs)
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            outputs.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return outputs

    def unload(self) -> None:
        """Free VRAM. The 4B rewriter and SD v1-5 should not be co-resident."""
        del self.model
        self.torch.cuda.empty_cache()


class OpenAIRewriter(BaseRewriter):
    """Any OpenAI-compatible chat endpoint (the paper's original setup)."""

    name = "openai"

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
        max_retries: int = 6,
        **_,
    ):
        from openai import OpenAI

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def rewrite_batch(self, prompts: Sequence[str]) -> list[str]:
        import time

        results = []
        for prompt in prompts:
            delay = 1.0
            for attempt in range(self.max_retries):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    results.append(resp.choices[0].message.content or "")
                    break
                except Exception as exc:  # noqa: BLE001 -- transient API errors
                    if attempt == self.max_retries - 1:
                        raise
                    print(f"  API error ({exc}); retry {attempt + 1} in {delay:.1f}s")
                    time.sleep(delay)
                    delay *= 2
        return results


def build_rewriter(backend: str = "local", **kwargs) -> BaseRewriter:
    backend = backend.lower()
    if backend == "echo":
        return EchoRewriter()
    if backend == "local":
        return LocalRewriter(**kwargs)
    if backend == "openai":
        return OpenAIRewriter(**kwargs)
    raise ValueError(f"unknown rewriter backend: {backend!r}")
