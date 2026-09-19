#!/usr/bin/env python
"""A minimal OpenAI-compatible chat server, so this repo runs without an API key.

The paper's rewriter is ChatGPT, reached through ``apiuse.py``. This server
provides the same OpenAI-compatible endpoint with a local model:

    python serve_local_llm.py                 # http://127.0.0.1:8000/v1
    export TV_OPENAI_BASE=http://127.0.0.1:8000/v1
    export TV_OPENAI_KEY=local
    python demo.py --input_prompt='a cat'

The compatibility patch updates ``apiuse.py`` to the current ``OpenAI`` client,
but the request still uses the standard chat-completions HTTP contract. Point
``TV_OPENAI_BASE`` at the real API instead and nothing else changes.

Only ``POST /v1/chat/completions`` is implemented, which is all this repo uses.
Requests are batched over a short window because ``main.py`` issues them one at
a time; batching turns a 6,232-sample run from days into hours.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"

_state: dict = {}


class Engine:
    """Loads the model once and serves requests, coalescing concurrent ones."""

    def __init__(self, model_name: str, max_new_tokens: int, batch_window: float, batch_size: int):
        self.max_new_tokens = max_new_tokens
        self.batch_window = batch_window
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[serve] loading {model_name} on {self.device} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=self.device
        ).eval()
        self.model_name = model_name
        self._lock = threading.Lock()
        self._queue: list[tuple[list[dict], int, threading.Event, list]] = []
        threading.Thread(target=self._loop, daemon=True).start()
        print("[serve] ready", flush=True)

    def _render(self, messages: list[dict]) -> str:
        try:
            # Qwen3.5 opens a <think> block by default; reasoning text leaking
            # into a rewritten prompt would corrupt every downstream metric.
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def submit(self, messages: list[dict], max_tokens: int) -> str:
        event = threading.Event()
        slot: list = []
        with self._lock:
            self._queue.append((messages, max_tokens, event, slot))
        event.wait()
        return slot[0]

    def _loop(self) -> None:
        while True:
            time.sleep(self.batch_window)
            with self._lock:
                if not self._queue:
                    continue
                batch, self._queue = self._queue[: self.batch_size], self._queue[self.batch_size :]
            prompts = [self._render(m) for m, _, _, _ in batch]
            budget = max(mt for _, mt, _, _ in batch)
            try:
                enc = self.tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192
                ).to(self.model.device)
                with torch.no_grad():
                    out = self.model.generate(
                        **enc,
                        max_new_tokens=min(budget, self.max_new_tokens),
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                texts = self.tokenizer.batch_decode(
                    out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
                )
            except Exception as exc:  # surface it to every caller in the batch
                texts = [f"[generation error: {exc}]"] * len(batch)
            for (_, _, event, slot), text in zip(batch, texts):
                slot.append(text.strip())
                event.set()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):  # quiet; main.py makes thousands of calls
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            engine: Engine = _state["engine"]
            self._send(200, {"object": "list", "data": [{"id": engine.model_name, "object": "model"}]})
        else:
            self._send(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": f"no route {self.path}"}})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send(400, {"error": {"message": str(exc)}})
            return
        engine: Engine = _state["engine"]
        text = engine.submit(req.get("messages", []), int(req.get("max_tokens") or 512))
        self._send(
            200,
            {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model", engine.model_name),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-window", type=float, default=0.05, help="seconds to coalesce requests")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    _state["engine"] = Engine(args.model, args.max_new_tokens, args.batch_window, args.batch_size)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] listening on http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
