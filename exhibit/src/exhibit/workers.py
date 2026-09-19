"""Offline inference entry point. No model is loaded into the web process."""

import json
import resource
import sys
import time
from pathlib import Path

from .config import CONFIG
from .domain import AXES, file_hash, validate_prompt


def emit(kind, **data):
    print(json.dumps({"type": kind, **data}, ensure_ascii=False), flush=True)


def llm_load(vision=False):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        Qwen3_5ForConditionalGeneration,
    )

    c = CONFIG["llm"]
    kw = {"revision": c["revision"], "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(c["model"], **kw)
    if vision:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            c["model"], dtype=torch.bfloat16, device_map="cuda", **kw
        ).eval()
        processor = AutoProcessor.from_pretrained(c["model"], **kw)
        return model, processor
    return AutoModelForCausalLM.from_pretrained(
        c["model"], dtype=torch.bfloat16, device_map="cuda", **kw
    ).eval(), tokenizer


def llm_reply(model, tokenizer, text, budget=96):
    import torch

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=budget,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


def rewrite(request):
    from transformers import CLIPTokenizer

    settings = CONFIG["generation"]
    tokenizers = [
        CLIPTokenizer.from_pretrained(
            settings["model"],
            revision=settings["revision"],
            subfolder=s,
            local_files_only=True,
        )
        for s in ("tokenizer", "tokenizer_2")
    ]
    model, tokenizer = llm_load()
    for item in request["items"]:
        context = item.get("context")
        phrases = (
            [AXES[k]["values"][v][1] for k, v in context["preferences"].items()]
            if context and context["preferences"]
            else ["balanced composition", "natural lighting", "detailed textures"]
        )
        # The model orders only approved aesthetic phrases. It cannot rewrite the subject.
        instruction = (
            "Arrange these aesthetic phrases into a comma-separated image prompt suffix. "
            "Use each phrase exactly once. Add no other words, punctuation, quotation marks or explanation. "
            f"Keep the subject unchanged: {item['topic']['basic_prompt_en']}\n"
            f"Preference evidence: {context['text'] if context else 'No personal preferences.'}\n"
            f"Phrases: {', '.join(phrases)}"
        )
        valid = False
        raw = ""
        for attempt in range(2):
            raw = llm_reply(
                model,
                tokenizer,
                instruction
                + ("\nReturn ONLY the supplied phrases." if attempt else ""),
            )
            pieces = [s.strip().rstrip(".") for s in raw.split(",")]
            prompt = item["topic"]["basic_prompt_en"] + " " + ", ".join(pieces) + "."
            valid = sorted(pieces) == sorted(phrases) and validate_prompt(
                prompt, item["topic"], tokenizers
            )
            if valid:
                break
        emit(
            "rewrite",
            id=item["id"],
            prompt=prompt if valid else None,
            raw=raw,
            valid=valid,
            context_hash=context["hash"] if context else None,
            model=CONFIG["llm"],
            token_lengths=[len(t(prompt)["input_ids"]) for t in tokenizers],
        )


def generate(request):
    import torch
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline

    s = request.get("settings", CONFIG["generation"])
    pipe = StableDiffusionXLPipeline.from_pretrained(
        s["model"],
        revision=s["revision"],
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    emit("loaded", scheduler=dict(pipe.scheduler.config))
    for item in request["items"]:
        if not all(
            len(t(item["prompt"], truncation=False)["input_ids"]) <= t.model_max_length
            for t in (pipe.tokenizer, pipe.tokenizer_2)
        ):
            raise ValueError("SDXL tokenizer overflow")
        t = time.monotonic()
        image = pipe(
            item["prompt"],
            negative_prompt=s["negative_prompt"],
            num_inference_steps=s["steps"],
            guidance_scale=s["guidance_scale"],
            width=s["width"],
            height=s["height"],
            generator=torch.Generator(device="cuda").manual_seed(item["seed"]),
        ).images[0]
        path = Path(item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.png")
        image.save(temporary)
        temporary.replace(path)
        emit(
            "image",
            id=item["id"],
            path=str(path),
            sha256=file_hash(path),
            seed=item["seed"],
            prompt=item["prompt"],
            settings=s,
            seconds=round(time.monotonic() - t, 3),
            context_hash=item.get("context_hash", request.get("context_hash")),
        )


def analyze(request):
    import torch
    from PIL import Image

    model, processor = llm_load(vision=True)
    for item in request["items"]:
        images = []
        for path in item["images"]:
            image = Image.open(path).convert("RGB")
            image.thumbnail((448, 448))
            images.append(image)
        prompt = (
            f"These images share this subject: {item['basic_prompt_en']} "
            f"The user chose Image {item['chosen_position']}. Compare only the {item['dimension']} in the two images. "
            "In one short sentence state an observable relative visual preference supported by this choice. "
            "Do not infer personality or dislike. No headings or lists."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to("cuda")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=110, do_sample=False)
        raw = processor.decode(
            out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        emit(
            "analysis",
            id=item["id"],
            raw=raw,
            source=CONFIG["llm"],
            context_source="generic_vlm",
        )


def evaluate(request):
    from pigreward_repro.worker import evaluate_request

    evaluate_request(request, emit)


def main():
    import torch

    started = time.monotonic()
    request = json.loads(Path(sys.argv[1]).read_text())
    torch.set_num_threads(4)
    {
        "rewrite": rewrite,
        "generate": generate,
        "analyze": analyze,
        "evaluate": evaluate,
    }[request["stage"]](request)
    torch.cuda.synchronize()
    emit(
        "metrics",
        stage=request["stage"],
        worker_seconds=round(time.monotonic() - started, 3),
        peak_vram_mib=round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        peak_rss_mib=round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
    )


if __name__ == "__main__":
    main()
