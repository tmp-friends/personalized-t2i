"""Local evaluator using a documented, independent task instruction."""

import json
from pathlib import Path

from .adapter import parse_judgment, tournament

TEMPLATE = """Compare Image 1 and Image 2 for the following user.
Original image prompt: {prompt}
User preference context (relative choices and explicit corrections):
{context}
Choose criteria supported by this context and compare observable features.
For each criterion use exactly this format:
Criterion: comparison reason. (Score 1: integer, Score 2: integer)
Then give the sums of the criterion scores on separate lines:
Score 1: total
Score 2: total
Finish with exactly "Image 1 is better" or "Image 2 is better". If tied, say "tie".
Do not infer personality. Evaluate the original prompt, not a rewritten prompt."""


def evaluate_request(request, emit):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs/model.json").read_text())
    path = root / "artifacts/evaluator"
    processor = AutoProcessor.from_pretrained(
        path, local_files_only=True, min_pixels=4 * 28 * 28, max_pixels=256 * 28 * 28
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    candidates = {c["id"]: c for c in request["candidates"]}

    def compare(a, b):
        images = []
        for key in (a, b):
            image = Image.open(candidates[key]["path"]).convert("RGB")
            image.thumbnail((config["max_image_edge"],) * 2)
            images.append(image)
        text = TEMPLATE.format(
            prompt=request["basic_prompt_en"], context=request["context"]["text"]
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ]
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[rendered], images=images, return_tensors="pt").to(
            "cuda"
        )
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=config["max_new_tokens"], do_sample=False
            )
        generated = output[0, inputs.input_ids.shape[1] :]
        raw = processor.decode(generated, skip_special_tokens=True)
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        record = parse_judgment(raw, [a, b], finished=int(generated[-1]) in eos)
        record.update(
            context_hash=request["context"]["hash"],
            model=config,
            template_hash=__import__("hashlib").sha256(TEMPLATE.encode()).hexdigest(),
        )
        emit("judgment", **record)
        return record

    if request.get("pairs"):
        for a, b in request["pairs"]:
            compare(a, b)
    else:
        result = tournament(list(candidates), compare, seed=request["shuffle_seed"])
        emit("recommendation", **result, context_hash=request["context"]["hash"])
