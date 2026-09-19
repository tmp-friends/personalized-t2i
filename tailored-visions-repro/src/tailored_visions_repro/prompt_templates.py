"""Rewriter input templates.

The instruction block and the five in-context demonstrations are copied verbatim
from the official release (``prompts.py``); the instruction also appears as
Table 1 of the paper. Do not "improve" this wording -- it is the method.
"""

from __future__ import annotations

from typing import Sequence

# Table 1 of the paper / Example_template in the reference implementation.
INSTRUCTION = "\n".join(
    [
        "Prompt in text-to-image generation describes the detailed attributes of the object it plans to draw. User's preference in text-to-image generation is shown in history prompts.",
        "Given 3 history prompts, your task is to rewrite the current prompt so that it matches the user’s preference. The rewritten prompt should retain primary objects in the original prompt and conform to the user’s preference. Please avoid being too diffused and restrict your output within 70 words.",
    ]
)

# Reference: History_template. Six slots: example index, 3 history prompts,
# the query, and the gold rewrite.
DEMO_TEMPLATE = "\n".join(
    [
        "Example {}:",
        "The history prompts are:",
        "1. {}",
        "2. {}",
        "3. {} \n",
        "The new query is: {}",
        "The rewritten prompt is :{}\n\n",
    ]
)

# The five hand-written demonstrations shipped with the reference implementation.
# Each is [history_1, history_2, history_3, query, gold_rewrite].
EXAMPLE_1 = [
    "Over exposure, sharp depth of field, cinematic lighting, bright, natural colors, real colors, iso noise Volumetric Lighting,\nBacklight, night, 3D, realistic, midjournal portal, a girl sitting on a park lounge chair with a delicate Eastern face, big eyes, long eyelashes, delicate lips, and a few ducklings swimming in the river in the distance. There are flowers and plants by the river, and a small pavilion nearby. HDR, the highest image quality, high,definition portrayal, sunset light, and a sense of atmosphere,trending in artstation",
    "A woman rides a white horse galloping on the green grassland, with long golden hair flying, wearing a dress on her head, beautiful green eyes, delete face, delete hands, wearing a red transparent coat Inside the gauge coat, she wears a white long skirt, and behind her fly a white eagle, with 8K image quality, HDR, UHD, lighting, the highest image quality, masterpiece, fine portal, exit CG, high light, ultra clear and detailed, with a sense of atmosphere, depth of field, bokeh, pristine lighting, photosensitive environment, trending on Artstation, 4k, 8k,CG_Render",
    "Woman, with golden long hair, wild flower wreath on her head, beautiful blue eyes, exquisite duck egg face, slightly longer face shape, exquisite mouth, pink lip glaze, wearing a red transparent gauze garment, wearing a white long skirt inside the gauze garment, exquisite necklace, silver hair accessories, high mountain waterfalls, faint castles, sunset on the horizon, birds, light, dreamy and charming, looking at the waterfall, back to the camera, full body, highest picture quality, masterpiece, fine portrayal, exquisite CG, Natural light, ultra clear and detailed,CG_Render",
    "A woman stands in front of a castle",
    "A woman stands in front of a high mountain castle, with long brown hair, beautiful blue eyes, exquisite duck egg face, slightly longer face shape, exquisite mouth, pink lip glaze, wearing a transparent red gauze garment. Inside the gauze garment, she is wearing a white long skirt, exquisite necklace, silver hair accessories, flowing water waterfall, sunset on the horizon, beautiful birds, and light. The woman looks at the waterfall, smiling and not smiling, with her back to the camera, her whole body, the highest picture quality, masterpiece, fine portrayal, and exquisite CG, Natural light, ultra clear and detailed,CG_Render,trending in artstation",
]
EXAMPLE_2 = [
    "Close,up of a boy with blue eyes and white hair, surrounded by a golden halo and an angel halo, Impressionism, Claude Monet, Alfred Sisley, Vincent Willem van Gogh",
    "Back view, a boy carrying a backpack home, with a white Samoye at his feet. The sky is blue, and on the green grass, there are many scattered pink flowers,UnrealEngine, CG_Render",
    "Gentle white Samoye with pink and blue flowers and azure sky dragon sky next to it, Gamescene, trending in artstation ",
    "Little Samoye followed behind the hostess",
    "Little Samoye followed behind the hostess, surrounded by pink flowers and azure sky, Gamescene, trending in artstation",
]
EXAMPLE_3 = [
    "Best Quality, Masterpiece photo of a girl, Single Person, Perfect Eyes, Acquire Face, Acquire Skin, Black Hair, Long Hair, Necklace, Looking at Camera, City, Street, Night, Hotel, Skyscraper, Skyline, Top Shoulder, Bottom Split Short Skirt, Black Socks, A Pair of Red High Heels, 4k Ultra Clear, Highly detailed, pro Professional digital painting, Unreal Engine 5, Photorealism, HD quality, 8k resolution, cinema 4d, 3D, cinematic, professional photography, art by art and greg rutkowski and alphonse mucha and finish and WLOP,Photography",
    "Photography exquisitely portrays realistic characters in the style of Pino daeni, a female college student fashion masterpiece, a classic masterpiece, a close,up of a beautiful girl, black hair, side bun, red dress, happy, long hair fluttering, full moon, dark night, glowing red fireflies, amidst plum blossoms, forest in the distance, colorful and colorful oil painting techniques, the best composition,Photography, realistic.",
    "A beautiful photo of a European girl, exquisite and complex skirt, colorful clothing, mature and beautiful, with clear and moving eyes, sitting outdoors, (masterpiece), best quality, midjournal portrait, masterpiece, close,up, by Paul Hedley.",
    "A beautiful girl",
    "A delicate and beautiful girl with a pure temperament, red lips, black hair, straight hair, wearing a black off shoulder top, red split mini skirt, high heels, modern urban clothing. Standing on the evening street, with trees on both sides of the street and leaves falling all over the street, the whole body is photographed, realistic, 8k, The best quality, masterpiece, highlights, beautiful.",
]
EXAMPLE_4 = [
    "Black hair, handsome and cute esports man, best quality, full details, realistic photography",
    "(masterpiece), best quality, close up of a boy with black eyes and hair, smiling face, cat ears, CG_rendering, 8k uhd, trending in artstation",
    "A handsome and cute teenager with yellow hair and white cloth clothes and a sword, 8k, high quality, trending in artstation.",
    "Young man with black hair",
    "(Masterpiece), best quality, handsome and cute young man with black hair and white cloth clothes holding a sword, exquisite details, trending in artstation.",
]
EXAMPLE_5 = [
    "The character is a young girl. The background is a lush forest, surrounded by flowers and trees, the setting sun, dusk, and halo. The painting style is Hayao Miyazaki's style. Emotions are surprises, happiness, doubts, and looking,Cinematic",
    "Ghibli, a work by Hayao Miyazaki, features gouache colors, high saturation, and soft lines. The blue sky character is a young girl. The posture is wearing glasses, looking very gentle, with eyes focused on the front, and holding a pile of books in one's arms. Wearing simple, expressionless, with exquisite facial features and tied high ponytail. The background is the library,GhibliStudio",
    "The character is a Japanese romantic girl. Wearing a jk outfit, black stockings, and wearing eyes. Sunny and outgoing personality, yet full of scheming. The background is the football field on campus,undefined",
    "A little girl next door.",
    "The character is a little girl next door. The action is to ride a bicycle leisurely. Emotionally reserved, obedient, happy. Hayao Miyazaki's Painting Style. The scene is the setting sun, dusk, the seaside,Cinematic",
]

EXAMPLES = [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3, EXAMPLE_4, EXAMPLE_5]

TAIL = "The rewritten prompt (one sentence less than 70 words) is :"


# Reference ``get_ZS_example`` / ``get_ICL_example`` hardcode one block per
# supported k. Their trailing whitespace is inconsistent (k=1 and k=5 end
# "{}\n", k=3 and k=7 end "{} \n"); reproduced verbatim rather than normalized.
_QUESTION_BLOCKS: dict[int, list[str]] = {
    1: ["The history prompts are:", "1. {}\n", "The new query is: {}", TAIL],
    3: [
        "The history prompts are:",
        "1. {}",
        "2. {}",
        "3. {} \n",
        "The new query is: {}",
        TAIL,
    ],
    5: [
        "The history prompts are:",
        "1. {}",
        "2. {}",
        "3. {}",
        "4. {}",
        "5. {}\n",
        "The new query is: {}",
        TAIL,
    ],
    7: [
        "The history prompts are:",
        "1. {}",
        "2. {}",
        "3. {}",
        "4. {}",
        "5. {}",
        "6. {}",
        "7. {} \n",
        "The new query is: {}",
        TAIL,
    ],
}


def question_block(num_retrieval: int) -> str:
    """The slot block listing ``num_retrieval`` history prompts plus the query."""
    if num_retrieval in _QUESTION_BLOCKS:
        return "\n".join(_QUESTION_BLOCKS[num_retrieval])
    # The reference falls back to its k=7 block for any unlisted k; we generate
    # a matching block instead so arbitrary k is at least well-formed.
    lines = ["The history prompts are:"]
    for i in range(num_retrieval):
        lines.append(f"{i + 1}. {{}}" + (" \n" if i == num_retrieval - 1 else ""))
    lines += ["The new query is: {}", TAIL]
    return "\n".join(lines)


def build_template(num_retrieval: int = 3, demos: Sequence[Sequence[str]] = ()) -> str:
    """Assemble the full rewriter input template.

    ``demos`` is a list of demonstrations already ordered by decreasing
    similarity to the current query (the paper's order-sensitivity handling).
    An empty ``demos`` gives the context-independent ("naive") template.

    One deliberate departure from the released code: ``get_ZS_example`` returns
    ``instruction + question_block`` with no separator, gluing "...within 70
    words." onto "The history prompts are:". Table 1 of the paper shows them on
    separate lines, so we insert the newline.
    """
    text = INSTRUCTION
    for i, demo in enumerate(demos):
        text += "\n" + DEMO_TEMPLATE.format(i + 1, *demo)
    if demos:
        text += "\n" + "Question:\n"
    else:
        text += "\n"
    return text + question_block(num_retrieval)


def fill_template(template: str, history_prompts: Sequence[str], query: str) -> str:
    """Fill a template's slots, padding/truncating history to the slot count."""
    n_slots = template.count("{}") - 1
    hist = list(history_prompts)[:n_slots]
    hist += [""] * (n_slots - len(hist))
    return template.format(*hist, query)


def rank_demos(query_emb, demo_query_embs, examples: Sequence[Sequence[str]] = EXAMPLES):
    """Order demonstrations by decreasing similarity of their query to ``query``.

    Both embedding arrays must be L2-normalized. Mirrors ``prompts.rank_examples``
    (which scores each example by its *query* field, ``e[-2]``).
    """
    import numpy as np

    scores = np.asarray(demo_query_embs) @ np.asarray(query_emb)
    order = np.lexsort((np.arange(len(scores)), -scores))
    return [examples[i] for i in order]


# --------------------------------------------------------------------------
# Baseline: General Prompt Rewriting (no user history)
# --------------------------------------------------------------------------
# The paper compares against "General PR", a general-purpose T2I prompt
# rewriter that ignores the user entirely, but does not publish its prompt.
# This is our reconstruction, deliberately mirroring the personalized
# instruction's constraints (retain the primary object, <= 70 words) so the two
# differ only in the presence of user history.
GENERAL_PR_TEMPLATE = "\n".join(
    [
        "Prompt in text-to-image generation describes the detailed attributes of the object it plans to draw.",
        "Your task is to rewrite the current prompt into a better text-to-image prompt. The rewritten prompt should retain primary objects in the original prompt and add expressive details about style, lighting, composition and quality. Please avoid being too diffused and restrict your output within 70 words.",
        "The new query is: {}",
        TAIL,
    ]
)

# --------------------------------------------------------------------------
# User preference summarization (for the PMS metric)
# --------------------------------------------------------------------------
# Paper, Sec. 3.3: "For each user u in PIP dataset, we summarize his preference
# P_u into 5 phrases from his history prompts using ChatGPT." The exact wording
# is not published; this is our reconstruction of that instruction.
PREFERENCE_SUMMARY_TEMPLATE = "\n".join(
    [
        "Below are text-to-image prompts written by one user. They reveal the user's preference: the objects, styles, colors, lighting and quality attributes this user keeps asking for.",
        "Your task is to summarize this user's preference into exactly 5 short phrases, comma-separated, on a single line. Output only the 5 phrases, nothing else.",
        "",
        "The user's history prompts are:",
        "{}",
        "",
        "The user's preference in 5 comma-separated phrases is:",
    ]
)
