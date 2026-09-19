"""Derive the paper's shorter query variants (Table 3).

The PIP release ships only one query per test sample -- the "Short Sentence"
scale. Table 3 additionally reports "Noun Phrase" and "Noun", described only as
"derived using spaCy from our dataset, adhering to the principle of minimizing
word count while maintaining the main entities". The derivation code is not
released, so this is a **reconstruction**, not a reproduction: the rule below is
ours.

* ``noun_phrase`` -- spaCy noun chunks and named entities, in order, deduplicated.
* ``noun``        -- the head nouns and proper nouns of those chunks.

Both fall back to the original query when extraction yields nothing, which is
what "maintaining the main entities" requires: an empty prompt is not a shorter
prompt.
"""

from __future__ import annotations

from typing import Iterable, Sequence

DEFAULT_SPACY_MODEL = "en_core_web_sm"


class Shortener:
    def __init__(self, model: str = DEFAULT_SPACY_MODEL):
        import spacy

        try:
            self.nlp = spacy.load(model, disable=["lemmatizer", "textcat"])
        except OSError as exc:  # pragma: no cover - depends on local install
            raise SystemExit(
                f"spaCy model {model!r} is missing. Install it with:\n"
                f"    python -m spacy download {model}"
            ) from exc

    @staticmethod
    def _dedup(items: Iterable[str]) -> list[str]:
        seen: dict[str, None] = {}
        for item in items:
            item = item.strip()
            if item:
                seen.setdefault(item, None)
        return list(seen)

    def noun_phrases(self, doc) -> list[str]:
        chunks = [c.text for c in doc.noun_chunks]
        ents = [e.text for e in doc.ents]
        return self._dedup(chunks + [e for e in ents if e not in chunks])

    def nouns(self, doc) -> list[str]:
        heads = [c.root.text for c in doc.noun_chunks if c.root.pos_ in ("NOUN", "PROPN")]
        if not heads:
            heads = [t.text for t in doc if t.pos_ in ("NOUN", "PROPN")]
        return self._dedup(heads)

    def shorten_batch(self, texts: Sequence[str], kind: str, batch_size: int = 256) -> list[str]:
        """``kind`` is 'noun_phrase' or 'noun'; anything else returns the input."""
        if kind not in ("noun_phrase", "noun"):
            return list(texts)
        out = []
        for text, doc in zip(texts, self.nlp.pipe(texts, batch_size=batch_size)):
            parts = self.noun_phrases(doc) if kind == "noun_phrase" else self.nouns(doc)
            out.append(", ".join(parts) if parts else text)
        return out
