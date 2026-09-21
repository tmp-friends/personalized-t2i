#!/usr/bin/env python3
"""Build a single, offline HTML page for the human review of catalog-v2.

catalog-v2 has 64 cards (4 subjects x 16 profiles). Each card claims four
aspects (color, lighting, texture, mood) as an English phrase plus a Japanese
label. The exhibit owner opens the built page, looks at every card, ticks the
aspects a museum visitor would actually see without being told, ticks a
subject-quality box, and exports ``cards-v2-review.json`` -- the file
``exhibit/src/exhibit/catalog.py`` (``_valid_v2``) requires before catalog-v2
can be shown.

This script only builds the page. It never marks anything reviewed itself:
every checkbox starts unticked unless an earlier, hash-matching review is
imported. ``image_sha256`` and ``description_hash`` are computed from the
files on disk at build time and embedded into the page, so an export made
against older images or an edited catalog definition can never validate
against the current ones (``catalog.py`` re-checks both on load).

    PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_review_sheet.py
    PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_review_sheet.py \\
        --out /tmp/review.html --embed-thumbnails
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from exhibit.catalog import description_hash, load_catalog
from exhibit.config import ASSETS, OUTPUTS, ROOT
from exhibit.domain import ASPECTS, file_hash

DEFAULT_OUT = OUTPUTS / "review/cards-v2-review.html"
DEFAULT_REVIEW = ROOT / "configs/cards-v2-review.json"
ASPECT_JA_LABELS = {
    "color": "色",
    "lighting": "光",
    "texture": "質感",
    "mood": "雰囲気",
}
SUBJECT_OK_LABEL = (
    "被写体OK（顔が見える・小さすぎない・肌の変色なし・場面がすり替わっていない）"
)


# ------------------------------------------------------------- data assembly


def gather_cards(*, assets, review_path):
    """Every catalog-v2 card, bound to its on-disk image bytes and description.

    Refuses clearly -- no traceback, no silently empty page -- if the
    generated manifest does not match the current catalog-v2 definition or
    its settings (``exhibit.catalog.card_settings``): the same check
    ``catalog.py``'s own loader performs before it will bind any image to a
    card, whether or not that card is being filtered to reviewed-only.
    """
    assets = Path(assets)
    manifest_path = assets / "catalog-v2.json"
    try:
        catalog = load_catalog(
            "catalog-v2", reviewed_only=False, assets=assets, review_path=review_path
        )
    except (ValueError, TypeError) as exc:
        raise SystemExit(
            f"{manifest_path} does not match the current catalog-v2 definition "
            "(exhibit/configs/catalog-v2.json) or its settings. Regenerate the "
            f"catalog-v2 assets before building the review sheet.\n"
            f"  underlying error: {exc}"
        ) from exc
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot read {manifest_path}: {exc}") from exc
    images = manifest.get("images") or {}
    cards = []
    for card in catalog["all_cards"]:
        image = images.get(card["id"]) or {}
        image_path = assets / card["path"]
        exists = image_path.is_file()
        try:
            sha256 = file_hash(image_path) if exists else None
        except OSError:
            sha256 = None
        cards.append(
            {
                **card,
                "manifest_sha256": image.get("sha256"),
                "image_path": image_path,
                "image_exists": exists,
                "image_sha256": sha256,
                "description_hash": description_hash(card),
            }
        )
    return cards


def read_review_dict(path):
    """The raw review file, tolerant of it being absent, empty, or ``{}``."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def imported_state(cards, review):
    """Prefill only where both hashes still match the built card exactly."""
    state = {}
    for card in cards:
        entry = review.get(card["id"])
        if not isinstance(entry, dict):
            continue
        if (
            entry.get("image_sha256") != card["image_sha256"]
            or entry.get("description_hash") != card["description_hash"]
        ):
            continue
        aspects = entry.get("aspects")
        if not isinstance(aspects, dict):
            aspects = {}
        state[card["id"]] = {
            "aspects": {axis: bool(aspects.get(axis)) for axis in ASPECTS},
            "subject_ok": bool(entry.get("subject_ok")),
            "note": entry.get("note") if isinstance(entry.get("note"), str) else "",
        }
    return state


# --------------------------------------------------------- export shaping
#
# `shape_review_entry` / `build_export` are the Python source of truth for
# the export schema `catalog.py`'s `_valid_v2` requires. The page's export
# button runs the JS function `shapeEntry` embedded below, which mirrors this
# logic exactly; keep the two in sync if either changes.


def shape_review_entry(*, image_sha256, description_hash, aspects, subject_ok, note):
    """One card's export entry. ``reviewed`` is true only if everything is."""
    aspects = {axis: bool((aspects or {}).get(axis)) for axis in ASPECTS}
    subject_ok = bool(subject_ok)
    reviewed = subject_ok and all(aspects.values())
    return {
        "reviewed": reviewed,
        "image_sha256": image_sha256,
        "description_hash": description_hash,
        "aspects": aspects,
        "subject_ok": subject_ok,
        "note": note or "",
    }


def build_export(cards, states, *, reviewer="", date=""):
    """The full ``cards-v2-review.json`` shape: every card id plus metadata."""
    review = {}
    for card in cards:
        state = states.get(card["id"]) or {}
        review[card["id"]] = shape_review_entry(
            image_sha256=card["image_sha256"],
            description_hash=card["description_hash"],
            aspects=state.get("aspects"),
            subject_ok=state.get("subject_ok"),
            note=state.get("note"),
        )
    review["_metadata"] = {
        "reviewer": reviewer,
        "date": date,
        "generated_by": "build_review_sheet.py",
    }
    return review


# --------------------------------------------------------------- thumbnails


def have_pillow():
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


def thumbnail_data_uri(path, *, max_size=640, quality=82):
    from PIL import Image

    try:
        image = Image.open(path).convert("RGB")
    except (OSError, ValueError):
        return None
    image.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def relative_url(target, start):
    """A forward-slash relative path from the HTML's directory to ``target``."""
    return Path(
        os.path.relpath(Path(target).resolve(), start=Path(start).resolve())
    ).as_posix()


# ------------------------------------------------------------------- HTML

STYLE = """
:root{color-scheme:dark;--bg:#100d0c;--panel:#191514;--raise:#221c1a;--line:#30282a;
--ink:#f8f1eb;--sub:#ab9f97;--mute:#75695f;--a1:#ff9a6b;--a2:#ff6f93;--on:#1a0e0a;
--ok:#8fd6a4;--warn:#ffd089;--bad:#ff8c8c;
--sans:"Noto Sans CJK JP","Noto Sans JP","Hiragino Kaku Gothic ProN","Yu Gothic",system-ui,sans-serif;
--mono:ui-monospace,"SFMono-Regular",Consolas,"Liberation Mono",monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6;font-size:14px}
header{position:sticky;top:0;z-index:5;background:rgba(16,13,12,.96);backdrop-filter:blur(6px);
border-bottom:1px solid var(--line);padding:14px 20px}
.brand{font-size:18px;font-weight:800;letter-spacing:-.5px;
background:linear-gradient(135deg,var(--a1),var(--a2));-webkit-background-clip:text;
background-clip:text;color:transparent;margin-right:14px}
.bar{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center}
.bar .grow{flex:1 1 auto}
.progress{font-family:var(--mono);font-size:12px;color:var(--sub);display:flex;gap:14px;flex-wrap:wrap}
.progress b{color:var(--ink)}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;align-items:center}
select,input[type=text],textarea,button{font-family:var(--sans);font-size:12.5px;
background:var(--raise);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:7px 10px}
button{cursor:pointer;font-weight:700}
button.primary{background:linear-gradient(135deg,var(--a1),var(--a2));color:var(--on);border:none}
button.primary:hover{filter:brightness(1.08)}
button:hover{border-color:var(--mute)}
label.check{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;color:var(--sub)}
main{max-width:1400px;margin:auto;padding:18px 20px 96px}
h1{font-size:20px;font-weight:800;margin:22px 0 4px}
p.lead{color:var(--sub);margin:0 0 18px;font-size:13px}
section.subject{margin-top:30px}
section.subject h2{font-size:16px;font-weight:800;border-bottom:1px solid var(--line);
padding-bottom:8px;display:flex;align-items:center;gap:10px}
section.subject h2 .count{font-family:var(--mono);font-size:11px;color:var(--mute);font-weight:400}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;margin-top:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;
display:flex;flex-direction:column}
.card.confirmed{border-color:var(--ok)}
.card.hidden{display:none}
.thumb{position:relative;background:var(--raise);cursor:zoom-in}
.thumb img{width:100%;display:block;aspect-ratio:4/5;object-fit:cover}
.thumb .missing{aspect-ratio:4/5;display:flex;align-items:center;justify-content:center;
color:var(--bad);font-size:12px;padding:12px;text-align:center}
.card-body{padding:12px 14px 14px;display:flex;flex-direction:column;gap:8px}
.card-title{font-weight:700;font-size:13.5px}
.card-id{font-family:var(--mono);font-size:10.5px;color:var(--mute);overflow-wrap:anywhere}
.aspect{display:flex;align-items:flex-start;gap:8px;font-size:12px;border:1px solid var(--line);
border-radius:8px;padding:6px 8px;background:var(--raise)}
.aspect input{margin-top:2px}
.aspect .ja{font-weight:700;white-space:nowrap}
.aspect .en{color:var(--sub);overflow-wrap:anywhere}
.subject-ok{display:flex;align-items:flex-start;gap:8px;font-size:12px;border:1px dashed var(--a1);
border-radius:8px;padding:7px 8px;color:var(--ink)}
.stale{color:var(--warn);font-size:10.5px}
textarea{width:100%;min-height:44px;resize:vertical}
.modal{position:fixed;inset:0;background:rgba(8,6,5,.9);display:none;align-items:center;
justify-content:center;z-index:20;padding:24px}
.modal.open{display:flex}
.modal img{max-width:94vw;max-height:92vh;border-radius:10px}
footer{margin-top:60px;border-top:1px solid var(--line);padding-top:18px;
font-size:11px;font-family:var(--mono);color:var(--mute)}
@media(max-width:640px){main{padding:14px 12px 80px}.grid{grid-template-columns:1fr}}
"""

# The JS export function mirrors `shape_review_entry` / `build_export` above.
SCRIPT = r"""
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("review-data").textContent);
  var STORAGE_KEY = "fanCardsV2ReviewState";
  var ASPECTS = DATA.aspect_order;

  function blankState() {
    var aspects = {};
    ASPECTS.forEach(function (a) { aspects[a] = false; });
    return { aspects: aspects, subject_ok: false, note: "" };
  }

  function loadStorage() {
    try {
      var raw = window.localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (e) {
      return {};
    }
  }

  function saveStorage() {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(STATE));
    } catch (e) {
      /* private mode / storage disabled: state stays in memory only */
    }
  }

  var stored = loadStorage();
  var STATE = {};
  DATA.cards.forEach(function (card) {
    var entry = stored[card.id];
    if (
      entry &&
      entry.image_sha256 === card.image_sha256 &&
      entry.description_hash === card.description_hash
    ) {
      STATE[card.id] = {
        aspects: Object.assign(blankState().aspects, entry.aspects || {}),
        subject_ok: !!entry.subject_ok,
        note: entry.note || "",
      };
      return;
    }
    var imported = DATA.imported[card.id];
    if (imported) {
      STATE[card.id] = {
        aspects: Object.assign(blankState().aspects, imported.aspects || {}),
        subject_ok: !!imported.subject_ok,
        note: imported.note || "",
      };
      return;
    }
    STATE[card.id] = blankState();
  });

  function persist(cardId) {
    var card = DATA.cardsById[cardId];
    STATE[cardId].image_sha256 = card.image_sha256;
    STATE[cardId].description_hash = card.description_hash;
    saveStorage();
  }

  function isConfirmed(cardId) {
    var s = STATE[cardId];
    return !!s.subject_ok && ASPECTS.every(function (a) { return !!s.aspects[a]; });
  }

  function shapeEntry(card, s) {
    var aspects = {};
    ASPECTS.forEach(function (a) { aspects[a] = !!(s.aspects && s.aspects[a]); });
    var subjectOk = !!s.subject_ok;
    var reviewed = subjectOk && ASPECTS.every(function (a) { return aspects[a]; });
    return {
      reviewed: reviewed,
      image_sha256: card.image_sha256,
      description_hash: card.description_hash,
      aspects: aspects,
      subject_ok: subjectOk,
      note: (s.note || "").toString(),
    };
  }

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (key) {
      if (key === "text") node.textContent = attrs[key];
      else node.setAttribute(key, attrs[key]);
    });
    (children || []).forEach(function (child) { node.appendChild(child); });
    return node;
  }

  var main = document.getElementById("main");
  var modal = document.getElementById("modal");
  var modalImg = document.getElementById("modal-img");

  function openModal(card) {
    modalImg.src = card.image_full || card.image_src || "";
    modal.classList.add("open");
  }
  modal.addEventListener("click", function () { modal.classList.remove("open"); });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") modal.classList.remove("open");
  });

  function cardNode(card) {
    var s = STATE[card.id];
    var thumb = el("div", { class: "thumb" });
    if (card.image_exists) {
      var img = el("img", { src: card.image_src, alt: card.label, loading: "lazy" });
      thumb.appendChild(img);
      thumb.addEventListener("click", function () { openModal(card); });
    } else {
      thumb.appendChild(el("div", { class: "missing", text: "画像が見つかりません: " + card.path }));
    }
    var body = el("div", { class: "card-body" });
    body.appendChild(el("div", { class: "card-title", text: card.label }));
    body.appendChild(el("div", { class: "card-id", text: card.id }));

    ASPECTS.forEach(function (axis) {
      var input = el("input", { type: "checkbox", "data-card": card.id, "data-aspect": axis });
      if (s.aspects[axis]) input.setAttribute("checked", "checked");
      if (!card.image_exists) input.setAttribute("disabled", "disabled");
      input.addEventListener("change", function () {
        s.aspects[axis] = input.checked;
        persist(card.id);
        refreshCard(card.id);
        refreshProgress();
      });
      var label = el("label", { class: "aspect" }, [
        input,
        el("span", { class: "ja", text: DATA.aspect_ja_labels[axis] }),
        el("span", { class: "en", text: card.aspects[axis] + "（" + card.aspects_ja[axis] + "）" }),
      ]);
      body.appendChild(label);
    });

    var subjectInput = el("input", { type: "checkbox", "data-card": card.id, "data-subject": "1" });
    if (s.subject_ok) subjectInput.setAttribute("checked", "checked");
    if (!card.image_exists) subjectInput.setAttribute("disabled", "disabled");
    subjectInput.addEventListener("change", function () {
      s.subject_ok = subjectInput.checked;
      persist(card.id);
      refreshCard(card.id);
      refreshProgress();
    });
    body.appendChild(el("label", { class: "subject-ok" }, [
      subjectInput,
      el("span", { text: DATA.subject_ok_label }),
    ]));

    var note = el("textarea", { placeholder: "メモ（任意）", "data-card": card.id });
    note.value = s.note || "";
    note.addEventListener("input", function () {
      s.note = note.value;
      persist(card.id);
    });
    body.appendChild(note);

    var root = el("div", {
      class: "card" + (isConfirmed(card.id) ? " confirmed" : ""),
      id: "card-" + card.id,
      "data-subject": card.subject_id,
      "data-c": card.axis_levels.color,
      "data-l": card.axis_levels.lighting,
      "data-t": card.axis_levels.texture,
      "data-m": card.axis_levels.mood,
    });
    root.appendChild(thumb);
    root.appendChild(body);
    return root;
  }

  function refreshCard(cardId) {
    var node = document.getElementById("card-" + cardId);
    if (node) node.classList.toggle("confirmed", isConfirmed(cardId));
  }

  function refreshProgress() {
    var total = DATA.cards.length;
    var confirmed = 0;
    var perAspect = {};
    ASPECTS.forEach(function (a) { perAspect[a] = 0; });
    var subjectOkCount = 0;
    DATA.cards.forEach(function (card) {
      var s = STATE[card.id];
      ASPECTS.forEach(function (a) { if (s.aspects[a]) perAspect[a] += 1; });
      if (s.subject_ok) subjectOkCount += 1;
      if (isConfirmed(card.id)) confirmed += 1;
    });
    var bits = ["完了 " + confirmed + " / " + total];
    ASPECTS.forEach(function (a) {
      bits.push(DATA.aspect_ja_labels[a] + " " + perAspect[a] + " / " + total);
    });
    bits.push("被写体 " + subjectOkCount + " / " + total);
    document.getElementById("progress").innerHTML = bits
      .map(function (b) { return "<span><b>" + b.split(" / ")[0] + "</b> / " + b.split(" / ")[1] + "</span>"; })
      .join("");
  }

  function applyFilters() {
    var subject = document.getElementById("f-subject").value;
    var onlyUnchecked = document.getElementById("f-unchecked").checked;
    var axisFilters = {};
    ASPECTS.forEach(function (a) {
      var v = document.getElementById("f-" + a).value;
      axisFilters[a] = v === "" ? null : v;
    });
    DATA.cards.forEach(function (card) {
      var node = document.getElementById("card-" + card.id);
      if (!node) return;
      var show = true;
      if (subject && card.subject_id !== subject) show = false;
      ASPECTS.forEach(function (a) {
        if (axisFilters[a] !== null && String(card.axis_levels[a]) !== axisFilters[a]) show = false;
      });
      if (onlyUnchecked && isConfirmed(card.id)) show = false;
      node.classList.toggle("hidden", !show);
    });
  }

  function buildFilters() {
    var subjectSelect = document.getElementById("f-subject");
    DATA.subjects.forEach(function (subject) {
      subjectSelect.appendChild(el("option", { value: subject.id, text: subject.label }));
    });
    ASPECTS.forEach(function (axis) {
      var select = document.getElementById("f-" + axis);
      [0, 1, 2, 3].forEach(function (level) {
        select.appendChild(el("option", { value: String(level), text: String(level) }));
      });
    });
    ["f-subject", "f-color", "f-lighting", "f-texture", "f-mood", "f-unchecked"].forEach(
      function (id) {
        document.getElementById(id).addEventListener("change", applyFilters);
      }
    );
  }

  function buildPage() {
    DATA.subjects.forEach(function (subject) {
      var cards = DATA.cards.filter(function (c) { return c.subject_id === subject.id; });
      var section = el("section", { class: "subject" });
      section.appendChild(
        el("h2", {}, [
          document.createTextNode(subject.label),
          el("span", { class: "count", text: subject.id + " · " + cards.length + " 枚" }),
        ])
      );
      var grid = el("div", { class: "grid" });
      cards.forEach(function (card) { grid.appendChild(cardNode(card)); });
      section.appendChild(grid);
      main.appendChild(section);
    });
  }

  function download(filename, text) {
    var blob = new Blob([text], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var a = el("a", { href: url, download: filename });
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function doExport() {
    var review = {};
    DATA.cards.forEach(function (card) {
      review[card.id] = shapeEntry(card, STATE[card.id]);
    });
    review._metadata = {
      reviewer: document.getElementById("reviewer").value || "",
      date: document.getElementById("review-date").value || "",
      generated_by: "build_review_sheet.py (export)",
      exported_at: new Date().toISOString(),
      source_generated_at: DATA.generated_at,
    };
    download("cards-v2-review.json", JSON.stringify(review, null, 2) + "\n");
  }

  buildFilters();
  buildPage();
  refreshProgress();
  applyFilters();
  document.getElementById("export").addEventListener("click", doExport);
})();
"""


def render_html(cards, *, out_dir, embed_thumbnails, imported, generated_at):
    out_dir = Path(out_dir)
    subjects = []
    seen_subjects = set()
    for card in cards:
        if card["subject_id"] not in seen_subjects:
            seen_subjects.add(card["subject_id"])
            subjects.append({"id": card["subject_id"], "label": card["subject_label"]})

    card_payload = []
    for card in cards:
        image_src = None
        image_full = None
        if card["image_exists"]:
            image_full = relative_url(card["image_path"], out_dir)
            if embed_thumbnails:
                image_src = thumbnail_data_uri(card["image_path"]) or image_full
            else:
                image_src = image_full
        card_payload.append(
            {
                "id": card["id"],
                "subject_id": card["subject_id"],
                "subject_label": card["subject_label"],
                "profile_id": card["profile_id"],
                "profile_label": card["profile_label"],
                "label": card["label"],
                "path": card["path"],
                "axis_levels": card["axis_levels"],
                "aspects": card["aspects"],
                "aspects_ja": card["aspects_ja"],
                "image_exists": card["image_exists"],
                "image_src": image_src,
                "image_full": image_full,
                "image_sha256": card["image_sha256"],
                "description_hash": card["description_hash"],
            }
        )

    payload = {
        "generated_at": generated_at,
        "aspect_order": list(ASPECTS),
        "aspect_ja_labels": ASPECT_JA_LABELS,
        "subject_ok_label": SUBJECT_OK_LABEL,
        "subjects": subjects,
        "cards": card_payload,
        "cardsById": {card["id"]: card for card in card_payload},
        "imported": imported,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data_json = data_json.replace("</", "<\\/")

    total = len(cards)
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>catalog-v2 目視確認シート</title><style>{STYLE}</style></head>
<body>
<header>
<div class="bar">
<span class="brand">catalog-v2 review</span>
<div id="progress" class="progress grow"></div>
</div>
<div class="controls">
<label class="check">被写体<select id="f-subject"><option value="">すべて</option></select></label>
<label class="check">色<select id="f-color"><option value="">すべて</option></select></label>
<label class="check">光<select id="f-lighting"><option value="">すべて</option></select></label>
<label class="check">質感<select id="f-texture"><option value="">すべて</option></select></label>
<label class="check">雰囲気<select id="f-mood"><option value="">すべて</option></select></label>
<label class="check"><input type="checkbox" id="f-unchecked">未確認のみ</label>
<span class="grow"></span>
<label class="check">確認者<input type="text" id="reviewer" size="10"></label>
<label class="check">日付<input type="text" id="review-date" placeholder="YYYY-MM-DD" size="10"></label>
<button class="primary" id="export">エクスポート</button>
</div>
</header>
<main id="main">
<h1>catalog-v2 の目視確認（{total} 枚）</h1>
<p class="lead">来場者が説明されなくても分かる場合だけ、各カードの4項目にチェックを入れてください。
すべての項目と被写体OKにチェックが入ったカードだけが「確認済み」として書き出されます。
入力内容はブラウザーのローカルストレージに保存され、閉じても再開できます。
このページは何もチェック済みにしません。</p>
</main>
<div class="modal" id="modal"><img id="modal-img" alt=""></div>
<script id="review-data" type="application/json">{data_json}</script>
<script>{SCRIPT}</script>
<footer>build_review_sheet.py が生成 · {generated_at} ·
exhibit/configs/cards-v2-review.json のスキーマは exhibit/src/exhibit/catalog.py が検査します。</footer>
</body></html>"""


# ----------------------------------------------------------------------- CLI


def build(
    *,
    assets=ASSETS,
    review_path=None,
    out=DEFAULT_OUT,
    embed_thumbnails=False,
):
    assets = Path(assets)
    out = Path(out)
    if embed_thumbnails and not have_pillow():
        raise SystemExit(
            "--embed-thumbnails requires Pillow in exhibit/.venv (not found)"
        )
    if review_path is None:
        review_path = DEFAULT_REVIEW if DEFAULT_REVIEW.is_file() else None
    review = read_review_dict(review_path) if review_path else {}

    cards = gather_cards(assets=assets, review_path=review_path)
    imported = imported_state(cards, review)

    out.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    html_text = render_html(
        cards,
        out_dir=out.parent,
        embed_thumbnails=embed_thumbnails,
        imported=imported,
        generated_at=generated_at,
    )
    out.write_text(html_text)
    missing = [card["id"] for card in cards if not card["image_exists"]]
    return {
        "out": str(out),
        "cards": len(cards),
        "missing_images": missing,
        "imported": len(imported),
        "embed_thumbnails": embed_thumbnails,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--review",
        type=Path,
        default=None,
        help="existing review JSON to pre-fill hash-matching boxes from "
        "(default: exhibit/configs/cards-v2-review.json if non-empty)",
    )
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument(
        "--embed-thumbnails",
        action="store_true",
        help="embed downscaled JPEG thumbnails instead of referencing "
        "exhibit/assets/cards-v2/*.png by relative path (needs Pillow)",
    )
    args = parser.parse_args(argv)

    report = build(
        assets=args.assets,
        review_path=args.review,
        out=args.out,
        embed_thumbnails=args.embed_thumbnails,
    )
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
