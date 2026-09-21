"""Server-owned, content-addressed card catalogs."""
from __future__ import annotations

import json
from pathlib import Path

from .config import ASSETS, CARDS_REVIEW, CONFIG, ROOT, read_json
from .domain import ASPECTS, build_cards, compose_prompt, digest, file_hash

V2 = ROOT / "configs/catalog-v2.json"
V2_REVIEW = ROOT / "configs/cards-v2-review.json"

def _copy(value): return json.loads(json.dumps(value))
def _description_hash(card): return digest({"aspects": card["aspects"], "aspects_ja": card.get("aspects_ja", {}), "ref_en": card["ref_en"], "label": card["label"], "profile_label": card["profile_label"]})

def build_catalog(definition):
    if not isinstance(definition, dict) or set(definition) != {"catalog_id", "subjects", "axes", "axes_ja", "generation"}: raise ValueError("Invalid catalog definition")
    axes=definition["axes"]
    axes_ja=definition["axes_ja"]
    if set(axes) != set(ASPECTS) or set(axes_ja) != set(ASPECTS) or any(not isinstance(axes[a], list) or len(axes[a]) != 4 or any(not isinstance(x,str) or not x.strip() for x in axes[a]) for a in ASPECTS): raise ValueError("Catalog requires four non-empty levels per axis")
    subjects=definition["subjects"]
    if not isinstance(subjects,list) or len(subjects)!=4 or len({x.get("id") for x in subjects}) != 4: raise ValueError("Catalog requires four unique subjects")
    cards=[]; mul2=(0,2,3,1)
    for subject in subjects:
      if not all(isinstance(subject.get(k),str) and subject[k] for k in ("id","label","basic_prompt_en")) or type(subject.get("seed")) is not int: raise ValueError("Invalid subject")
      for a in range(4):
       for b in range(4):
        levels={"color":a,"lighting":b,"texture":a^b,"mood":a^mul2[b]}
        profile=f"c{a}-l{b}-t{levels['texture']}-m{levels['mood']}"
        aspects={axis:axes[axis][levels[axis]] for axis in ASPECTS}
        aspects_ja={axis:axes_ja[axis][levels[axis]] for axis in ASPECTS}
        if any(not isinstance(value, str) or not value.strip() for value in aspects_ja.values()): raise ValueError("Catalog requires Japanese aspect labels")
        cid=f"{subject['id']}-{profile}"
        profile_label=f"{aspects_ja['color']}・{aspects_ja['texture']}"
        cards.append({"id":cid,"subject_id":subject["id"],"profile_id":profile,"subject_label":subject["label"],"profile_label":profile_label,"label":f"{subject['label']} · {profile_label}","axis_levels":levels,"aspects":aspects,"aspects_ja":aspects_ja,"ref_en":", ".join(aspects.values()),"prompt":compose_prompt(subject["basic_prompt_en"],list(aspects.values()),definition["generation"]),"seed":subject["seed"],"path":f"cards-v2/{cid}.png"})
    return cards

def _v1_cards():
    cards=[]
    # Stable categorical levels come from the distinct phrase per axis, not profile position.
    base=build_cards(); maps={a:{v:i for i,v in enumerate(dict.fromkeys(c["aspects"][a] for c in base))} for a in ASPECTS}
    for card in base:
      c=_copy(card); c["axis_levels"]={a:maps[a][c["aspects"][a]] for a in ASPECTS}; cards.append(c)
    return cards

def _v2_manifest(assets=ASSETS): return read_json(Path(assets) / "catalog-v2.json", {}) or {}
def _valid_v1(card, image, review, assets=ASSETS):
    if not isinstance(image, dict) or not isinstance(review, dict) or not review.get("reviewed"):
        return False
    try:
        path = (Path(assets) / image["path"]).resolve()
        image_ok = path.is_relative_to(Path(assets).resolve()) and file_hash(path) == image["sha256"]
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return image_ok and all(image.get(k) == card[k] for k in ("path", "seed", "prompt", "ref_en", "aspects")) and image.get("settings") == CONFIG["generation"]

def _valid_v2(card, image, review, assets=ASSETS):
    if not isinstance(image,dict) or not isinstance(review,dict): return False
    try:
      path=(Path(assets) / image["path"]).resolve()
      image_ok=path.is_relative_to(Path(assets).resolve()) and file_hash(path)==image["sha256"]
    except (OSError,KeyError,TypeError,ValueError): image_ok=False
    return image_ok and all(image.get(k)==card[k] for k in ("path","seed","prompt","ref_en","aspects")) and image.get("settings")==CONFIG["generation"] and review.get("reviewed") is True and review.get("image_sha256")==image.get("sha256") and review.get("description_hash")==_description_hash(card) and review.get("aspects")=={a:True for a in ASPECTS}

def load_catalog(catalog_id, *, reviewed_only=True, assets=ASSETS, review_path=None):
    if catalog_id == "catalog-v1":
      all_cards = _v1_cards()
      images = (read_json(Path(assets) / "manifest.json", {}) or {}).get("images", {})
      review = read_json(review_path or CARDS_REVIEW, {}) or {}
      eligible = [c for c in all_cards if _valid_v1(c, images.get(c["id"]), review.get(c["id"]), assets)]
    elif catalog_id == "catalog-v2":
      all_cards = build_catalog(read_json(V2))
      manifest = _v2_manifest(assets)
      if not isinstance(manifest, dict) or manifest.get("catalog_id") not in (None, "catalog-v2") or manifest.get("generation") not in (None, CONFIG["generation"]):
          raise ValueError("Invalid v2 manifest")
      images = manifest.get("images", {})
      review = read_json(review_path or V2_REVIEW, {}) or {}
      if not isinstance(images, dict) or not isinstance(review, dict):
          raise ValueError("Invalid v2 image or review manifest")
      eligible = [c for c in all_cards if _valid_v2(c, images.get(c["id"]), review.get(c["id"]), assets)]
    else: raise ValueError("Unknown catalog")
    # Content identity includes generated image hashes when available, never review metadata.
    hashes={c["id"]:(images.get(c["id"],{}) or {}).get("sha256") for c in all_cards}
    value={"catalog_id":catalog_id,"all_cards":all_cards,"image_hashes":hashes}
    return {"catalog_id":catalog_id,"catalog_hash":digest(value),"all_cards":_copy(all_cards),"cards":_copy(eligible if reviewed_only else all_cards)}

def validate_card_tokens(cards, tokenizers):
    rows=[]
    for card in cards:
      for name, tokenizer in tokenizers.items():
       encoded=tokenizer(card["prompt"], add_special_tokens=True)
       ids=encoded["input_ids"] if isinstance(encoded,dict) else encoded.input_ids
       rows.append({"card_id":card["id"],"tokenizer":name,"token_ids":list(ids),"tokens":len(ids),"limit":77,"overflow":len(ids)>77})
    return rows
