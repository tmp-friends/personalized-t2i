"""Session lifecycle, blind comparison identity and single-GPU job orchestration."""

from __future__ import annotations

import copy
import os
import random
import secrets
import shutil
import threading
import time
import uuid
from pathlib import Path

from .config import ASSETS, CONFIG, FAN_UPSTREAM, OUTPUTS, read_json, write_json
from .domain import (
    build_cards,
    build_personalization,
    file_hash,
    normalize_selection,
    target_prompt,
    variant_cache_key,
)
from .gpu import GPUError, run_stage
from .preflight import sample_errors


class Conflict(RuntimeError):
    pass


def _link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


class Service:
    def __init__(self, root=OUTPUTS, runner=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.session = None
        self.busy = False
        self.cancel = threading.Event()
        self.runner = runner or self._execute

    # ---------------------------------------------------------------- session

    def _current(self, sid):
        if not self.session or self.session["id"] != sid:
            raise KeyError("Session expired")
        return self.session

    def create_session(self):
        with self.lock:
            self.expire_idle()
            if self.session or self.busy:
                raise Conflict(
                    "この端末で体験中です。元の画面を再開するか、処理終了までお待ちください。"
                )
            order = [card["id"] for card in build_cards()]
            random.SystemRandom().shuffle(order)
            self.session = {
                "id": uuid.uuid4().hex,
                "card_order": order,
                "selection": [],
                "run": None,
                "cache": {},
                "last_active": time.monotonic(),
            }
            (self.root / "sessions" / self.session["id"]).mkdir(parents=True)
            return self._snapshot()

    def snapshot(self, sid):
        with self.lock:
            self._current(sid)
            return self._snapshot()

    def touch(self, sid):
        with self.lock:
            self._current(sid)["last_active"] = time.monotonic()

    def set_selection(self, sid, cards):
        with self.lock:
            session = self._current(sid)
            # A cancelled worker may still be exiting; editing needs no GPU.
            if session["run"]:
                raise Conflict("選択をやり直すには終了してください。")
            session["selection"] = normalize_selection(cards)
            session["last_active"] = time.monotonic()
            return self._snapshot()

    # --------------------------------------------------------------- snapshot

    def _snapshot(self):
        session = self.session
        return {
            "id": session["id"],
            "card_order": list(session["card_order"]),
            "selection": copy.deepcopy(session["selection"]),
            "run": self._public_run(session),
        }

    def _public_run(self, session):
        run = session["run"]
        if not run:
            return None
        revealed = run["blind"]["revealed"]
        pairs = []
        for pair in run["blind"]["pairs"]:
            pairs.append(
                {
                    "index": pair["index"],
                    "seed": pair["seed"],
                    "ready": pair["ready"],
                    "items": [
                        {"token": i["token"], "url": i["url"]} for i in pair["items"]
                    ]
                    if pair["ready"]
                    else [],
                    "pick": pair["pick"],
                }
            )
        answered = sum(1 for pair in run["blind"]["pairs"] if pair["pick"] is not None)
        blind = {
            "revealed": revealed,
            "pairs": pairs,
            "answered": answered,
            "mapping": self._mapping(run) if revealed else None,
            "score": self._score(run) if revealed else None,
        }
        variants = []
        for index, variant in enumerate(run["variants"]):
            hidden = index == 0 and not revealed
            variants.append(
                {
                    "id": variant["id"],
                    "request_id": variant["request_id"],
                    "alpha_key": variant["alpha_key"],
                    "weights": dict(variant["weights"]),
                    "status": variant["status"],
                    "mode": variant["mode"],
                    "done_count": len(variant["images"]),
                    "images": [] if hidden else copy.deepcopy(variant["images"]),
                    "personalization": None
                    if hidden
                    else copy.deepcopy(variant["personalization"]),
                    "prompt": variant["prompt"],
                    "timings": copy.deepcopy(variant["timings"]),
                    "error": variant["error"],
                }
            )
        return {
            "id": run["id"],
            "topic_id": run["topic_id"],
            "status": run["status"],
            "message": run["message"],
            "elapsed_seconds": run["elapsed_seconds"],
            "error": run["error"],
            "blind": blind,
            "plain": copy.deepcopy(run["plain"]) if revealed else [],
            "variants": variants,
        }

    @staticmethod
    def _mapping(run):
        return {
            item["token"]: item["kind"]
            for pair in run["blind"]["pairs"]
            for item in pair["items"]
        }

    @staticmethod
    def _score(run):
        mapping = Service._mapping(run)
        picks = [pair["pick"] for pair in run["blind"]["pairs"]]
        return {
            "personal": sum(1 for p in picks if mapping.get(p) == "personal"),
            "plain": sum(1 for p in picks if mapping.get(p) == "plain"),
            "tie": sum(1 for p in picks if p == "tie"),
            "answered": sum(1 for p in picks if p is not None),
        }

    # ------------------------------------------------------------------- runs

    def _plain(self, topic_id, directory, sid):
        """Generic cache copied into the session so its URL reveals no method."""
        manifest = read_json(ASSETS / "manifest.json", {}) or {}
        if manifest.get("generation") != CONFIG["generation"]:
            raise GPUError("Generic cache settings mismatch")
        topic = next(t for t in CONFIG["topics"] if t["id"] == topic_id)
        prompt = target_prompt(topic)
        result = []
        for index, seed in enumerate(CONFIG["seeds"]):
            image = manifest.get("images", {}).get(f"{topic_id}-{index}")
            if (
                not image
                or image.get("seed") != seed
                or image.get("prompt") != prompt
                or image.get("settings") != CONFIG["generation"]
                or file_hash(ASSETS / image["path"]) != image["sha256"]
            ):
                raise GPUError("Generic cache missing or corrupt")
            relative = f"{directory.name}/plain-{index}.png"
            _link(ASSETS / image["path"], directory / f"plain-{index}.png")
            result.append(
                {
                    "id": f"plain-{index}",
                    "seed": seed,
                    "prompt": prompt,
                    "sha256": image["sha256"],
                    "relative_path": relative,
                    "url": f"/api/sessions/{sid}/images/{relative}",
                }
            )
        return result

    def _new_variant(
        self, index, request_id, alpha_key, weights, personalization, prompt
    ):
        return {
            "id": f"v{index}",
            "request_id": request_id,
            "alpha_key": alpha_key,
            "weights": dict(weights),
            "personalization": personalization,
            "status": "queued",
            "mode": "live",
            "images": [],
            "prompt": prompt,
            "timings": {},
            "metrics": [],
            "error": None,
            "cancelled": False,
        }

    def start_run(self, sid, topic_id, alpha_key, weights, request_id):
        with self.lock:
            session = self._current(sid)
            if not request_id or len(request_id) > 100:
                raise ValueError("Invalid request ID")
            if topic_id not in {t["id"] for t in CONFIG["topics"]}:
                raise ValueError("Unknown topic")
            selection = session["selection"]
            if len(selection) < CONFIG["selection"]["min"]:
                raise ValueError(
                    f"{CONFIG['selection']['min']}枚以上の画像選択が必要です。"
                )
            personalization = build_personalization(selection, weights, alpha_key)
            run = session["run"]
            if run and run["topic_id"] == topic_id and run["mode"] != "sample":
                existing = next(
                    (v for v in run["variants"] if v["request_id"] == request_id), None
                )
                if existing:
                    if existing["personalization"]["hash"] != personalization["hash"]:
                        raise Conflict("Request ID was reused with different input")
                    return self._snapshot()
            if self.busy:
                raise Conflict("処理中です。")
            if run and (run["topic_id"] != topic_id or run["mode"] == "sample"):
                # A new topic, or leaving the sample, starts a fresh comparison.
                if run["status"] != "done":
                    raise Conflict("処理中です。")
                session["run"] = run = None
            if run:
                if run["status"] != "done":
                    raise Conflict("処理中です。")
                if not run["blind"]["revealed"]:
                    raise Conflict("答えを見てから調整できます。")
                if len(run["variants"]) >= CONFIG["max_variants"]:
                    raise Conflict(f"描き直しは{CONFIG['max_variants']}回までです。")
            else:
                run = self._create_run(sid, topic_id)
                session["run"] = run
            variant = self._new_variant(
                len(run["variants"]),
                request_id,
                alpha_key,
                weights or {},
                personalization,
                run["prompt"],
            )
            run["variants"].append(variant)
            run["status"] = "generating"
            run["error"] = None
            run["message"] = "描く準備をしています。"
            session["last_active"] = time.monotonic()
            self.busy = True
            self.cancel = threading.Event()
            threading.Thread(
                target=self._work,
                args=(
                    copy.deepcopy(session),
                    run["id"],
                    copy.deepcopy(variant),
                    self.cancel,
                ),
                daemon=True,
            ).start()
            return self._snapshot()

    def _create_run(self, sid, topic_id):
        run_id = uuid.uuid4().hex
        directory = self.root / "sessions" / sid / run_id
        directory.mkdir(parents=True, exist_ok=True)
        topic = next(t for t in CONFIG["topics"] if t["id"] == topic_id)
        try:
            plain = self._plain(topic_id, directory, sid)
        except GPUError as exc:
            raise ValueError(f"通常画像のキャッシュを使えません: {exc}") from exc
        random_source = random.SystemRandom()
        pairs = []
        for index, seed in enumerate(CONFIG["seeds"]):
            personal = f"v0/v0-{index}.png"
            items = [
                {
                    "token": secrets.token_urlsafe(12),
                    "kind": "plain",
                    "relative_path": plain[index]["relative_path"],
                },
                {
                    "token": secrets.token_urlsafe(12),
                    "kind": "personal",
                    "relative_path": f"{run_id}/{personal}",
                },
            ]
            random_source.shuffle(items)
            for item in items:
                item["url"] = f"/api/sessions/{sid}/images/blind/{item['token']}.png"
            pairs.append(
                {
                    "index": index,
                    "seed": seed,
                    "ready": False,
                    "items": items,
                    "pick": None,
                }
            )
        return {
            "id": run_id,
            "topic_id": topic_id,
            "prompt": target_prompt(topic),
            "status": "queued",
            "mode": "live",
            "message": "描く準備をしています。",
            "elapsed_seconds": 0,
            "error": None,
            "plain": plain,
            "variants": [],
            "blind": {"revealed": False, "pairs": pairs},
        }

    # ------------------------------------------------------------- publishing

    def publish(self, sid, run_id, variant_id, personalization_hash, updates):
        """A result only reaches the screen if its session, run and preference match."""
        with self.lock:
            if not self.session or self.session["id"] != sid:
                return False
            run = self.session["run"]
            if not run or run["id"] != run_id:
                return False
            variant = next((v for v in run["variants"] if v["id"] == variant_id), None)
            if (
                not variant
                or variant["cancelled"]
                or variant["personalization"]["hash"] != personalization_hash
            ):
                return False
            variant.update(copy.deepcopy(updates))
            if variant["id"] == "v0":
                ready = {image["id"] for image in variant["images"]}
                for pair in run["blind"]["pairs"]:
                    pair["ready"] = f"v0-{pair['index']}" in ready
            if "message" in updates:
                run["message"] = updates["message"]
            run["status"] = (
                "generating"
                if any(v["status"] != "done" for v in run["variants"])
                else "done"
            )
            run["error"] = next(
                (v["error"] for v in run["variants"] if v["error"]), None
            )
            return True

    def _work(self, session, run_id, variant, cancel):
        started = time.monotonic()
        sid, vid = session["id"], variant["id"]
        key = variant["personalization"]["hash"]
        try:
            self.runner(session, run_id, variant, cancel) if self.runner == (
                self._execute
            ) else self.runner(self, session, run_id, variant)
        except Exception as exc:  # noqa: BLE001 - job boundary publishes and releases
            self.publish(
                sid,
                run_id,
                vid,
                key,
                {
                    "status": "done",
                    "error": str(exc),
                    "message": "処理を完了できませんでした。できた画像はそのまま見られます。",
                },
            )
        finally:
            with self.lock:
                # The run may have been dropped (cancel, topic change) while the
                # worker was still exiting; only the session decides cleanup.
                current = self.session["run"] if self.session else None
                if self.session and self.session["id"] == sid:
                    if current and current["id"] == run_id:
                        current["elapsed_seconds"] = round(
                            time.monotonic() - started, 2
                        )
                    self.session["last_active"] = time.monotonic()
                else:
                    shutil.rmtree(self.root / "sessions" / sid, ignore_errors=True)
                self.busy = False

    def _execute(self, session, run_id, variant, cancel):
        sid, vid = session["id"], variant["id"]
        personalization = variant["personalization"]
        key = personalization["hash"]
        publish = lambda **updates: self.publish(sid, run_id, vid, key, updates)
        started = time.monotonic()
        deadline = started + CONFIG["timeout_seconds"]
        session_root = self.root / "sessions" / sid
        directory = session_root / run_id / vid
        directory.mkdir(parents=True, exist_ok=True)
        cache_key = variant_cache_key(session["run"]["topic_id"], personalization)
        cached = session["cache"].get(cache_key)
        if cached and all(
            (session_root / image["relative_path"]).is_file()
            and file_hash(session_root / image["relative_path"]) == image["sha256"]
            for image in cached["images"]
        ):
            publish(
                images=copy.deepcopy(cached["images"]),
                timings=copy.deepcopy(cached["timings"]),
                status="done",
                mode="exact-cache",
                message="同じ設定・同じお題で、この体験中に生成した結果です。",
            )
            return
        write_json(directory / "personalization.json", personalization)
        images = []
        timings = {}
        metrics = []
        publish(status="generating", message="1枚ずつ描いています。")
        items = [
            {
                "id": f"{vid}-{index}",
                "prompt": variant["prompt"],
                "seed": seed,
                "path": str(directory / f"{vid}-{index}.png"),
                "personalization": personalization,
            }
            for index, seed in enumerate(CONFIG["seeds"])
        ]

        def image_event(event):
            if event["type"] == "image":
                if event.get("personalization_hash") != key:
                    raise GPUError("Personalization mismatch")
                relative = str(Path(event["path"]).relative_to(session_root))
                images.append(
                    {
                        "id": event["id"],
                        "seed": event["seed"],
                        "sha256": event["sha256"],
                        "relative_path": relative,
                        "url": f"/api/sessions/{sid}/images/{relative}",
                    }
                )
                publish(
                    images=images,
                    message=f"{len(images)} / {len(items)}枚ができました。",
                )
            elif event["type"] == "metrics":
                metrics.append(event)

        try:
            timings["generation"] = run_stage(
                {
                    "stage": "generate",
                    "items": items,
                    "upstream": str(FAN_UPSTREAM),
                    "personalization_hash": key,
                },
                directory,
                cancel,
                deadline,
                image_event,
            )
        except GPUError as exc:
            publish(
                status="done",
                error=str(exc),
                images=images,
                timings=timings,
                metrics=metrics,
                message=f"{len(images)}枚のみ生成しました。",
            )
            return
        publish(
            images=images,
            timings=timings,
            metrics=metrics,
            status="done",
            message="4枚ができました。",
        )
        write_json(
            directory / "variant.json",
            {**variant, "images": images, "timings": timings, "metrics": metrics},
        )
        with self.lock:
            current = self.session["run"] if self.session else None
            if self.session and self.session["id"] == sid and current:
                self.session["cache"][cache_key] = copy.deepcopy(
                    {"images": images, "timings": timings}
                )

    # ------------------------------------------------------- blind comparison

    def pick_blind(self, sid, pair_index, pick):
        with self.lock:
            session = self._current(sid)
            run = session["run"]
            if not run:
                raise Conflict("比較はまだ始まっていません。")
            if run["blind"]["revealed"]:
                raise Conflict("答えの表示後は変更できません。")
            pair = next(
                (p for p in run["blind"]["pairs"] if p["index"] == pair_index), None
            )
            if not pair:
                raise ValueError("Unknown pair")
            if not pair["ready"]:
                raise Conflict("この対はまだ描けていません。")
            if pick != "tie" and pick not in {i["token"] for i in pair["items"]}:
                raise ValueError("Unknown image token")
            pair["pick"] = pick
            session["last_active"] = time.monotonic()
            return self._snapshot()

    def reveal(self, sid):
        with self.lock:
            session = self._current(sid)
            run = session["run"]
            if not run:
                raise Conflict("比較はまだ始まっていません。")
            run["blind"]["revealed"] = True
            run["message"] = (
                "左右の答えです。お題も生成モデルも同じで、違うのは参照と強さだけです。"
            )
            session["last_active"] = time.monotonic()
            return self._snapshot()

    # ---------------------------------------------------------------- samples

    def sample(self, sid, sample_id):
        with self.lock:
            session = self._current(sid)
            if self.busy:
                raise Conflict("処理中です。")
            item = next(
                (
                    x
                    for x in (read_json(ASSETS / "samples.json", []) or [])
                    if x.get("id") == sample_id
                ),
                None,
            )
            if not item or sample_errors(item, ASSETS):
                raise ValueError("Sample unavailable or inconsistent")
            run_id = uuid.uuid4().hex
            directory = self.root / "sessions" / sid / run_id
            directory.mkdir(parents=True, exist_ok=True)
            plain = self._plain(item["topic_id"], directory, sid)
            images = []
            for index, image in enumerate(item["images"]):
                relative = f"{run_id}/v0/v0-{index}.png"
                _link(ASSETS / image["path"], directory / "v0" / f"v0-{index}.png")
                images.append(
                    {
                        "id": f"v0-{index}",
                        "seed": image["seed"],
                        "sha256": image["sha256"],
                        "relative_path": relative,
                        "url": f"/api/sessions/{sid}/images/{relative}",
                    }
                )
            topic = next(t for t in CONFIG["topics"] if t["id"] == item["topic_id"])
            pairs = []
            for index, seed in enumerate(CONFIG["seeds"]):
                entries = [
                    {
                        "token": secrets.token_urlsafe(12),
                        "kind": "plain",
                        "relative_path": plain[index]["relative_path"],
                    },
                    {
                        "token": secrets.token_urlsafe(12),
                        "kind": "personal",
                        "relative_path": images[index]["relative_path"],
                    },
                ]
                for entry in entries:
                    entry["url"] = (
                        f"/api/sessions/{sid}/images/blind/{entry['token']}.png"
                    )
                pairs.append(
                    {
                        "index": index,
                        "seed": seed,
                        "ready": True,
                        "items": entries,
                        "pick": None,
                    }
                )
            session["selection"] = item["selection"]
            session["run"] = {
                "id": run_id,
                "topic_id": item["topic_id"],
                "prompt": target_prompt(topic),
                "status": "done",
                "mode": "sample",
                "message": "代表的な選択から事前に生成したサンプルです。あなたの選択を反映した結果ではありません。",
                "elapsed_seconds": 0,
                "error": None,
                "plain": plain,
                "blind": {"revealed": True, "pairs": pairs},
                "variants": [
                    {
                        "id": "v0",
                        "request_id": f"sample-{sample_id}",
                        "alpha_key": item.get("alpha_key", "mid"),
                        "weights": {
                            entry["card_id"]: "normal" for entry in item["selection"]
                        },
                        "personalization": item["personalization"],
                        "status": "done",
                        "mode": "sample",
                        "images": images,
                        "prompt": target_prompt(topic),
                        "timings": {},
                        "metrics": [],
                        "error": None,
                        "cancelled": False,
                    }
                ],
            }
            session["last_active"] = time.monotonic()
            return self._snapshot()

    # ------------------------------------------------------------- lifecycle

    def cancel_run(self, sid):
        with self.lock:
            session = self._current(sid)
            self.cancel.set()
            run = session["run"]
            if run:
                running = [v for v in run["variants"] if v["status"] != "done"]
                for variant in running:
                    variant["cancelled"] = True
                    variant["status"] = "done"
                    variant["error"] = "cancelled"
                if (
                    any(v["id"] == "v0" for v in running)
                    and not run["blind"]["revealed"]
                ):
                    # Nothing was ever shown, so drop the run and let the visitor
                    # change their selection. The worker still owns its directory.
                    session["run"] = None
                else:
                    run["status"] = "done"
                    run["message"] = "中止しました。できた画像はそのまま見られます。"
            session["last_active"] = time.monotonic()
            return self._snapshot()

    def artifact(self, sid, name):
        with self.lock:
            session = self._current(sid)
            base = (self.root / "sessions" / sid).resolve()
            run = session["run"]
            if name.startswith("blind/"):
                token = name[len("blind/") :].removesuffix(".png")
                item = next(
                    (
                        entry
                        for pair in (run["blind"]["pairs"] if run else [])
                        for entry in pair["items"]
                        if entry["token"] == token and pair["ready"]
                    ),
                    None,
                )
                if not item:
                    raise FileNotFoundError(name)
                name = item["relative_path"]
            elif run and not run["blind"]["revealed"] and name.startswith(run["id"]):
                # Direct paths would let a client de-blind a pair by comparing bytes.
                raise FileNotFoundError(name)
            path = (base / name).resolve()
            if not path.is_relative_to(base) or path.suffix != ".png":
                raise ValueError("Invalid image path")
            if not path.is_file():
                raise FileNotFoundError(name)
            return path

    def reset(self, sid):
        with self.lock:
            self._current(sid)
            self.cancel.set()
            self.session = None
            if not self.busy:
                shutil.rmtree(self.root / "sessions" / sid, ignore_errors=True)

    def expire_idle(self):
        with self.lock:
            if (
                self.session
                and not self.busy
                and time.monotonic() - self.session["last_active"]
                > CONFIG["idle_seconds"]
            ):
                self.reset(self.session["id"])
