"""Session lifecycle and single-GPU job orchestration."""

from __future__ import annotations

import copy
import os
import random
import shutil
import threading
import time
import uuid
from pathlib import Path

from .config import ASSETS, CONFIG, FAN_UPSTREAM, OUTPUTS, read_json, write_json
from .domain import (
    build_cards,
    build_legacy_personalization,
    file_hash,
    normalize_selection,
    run_cache_key,
    target_prompt,
)
from .gpu import GPUError, run_stage
from .preflight import sample_errors


class Conflict(RuntimeError):
    pass


# What a snapshot shows of a run; the rest is bookkeeping.
PUBLIC_RUN = (
    "id",
    "topic_id",
    "status",
    "mode",
    "message",
    "elapsed_seconds",
    "error",
    "prompt",
    "plain",
    "personal",
    "personalization",
    "timings",
)


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
        return copy.deepcopy({key: run[key] for key in PUBLIC_RUN})

    # ------------------------------------------------------------------- runs

    def _plain(self, topic_id, directory, sid):
        """Generic cache copied into the session, so every image URL is session-scoped."""
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

    def start_run(self, sid, topic_id, request_id):
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
            personalization = build_legacy_personalization(selection)
            run = session["run"]
            if run and run["request_id"] == request_id:
                if (
                    run["topic_id"] != topic_id
                    or run["personalization"]["hash"] != personalization["hash"]
                ):
                    raise Conflict("Request ID was reused with different input")
                return self._snapshot()
            if self.busy or (run and run["status"] != "done"):
                raise Conflict("処理中です。")
            # Every request is a fresh comparison; the finished one is replaced.
            session["run"] = self._create_run(
                sid, topic_id, request_id, personalization
            )
            session["last_active"] = time.monotonic()
            self.busy = True
            self.cancel = threading.Event()
            threading.Thread(
                target=self._work,
                args=(copy.deepcopy(session), self.cancel),
                daemon=True,
            ).start()
            return self._snapshot()

    def _create_run(self, sid, topic_id, request_id, personalization):
        run_id = uuid.uuid4().hex
        directory = self.root / "sessions" / sid / run_id
        directory.mkdir(parents=True, exist_ok=True)
        topic = next(t for t in CONFIG["topics"] if t["id"] == topic_id)
        try:
            plain = self._plain(topic_id, directory, sid)
        except GPUError as exc:
            raise ValueError(f"通常画像のキャッシュを使えません: {exc}") from exc
        return {
            "id": run_id,
            "request_id": request_id,
            "topic_id": topic_id,
            "prompt": target_prompt(topic),
            "status": "generating",
            "mode": "live",
            "message": "描く準備をしています。",
            "elapsed_seconds": 0,
            "error": None,
            "plain": plain,
            "personal": [],
            "personalization": personalization,
            "timings": {},
            "metrics": [],
        }

    # ------------------------------------------------------------- publishing

    def publish(self, sid, run_id, personalization_hash, updates):
        """A result only reaches the screen if its session, run and preference match."""
        with self.lock:
            if not self.session or self.session["id"] != sid:
                return False
            run = self.session["run"]
            if (
                not run
                or run["id"] != run_id
                or run["personalization"]["hash"] != personalization_hash
            ):
                return False
            run.update(copy.deepcopy(updates))
            return True

    def _work(self, session, cancel):
        started = time.monotonic()
        sid, run_id = session["id"], session["run"]["id"]
        key = session["run"]["personalization"]["hash"]
        try:
            self.runner(session, cancel) if self.runner == (
                self._execute
            ) else self.runner(self, session)
        except Exception as exc:  # noqa: BLE001 - job boundary publishes and releases
            self.publish(
                sid,
                run_id,
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

    def _execute(self, session, cancel):
        sid, run = session["id"], session["run"]
        run_id = run["id"]
        personalization = run["personalization"]
        key = personalization["hash"]
        publish = lambda **updates: self.publish(sid, run_id, key, updates)
        started = time.monotonic()
        deadline = started + CONFIG["timeout_seconds"]
        session_root = self.root / "sessions" / sid
        directory = session_root / run_id / "personal"
        directory.mkdir(parents=True, exist_ok=True)
        cache_key = run_cache_key(run["topic_id"], personalization)
        cached = session["cache"].get(cache_key)
        if cached and all(
            (session_root / image["relative_path"]).is_file()
            and file_hash(session_root / image["relative_path"]) == image["sha256"]
            for image in cached["images"]
        ):
            publish(
                personal=copy.deepcopy(cached["images"]),
                timings=copy.deepcopy(cached["timings"]),
                status="done",
                mode="exact-cache",
                message="同じ好み・同じお題で、この体験中に生成した結果です。",
            )
            return
        write_json(directory / "personalization.json", personalization)
        images = []
        timings = {}
        metrics = []
        publish(message="1枚ずつ描いています。")
        items = [
            {
                "id": f"personal-{index}",
                "prompt": run["prompt"],
                "seed": seed,
                "path": str(directory / f"personal-{index}.png"),
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
                    personal=images,
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
                personal=images,
                timings=timings,
                metrics=metrics,
                message=f"{len(images)}枚のみ生成しました。",
            )
            return
        publish(
            personal=images,
            timings=timings,
            metrics=metrics,
            status="done",
            message="4枚ができました。",
        )
        write_json(
            directory / "run.json",
            {**run, "personal": images, "timings": timings, "metrics": metrics},
        )
        with self.lock:
            if self.session and self.session["id"] == sid:
                self.session["cache"][cache_key] = copy.deepcopy(
                    {"images": images, "timings": timings}
                )

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
                relative = f"{run_id}/personal/personal-{index}.png"
                _link(ASSETS / image["path"], directory / relative.split("/", 1)[1])
                images.append(
                    {
                        "id": f"personal-{index}",
                        "seed": image["seed"],
                        "sha256": image["sha256"],
                        "relative_path": relative,
                        "url": f"/api/sessions/{sid}/images/{relative}",
                    }
                )
            topic = next(t for t in CONFIG["topics"] if t["id"] == item["topic_id"])
            session["selection"] = item["selection"]
            session["run"] = {
                "id": run_id,
                "request_id": f"sample-{sample_id}",
                "topic_id": item["topic_id"],
                "prompt": target_prompt(topic),
                "status": "done",
                "mode": "sample",
                "message": "代表的な選択から事前に生成したサンプルです。あなたの選択を反映した結果ではありません。",
                "elapsed_seconds": 0,
                "error": None,
                "plain": plain,
                "personal": images,
                "personalization": item["personalization"],
                "timings": {},
                "metrics": [],
            }
            session["last_active"] = time.monotonic()
            return self._snapshot()

    # ------------------------------------------------------------- lifecycle

    def cancel_run(self, sid):
        with self.lock:
            session = self._current(sid)
            self.cancel.set()
            run = session["run"]
            if run and run["status"] != "done":
                # The comparison never completed, so drop the run and let the
                # visitor start over. The worker still owns its directory.
                session["run"] = None
            session["last_active"] = time.monotonic()
            return self._snapshot()

    def artifact(self, sid, name):
        with self.lock:
            self._current(sid)
            base = (self.root / "sessions" / sid).resolve()
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
