"""Session lifecycle, snapshot identity and single-GPU job orchestration."""

from __future__ import annotations

import copy
import random
import shutil
import threading
import time
import uuid
from pathlib import Path

from .config import ASSETS, CONFIG, OUTPUTS, REPO, read_json, write_json
from .domain import build_persona, digest, effective_context, file_hash
from .gpu import GPUError, run_stage
from .preflight import sample_errors


class Conflict(RuntimeError):
    pass


class Service:
    def __init__(self, root=OUTPUTS, runner=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.session = None
        self.busy = False
        self.cancel = threading.Event()
        self.runner = runner or self._execute

    def _current(self, sid):
        if not self.session or self.session["id"] != sid:
            raise KeyError("Session expired")
        return self.session

    def _snapshot(self):
        return copy.deepcopy(
            {k: v for k, v in self.session.items() if k not in ("cache", "last_active")}
        )

    def create_session(self):
        with self.lock:
            self.expire_idle()
            if self.session or self.busy:
                raise Conflict(
                    "この端末で体験中です。元の画面を再開するか、処理終了までお待ちください。"
                )
            pairs = copy.deepcopy(CONFIG["pairs"])
            for pair in pairs:
                random.SystemRandom().shuffle(pair["image_ids"])
            self.session = {
                "id": uuid.uuid4().hex,
                "pairs": pairs,
                "choices": [],
                "persona": None,
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

    def answer(self, sid, pair_id, chosen_id):
        with self.lock:
            session = self._current(sid)
            if self.busy or session["run"]:
                raise Conflict("選択をやり直すには終了してください。")
            pair = next((p for p in session["pairs"] if p["id"] == pair_id), None)
            if not pair or (
                chosen_id is not None and chosen_id not in pair["image_ids"]
            ):
                raise ValueError("Invalid pair or image")
            choice = {
                "pair_id": pair_id,
                "chosen_id": chosen_id,
                "other_id": next((i for i in pair["image_ids"] if i != chosen_id), None)
                if chosen_id
                else None,
                "skip": chosen_id is None,
                "display_order": pair["image_ids"][:],
            }
            session["choices"] = [
                c for c in session["choices"] if c["pair_id"] != pair_id
            ] + [choice]
            session["persona"] = build_persona(
                session["choices"], read_json(ASSETS / "evidence.json", [])
            )
            session["last_active"] = time.monotonic()
            return self._snapshot()

    def start_run(self, sid, topic_id, edits, request_id):
        with self.lock:
            s = self._current(sid)
            context = effective_context(s["persona"] or build_persona([], []), edits)
            fingerprint = digest({"topic": topic_id, "context": context["hash"]})
            if s["run"] and s["run"].get("request_id") == request_id:
                if s["run"]["fingerprint"] != fingerprint:
                    raise Conflict("Request ID was reused with different input")
                return self._snapshot()
            if self.busy:
                raise Conflict("処理中です。")
            if sum(c["chosen_id"] is not None for c in s["choices"]) < 3:
                raise ValueError("3回以上の画像選択が必要です。")
            if topic_id not in {t["id"] for t in CONFIG["topics"]}:
                raise ValueError("Unknown topic")
            if not request_id or len(request_id) > 100:
                raise ValueError("Invalid request ID")
            job = {
                "id": uuid.uuid4().hex,
                "request_id": request_id,
                "fingerprint": fingerprint,
                "context": context,
                "topic_id": topic_id,
                "status": "queued",
                "mode": "live",
                "generic": [],
                "personalized": [],
                "winner_id": None,
                "judgments": [],
                "message": "描く準備をしています。",
                "timings": {},
                "elapsed_seconds": 0,
                "selected_id": None,
            }
            s["run"] = job
            s["last_active"] = time.monotonic()
            self.busy = True
            self.cancel = threading.Event()
            threading.Thread(
                target=self._work,
                args=(copy.deepcopy(s), copy.deepcopy(job), self.cancel),
                daemon=True,
            ).start()
            return self._snapshot()

    def publish(self, sid, jid, context_hash, updates):
        with self.lock:
            if not self.session or self.session["id"] != sid:
                return False
            run = self.session["run"]
            if not run or run["id"] != jid or run["context"]["hash"] != context_hash:
                return False
            run.update(copy.deepcopy(updates))
            return True

    def _work(self, session, job, cancel):
        started = time.monotonic()
        try:
            self.runner(
                session, job, cancel
            ) if self.runner == self._execute else self.runner(self, session, job)
        except Exception as exc:  # noqa: BLE001 - job boundary must publish failure and release lease
            self.publish(
                session["id"],
                job["id"],
                job["context"]["hash"],
                {
                    "status": "done",
                    "winner_id": None,
                    "error": str(exc),
                    "message": "処理を完了できませんでした。完成した画像から選べます。画像がない場合はサンプル体験をご利用ください。",
                },
            )
        finally:
            with self.lock:
                self.publish(
                    session["id"],
                    job["id"],
                    job["context"]["hash"],
                    {"elapsed_seconds": round(time.monotonic() - started, 2)},
                )
                self.busy = False
                if self.session and self.session["id"] == session["id"]:
                    self.session["last_active"] = time.monotonic()
                else:
                    shutil.rmtree(
                        self.root / "sessions" / session["id"], ignore_errors=True
                    )

    def _generic(self, topic_id):
        manifest = read_json(ASSETS / "manifest.json", {})
        if manifest.get("generation") != CONFIG["generation"]:
            raise GPUError("Generic cache settings mismatch")
        result = []
        prompts = read_json(ASSETS / "generic-prompts.json", {})
        for i, seed in enumerate(CONFIG["seeds"]):
            image = manifest.get("images", {}).get(f"{topic_id}-{i}")
            if (
                not image
                or image["seed"] != seed
                or image["prompt"] != prompts[topic_id]["prompt"]
                or file_hash(ASSETS / image["path"]) != image["sha256"]
            ):
                raise GPUError("Generic cache missing or corrupt")
            result.append(
                {**image, "id": f"generic-{i}", "url": "/assets/" + image["path"]}
            )
        return result

    def _execute(self, s, job, cancel):
        sid, jid, context = s["id"], job["id"], job["context"]
        publish = lambda **u: self.publish(sid, jid, context["hash"], u)
        started = time.monotonic()
        deadline = started + CONFIG["timeout_seconds"]
        directory = self.root / "sessions" / sid / jid
        directory.mkdir(parents=True)
        topic = next(t for t in CONFIG["topics"] if t["id"] == job["topic_id"])
        generic = self._generic(topic["id"])
        publish(generic=generic, generic_prompt=generic[0]["prompt"])
        if not context["preferences"]:
            publish(
                status="done",
                mode="generic",
                message="好みを使わずに描いた4枚です。好きな1枚を選んでください。",
            )
            return
        key = digest(
            {
                "context": context,
                "topic": topic,
                "llm": CONFIG["llm"],
                "generation": CONFIG["generation"],
                "seeds": CONFIG["seeds"],
                "generic_prompt": generic[0]["prompt"],
            }
        )
        cached = s["cache"].get(key)
        if cached and all(
            (self.root / "sessions" / sid / x["relative_path"]).is_file()
            and file_hash(self.root / "sessions" / sid / x["relative_path"])
            == x["sha256"]
            for x in cached["personalized"]
        ):
            publish(
                **cached,
                status="done",
                mode="exact-cache",
                message="同じ好み・お題・設定で、この体験中に生成した結果です。",
            )
            return
        write_json(directory / "context.json", context)
        rewrites = []
        timings = {}
        metrics = []
        publish(
            status="rewriting", message="あなたの好みを、描き方の言葉にしています。"
        )

        def rewrite_event(e):
            if e["type"] == "rewrite":
                rewrites.append(e)
            elif e["type"] == "metrics":
                metrics.append(e)

        timings["rewrite"] = run_stage(
            {
                "stage": "rewrite",
                "items": [{"id": jid, "topic": topic, "context": context}],
            },
            directory,
            cancel,
            deadline,
            rewrite_event,
        )
        if not rewrites or not rewrites[-1]["valid"]:
            publish(
                status="done",
                mode="generic",
                error="rewrite_invalid",
                message="今回は好みを反映できませんでした。通常の4枚から選んでください。",
                timings=timings,
            )
            return
        prompt = rewrites[-1]["prompt"]
        images = []
        publish(
            status="generating",
            personalized_prompt=prompt,
            message="あなた向けの画像を1枚ずつ描いています。",
            timings=timings,
        )
        items = [
            {
                "id": f"personal-{i}",
                "prompt": prompt,
                "seed": seed,
                "path": str(directory / f"personal-{i}.png"),
            }
            for i, seed in enumerate(CONFIG["seeds"])
        ]

        def image_event(e):
            if e["type"] == "image":
                if e.get("context_hash") != context["hash"]:
                    raise GPUError("Context mismatch")
                relative = str(
                    Path(e["path"]).relative_to(self.root / "sessions" / sid)
                )
                images.append(
                    {
                        **e,
                        "relative_path": relative,
                        "url": f"/api/sessions/{sid}/images/{relative}",
                    }
                )
                publish(
                    personalized=images, message=f"{len(images)} / 4枚ができました。"
                )
            elif e["type"] == "metrics":
                metrics.append(e)

        try:
            timings["generation"] = run_stage(
                {"stage": "generate", "items": items, "context_hash": context["hash"]},
                directory,
                cancel,
                deadline,
                image_event,
            )
        except GPUError as e:
            publish(
                status="done",
                error=str(e),
                message=f"{len(images)}枚のみ生成しました。自動推薦は行いません。",
                personalized=images,
                timings=timings,
            )
            return
        recommendation = {"winner_id": None, "judgments": [], "reason": ""}
        pig = read_json(REPO / "pigreward-repro/configs/model.json", {})
        if (
            CONFIG["recommendation_enabled"]
            and pig.get("live_approved")
            and len(images) == 4
        ):
            publish(status="evaluating", message="この4枚からおすすめを選んでいます。")

            def judgment_event(e):
                if e["type"] == "judgment":
                    recommendation["judgments"].append(e)
                    publish(judgments=recommendation["judgments"])
                elif e["type"] == "recommendation":
                    recommendation.update(e)
                elif e["type"] == "metrics":
                    metrics.append(e)

            try:
                timings["evaluation"] = run_stage(
                    {
                        "stage": "evaluate",
                        "candidates": images,
                        "context": context,
                        "basic_prompt_en": topic["basic_prompt_en"],
                        "shuffle_seed": int(jid[:8], 16),
                    },
                    directory,
                    cancel,
                    deadline,
                    judgment_event,
                )
            except GPUError as e:
                recommendation["winner_id"] = None
                recommendation["error"] = str(e)
        result = {
            "generic": generic,
            "personalized": images,
            "generic_prompt": generic[0]["prompt"],
            "personalized_prompt": prompt,
            "winner_id": recommendation["winner_id"],
            "judgments": recommendation["judgments"],
            "reason": recommendation.get("reason", ""),
            "timings": timings,
            "metrics": metrics,
        }
        publish(
            **result,
            status="done",
            message="4枚ができました。最後は、あなた自身の好みで選んでください。"
            if result["winner_id"]
            else "4枚ができました。自動推薦は利用していません。好きな1枚を選んでください。",
        )
        write_json(directory / "run.json", {**job, **result, "rewrite": rewrites[-1]})
        with self.lock:
            if self.session and self.session["id"] == sid:
                self.session["cache"][key] = copy.deepcopy(result)

    def sample(self, sid, sample_id):
        with self.lock:
            s = self._current(sid)
            if self.busy:
                raise Conflict("処理中です。")
            item = next(
                (
                    x
                    for x in read_json(ASSETS / "samples.json", [])
                    if x["id"] == sample_id
                ),
                None,
            )
            if not item or sample_errors(item):
                raise ValueError("Sample unavailable or inconsistent")
            s["run"] = {
                "id": uuid.uuid4().hex,
                "status": "done",
                "mode": "sample",
                "topic_id": item["topic_id"],
                "context": item["context"],
                "personalized": [
                    {**x, "url": "/assets/" + x["path"]} for x in item["images"]
                ],
                "generic": self._generic(item["topic_id"]),
                "generic_prompt": self._generic(item["topic_id"])[0]["prompt"],
                "personalized_prompt": item["rewrite"]["prompt"],
                "winner_id": None,
                "judgments": [],
                "message": "代表的な選択履歴から事前に生成したサンプルです。あなたの選択を反映した結果ではありません。",
                "sample_choices": item["choices"],
                "elapsed_seconds": 0,
                "selected_id": None,
                "timings": {},
            }
            s["last_active"] = time.monotonic()
            return self._snapshot()

    def cancel_run(self, sid):
        with self.lock:
            s = self._current(sid)
            self.cancel.set()
            s["run"] = None
            s["last_active"] = time.monotonic()
            return self._snapshot()

    def select(self, sid, image_id):
        with self.lock:
            s = self._current(sid)
            run = s["run"]
            if not run or run["status"] != "done":
                raise Conflict("生成終了後に選べます。")
            if image_id != "none" and image_id not in {
                x["id"] for x in run["generic"] + run["personalized"]
            }:
                raise ValueError("Unknown image")
            run["selected_id"] = image_id
            s["last_active"] = time.monotonic()
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
