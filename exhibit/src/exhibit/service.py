"""Session lifecycle and single-GPU job orchestration."""

from __future__ import annotations

import copy
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from .catalog import load_catalog
from .config import (
    ASSETS,
    CONFIG,
    FAN_POLICIES,
    FAN_UPSTREAM,
    OUTPUTS,
    read_json,
    write_json,
)
from .domain import (
    ASPECTS,
    build_personalization,
    digest,
    file_hash,
    run_cache_key,
    target_prompt,
)
from .elicitation import PreferenceError, RoundError, next_round, normalize_preferences
from .evaluation import runtime_tokenizer_provenance
from .fan_adapter import resolve_policy, thaw_policy
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
    "preference_revision",
    "preference",
    "policy_id",
    "policy_hash",
    "personalization_hash",
    "personalization",
    "feedback",
    "timings",
)
# Provenance and the profiling argument stay server-side; identity does not.
PUBLIC_PERSONALIZATION = (
    "refs",
    "alpha",
    "policy_id",
    "policy_hash",
    "personalization_hash",
    "hash",
    "effective_policy",
)
PUBLIC_ROUND = ("round_id", "round_index", "card_ids", "cards", "shortfall_reason")
PREFERENCES = ("plain", "personal", "tie")
# An older cache entry cannot prove which policy produced it, so it is refused.
CACHE_FIELDS = ("images", "timings", "personalization_hash", "policy_hash", "policy_id")
# The web-side sources that decide how a prompt and its references are encoded.
ENCODING_SOURCES = ("fan_adapter.py", "workers.py", "gpu.py")
IMAGE_IDENTITY = (
    "policy_id",
    "policy_hash",
    "effective_policy",
    "personalization_hash",
)


def _link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def active_catalog():
    """The reviewed catalog this exhibit serves; the client never names one."""
    return load_catalog(reviewed_only=True, assets=ASSETS)


def default_policy():
    """The server resolves the policy; a request can never carry encoder settings."""
    policy_id = FAN_POLICIES["default_policy_id"]
    return policy_id, thaw_policy(resolve_policy(policy_id, FAN_POLICIES))


def web_provenance():
    """Everything beside the preference that decides what an image will be."""
    source = Path(__file__).resolve().parent
    return {
        "fan_pin": CONFIG["fan"]["commit"],
        "adapter_hash": digest(
            {name: file_hash(source / name) for name in ENCODING_SOURCES}
        ),
        "decoder_hash": digest(CONFIG["fan"]["decoders"]),
        "tokenizer_hash": digest(runtime_tokenizer_provenance(CONFIG["generation"])),
        "generation": copy.deepcopy(CONFIG["generation"]),
        "seeds": list(CONFIG["seeds"]),
    }


class Service:
    def __init__(self, root=OUTPUTS, runner=None, provenance=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.session = None
        self.busy = False
        self.cancel = threading.Event()
        self.runner = runner or self._execute
        self._provenance = copy.deepcopy(provenance) if provenance else None

    # ---------------------------------------------------------------- session

    def _current(self, sid):
        if not self.session or self.session["id"] != sid:
            raise KeyError("Session expired")
        return self.session

    def provenance(self):
        """Computed once per process; a run never starts on a placeholder hash."""
        with self.lock:
            if self._provenance is None:
                try:
                    self._provenance = web_provenance()
                except (OSError, TypeError, ValueError) as exc:
                    raise ValueError(f"生成の由来を確認できません: {exc}") from exc
            return copy.deepcopy(self._provenance)

    def create_session(self):
        with self.lock:
            self.expire_idle()
            if self.session or self.busy:
                raise Conflict(
                    "この端末で体験中です。元の画面を再開するか、処理終了までお待ちください。"
                )
            catalog = active_catalog()
            self.session = {
                "id": uuid.uuid4().hex,
                "session_seed": uuid.uuid4().hex,
                "catalog_id": catalog["catalog_id"],
                "catalog_hash": catalog["catalog_hash"],
                "available_ids": [card["id"] for card in catalog["cards"]],
                "revision": 0,
                "selection": [],
                "aspect_gains": {aspect: 1.0 for aspect in ASPECTS},
                "committed": False,
                "rounds": [],
                "shown_ids": [],
                "ledger": {},
                "run": None,
                "cache": {},
                "last_active": time.monotonic(),
            }
            (self.root / "sessions" / self.session["id"]).mkdir(parents=True)
            self._serve_round(self.session, catalog, allow_empty=True)
            return self._snapshot()

    def snapshot(self, sid):
        with self.lock:
            self._current(sid)
            return self._snapshot()

    def touch(self, sid):
        with self.lock:
            self._current(sid)["last_active"] = time.monotonic()

    # ------------------------------------------------------------ preferences

    def _busy_guard(self, session):
        """A cancelled worker still owns the GPU until its child has exited."""
        run = session["run"]
        if self.busy or (run and run["status"] != "done"):
            raise Conflict("生成中です。終わるまで選択は変えられません。")

    def _revision_guard(self, session, expected_revision):
        if type(expected_revision) is not int:
            raise PreferenceError("expected_revision must be an integer")
        if expected_revision != session["revision"]:
            raise Conflict("画面が古くなっています。最新の状態を読み込んでください。")

    def _ledger(self, session, request_id, kind, payload):
        """`request_id` is bound to one kind and one payload for the whole session."""
        if not isinstance(request_id, str) or not request_id or len(request_id) > 100:
            raise PreferenceError("Invalid request ID")
        payload_hash = digest({"kind": kind, "payload": payload})
        entry = session["ledger"].get(request_id)
        if entry and entry["payload_hash"] != payload_hash:
            raise Conflict("Request ID was reused with different input")
        return payload_hash, entry

    def _remember_catalog(self, session, catalog):
        """Snapshots stay free of disk work; every mutation refreshes the catalog."""
        session["catalog_hash"] = catalog["catalog_hash"]
        session["available_ids"] = [card["id"] for card in catalog["cards"]]

    def _draft(self, session):
        return {
            "selection": copy.deepcopy(session["selection"]),
            "aspect_gains": dict(session["aspect_gains"]),
            "committed": session["committed"],
        }

    def _preference_snapshot(self, session, catalog):
        """The server-owned snapshot; the catalog hash never comes from a client."""
        return {
            "revision": session["revision"],
            "catalog_id": catalog["catalog_id"],
            "catalog_hash": catalog["catalog_hash"],
            "selection": copy.deepcopy(session["selection"]),
            "aspect_gains": dict(session["aspect_gains"]),
        }

    def set_selection(self, sid, *, expected_revision, cards, aspect_gains, commit):
        """A full replace. Identical content keeps the revision and the run."""
        with self.lock:
            session = self._current(sid)
            self._busy_guard(session)
            self._revision_guard(session, expected_revision)
            if type(commit) is not bool:
                raise PreferenceError("commit must be a boolean")
            catalog = active_catalog()
            normalized = normalize_preferences(
                {"cards": cards, "aspect_gains": aspect_gains},
                catalog,
                commit=commit,
                selection=CONFIG["selection"],
            )
            shown = set(session["shown_ids"])
            if any(entry["card_id"] not in shown for entry in normalized["selection"]):
                raise PreferenceError("まだ表示していない画像は選べません。")
            content = {
                "selection": normalized["selection"],
                "aspect_gains": normalized["aspect_gains"],
                "committed": commit,
            }
            if content != self._draft(session):
                session.update(copy.deepcopy(content))
                session["revision"] += 1
                # The finished comparison belongs to the previous preference; its
                # images stay on disk as this session's cache.
                session["run"] = None
            self._remember_catalog(session, catalog)
            session["last_active"] = time.monotonic()
            return self._snapshot()

    # ----------------------------------------------------------------- rounds

    def _serve_round(self, session, catalog, *, allow_empty=False):
        record = next_round(
            catalog,
            {"selection": copy.deepcopy(session["selection"])},
            shown_ids=list(session["shown_ids"]),
            round_index=len(session["rounds"]),
            session_seed=session["session_seed"],
            selection=CONFIG["selection"],
        )
        if not record["card_ids"] and not allow_empty:
            raise RoundError("お見せできる画像がもうありません。")
        session["rounds"].append(record)
        session["shown_ids"].extend(record["card_ids"])
        self._remember_catalog(session, catalog)
        return record

    def _round_available(self, session):
        if len(session["rounds"]) >= CONFIG["selection"]["max_rounds"]:
            return False
        shown = set(session["shown_ids"])
        return any(card_id not in shown for card_id in session["available_ids"])

    def request_round(self, sid, request_id, expected_revision):
        with self.lock:
            session = self._current(sid)
            payload = {"expected_revision": expected_revision}
            payload_hash, entry = self._ledger(session, request_id, "round", payload)
            if entry:
                # The same request never shows a card twice.
                return self._snapshot()
            self._busy_guard(session)
            self._revision_guard(session, expected_revision)
            if len(session["rounds"]) >= CONFIG["selection"]["max_rounds"]:
                raise RoundError(
                    f"選択は最大{CONFIG['selection']['max_rounds']}回までです。"
                )
            catalog = active_catalog()
            record = self._serve_round(session, catalog)
            session["ledger"][request_id] = {
                "kind": "round",
                "payload_hash": payload_hash,
                "round_id": record["round_id"],
            }
            session["last_active"] = time.monotonic()
            return self._snapshot()

    # --------------------------------------------------------------- snapshot

    def _snapshot(self):
        session = self.session
        return {
            "id": session["id"],
            "revision": session["revision"],
            "catalog_id": session["catalog_id"],
            "catalog_hash": session["catalog_hash"],
            "committed": session["committed"],
            "selection": copy.deepcopy(session["selection"]),
            "aspect_gains": dict(session["aspect_gains"]),
            "rounds": [
                {key: copy.deepcopy(record[key]) for key in PUBLIC_ROUND}
                for record in session["rounds"]
            ],
            "shown_ids": list(session["shown_ids"]),
            "next_round_available": self._round_available(session),
            "run": self._public_run(session),
        }

    def _public_run(self, session):
        run = session["run"]
        if not run:
            return None
        public = copy.deepcopy({key: run[key] for key in PUBLIC_RUN})
        public["personalization"] = {
            key: value
            for key, value in public["personalization"].items()
            if key in PUBLIC_PERSONALIZATION
        }
        return public

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

    def start_run(self, sid, topic_id, request_id, expected_revision):
        with self.lock:
            session = self._current(sid)
            payload = {"topic_id": topic_id, "expected_revision": expected_revision}
            payload_hash, entry = self._ledger(session, request_id, "run", payload)
            run = session["run"]
            if entry:
                if run and run["id"] == entry["run_id"]:
                    return self._snapshot()
                raise Conflict("この生成はすでに終わっています。")
            if topic_id not in {t["id"] for t in CONFIG["topics"]}:
                raise ValueError("Unknown topic")
            self._revision_guard(session, expected_revision)
            if not session["committed"]:
                raise ValueError(
                    f"{CONFIG['selection']['min']}枚以上を選んで決定してください。"
                )
            if self.busy or (run and run["status"] != "done"):
                raise Conflict("処理中です。")
            catalog = active_catalog()
            self._remember_catalog(session, catalog)
            topic = next(t for t in CONFIG["topics"] if t["id"] == topic_id)
            policy_id, policy = default_policy()
            personalization = build_personalization(
                self._preference_snapshot(session, catalog),
                prompt=target_prompt(topic),
                policy=policy,
                provenance=self.provenance(),
                catalog=catalog,
            )
            personalization["policy_id"] = policy_id
            # Every request is a fresh comparison; the finished one is replaced.
            session["run"] = self._create_run(
                sid, topic, request_id, session, catalog, personalization
            )
            session["ledger"][request_id] = {
                "kind": "run",
                "payload_hash": payload_hash,
                "run_id": session["run"]["id"],
            }
            session["last_active"] = time.monotonic()
            self.busy = True
            self.cancel = threading.Event()
            threading.Thread(
                target=self._work,
                args=(copy.deepcopy(session), self.cancel),
                daemon=True,
            ).start()
            return self._snapshot()

    def _create_run(self, sid, topic, request_id, session, catalog, personalization):
        run_id = uuid.uuid4().hex
        directory = self.root / "sessions" / sid / run_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            plain = self._plain(topic["id"], directory, sid)
        except GPUError as exc:
            raise ValueError(f"通常画像のキャッシュを使えません: {exc}") from exc
        return {
            "id": run_id,
            "request_id": request_id,
            "topic_id": topic["id"],
            "prompt": target_prompt(topic),
            "status": "generating",
            "mode": "live",
            "message": "描く準備をしています。",
            "elapsed_seconds": 0,
            "error": None,
            "plain": plain,
            "personal": [],
            "preference_revision": session["revision"],
            "preference": self._preference_snapshot(session, catalog),
            "policy_id": personalization["policy_id"],
            "policy_hash": personalization["policy_hash"],
            "personalization_hash": personalization["hash"],
            "personalization": personalization,
            "feedback": None,
            "timings": {},
            "metrics": [],
        }

    # ------------------------------------------------------------- publishing

    def publish(self, sid, run_id, revision, personalization_hash, updates):
        """A result only reaches the screen if session, run and preference match."""
        with self.lock:
            if not self.session or self.session["id"] != sid:
                return False
            run = self.session["run"]
            if (
                not run
                or run["id"] != run_id
                or run["preference_revision"] != revision
                or run["personalization"]["hash"] != personalization_hash
            ):
                return False
            run.update(copy.deepcopy(updates))
            return True

    def _work(self, session, cancel):
        started = time.monotonic()
        sid, run = session["id"], session["run"]
        run_id, revision = run["id"], run["preference_revision"]
        key = run["personalization"]["hash"]
        try:
            self.runner(session, cancel) if self.runner == (
                self._execute
            ) else self.runner(self, session)
        except Exception as exc:  # noqa: BLE001 - job boundary publishes and releases
            self.publish(
                sid,
                run_id,
                revision,
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

    def _run_json(self, sid, run):
        write_json(
            self.root / "sessions" / sid / run["id"] / "personal" / "run.json",
            copy.deepcopy(run),
        )

    def _usable_cache(self, entry, run, session_root):
        """Content-addressed, still on disk, and provably from this policy."""
        if not isinstance(entry, dict) or any(key not in entry for key in CACHE_FIELDS):
            return False
        if (
            entry["personalization_hash"] != run["personalization"]["hash"]
            or entry["policy_hash"] != run["policy_hash"]
            or entry["policy_id"] != run["policy_id"]
        ):
            return False
        return all(
            (session_root / image["relative_path"]).is_file()
            and file_hash(session_root / image["relative_path"]) == image["sha256"]
            for image in entry["images"]
        )

    def _execute(self, session, cancel):
        sid, run = session["id"], session["run"]
        run_id, revision = run["id"], run["preference_revision"]
        personalization = run["personalization"]
        key = personalization["hash"]
        publish = lambda **updates: self.publish(sid, run_id, revision, key, updates)
        started = time.monotonic()
        deadline = started + CONFIG["timeout_seconds"]
        session_root = self.root / "sessions" / sid
        directory = session_root / run_id / "personal"
        directory.mkdir(parents=True, exist_ok=True)
        cache_key = run_cache_key(run["topic_id"], personalization)
        cached = session["cache"].get(cache_key)
        if self._usable_cache(cached, run, session_root):
            images = copy.deepcopy(cached["images"])
            timings = copy.deepcopy(cached["timings"])
            self._run_json(
                sid,
                {
                    **run,
                    "personal": images,
                    "timings": timings,
                    "status": "done",
                    "mode": "exact-cache",
                },
            )
            publish(
                personal=images,
                timings=timings,
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
                expected = {
                    "policy_id": run["policy_id"],
                    "policy_hash": run["policy_hash"],
                    "effective_policy": personalization["effective_policy"],
                    "personalization_hash": key,
                }
                if any(event.get(field) != expected[field] for field in IMAGE_IDENTITY):
                    raise GPUError("Personalization mismatch")
                relative = str(Path(event["path"]).relative_to(session_root))
                images.append(
                    {
                        "id": event["id"],
                        "seed": event["seed"],
                        "sha256": event["sha256"],
                        "relative_path": relative,
                        "url": f"/api/sessions/{sid}/images/{relative}",
                        "policy_id": event["policy_id"],
                        "policy_hash": event["policy_hash"],
                        "personalization_hash": event["personalization_hash"],
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
            self._run_json(
                sid,
                {
                    **run,
                    "personal": images,
                    "timings": timings,
                    "metrics": metrics,
                    "status": "done",
                    "error": str(exc),
                },
            )
            publish(
                status="done",
                error=str(exc),
                personal=images,
                timings=timings,
                metrics=metrics,
                message=f"{len(images)}枚のみ生成しました。",
            )
            return
        self._run_json(
            sid,
            {
                **run,
                "personal": images,
                "timings": timings,
                "metrics": metrics,
                "status": "done",
            },
        )
        publish(
            personal=images,
            timings=timings,
            metrics=metrics,
            status="done",
            message="4枚ができました。",
        )
        with self.lock:
            if self.session and self.session["id"] == sid:
                self.session["cache"][cache_key] = copy.deepcopy(
                    {
                        "images": images,
                        "timings": timings,
                        "personalization_hash": key,
                        "policy_hash": run["policy_hash"],
                        "policy_id": run["policy_id"],
                    }
                )

    # --------------------------------------------------------------- feedback

    def set_feedback(self, sid, run_id, expected_revision, preference):
        """Optional, label-shown answer about one finished comparison."""
        with self.lock:
            session = self._current(sid)
            if preference not in PREFERENCES:
                raise ValueError("Unknown preference")
            if type(expected_revision) is not int:
                raise PreferenceError("expected_revision must be an integer")
            run = session["run"]
            if (
                not run
                or run["id"] != run_id
                or run["mode"] == "sample"
                or run["status"] != "done"
                or run["error"]
                or run["preference_revision"] != expected_revision
                or expected_revision != session["revision"]
            ):
                raise Conflict("この結果には回答できません。")
            if not run["feedback"] or run["feedback"]["preference"] != preference:
                run["feedback"] = {
                    "preference": preference,
                    "preference_revision": run["preference_revision"],
                    "personalization_hash": run["personalization"]["hash"],
                }
                self._run_json(sid, run)
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
            if not item or sample_errors(item, ASSETS, catalog=active_catalog()):
                raise ValueError("Sample unavailable or inconsistent")
            preference = copy.deepcopy(item["preference"])
            run_id = uuid.uuid4().hex
            directory = self.root / "sessions" / sid / run_id
            directory.mkdir(parents=True, exist_ok=True)
            plain = self._plain(item["topic_id"], directory, sid)
            personalization_hash = item["personalization"].get("hash")
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
                        "policy_id": None,
                        "policy_hash": None,
                        "personalization_hash": personalization_hash,
                    }
                )
            topic = next(t for t in CONFIG["topics"] if t["id"] == item["topic_id"])
            # A sample is somebody else's preference; the visitor's draft, revision
            # and rounds stay exactly as they were.
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
                "preference_revision": None,
                "preference": {
                    **preference,
                    "source": "sample",
                    "sample_id": sample_id,
                },
                "policy_id": None,
                "policy_hash": None,
                "personalization_hash": personalization_hash,
                "personalization": copy.deepcopy(item["personalization"]),
                "feedback": None,
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
            # Draft, rounds, request ledger, cache and images all go at once.
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
