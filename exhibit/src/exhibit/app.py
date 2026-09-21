"""Local-only exhibit API. All inference happens in managed subprocesses."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from .config import ASSETS, CONFIG, REPO, read_json
from .domain import digest
from .elicitation import PreferenceError, RoundError
from .preflight import sample_errors
from .service import Conflict, Service, active_catalog, default_policy

service = Service()
STATIC = Path(__file__).parent / "static"
REPORT = REPO / "docs/reports/fan-demo"


@asynccontextmanager
async def lifespan(app):
    async def reap():
        while True:
            await asyncio.sleep(1)
            service.expire_idle()

    task = asyncio.create_task(reap())
    yield
    task.cancel()
    if service.session:
        service.reset(service.session["id"])


app = FastAPI(
    title="FAN / 同じ一文から、あなたの一枚を",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)


@app.middleware("http")
async def local_origin(request: Request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Origin mismatch"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    if request.url.path.startswith("/api/sessions"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(Conflict)
async def conflict(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=409)


# Both preference errors subclass ValueError; the closest handler wins.
@app.exception_handler(RoundError)
async def round_conflict(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=409)


@app.exception_handler(PreferenceError)
async def preference(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.exception_handler(ValueError)
async def invalid(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.exception_handler(KeyError)
async def missing(_, exc):
    return JSONResponse(
        {"detail": "体験が終了しました。最初から始めてください。"}, status_code=404
    )


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Card(Body):
    # Strength and aspects stay raw: the preference normalizer, not pydantic,
    # decides whether `true` or `1.0` is a strength.
    card_id: str = Field(max_length=60)
    strength: Any = Field(...)
    aspects: list[Any] = Field(max_length=8)


class Selection(Body):
    expected_revision: StrictInt
    cards: list[Card] = Field(max_length=32)
    aspect_gains: dict[str, Any]
    commit: StrictBool


class Round(Body):
    request_id: str = Field(min_length=1, max_length=100)
    expected_revision: StrictInt


class Generate(Body):
    topic_id: str = Field(max_length=30)
    request_id: str = Field(min_length=1, max_length=100)
    expected_revision: StrictInt


class Feedback(Body):
    expected_revision: StrictInt
    preference: str = Field(max_length=20)


class Sample(Body):
    sample_id: str = Field(max_length=100)


@app.get("/api/config")
def config():
    catalog = active_catalog()
    policy_id, policy = default_policy()
    samples = read_json(ASSETS / "samples.json", []) or []
    manifest = read_json(ASSETS / "manifest.json", {}) or {}
    return {
        "schema_version": CONFIG["schema_version"],
        "catalog_id": catalog["catalog_id"],
        "catalog_hash": catalog["catalog_hash"],
        "cards": [
            {
                "id": card["id"],
                "subject_id": card["subject_id"],
                "profile_id": card["profile_id"],
                "label": card["label"],
                "subject_label": card["subject_label"],
                "profile_label": card["profile_label"],
                "axis_levels": card["axis_levels"],
                "aspects": card["aspects"],
                "aspects_ja": card["aspects_ja"],
                "url": "/assets/" + card["path"],
            }
            for card in catalog["cards"]
        ],
        "aspects": CONFIG["aspect_labels"],
        "topics": [
            {
                "id": topic["id"],
                "label": topic["label"],
                "preview_url": f"/assets/generic/{topic['id']}-0.png",
            }
            for topic in CONFIG["topics"]
        ],
        # Display values come from the resolved policy, never from the root config.
        "alpha": policy["alpha"],
        "policy": {
            "policy_id": policy_id,
            "policy_hash": digest(policy),
            "alpha": policy["alpha"],
            "pooled_mode": policy["pooled_mode"],
            "reference_unit": policy["reference_unit"],
            "profiling": policy["profiling"],
        },
        "strengths": [1, 2],
        "aspect_gains": [0.5, 1, 2],
        "selection": CONFIG["selection"],
        "idle_seconds": CONFIG["idle_seconds"],
        "timeout_seconds": CONFIG["timeout_seconds"],
        "samples": [
            {
                "id": s["id"],
                "topic_id": s["topic_id"],
                "label": s.get("label", s["id"]),
                "preview_url": "/assets/" + s["images"][0]["path"],
            }
            for s in samples
            if not sample_errors(s, ASSETS)
        ],
        "ready": len(catalog["cards"]) >= CONFIG["selection"]["min"]
        and manifest.get("generation") == CONFIG["generation"],
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "gpu_busy": service.busy,
        "mode": "fan-live",
        "offline": True,
    }


@app.post("/api/sessions")
def create():
    return service.create_session()


@app.get("/api/sessions/{sid}")
def snapshot(sid: str):
    return service.snapshot(sid)


@app.post("/api/sessions/{sid}/touch")
def touch(sid: str):
    service.touch(sid)
    return {"ok": True}


@app.put("/api/sessions/{sid}/selection")
def selection(sid: str, body: Selection):
    return service.set_selection(
        sid,
        expected_revision=body.expected_revision,
        cards=[card.model_dump() for card in body.cards],
        aspect_gains=body.aspect_gains,
        commit=body.commit,
    )


@app.post("/api/sessions/{sid}/rounds")
def rounds(sid: str, body: Round):
    return service.request_round(sid, body.request_id, body.expected_revision)


@app.post("/api/sessions/{sid}/runs")
def generate(sid: str, body: Generate):
    return service.start_run(
        sid, body.topic_id, body.request_id, body.expected_revision
    )


@app.put("/api/sessions/{sid}/runs/{rid}/feedback")
def feedback(sid: str, rid: str, body: Feedback):
    return service.set_feedback(sid, rid, body.expected_revision, body.preference)


@app.post("/api/sessions/{sid}/sample")
def sample(sid: str, body: Sample):
    return service.sample(sid, body.sample_id)


@app.post("/api/sessions/{sid}/cancel")
def cancel(sid: str):
    return service.cancel_run(sid)


@app.delete("/api/sessions/{sid}")
def reset(sid: str):
    service.reset(sid)
    return {"ok": True, "releasing_gpu": service.busy}


@app.get("/api/sessions/{sid}/images/{name:path}")
def image(sid: str, name: str):
    try:
        return FileResponse(
            service.artifact(sid, name), headers={"Cache-Control": "no-store"}
        )
    except (ValueError, FileNotFoundError):
        raise HTTPException(404)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/fallback")
def fallback():
    return FileResponse(ASSETS / "fallback.html")


@app.get("/reference/{name}")
def reference(name: str):
    references = {
        "readme": REPO / "exhibit/README.md",
        "fan": REPO / "fan-repro/README.md",
        "spec": REPO
        / "docs/superpowers/specs/2026-09-21-fan-exhibition-demo-design.md",
    }
    if name not in references:
        raise HTTPException(404)
    return FileResponse(references[name], media_type="text/plain; charset=utf-8")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
REPORT.mkdir(parents=True, exist_ok=True)
app.mount("/report", StaticFiles(directory=REPORT, html=True), name="report")
