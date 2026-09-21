"""Local-only exhibit API. All inference happens in managed subprocesses."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .config import ASSETS, CONFIG, REPO, read_json
from .domain import CARDS, reviewed_ids
from .preflight import sample_errors
from .service import Conflict, Service

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
    card_id: str = Field(max_length=60)
    aspects_off: list[str] = Field(default_factory=list, max_length=4)


class Selection(Body):
    cards: list[Card] = Field(max_length=16)


class Generate(Body):
    topic_id: str = Field(max_length=30)
    request_id: str = Field(min_length=1, max_length=100)


class Sample(Body):
    sample_id: str = Field(max_length=100)


@app.get("/api/config")
def config():
    reviewed = reviewed_ids()
    samples = read_json(ASSETS / "samples.json", []) or []
    manifest = read_json(ASSETS / "manifest.json", {}) or {}
    return {
        "cards": [
            {
                "id": card["id"],
                "subject_id": card["subject_id"],
                "profile_id": card["profile_id"],
                "label": card["label"],
                "subject_label": card["subject_label"],
                "profile_label": card["profile_label"],
                "aspects": card["aspects"],
                "aspects_ja": card["aspects_ja"],
                "url": "/assets/" + card["path"],
            }
            for card in CARDS.values()
            if card["id"] in reviewed
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
        "alpha": CONFIG["alpha"],
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
        "ready": bool(reviewed) and manifest.get("generation") == CONFIG["generation"],
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
    return service.set_selection(sid, [card.model_dump() for card in body.cards])


@app.post("/api/sessions/{sid}/runs")
def generate(sid: str, body: Generate):
    return service.start_run(sid, body.topic_id, body.request_id)


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
