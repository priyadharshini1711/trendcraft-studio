import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.config import ROOT_DIR, settings
from app.services import image, library, llm, news, reel

logger = logging.getLogger(__name__)

MAX_STYLE_IMAGES = 3
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_AUDIO_BYTES = 10 * 1024 * 1024

library.init()
_background_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    for task in _background_tasks:
        task.cancel()


app = FastAPI(title="TrendCraft", lifespan=lifespan)
app.mount("/media", StaticFiles(directory=library.MEDIA_DIR), name="media")


class AnglesRequest(BaseModel):
    niche: str | None = None
    custom_query: str | None = Field(default=None, max_length=100)


class ArticleIn(BaseModel):
    id: str
    title: str
    description: str = ""
    url: str
    source: str = ""
    published_at: str = ""

    @field_validator("url")
    @classmethod
    def http_only(cls, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError("url must be http(s)")
        return url


class AngleIn(BaseModel):
    title: str = Field(max_length=300)
    why_now: str = Field(max_length=1000)
    articles: list[ArticleIn] = Field(min_length=1, max_length=20)


class KitRequest(BaseModel):
    angle: AngleIn
    format: Literal["reel", "carousel", "single", "thread"]
    style: llm.Style | None = None
    niche: str = Field(default="", max_length=100)


class CoverRequest(BaseModel):
    cover_image_prompt: str = Field(min_length=1, max_length=2000)
    hook: str = Field(default="", max_length=300)
    palette: list[str] | None = Field(default=None, max_length=8)
    kit_id: str | None = None


class StatusRequest(BaseModel):
    status: Literal["new", "posted", "archived"]


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)


def _require_row(kit_id: str):
    row = library.get_row(kit_id)
    if not row:
        raise HTTPException(404, "That kit wasn't found.")
    return row


@app.exception_handler(news.NewsUnavailable)
async def news_error(_: Request, exc: news.NewsUnavailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(llm.LLMError)
async def llm_error(_: Request, exc: llm.LLMError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error: %s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong. Please try again."})


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "demo_mode": settings.demo_mode,
        "text_ready": settings.text_ready,
        "vision_ready": settings.vision_ready,
        "image_ready": settings.image_ready,
        "news_ready": bool(settings.news_api_key),
        "voice_ready": bool(settings.meta_transcribe_model and settings.meta_api_key),
    }


@app.get("/api/niches")
async def niches() -> dict:
    return {"niches": [{"id": n.id, "label": n.label} for n in news.NICHES.values()]}


@app.post("/api/news")
async def latest_news(body: AnglesRequest) -> dict:
    """Headlines on their own, so the UI can show real stories while angles are being found."""
    result = await news.get_articles(body.niche, body.custom_query)
    headlines = [{"title": a["title"], "source": a["source"]} for a in result["articles"]]
    return {"headlines": headlines, "origin": result["origin"]}


@app.post("/api/angles")
async def angles(body: AnglesRequest) -> dict:
    result = await news.get_articles(body.niche, body.custom_query)
    label = news.NICHES[body.niche].label if body.niche else (body.custom_query or "").strip()
    cache_key = f"angles:{news.cache_key(body.niche, body.custom_query)}"
    found = news.cache_get(cache_key)
    if not found:
        found = await llm.find_angles(label, result["articles"])
        if result["origin"] != "fixture":
            news.cache_put(cache_key, found)
    return {**found, "niche": label, "origin": result["origin"], "article_count": len(result["articles"])}


def _image_mime(data: bytes) -> str | None:
    """Identify an image from its leading bytes rather than trusting the upload's content type."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


@app.post("/api/style")
async def style(images: list[UploadFile] = File(...)) -> dict:
    if len(images) > MAX_STYLE_IMAGES:
        raise HTTPException(400, f"Upload at most {MAX_STYLE_IMAGES} screenshots.")
    prepared: list[tuple[str, bytes]] = []
    for upload in images:
        data = await upload.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(400, "Each screenshot must be 5 MB or smaller.")
        mime = _image_mime(data)
        if not mime:
            raise HTTPException(400, "Screenshots must be PNG, JPEG or WebP images.")
        prepared.append((mime, data))
    return await llm.extract_style(prepared)


@app.post("/api/kit")
async def kit(body: KitRequest) -> dict:
    """Write the kit and save it to the library. The fact-check and cover follow as separate calls."""
    angle = body.angle.model_dump()
    chosen_style = body.style.model_dump() if body.style else None
    built = await llm.build_kit(angle, angle["articles"], body.format, chosen_style)
    return library.save_kit(built, angle["title"], angle["why_now"], body.niche, angle["articles"], chosen_style)


@app.post("/api/cover")
async def cover(body: CoverRequest) -> dict:
    size = None
    row = None
    if body.kit_id:
        row = _require_row(body.kit_id)
        size = "1024x1536" if row["format"] == "reel" else "1024x1024"
    result = await image.generate_cover(body.cover_image_prompt, body.hook, body.palette, size)
    if not row:
        encoded = base64.b64encode(result["data"]).decode("ascii")
        return {"image_url": f"data:{result['mime']};base64,{encoded}", "placeholder": result["placeholder"]}
    filename = f"cover.{result['extension']}"
    (library.kit_dir(row["id"]) / filename).write_bytes(result["data"])
    library.update(row["id"], cover_file=filename, cover_placeholder=int(result["placeholder"]))
    return {"image_url": f"/media/{row['id']}/{filename}", "placeholder": result["placeholder"]}


@app.get("/api/library")
async def library_list() -> dict:
    return {"kits": library.list_kits()}


@app.get("/api/library/{kit_id}")
async def library_get(kit_id: str) -> dict:
    _require_row(kit_id)
    return library.get_kit(kit_id)


@app.patch("/api/library/{kit_id}")
async def library_status(kit_id: str, body: StatusRequest) -> dict:
    _require_row(kit_id)
    library.update(kit_id, status=body.status)
    return library.get_kit(kit_id)


@app.post("/api/library/{kit_id}/factcheck")
async def library_factcheck(kit_id: str) -> dict:
    row = _require_row(kit_id)
    result = await llm.check_kit(json.loads(row["kit_json"]), json.loads(row["articles_json"]))
    library.update_kit_fields(kit_id, **result)
    return library.get_kit(kit_id)


@app.post("/api/library/{kit_id}/reel")
async def library_reel(kit_id: str) -> dict:
    row = _require_row(kit_id)
    if row["format"] != "reel":
        raise HTTPException(400, "Reels can only be made from a Reel kit.")
    if row["reel_status"] != "rendering":
        library.update(kit_id, reel_status="rendering", reel_stage="Getting started", reel_progress=1)
        task = asyncio.create_task(reel.render_reel(kit_id))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    return library.get_kit(kit_id)


@app.get("/api/library/{kit_id}/bundle")
async def library_bundle(kit_id: str) -> Response:
    _require_row(kit_id)
    data = await asyncio.to_thread(library.bundle, library.get_kit(kit_id))
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="trendcraft-{kit_id[:8]}.zip"'},
    )


@app.post("/api/chat")
async def chat(body: ChatRequest) -> dict:
    history = [turn.model_dump() for turn in body.history]
    return await llm.chat_answer(body.message, history, library.list_kits())


@app.post("/api/chat/transcribe")
async def chat_transcribe(audio: UploadFile = File(...)) -> dict:
    data = await audio.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(400, "That recording is too long.")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise HTTPException(400, "Audio must be a WAV recording.")
    return {"text": await llm.transcribe(data)}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(ROOT_DIR / "app" / "static" / "index.html")
