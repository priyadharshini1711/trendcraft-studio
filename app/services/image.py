"""Images from Muse Image and subject detection from SAM, with fallbacks so the UI never breaks."""
import base64
import hashlib
import logging
import re
import textwrap
from html import escape

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

GRADIENTS = [("#E5482D", "#F9A826"), ("#0F766E", "#84CC16"), ("#4338CA", "#EC4899"), ("#1E1A16", "#E5482D")]
HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
SAM_BOX = re.compile(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>")
RECOVERABLE = (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError)


async def generate_image(prompt: str, size: str | None = None, fast: bool = False) -> bytes:
    """Return JPEG bytes from POST /images/generations, as documented at dev.meta.ai/docs/image-generation."""
    body: dict = {"model": settings.muse_image_model, "prompt": prompt, "n": 1, "output_format": "jpeg"}
    if size:
        body["size"] = size
    if fast:
        body["reasoning_strength"] = "low"
        body["tool_enablement"] = {"enable_image_search": False, "enable_web_search": False, "enable_shell": False}
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            settings.muse_image_url,
            headers={"Authorization": f"Bearer {settings.meta_api_key}"},
            json=body,
        )
        response.raise_for_status()
        first = response.json()["data"][0]
        if first.get("b64_json"):
            return base64.b64decode(first["b64_json"])
        if str(first.get("url", "")).startswith("https://"):
            download = await client.get(first["url"])
            download.raise_for_status()
            return download.content
    raise ValueError("no usable image in response")


async def find_subject(image_bytes: bytes, phrase: str) -> tuple[float, float] | None:
    """Return the subject's centre as fractions of the image, using SAM. None if it can't be found."""
    if not (settings.meta_segment_model and settings.meta_base_url and phrase.strip()):
        return None
    encoded = base64.b64encode(image_bytes).decode("ascii")
    body = {
        "model": settings.meta_segment_model,
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": phrase.strip()[:80]},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}"},
            ],
        }],
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{settings.meta_base_url.rstrip('/')}/responses",
                headers={"Authorization": f"Bearer {settings.meta_api_key}"},
                json=body,
            )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("SAM request failed: %s", type(exc).__name__)
        return None
    match = SAM_BOX.search(response.text)
    if not match:
        return None
    x1, y1, x2, y2, width, height = map(int, match.groups())
    if not width or not height:
        return None
    return (x1 + x2) / 2 / width, (y1 + y2) / 2 / height


def placeholder_svg(hook: str, palette: list[str] | None = None) -> bytes:
    """Render a 1080x1080 gradient SVG with the hook text."""
    colors = [c for c in (palette or []) if HEX_COLOR.match(c)]
    if len(colors) >= 2:
        start, end = colors[0], colors[1]
    else:
        digest = hashlib.sha256(hook.encode("utf-8")).digest()[0]
        start, end = GRADIENTS[digest % len(GRADIENTS)]

    lines = textwrap.wrap(hook or "Your next post", width=18)[:6]
    top = 540 - (len(lines) - 1) * 50
    text = "".join(
        f'<text x="90" y="{top + i * 100}" paint-order="stroke" stroke="#00000033" stroke-width="4">{escape(line)}</text>'
        for i, line in enumerate(lines)
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1080" width="1080" height="1080">'
        f'<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{start}"/>'
        f'<stop offset="1" stop-color="{end}"/></linearGradient></defs>'
        '<rect width="1080" height="1080" fill="url(#g)"/>'
        f'<g font-family="Arial, Helvetica, sans-serif" font-size="84" font-weight="800" fill="#ffffff">{text}</g>'
        "</svg>"
    )
    return svg.encode("utf-8")


async def generate_cover(prompt: str, hook: str = "", palette: list[str] | None = None, size: str | None = None) -> dict:
    """Return {data, extension, mime, placeholder}. Never raises for Muse problems."""
    if settings.image_ready:
        try:
            data = await generate_image(prompt, size)
            return {"data": data, "extension": "jpg", "mime": "image/jpeg", "placeholder": False}
        except RECOVERABLE as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            logger.warning("Muse Image failed: %s status=%s", type(exc).__name__, status)
    return {"data": placeholder_svg(hook, palette), "extension": "svg", "mime": "image/svg+xml", "placeholder": True}
