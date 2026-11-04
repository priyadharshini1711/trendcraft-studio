"""Render a silent vertical reel: Muse Image scenes, SAM-guided camera moves and animated captions."""
import asyncio
import io
import json
import logging
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageOps

from app.config import settings
from app.services import image, library

logger = logging.getLogger(__name__)

WIDTH, HEIGHT = 720, 1280
FPS = 24
OVERSCAN = 1.18  # scene is rendered larger than the frame so the camera has room to move
FADE_FRAMES = 8
MAX_CONCURRENT_IMAGES = 3
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_REGULAR = "C:/Windows/Fonts/arial.ttf"


def _font(path: str, size: int) -> ImageFont.ImageFont:
    for candidate in (path, Path(path).name, "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _printable(text: str) -> str:
    """Drop emoji and other characters the caption font can't draw."""
    return re.sub(r"[^\u0000-\uFFFF]|[\uFE00-\uFE0F\u200D]", "", text).strip()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    return lines + [current] if current else lines


def _gradient_scene(index: int, palette: list[str]) -> bytes:
    """Fallback scene when Muse Image is unavailable."""
    colors = [c for c in palette if image.HEX_COLOR.match(c)]
    start, end = (colors[0], colors[1]) if len(colors) >= 2 else image.GRADIENTS[index % len(image.GRADIENTS)]
    picture = ImageOps.colorize(
        Image.linear_gradient("L").resize((1024, 1536)), black=start, white=end,
    )
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def _caption_overlay(headline: str, spoken: str, subject_y: float) -> Image.Image:
    """Draw captions away from the subject, over a soft scrim, keeping clear of platform UI."""
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    head_font, body_font = _font(FONT_BOLD, 60), _font(FONT_REGULAR, 30)
    head_lines = _wrap(draw, _printable(headline), head_font, WIDTH - 110)[:4]
    body_lines = _wrap(draw, _printable(spoken), body_font, WIDTH - 110)[:4]
    block = len(head_lines) * 70 + 24 + len(body_lines) * 38
    at_bottom = subject_y < 0.55
    top = HEIGHT - 190 - block if at_bottom else 130

    scrim = Image.linear_gradient("L").resize((WIDTH, block + 300))
    if not at_bottom:
        scrim = ImageOps.flip(scrim)
    scrim = scrim.point(lambda value: int(value * 0.72))
    shade = Image.new("RGBA", scrim.size, (0, 0, 0, 255))
    shade.putalpha(scrim)
    overlay.alpha_composite(shade, (0, HEIGHT - scrim.height if at_bottom else 0))

    y = top
    for line in head_lines:
        draw.text((55, y), line, font=head_font, fill="white", stroke_width=3, stroke_fill=(0, 0, 0, 180))
        y += 70
    y += 24
    for line in body_lines:
        draw.text((55, y), line, font=body_font, fill=(255, 255, 255, 235), stroke_width=1, stroke_fill=(0, 0, 0, 160))
        y += 38
    return overlay


def _beat_seconds(spoken: str) -> float:
    return max(3.0, min(6.0, len(spoken.split()) / 2.5))


def _camera_frame(scene: Image.Image, focus: tuple[float, float], progress: float) -> Image.Image:
    """Slowly push in from the full scene towards the subject."""
    width, height = scene.size
    zoom = 1 + (OVERSCAN - 1) * progress
    crop_w, crop_h = width / zoom, height / zoom
    center_x = width / 2 + (focus[0] * width - width / 2) * progress
    center_y = height / 2 + (focus[1] * height - height / 2) * progress
    left = min(max(center_x - crop_w / 2, 0), width - crop_w)
    top = min(max(center_y - crop_h / 2, 0), height - crop_h)
    return scene.resize((WIDTH, HEIGHT), Image.BILINEAR, box=(left, top, left + crop_w, top + crop_h))


def _encode(beats: list[dict], output: Path) -> None:
    """Pipe frames into ffmpeg (bundled by imageio-ffmpeg, so no system install is needed)."""
    temp = output.with_name("reel.tmp.mp4")
    process = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "21",
         "-movflags", "+faststart", str(temp)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for index, beat in enumerate(beats):
            total = int(beat["seconds"] * FPS)
            upcoming = None
            if index + 1 < len(beats):
                upcoming = _camera_frame(beats[index + 1]["scene"], beats[index + 1]["focus"], 0)
            for frame_number in range(total):
                background = _camera_frame(beat["scene"], beat["focus"], frame_number / total)
                frame = background.convert("RGBA")
                frame.alpha_composite(beat["overlay"])
                frame = frame.convert("RGB")
                if frame_number < FADE_FRAMES:  # captions fade in
                    frame = Image.blend(background, frame, (frame_number + 1) / FADE_FRAMES)
                remaining = total - frame_number
                if upcoming is not None and remaining <= FADE_FRAMES:  # scenes cross-fade
                    frame = Image.blend(frame, upcoming, (FADE_FRAMES - remaining + 1) / (FADE_FRAMES + 1))
                process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait(timeout=120) != 0:
            raise RuntimeError("ffmpeg failed")
    except BaseException:
        process.kill()
        temp.unlink(missing_ok=True)
        raise
    temp.replace(output)


def _timestamp(seconds: float) -> str:
    millis = int(seconds * 1000)
    return f"{millis // 3600000:02}:{millis // 60000 % 60:02}:{millis // 1000 % 60:02},{millis % 1000:03}"


def _write_scripts(folder: Path, items: list[dict], durations: list[float]) -> None:
    captions, voiceover = [], []
    start = 0.0
    for number, (item, seconds) in enumerate(zip(items, durations), start=1):
        end = start + seconds
        captions.append(f"{number}\n{_timestamp(start)} --> {_timestamp(end)}\n{item['text']}\n")
        voiceover.append(f"[{start:.1f}s to {end:.1f}s] {item['label']}\n{item['text']}\n")
        start = end
    (folder / "captions.srt").write_text("\n".join(captions), encoding="utf-8")
    (folder / "voiceover.txt").write_text("\n".join(voiceover), encoding="utf-8")


async def render_reel(kit_id: str) -> None:
    """Generate every artifact for a reel, recording progress on the library row."""
    kit = library.get_kit(kit_id)
    row = library.get_row(kit_id)
    items = kit["script_or_slides"]
    style = _parse_style(row["style_json"])
    folder = library.kit_dir(kit_id)
    scenes_folder = folder / "scenes"
    scenes_folder.mkdir(exist_ok=True)
    limit = asyncio.Semaphore(MAX_CONCURRENT_IMAGES)
    finished = {"count": 0}

    async def make_scene(index: int, item: dict) -> dict:
        async with limit:
            prompt = _scene_prompt(item, kit, style)
            try:
                if not settings.image_ready:
                    raise ValueError("Muse Image not configured")
                data = await image.generate_image(prompt, size="1024x1536", fast=True)
            except image.RECOVERABLE as exc:
                logger.warning("Scene %d fell back to a gradient: %s", index + 1, type(exc).__name__)
                data = _gradient_scene(index, style.get("palette", []))
            focus = await image.find_subject(data, item.get("subject", "")) or (0.5, 0.5)
        (scenes_folder / f"scene_{index + 1}.jpg").write_bytes(data)
        finished["count"] += 1
        library.update(
            kit_id,
            reel_stage=f"Painted scene {finished['count']} of {len(items)}",
            reel_progress=10 + int(65 * finished["count"] / len(items)),
        )
        return {"data": data, "focus": focus}

    try:
        library.update(kit_id, reel_status="rendering", reel_stage="Painting your scenes with Muse Image", reel_progress=5)
        scenes = await asyncio.gather(*(make_scene(i, item) for i, item in enumerate(items)))
        library.update(kit_id, reel_stage="Directing the camera and laying out captions", reel_progress=80)

        beats = []
        for item, scene in zip(items, scenes):
            picture = Image.open(io.BytesIO(scene["data"])).convert("RGB")
            size = (int(WIDTH * OVERSCAN), int(HEIGHT * OVERSCAN))
            beats.append({
                "scene": ImageOps.fit(picture, size, Image.LANCZOS, centering=scene["focus"]),
                "focus": scene["focus"],
                "overlay": _caption_overlay(item.get("on_screen_text") or item["label"], item["text"], scene["focus"][1]),
                "seconds": _beat_seconds(item["text"]),
            })
        library.update(kit_id, reel_stage="Rendering your video", reel_progress=88)
        await asyncio.to_thread(_encode, beats, folder / "reel.mp4")
        _write_scripts(folder, items, [beat["seconds"] for beat in beats])
        library.update(kit_id, reel_status="ready", reel_stage="", reel_progress=100)
    except Exception as exc:
        logger.error("Reel render failed: %s", type(exc).__name__)
        library.update(kit_id, reel_status="failed", reel_stage="The reel couldn't be rendered. Please try again.", reel_progress=0)


def _parse_style(style_json: str | None) -> dict:
    return json.loads(style_json) if style_json else {}


def _scene_prompt(item: dict, kit: dict, style: dict) -> str:
    visual = item.get("visual") or kit["cover_image_prompt"]
    style_note = ""
    if style:
        style_note = f" Mood: {style.get('tone', '')}. Colours: {', '.join(style.get('palette', []))}."
    return f"{visual}{style_note} Vertical composition with a clear main subject. No text, no logos, no real people."
