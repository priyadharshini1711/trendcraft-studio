"""Environment-driven settings. Secrets are read here and never logged."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


@dataclass(frozen=True)
class Settings:
    meta_api_key: str = _env("META_API_KEY")
    meta_base_url: str = _env("META_BASE_URL")
    meta_text_model: str = _env("META_TEXT_MODEL")
    meta_vision_model: str = _env("META_VISION_MODEL")
    muse_image_url: str = _env("MUSE_IMAGE_URL")
    muse_image_model: str = _env("MUSE_IMAGE_MODEL")
    meta_fact_check_model: str = _env("META_FACT_CHECK_MODEL") or _env("META_TEXT_MODEL")
    meta_transcribe_model: str = _env("META_TRANSCRIBE_MODEL")
    meta_segment_model: str = _env("META_SEGMENT_MODEL")
    news_api_key: str = _env("NEWS_API_KEY")
    demo_mode: bool = _env("DEMO_MODE").lower() == "true"

    @property
    def text_ready(self) -> bool:
        return bool(self.meta_api_key and self.meta_base_url and self.meta_text_model)

    @property
    def vision_ready(self) -> bool:
        return bool(self.meta_api_key and self.meta_base_url and self.meta_vision_model)

    @property
    def image_ready(self) -> bool:
        return bool(self.meta_api_key and self.muse_image_url and self.muse_image_model)


settings = Settings()
