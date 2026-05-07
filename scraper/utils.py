from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, TypeVar

from dotenv import load_dotenv
from playwright.async_api import Locator, Page

import os


T = TypeVar("T")
LOGGER = logging.getLogger(__name__)


WORKOUT_TYPE_MAP: dict[str, str] = {
    "10 meter walk": "walk",
    "Ten Meter Walk": "walk",
    "Sit To Stand": "sit_to_stand",
    "Stand To Sit": "stand_to_sit",
    "Left Turn 360": "left_turn",
    "Right Turn 360": "right_turn",
}


@dataclass(frozen=True)
class ScraperConfig:
    base_url: str
    username: str
    password: str
    user_id: str
    output_root: Path
    headless: bool
    timeout_ms: int
    max_retries: int
    retry_backoff_seconds: float
    log_level: str
    user_agent: str

    @classmethod
    def from_env(
        cls,
        *,
        user_id: str | None = None,
        headless: bool | None = None,
        output_root: Path | None = None,
        log_level: str | None = None,
    ) -> "ScraperConfig":
        load_dotenv()
        username = os.getenv("USERNAME")
        password = os.getenv("PASSWORD")
        if not username or not password:
            raise ValueError("USERNAME and PASSWORD must be set in .env")

        return cls(
            base_url=os.getenv("BASE_URL", "https://mywalksentinel.soangra.com/"),
            username=username,
            password=password,
            user_id=str(user_id or os.getenv("USER_ID", "12")),
            output_root=(output_root or Path(os.getenv("DOWNLOAD_DIR", "Scrapper"))).resolve(),
            headless=headless if headless is not None else env_bool("HEADLESS", True),
            timeout_ms=int(os.getenv("TIMEOUT_MS", "30000")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_backoff_seconds=float(os.getenv("RETRY_BACKOFF_SECONDS", "2")),
            log_level=(log_level or os.getenv("LOG_LEVEL", "INFO")).upper(),
            user_agent=os.getenv(
                "USER_AGENT",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
            ),
        )

    @property
    def patient_output_dir(self) -> Path:
        return self.output_root / f"patient_{self.user_id}"

    @property
    def staging_dir(self) -> Path:
        return self.output_root / ".downloads_tmp" / f"patient_{self.user_id}"

    @property
    def screenshot_dir(self) -> Path:
        return self.output_root / "_screenshots" / f"patient_{self.user_id}"

    def ensure_directories(self) -> None:
        self.patient_output_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    async def failure_screenshot(self, page: Page, label: str) -> Path:
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.screenshot_dir / f"{timestamp_slug()}_{safe_filename(label)}.png"
        await page.screenshot(path=path, full_page=True)
        return path


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


async def retry_async(
    action: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    backoff_seconds: float,
    label: str,
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await action()
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = backoff_seconds * attempt
            LOGGER.warning("%s failed on attempt %s/%s: %s; retrying in %.1fs", label, attempt, attempts, exc, delay)
            await asyncio.sleep(delay)
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last_error


def normalize_workout_type(raw: str) -> str | None:
    cleaned = " ".join(raw.split())
    for source, target in WORKOUT_TYPE_MAP.items():
        if cleaned.casefold() == source.casefold():
            return target
    return None


def recorded_date(recorded_time: str) -> str:
    text = normalize_recorded_time(recorded_time)
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %I:%M:%S %p",
        "%B %d, %Y, %I:%M %p",
        "%B %d, %Y %I:%M %p",
        "%B %d, %Y, %I %p",
        "%B %d, %Y %I %p",
        "%b %d, %Y, %I:%M %p",
        "%b %d, %Y %I:%M %p",
        "%b %d, %Y, %I %p",
        "%b %d, %Y %I %p",
    ):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    match = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", text)
    if not match:
        raise ValueError(f"Could not extract date from Recorded Time: {recorded_time!r}")
    return match.group(0).replace("/", "-")


def normalize_recorded_time(recorded_time: str) -> str:
    """Normalize common portal timestamp variants before parsing."""
    text = " ".join(recorded_time.split())
    text = re.sub(r"\ba\.m\.", "AM", text, flags=re.I)
    text = re.sub(r"\bp\.m\.", "PM", text, flags=re.I)
    text = re.sub(r"\bam\b", "AM", text, flags=re.I)
    text = re.sub(r"\bpm\b", "PM", text, flags=re.I)
    return text


def safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return safe or "download.csv"


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


async def first_visible(page_or_locator: Page | Locator, selectors: list[str]) -> Locator | None:
    for selector in selectors:
        locator = page_or_locator.locator(selector).first
        try:
            if await locator.count() and await locator.is_visible():
                return locator
        except Exception:
            continue
    return None


async def click_first_visible(page: Page, selectors: list[str], label: str) -> None:
    locator = await first_visible(page, selectors)
    if not locator:
        raise RuntimeError(f"Could not find visible {label}. Tried: {selectors}")
    await locator.click()


async def close_popups(page: Page) -> None:
    """Dismiss common blocking modal/popover controls when they exist."""
    labels = ["Close", "Cancel", "OK", "Ok", "Got it", "Dismiss"]
    for label in labels:
        button = page.get_by_role("button", name=re.compile(f"^{re.escape(label)}$", re.I)).first
        try:
            if await button.count() and await button.is_visible():
                await button.click(timeout=1500)
        except Exception:
            continue


async def table_headers(table: Locator) -> list[str]:
    headers = [clean_cell_text(text) for text in await table.locator("thead th").all_inner_texts()]
    if headers:
        return headers
    first_row = table.locator("tr").first
    return [clean_cell_text(text) for text in await first_row.locator("th,td").all_inner_texts()]


def clean_cell_text(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


def header_lookup(headers: list[str], candidates: list[str]) -> int | None:
    normalized = [h.casefold() for h in headers]
    for candidate in candidates:
        candidate_cf = candidate.casefold()
        for index, header in enumerate(normalized):
            if candidate_cf == header or candidate_cf in header:
                return index
    return None
