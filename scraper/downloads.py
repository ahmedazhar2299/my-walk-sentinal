from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from playwright.async_api import Download, Locator, Page

from scraper.utils import safe_filename, timestamp_slug


LOGGER = logging.getLogger(__name__)


def unique_path(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    safe = safe_filename(filename)
    target = directory / safe
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix or ".csv"
    return directory / f"{stem}_{timestamp_slug()}{suffix}"


async def save_download(download: Download, directory: Path) -> Path:
    suggested = download.suggested_filename or f"activity_{timestamp_slug()}.csv"
    target = unique_path(directory, suggested)
    await download.save_as(target)
    validate_csv(target)
    LOGGER.info("Saved download: %s", target)
    return target


async def click_and_save_download(page: Page, trigger: Locator, directory: Path) -> Path:
    async with page.expect_download() as download_info:
        await trigger.click()
    download = await download_info.value
    return await save_download(download, directory)


def validate_csv(path: Path) -> None:
    """Fail fast if the portal returns an error page or non-CSV file instead of sensor data."""
    try:
        pd.read_csv(path, nrows=5)
    except Exception as exc:
        raise RuntimeError(f"Downloaded file is not a readable CSV: {path}") from exc
