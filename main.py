from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import shutil

from playwright.async_api import async_playwright

from scraper.activities import ActivityScraper
from scraper.auth import login
from scraper.patients import PatientScraper
from scraper.session_manager import SessionManager
from scraper.utils import ScraperConfig, close_popups, setup_logging


LOGGER = logging.getLogger(__name__)


async def run(config: ScraperConfig) -> None:
    """Run the end-to-end MyWalkSentinel export workflow."""
    setup_logging(config.log_level)
    if config.overwrite_patient_dir and config.patient_output_dir.exists():
        LOGGER.warning("Removing existing output folder before rerun: %s", config.patient_output_dir)
        shutil.rmtree(config.patient_output_dir)
    config.ensure_directories()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=config.headless)
        context = await browser.new_context(
            accept_downloads=True,
            viewport={"width": 1440, "height": 1000},
            user_agent=config.user_agent,
        )
        page = await context.new_page()
        page.set_default_timeout(config.timeout_ms)

        try:
            await login(page, config)
            await close_popups(page)
            await process_patient(page, config)
        except Exception:
            screenshot = await config.failure_screenshot(page, "fatal")
            LOGGER.exception("Scraper failed. Screenshot saved to %s", screenshot)
            raise
        finally:
            await context.close()
            await browser.close()


async def run_many(base_config: ScraperConfig, user_ids: list[str]) -> dict[str, int]:
    setup_logging(base_config.log_level)
    base_config.ensure_directories()
    results: dict[str, int] = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=base_config.headless)
        context = await browser.new_context(
            accept_downloads=True,
            viewport={"width": 1440, "height": 1000},
            user_agent=base_config.user_agent,
        )
        page = await context.new_page()
        page.set_default_timeout(base_config.timeout_ms)

        try:
            await login(page, base_config)
            await close_popups(page)
            for user_id in user_ids:
                patient_config = base_config.with_user_id(user_id)
                patient_config.ensure_directories()
                try:
                    results[user_id] = await process_patient(page, patient_config)
                except Exception:
                    screenshot = await patient_config.failure_screenshot(page, f"patient_{user_id}_fatal")
                    LOGGER.exception("Failed patient_%s. Screenshot saved to %s", user_id, screenshot)
                    results[user_id] = -1
        finally:
            await context.close()
            await browser.close()

    return results


async def process_patient(page, config: ScraperConfig) -> int:
    if config.overwrite_patient_dir and config.patient_output_dir.exists():
        LOGGER.warning("Removing existing output folder before rerun: %s", config.patient_output_dir)
        shutil.rmtree(config.patient_output_dir)
        config.ensure_directories()

    patient_scraper = PatientScraper(page, config)
    full_name = await patient_scraper.find_full_name(config.user_id)
    LOGGER.info("Resolved patient_%s full name: %s", config.user_id, full_name)

    session_manager = SessionManager(config.patient_output_dir)
    activity_scraper = ActivityScraper(page, config, session_manager)
    downloaded_count = await activity_scraper.download_for_patient(full_name)

    session_manager.materialize()
    LOGGER.info(
        "Finished patient_%s: %s CSV file(s) downloaded and organized under %s",
        config.user_id,
        downloaded_count,
        config.patient_output_dir,
    )
    return downloaded_count


def existing_patient_ids(output_root: Path) -> list[str]:
    ids = []
    for path in sorted(output_root.glob("patient_*")):
        if path.is_dir() and path.name.removeprefix("patient_").isdigit():
            ids.append(path.name.removeprefix("patient_"))
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MyWalkSentinel exercise CSV files.")
    parser.add_argument("--user-id", default=None, help="Patient user id. Defaults to USER_ID env var or 12.")
    parser.add_argument("--patient-ids", nargs="+", default=None, help="One or more patient ids to process.")
    parser.add_argument(
        "--all-existing-patients",
        action="store_true",
        help="Process all patient_* folders already present under the download directory.",
    )
    parser.add_argument("--headed", action="store_true", help="Run browser visibly for debugging.")
    parser.add_argument("--download-dir", default=None, help="Output root. Defaults to Scrapper.")
    parser.add_argument("--log-level", default=None, help="Python logging level. Defaults to INFO.")
    parser.add_argument("--start-date", default=None, help="Only download records on or after this YYYY-MM-DD date.")
    parser.add_argument("--end-date", default=None, help="Only download records on or before this YYYY-MM-DD date.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing patient output folder before the run. Disabled by default.",
    )
    parser.add_argument(
        "--include-existing-dates",
        action="store_true",
        help="Download records even when that patient/date folder already contains CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ScraperConfig.from_env(
        user_id=args.user_id,
        headless=False if args.headed else None,
        output_root=Path(args.download_dir) if args.download_dir else None,
        log_level=args.log_level,
        overwrite_patient_dir=args.overwrite,
        start_date=args.start_date,
        end_date=args.end_date,
        skip_existing_dates=not args.include_existing_dates,
    )
    if args.all_existing_patients:
        user_ids = existing_patient_ids(config.output_root)
        if not user_ids:
            raise RuntimeError(f"No patient_* folders found under {config.output_root}")
        results = asyncio.run(run_many(config, user_ids))
        LOGGER.info("Multi-patient results: %s", results)
    elif args.patient_ids:
        results = asyncio.run(run_many(config, [str(pid) for pid in args.patient_ids]))
        LOGGER.info("Multi-patient results: %s", results)
    else:
        asyncio.run(run(config))


if __name__ == "__main__":
    main()
