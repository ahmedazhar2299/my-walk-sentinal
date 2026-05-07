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
        except Exception:
            screenshot = await config.failure_screenshot(page, "fatal")
            LOGGER.exception("Scraper failed. Screenshot saved to %s", screenshot)
            raise
        finally:
            await context.close()
            await browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MyWalkSentinel exercise CSV files.")
    parser.add_argument("--user-id", default=None, help="Patient user id. Defaults to USER_ID env var or 12.")
    parser.add_argument("--headed", action="store_true", help="Run browser visibly for debugging.")
    parser.add_argument("--download-dir", default=None, help="Output root. Defaults to Scrapper.")
    parser.add_argument("--log-level", default=None, help="Python logging level. Defaults to INFO.")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep an existing patient output folder instead of deleting it before the run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ScraperConfig.from_env(
        user_id=args.user_id,
        headless=False if args.headed else None,
        output_root=Path(args.download_dir) if args.download_dir else None,
        log_level=args.log_level,
        overwrite_patient_dir=not args.no_overwrite,
    )
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
