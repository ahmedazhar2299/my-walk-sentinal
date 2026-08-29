from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from playwright.async_api import Locator, Page

from scraper.downloads import click_and_save_download
from scraper.session_manager import SessionManager
from scraper.utils import (
    ScraperConfig,
    clean_cell_text,
    close_popups,
    header_lookup,
    normalize_workout_type,
    recorded_date,
    recorded_datetime,
    table_headers,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActivityRecord:
    row: Locator
    patient_name: str
    recorded_time: str
    recorded_date: date
    workout_type_raw: str
    workout_type: str


class ActivityScraper:
    def __init__(self, page: Page, config: ScraperConfig, session_manager: SessionManager) -> None:
        self.page = page
        self.config = config
        self.session_manager = session_manager

    async def download_for_patient(self, full_name: str) -> int:
        await self._open_exercise_data_list()
        await close_popups(self.page)

        downloaded = 0
        skipped_existing_dates: set[date] = set()
        skipped_out_of_range = 0
        for page_number in range(1, 100_000):
            LOGGER.info("Scanning Exercise Data List page %s", page_number)
            table = await self._current_table()
            page_dates = await self._table_record_dates(table)
            if page_dates and all(self._is_before_start(record_date) for record_date in page_dates):
                LOGGER.info(
                    "All records on page %s are before %s; stopping pagination.",
                    page_number,
                    self.config.start_date,
                )
                break

            records = await self._matching_records(table, full_name)
            if records and all(self._is_before_start(record.recorded_date) for record in records):
                LOGGER.info("All matching records on page %s are before %s; stopping pagination.", page_number, self.config.start_date)
                break

            for record in records:
                if not self._record_in_date_window(record.recorded_date):
                    skipped_out_of_range += 1
                    continue
                if self.config.skip_existing_dates and self._date_already_downloaded(record.recorded_date):
                    skipped_existing_dates.add(record.recorded_date)
                    continue
                try:
                    await self._download_record(record)
                    downloaded += 1
                except Exception:
                    screenshot = await self.config.failure_screenshot(self.page, f"download_page_{page_number}")
                    LOGGER.exception("Failed to download one record. Screenshot: %s", screenshot)

            if not await self._next_page():
                break

        if skipped_existing_dates:
            LOGGER.info(
                "Skipped %s existing date(s): %s",
                len(skipped_existing_dates),
                ", ".join(sorted(d.isoformat() for d in skipped_existing_dates)),
            )
        if skipped_out_of_range:
            LOGGER.info("Skipped %s record(s) outside requested date window.", skipped_out_of_range)
        return downloaded

    async def _open_exercise_data_list(self) -> None:
        LOGGER.info("Navigating to Exercise Data List")
        link = self.page.get_by_role("link", name=re.compile("exercise data list|exercise data|activities", re.I)).first
        if await link.count():
            await link.click()
        else:
            await self.page.locator("text=/Exercise\\s+Data\\s+List|Exercise\\s+Data|Activities/i").first.click()
        await self.page.wait_for_load_state("networkidle")

    async def _current_table(self) -> Locator:
        table = self.page.locator("table").first
        await table.wait_for(state="visible")
        return table

    async def _matching_records(self, table: Locator, full_name: str) -> list[ActivityRecord]:
        headers = await table_headers(table)
        patient_index = header_lookup(headers, ["Patient Name", "Full Name", "Name"])
        recorded_index = header_lookup(headers, ["Recorded Time", "Record Time", "Created At", "Date"])
        workout_index = header_lookup(headers, ["Workout Type", "Exercise Type", "Activity Type", "Type"])

        if patient_index is None or recorded_index is None or workout_index is None:
            LOGGER.warning("Could not confidently map table headers: %s", headers)

        records: list[ActivityRecord] = []
        rows = table.locator("tbody tr")
        for i in range(await rows.count()):
            row = rows.nth(i)
            cells = [clean_cell_text(text) for text in await row.locator("td").all_inner_texts()]
            if not cells:
                continue

            patient_name = self._cell(cells, patient_index, "")
            if patient_name != full_name:
                continue

            recorded_time = self._cell(cells, recorded_index, "")
            try:
                record_dt = recorded_datetime(recorded_time)
            except ValueError:
                LOGGER.warning("Skipping record with unparseable Recorded Time %r for %s", recorded_time, full_name)
                continue

            workout_type_raw = self._cell(cells, workout_index, "")
            workout_type = normalize_workout_type(workout_type_raw)
            if not workout_type:
                LOGGER.warning("Skipping unknown workout type %r for %s at %s", workout_type_raw, full_name, recorded_time)
                continue

            records.append(ActivityRecord(row, patient_name, recorded_time, record_dt.date(), workout_type_raw, workout_type))
        LOGGER.info("Found %s matching records for %s on current page", len(records), full_name)
        return records

    async def _table_record_dates(self, table: Locator) -> list[date]:
        headers = await table_headers(table)
        recorded_index = header_lookup(headers, ["Recorded Time", "Record Time", "Created At", "Date"])
        if recorded_index is None:
            return []

        dates: list[date] = []
        rows = table.locator("tbody tr")
        for i in range(await rows.count()):
            cells = [clean_cell_text(text) for text in await rows.nth(i).locator("td").all_inner_texts()]
            if recorded_index >= len(cells):
                continue
            try:
                dates.append(recorded_datetime(cells[recorded_index]).date())
            except ValueError:
                continue
        return dates

    async def _download_record(self, record: ActivityRecord) -> None:
        date = recorded_date(record.recorded_time)
        list_url = self.page.url
        trigger = await self._download_trigger(record.row)
        should_return_to_list = False

        if trigger is None:
            await self._open_record_details(record.row)
            should_return_to_list = True
            trigger = await self._download_trigger(self.page.locator("body"))

        if trigger is None:
            row_html = await record.row.evaluate("element => element.outerHTML")
            LOGGER.debug("No download trigger found. Row HTML: %s", row_html)
            raise RuntimeError("No CSV/download trigger found in matching exercise row or detail page.")

        if should_return_to_list:
            try:
                staged_path = await click_and_save_download(self.page, trigger, self.config.staging_dir)
            finally:
                await self.page.goto(list_url, wait_until="networkidle")
                await self._current_table()
        else:
            staged_path = await click_and_save_download(self.page, trigger, self.config.staging_dir)

        self.session_manager.stage(
            date=date,
            workout_type=record.workout_type,
            staged_path=staged_path,
            original_filename=staged_path.name,
            recorded_time=record.recorded_time,
        )
        LOGGER.info(
            "Downloaded %s | %s | %s | %s",
            record.patient_name,
            record.recorded_time,
            record.workout_type,
            staged_path.name,
        )

    async def _download_trigger(self, scope: Locator) -> Locator | None:
        selectors = [
            "a[download]",
            "a[href$='.csv' i]",
            "a[href*='csv' i]",
            "a[href*='download' i]",
            "a:has-text('CSV')",
            "button:has-text('CSV')",
            "a:has-text('Download')",
            "button:has-text('Download')",
            "a:has-text('Export')",
            "button:has-text('Export')",
            "[title*='download' i]",
            "[aria-label*='download' i]",
        ]
        for selector in selectors:
            trigger = scope.locator(selector).first
            if await trigger.count() and await trigger.is_visible():
                return trigger
        return None

    async def _open_record_details(self, row: Locator) -> None:
        selectors = [
            "a:has-text('View Data')",
            "button:has-text('View Data')",
            "a:has-text('View')",
            "button:has-text('View')",
            "a[href*='exercise' i]",
            "a[href*='data' i]",
            "[title*='view' i]",
            "[aria-label*='view' i]",
        ]
        for selector in selectors:
            trigger = row.locator(selector).first
            if await trigger.count() and await trigger.is_visible():
                LOGGER.info("Opening exercise detail via %s", selector)
                await trigger.click()
                await self.page.wait_for_load_state("networkidle")
                await close_popups(self.page)
                return

        row_html = await row.evaluate("element => element.outerHTML")
        LOGGER.debug("No detail trigger found. Row HTML: %s", row_html)
        raise RuntimeError("No View Data/detail trigger found in matching exercise row.")

    async def _next_page(self) -> bool:
        next_button = self.page.get_by_role("button", name=re.compile(r"next|>", re.I)).first
        if not await next_button.count():
            next_button = self.page.locator("a:has-text('Next'), .pagination a[rel='next'], li.next a").first
        if not await next_button.count() or not await next_button.is_visible() or await next_button.is_disabled():
            return False
        await next_button.click()
        await self.page.wait_for_load_state("networkidle")
        return True

    @staticmethod
    def _cell(cells: list[str], index: int | None, default: str) -> str:
        if index is None or index >= len(cells):
            return default
        return cells[index]

    def _record_in_date_window(self, record_date: date) -> bool:
        if self.config.start_date and record_date < self.config.start_date:
            return False
        if self.config.end_date and record_date > self.config.end_date:
            return False
        return True

    def _is_before_start(self, record_date: date) -> bool:
        return self.config.start_date is not None and record_date < self.config.start_date

    def _date_already_downloaded(self, record_date: date) -> bool:
        date_dir = self.config.patient_output_dir / record_date.isoformat()
        return date_dir.exists() and any(self._iter_csv_files(date_dir))

    @staticmethod
    def _iter_csv_files(date_dir: Path):
        yield from date_dir.rglob("*.csv")
