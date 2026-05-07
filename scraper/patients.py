from __future__ import annotations

import logging
import re

from playwright.async_api import Locator, Page

from scraper.utils import ScraperConfig, clean_cell_text, close_popups, header_lookup, table_headers


LOGGER = logging.getLogger(__name__)


class PatientScraper:
    def __init__(self, page: Page, config: ScraperConfig) -> None:
        self.page = page
        self.config = config

    async def find_full_name(self, user_id: str) -> str:
        await self._open_patient_board()
        await close_popups(self.page)
        await self._try_search(user_id)

        for page_number in range(1, 10_000):
            table = await self._current_table()
            full_name = await self._extract_from_table(table, user_id)
            if full_name:
                return full_name

            if not await self._next_page():
                break
            LOGGER.debug("Patient not found on page %s; advanced pagination", page_number)

        raise RuntimeError(f"Patient user id {user_id!r} was not found in Patient Board.")

    async def _open_patient_board(self) -> None:
        LOGGER.info("Navigating to Patient Board")
        link = self.page.get_by_role("link", name=re.compile("patient board|patients?", re.I)).first
        if await link.count():
            await link.click()
        else:
            await self.page.locator("text=/Patient\\s+Board|Patients?/i").first.click()
        await self.page.wait_for_load_state("networkidle")

    async def _try_search(self, user_id: str) -> None:
        search = self.page.locator(
            "input[type='search'], input[placeholder*='Search' i], input[aria-label*='Search' i]"
        ).first
        if await search.count() and await search.is_visible():
            LOGGER.info("Searching Patient Board for user id %s", user_id)
            await search.fill(user_id)
            await search.press("Enter")
            await self.page.wait_for_load_state("networkidle")

    async def _current_table(self) -> Locator:
        table = self.page.locator("table").first
        await table.wait_for(state="visible")
        return table

    async def _extract_from_table(self, table: Locator, user_id: str) -> str | None:
        headers = await table_headers(table)
        id_index = header_lookup(headers, ["User ID", "Patient ID", "ID"])
        name_index = header_lookup(headers, ["Full Name", "Patient Name", "Name"])

        rows = table.locator("tbody tr")
        for i in range(await rows.count()):
            cells = [clean_cell_text(text) for text in await rows.nth(i).locator("td").all_inner_texts()]
            if not cells:
                continue

            id_candidates = [cells[id_index]] if id_index is not None and id_index < len(cells) else cells
            if user_id not in id_candidates:
                continue

            if name_index is not None and name_index < len(cells):
                name = cells[name_index]
            else:
                name = next((cell for cell in cells if cell and cell != user_id and not cell.isdigit()), "")
            if name:
                return name
        return None

    async def _next_page(self) -> bool:
        next_button = self.page.get_by_role("button", name=re.compile(r"next|>", re.I)).first
        if not await next_button.count():
            next_button = self.page.locator("a:has-text('Next'), .pagination a[rel='next'], li.next a").first
        if not await next_button.count() or not await next_button.is_visible() or await next_button.is_disabled():
            return False
        await next_button.click()
        await self.page.wait_for_load_state("networkidle")
        return True
