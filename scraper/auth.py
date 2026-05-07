from __future__ import annotations

import logging
import re

from playwright.async_api import Page

from scraper.utils import ScraperConfig, click_first_visible, retry_async


LOGGER = logging.getLogger(__name__)


async def login(page: Page, config: ScraperConfig) -> None:
    async def attempt() -> None:
        LOGGER.info("Opening login page: %s", config.base_url)
        await page.goto(config.base_url, wait_until="domcontentloaded")
        await page.wait_for_load_state("networkidle")

        username = page.get_by_label(re.compile("user|email|login", re.I)).first
        if not await username.count():
            username = page.locator(
                "input[name*='user' i], input[name*='email' i], input[type='email'], input[type='text']"
            ).first

        password = page.get_by_label(re.compile("password", re.I)).first
        if not await password.count():
            password = page.locator("input[type='password'], input[name*='pass' i]").first

        await username.fill(config.username)
        await password.fill(config.password)

        LOGGER.info("Submitting login form")
        await click_first_visible(
            page,
            [
                "button[type='submit']",
                "input[type='submit']",
                "button:has-text('Login')",
                "button:has-text('Log in')",
                "button:has-text('Sign in')",
            ],
            "login submit button",
        )
        await page.wait_for_load_state("networkidle")

        visible_passwords = await page.locator("input[type='password']:visible").count()
        if visible_passwords:
            raise RuntimeError("Login appears unsuccessful; password field is still visible.")
        LOGGER.info("Login succeeded")

    await retry_async(
        attempt,
        attempts=config.max_retries,
        backoff_seconds=config.retry_backoff_seconds,
        label="login",
    )
