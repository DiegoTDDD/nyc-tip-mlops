"""Open a deployed Streamlit Community Cloud app in a real browser and fail if it does not work.

- wakes the app if it went to sleep
- waits for the app to load (up to 10 minutes: a rebuild after a dependency change takes a while)
- fails on a Streamlit exception, a Python import error or an empty page
Failure reasons are printed as ::error:: annotations, which are visible on the public run page.
"""
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ["APP_URL"]
DEADLINE = time.time() + 600
BAD = [
    "Error running app",
    "ModuleNotFoundError",
    "ImportError",
    "Traceback (most recent call last)",
    "Error installing requirements",
    "There was a problem installing",
]


def fail(msg):
    print(f"::error::{URL} - {msg}")
    sys.exit(1)


def app_frame(page):
    """The frame that holds the Streamlit app (an iframe on *.streamlit.app, or the page itself)."""
    for fr in page.frames:
        try:
            if fr.locator('[data-testid="stApp"]').count():
                return fr
        except Exception:
            pass
    return None


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(URL, wait_until="domcontentloaded", timeout=90000)

    woke = False
    frame = None
    last = ""
    while time.time() < DEADLINE:
        try:
            wake = page.get_by_text(re.compile(r"get this app back up", re.I))
            if wake.count() and not woke:
                print("app was asleep - waking it up")
                wake.first.click()
                woke = True
            frame = app_frame(page)
            if frame is not None:
                text = frame.locator("body").inner_text(timeout=15000)
                if "in the oven" not in text:
                    break
                last = "app is still starting (in the oven)"
            else:
                last = page.inner_text("body")[:300].replace("\n", " | ")
        except Exception as e:  # keep polling
            last = f"(poll error: {e})"[:300]
        time.sleep(5)
    else:
        fail(f"app did not load within 10 minutes. Page says: {last}")

    time.sleep(25)  # let the first script run finish
    frame = app_frame(page) or frame
    body = frame.locator("body").inner_text(timeout=30000)
    exc = frame.locator('[data-testid="stException"]')
    if exc.count():
        fail("Streamlit exception on the page: " + exc.first.inner_text()[:300].replace("\n", " | "))
    for marker in BAD:
        if marker in body:
            fail(f"page contains {marker!r}: " + body[:300].replace("\n", " | "))
    if len(body.strip()) < 50:
        fail("page body is almost empty")
    print(f"OK - {URL} loaded and rendered ({len(body)} characters of text; woke={woke})")
    browser.close()
