"""Open a deployed Streamlit Community Cloud app in a real browser and fail if it does not work.

- wakes the app if it went to sleep (the "get this app back up" button is clicked again every 25 seconds
  while the sleep screen is still showing, in any frame of the page)
- waits for the app to load, retrying up to 3 times with a fresh browser
- fails right away on a Streamlit exception, a Python import error or an empty page
- if the sleep screen never goes away: a push or a manual run FAILS (the deployment could not be verified),
  a scheduled run only prints a warning, because a sleeping app is normal for Streamlit Community Cloud
Failure reasons are printed as ::error:: annotations, which are visible on the public run page.
"""
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ["APP_URL"]
EVENT = os.environ.get("EVENT_NAME", "")
ATTEMPTS = 3
PER_ATTEMPT = 300  # seconds
WAKE = re.compile(r"get this app back up", re.I)
SLEEPING = re.compile(r"gone to sleep|get this app back up", re.I)
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


def sleep_screen(page):
    for fr in page.frames:
        try:
            if fr.get_by_text(SLEEPING).count():
                return True
        except Exception:
            pass
    return False


def diagnostics(page):
    info = [f"url={page.url[:110]}"]
    try:
        info.append(f"title={page.title()[:70]!r}")
        for fr in page.frames[:3]:
            try:
                txt = fr.locator("body").inner_text(timeout=3000)[:140].replace("\n", " | ")
            except Exception as e:
                txt = f"(no text: {str(e)[:40]})"
            info.append(f"frame[{fr.url[:60]}]: {txt!r}")
    except Exception as e:
        info.append(f"(diagnostics failed: {e})")
    return " || ".join(info)


def one_attempt(browser):
    """Returns ('loaded', frame, page, ctx, clicks) or ('timeout', None, page, ctx, clicks)."""
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    page = ctx.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=90000)
    try:
        page.wait_for_load_state("networkidle", timeout=20000)
    except Exception:
        pass
    deadline = time.time() + PER_ATTEMPT
    last_click = 0.0
    clicks = 0
    while time.time() < deadline:
        try:
            if time.time() - last_click > 25:
                for fr in page.frames:
                    btn = fr.get_by_role("button", name=WAKE)
                    if not btn.count():
                        btn = fr.get_by_text(WAKE)
                    if btn.count():
                        btn.first.click(timeout=5000, force=True)
                        clicks += 1
                        last_click = time.time()
                        print(f"sleep screen found - clicked the wake button (click {clicks}, frame {fr.url[:50]})")
                        break
            frame = app_frame(page)
            if frame is not None:
                text = frame.locator("body").inner_text(timeout=15000)
                if "in the oven" not in text:
                    return "loaded", frame, page, ctx, clicks
        except Exception as e:  # keep polling
            print(f"poll error: {str(e)[:100]}")
        time.sleep(5)
    return "timeout", None, page, ctx, clicks


with sync_playwright() as p:
    browser = p.chromium.launch()
    last_diag, still_asleep, total_clicks = "", False, 0
    for attempt in range(1, ATTEMPTS + 1):
        print(f"attempt {attempt} of {ATTEMPTS}")
        status, frame, page, ctx, clicks = one_attempt(browser)
        total_clicks += clicks
        if status == "loaded":
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
            print(f"OK - {URL} loaded and rendered ({len(body)} characters of text; wake clicks={total_clicks})")
            browser.close()
            sys.exit(0)
        still_asleep = sleep_screen(page)
        last_diag = diagnostics(page)
        print(f"attempt {attempt}: not loaded (sleep screen still showing: {still_asleep}); {last_diag}")
        ctx.close()
        time.sleep(20)
    browser.close()

if still_asleep and EVENT == "schedule":
    print(f"::warning::{URL} is still on the sleep screen after {ATTEMPTS} attempts and {total_clicks} wake clicks; "
          "scheduled runs do not fail for this (Streamlit puts idle apps to sleep). " + last_diag)
    sys.exit(0)
fail(f"app did not load after {ATTEMPTS} attempts ({total_clicks} wake clicks; sleep screen showing: {still_asleep}) || {last_diag}")
