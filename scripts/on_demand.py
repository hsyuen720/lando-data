"""
On-Demand Search — triggered by GitHub Issues with title:
  [SEARCH_REQUEST]: {country} - {visa_type}

1. Parses country and visa_type from the issue title.
2. Uses Gemini to find the official government URL.
3. Scrapes and structures the data.
4. Adds the URL to config/sources.json with "source": "manual".
5. Saves output to data/{country}_{visa_slug}.json.
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCES_PATH = ROOT_DIR / "config" / "sources.json"
APPS_PATH = ROOT_DIR / "config" / "apps.json"
DATA_DIR = ROOT_DIR / "data"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("on_demand")

# ---------------------------------------------------------------------------
# Title parsing
# ---------------------------------------------------------------------------
TITLE_PATTERN = re.compile(
    r"\[SEARCH_REQUEST\]\s*:\s*(.+?)\s*-\s*(.+)", re.IGNORECASE
)


def parse_issue_title(title: str) -> tuple[str, str] | None:
    """Extract (country_slug, visa_slug) from issue title.

    Expected: [SEARCH_REQUEST]: Japan - Digital Nomad Visa
    Returns:  ("japan", "digital-nomad-visa")
    """
    match = TITLE_PATTERN.match(title.strip())
    if not match:
        return None
    country_raw = match.group(1).strip()
    visa_raw = match.group(2).strip()

    country_slug = re.sub(r"[^a-z0-9]+", "-", country_raw.lower()).strip("-")
    visa_slug = re.sub(r"[^a-z0-9]+", "-", visa_raw.lower()).strip("-")
    # Remove trailing "visa" / "permit" for cleaner slugs
    visa_slug = re.sub(r"-(visa|permit)$", "", visa_slug)

    return country_slug, visa_slug


# ---------------------------------------------------------------------------
# Gemini helpers (shared patterns with pathfinder/engine_scraper)
# ---------------------------------------------------------------------------
URL_DISCOVERY_PROMPT = (
    "You are an immigration research assistant. "
    "Find the OFFICIAL government website URLs for the specified visa type "
    "in the given country.\n\n"
    "Rules:\n"
    "1. Only return URLs from official government domains.\n"
    "2. Return ONLY valid JSON — no markdown fences, no commentary.\n\n"
    "Return format (strict JSON):\n"
    "[\n"
    '  {"url": "<URL>", "title": "<page title>"}\n'
    "]\n"
)

STRUCTURING_SYSTEM_PROMPT = (
    "Extract visa details from the provided text in its original language. "
    "Output a valid JSON with fields: visa_category, visa_type_detail, "
    "processing_time, fee_amount, currency, last_updated_official, "
    "reference_urls. If data is missing, return null for that field.\n\n"
    "Rules:\n"
    "1. DO NOT translate any content. Keep all text in the original language.\n"
    "2. Return ONLY valid JSON — no markdown fences, no commentary.\n"
    "3. If multiple visa sub-types exist, return a JSON array of objects.\n"
    "4. Dates should be in ISO 8601 format where possible.\n"
)


def configure_gemini() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)
    return genai.Client(api_key=api_key)


MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]


def _is_transient(exc: Exception) -> bool:
    msg = str(exc)
    return "503" in msg or "429" in msg or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)
def _call_model(client: genai.Client, model: str, prompt: str, system_instruction: str) -> str:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
        ),
    )
    return response.text.strip()


def _call_gemini(client: genai.Client, prompt: str, system_instruction: str) -> str:
    """Try each model in the fallback chain until one succeeds."""
    last_exc = None
    for model in MODELS:
        try:
            result = _call_model(client, model, prompt, system_instruction)
            return result
        except Exception as exc:
            last_exc = exc
            logger.warning("Model %s failed: %s — trying next fallback", model, exc)
    raise last_exc


def _parse_json(raw: str) -> dict | list:
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw.strip())


# ---------------------------------------------------------------------------
# Scraping (reused from engine_scraper)
# ---------------------------------------------------------------------------
import requests
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = 30
USER_AGENT = "Mozilla/5.0 (compatible; LandoBot/1.0; +https://github.com/lando-data)"


def scrape_page(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        lines = [ln.strip() for ln in soup.get_text(separator="\n", strip=True).splitlines() if ln.strip()]
        return "\n".join(lines)
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def load_sources() -> dict:
    if SOURCES_PATH.exists():
        with open(SOURCES_PATH, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
            if content:
                return json.loads(content)
    return {}


def load_apps() -> dict:
    if APPS_PATH.exists():
        with open(APPS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def run(issue_title: str) -> str:
    """Run the on-demand pipeline. Returns a markdown comment for the issue."""
    parsed = parse_issue_title(issue_title)
    if not parsed:
        msg = f"Could not parse issue title: `{issue_title}`\nExpected format: `[SEARCH_REQUEST]: Country - Visa Type`"
        logger.error(msg)
        return msg

    country_slug, visa_slug = parsed
    country_display = country_slug.replace("-", " ").title()
    visa_display = visa_slug.replace("-", " ").title()
    logger.info("On-demand search: %s / %s", country_display, visa_display)

    client = configure_gemini()

    # Step 1: Discover official URL via Gemini
    prompt = (
        f"Find the official government visa/immigration URL for:\n"
        f"Country: {country_display}\n"
        f"Visa type: {visa_display}"
    )
    try:
        raw = _call_gemini(client, prompt, URL_DISCOVERY_PROMPT)
        url_entries = _parse_json(raw)
    except Exception as exc:
        msg = f"Failed to discover URLs for {country_display} / {visa_display}: {exc}"
        logger.error(msg)
        return msg

    if not url_entries:
        msg = f"No official URLs found for **{country_display}** — **{visa_display}**."
        logger.warning(msg)
        return msg

    logger.info("  Found %d URL(s): %s", len(url_entries), [e["url"] for e in url_entries])

    # Step 2: Add to sources.json with source=manual
    sources = load_sources()
    if country_slug not in sources:
        sources[country_slug] = {}
    sources[country_slug][visa_slug] = {
        "urls": url_entries,
        "source": "manual",
    }
    with open(SOURCES_PATH, "w", encoding="utf-8") as fh:
        json.dump(sources, fh, indent=2, ensure_ascii=False)
    logger.info("  Added to sources.json as manual entry: %s/%s", country_slug, visa_slug)

    time.sleep(1)

    # Step 3: Scrape the pages
    all_text: list[str] = []
    for entry in url_entries:
        url = entry.get("url")
        if not url:
            continue
        logger.info("  Scraping: %s", url)
        text = scrape_page(url)
        if text:
            all_text.append(f"--- Source: {url} ---\n{text}")

    if not all_text:
        msg = f"URLs found but could not scrape any content for **{country_display}** — **{visa_display}**."
        logger.warning(msg)
        return msg

    merged_text = "\n\n".join(all_text)
    if len(merged_text) > 60_000:
        merged_text = merged_text[:60_000]

    # Step 4: Structure with AI
    struct_prompt = (
        f"Country: {country_display}\n"
        f"Visa category: {visa_display}\n\n"
        f"--- BEGIN PAGE TEXT ---\n{merged_text}\n--- END PAGE TEXT ---"
    )
    try:
        raw = _call_gemini(client, struct_prompt, STRUCTURING_SYSTEM_PROMPT)
        structured = _parse_json(raw)
    except Exception as exc:
        msg = f"AI structuring failed for {country_display}/{visa_display}: {exc}"
        logger.error(msg)
        return msg

    # Step 5: Save output
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    apps = load_apps()
    country_apps = apps.get(country_slug, [])
    output = {
        "country": country_slug,
        "visa_category": visa_slug,
        "data": structured,
        "essential_apps": country_apps,
    }
    filepath = DATA_DIR / f"{country_slug}_{visa_slug}.json"
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    logger.info("  Saved %s", filepath)

    # Build summary comment
    urls_md = "\n".join(f"- [{e.get('title', e['url'])}]({e['url']})" for e in url_entries)
    comment = (
        f"## Lando Search Complete\n\n"
        f"**Country:** {country_display}  \n"
        f"**Visa Type:** {visa_display}  \n\n"
        f"### Sources Found\n{urls_md}\n\n"
        f"Data saved to `data/{country_slug}_{visa_slug}.json` and "
        f"registered in `config/sources.json` as a **manual** entry "
        f"(protected from automated overwrites)."
    )
    return comment


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/on_demand.py '<issue title>'")
        sys.exit(1)
    result = run(sys.argv[1])
    # Write comment to file for the workflow to pick up
    comment_path = ROOT_DIR / "issue_comment.md"
    comment_path.write_text(result, encoding="utf-8")
    print(result)
