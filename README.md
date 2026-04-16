# Lando Data Pipeline

A self-evolving, AI-driven data pipeline that discovers, scrapes, and structures immigration data for 30+ countries. Built to power **Lando** — an AI relocation assistant.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GitHub Actions                        │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Monthly     │  │   Weekly     │  │  On-Demand   │  │
│  │  Pathfinder   │  │   Scraper    │  │  via Issues  │  │
│  │  (1st of mo)  │  │  (Sundays)   │  │  (instant)   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
└─────────┼─────────────────┼─────────────────┼───────────┘
          │                 │                 │
          ▼                 ▼                 ▼
   config/sources.json   data/*.json    both + issue comment
   config/apps.json
```

### Scripts

| Script | Purpose | Schedule |
|--------|---------|----------|
| `pathfinder.py` | Discovers official government visa URLs and essential apps per country via Gemini AI | Monthly (1st) |
| `engine_scraper.py` | Scrapes discovered URLs and structures visa data via Gemini AI | Weekly (Sunday) |
| `on_demand.py` | Processes user-requested visa searches triggered by GitHub Issues | On issue open |

### Data Flow

1. **Pathfinder** reads `config/countries.json` → asks Gemini for 5–10 relevant visa categories per country → writes URLs to `config/sources.json` and apps to `config/apps.json`
2. **Engine Scraper** reads `config/sources.json` → scrapes each URL → sends text to Gemini for structured extraction → writes `data/{country}_{category}.json`
3. **On-Demand** parses a GitHub Issue title → discovers URL → scrapes → structures → saves data and registers as a protected manual entry

## Config Files

### `config/countries.json`

A flat JSON array of country slugs that drive the entire pipeline:

```json
["united-states", "united-kingdom", "japan", "taiwan", ...]
```

### `config/sources.json`

Discovered URLs per country per visa category. Each entry includes a `source` tag:

```json
{
  "japan": {
    "student": {
      "urls": [{"url": "https://www.mofa.go.jp/...", "title": "Student Visa"}],
      "source": "ai"
    },
    "digital-nomad": {
      "urls": [{"url": "https://www.mofa.go.jp/...", "title": "Digital Nomad Visa"}],
      "source": "manual"
    }
  }
}
```

- `"source": "ai"` — Discovered by pathfinder, updated monthly
- `"source": "manual"` — Added via GitHub Issues, **never overwritten** by the pathfinder

### `config/apps.json`

Essential relocation apps per country (government, healthcare, transport, banking).

## Output Format

Each `data/{country}_{category}.json` file:

```json
{
  "country": "japan",
  "visa_category": "student",
  "data": {
    "visa_type_detail": "留学ビザ (College Student Visa)",
    "processing_time": "5営業日〜2週間",
    "fee_amount": "3,000",
    "currency": "JPY",
    "last_updated_official": null,
    "reference_urls": ["https://www.mofa.go.jp/..."]
  },
  "essential_apps": [...]
}
```

All text values are kept in their **original language** — no translation.

## On-Demand Search via GitHub Issues

Request data for any country/visa combo by opening an issue:

**Title format:**
```
[SEARCH_REQUEST]: Country - Visa Type
```

**Examples:**
```
[SEARCH_REQUEST]: Japan - Digital Nomad Visa
[SEARCH_REQUEST]: Portugal - Golden Visa
[SEARCH_REQUEST]: Taiwan - Gold Card
```

The workflow will automatically:
1. Discover the official government URL
2. Scrape and structure the data
3. Save to `data/` and register in `sources.json` as `"source": "manual"`
4. Comment on the issue with results
5. Close the issue

## Safety Mechanisms

| Feature | Description |
|---------|-------------|
| **Manual protection** | Entries with `"source": "manual"` are never overwritten by the monthly pathfinder |
| **Diff check** | Files are only written/committed if content actually changed — saves repo storage |
| **Stale cleanup** | Data files for removed countries/categories are automatically deleted |
| **Retry with backoff** | All Gemini API calls retry 3× with exponential backoff (4–60s) on 429/503 errors |
| **Original language** | All scraped content preserved in original language — no translation |

## Local Development

### Prerequisites

- Python 3.12+
- A [Gemini API key](https://aistudio.google.com/apikey)

### Setup

```bash
# Clone
git clone https://github.com/hsyuen720/lando-data.git
cd lando-data

# Install dependencies
pip install -r requirements.txt

# Set your API key
export GEMINI_API_KEY="your-key-here"
```

### Run Locally

```bash
# Discover URLs and apps for all countries
python scripts/pathfinder.py

# Scrape and structure data from discovered URLs
python scripts/engine_scraper.py

# Test on-demand search
python scripts/on_demand.py "[SEARCH_REQUEST]: Japan - Digital Nomad Visa"
```

## GitHub Actions Workflows

| Workflow | Trigger | File |
|----------|---------|------|
| Monthly Pathfinder | `0 0 1 * *` + manual | `.github/workflows/monthly_pathfinder.yml` |
| Weekly Engine Sync | `0 0 * * 0` + manual | `.github/workflows/weekly_engine_sync.yml` |
| Full Pipeline | Manual only | `.github/workflows/full_pipeline.yml` |
| On-Demand Search | Issue opened with `[SEARCH_REQUEST]` | `.github/workflows/on_demand.yml` |

All workflows require a `GEMINI_API_KEY` secret in the **Production** environment.

## Project Structure

```
lando-data/
├── config/
│   ├── countries.json      # 30 country slugs
│   ├── sources.json        # Discovered URLs (ai + manual)
│   └── apps.json           # Essential apps per country
├── data/
│   ├── japan_student.json
│   ├── japan_work.json
│   └── ...                 # {country}_{category}.json
├── scripts/
│   ├── pathfinder.py       # URL + app discovery
│   ├── engine_scraper.py   # Scrape + AI structuring
│   └── on_demand.py        # Issue-triggered search
├── .github/workflows/
│   ├── monthly_pathfinder.yml
│   ├── weekly_engine_sync.yml
│   ├── full_pipeline.yml
│   └── on_demand.yml
├── requirements.txt
└── README.md
```
