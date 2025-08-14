import os
import sys
import json
import time
import logging
import base64
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()
# Pull API key from environment
API_KEY = os.getenv("LEGISCAN_API_KEY")  # Set your LegiScan API key in .env
if not API_KEY:
    print("Error: Please set LEGISCAN_API_KEY in your .env file.")
    sys.exit(1)

# Modes for testing
# Quick test: pulls only TEST_MAX_BILLS bills
TESTING_MODE = False
# Full test: pulls all bills for TEST_STATE and TEST_YEAR without bill count cap
FULL_TESTING_MODE = False
TEST_STATE = 'CA'
TEST_YEAR = 2023
TEST_MAX_BILLS = 3

# Output files
CACHE_FILE = "data/bill_cache.json"        # Stores bill_id -> change_hash
OUTPUT_FILE = "data/known_bills.json"     # Final bills data

# Query settings
QUERY = "artificial intelligence"
START_YEAR = 2023
END_YEAR = datetime.now(timezone.utc).year

# Include all state legislatures plus U.S. Congress (both chambers)
STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
    "US"  # U.S. Congress
]

# Rate limiting (seconds between requests)
RATE_LIMIT = 0.2

# Create logs directory if it doesn't exist
os.makedirs("data_updating_scripts/logs", exist_ok=True)

# Logging configuration
LOG_FILE = "data_updating_scripts/logs/fetch_ai_bills.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# Apply testing overrides
if TESTING_MODE:
    logger.info(f"*** TESTING MODE: fetching only {TEST_MAX_BILLS} bills from {TEST_STATE} ({TEST_YEAR}) ***")
    STATES = [TEST_STATE]
if FULL_TESTING_MODE:
    logger.info(f"*** FULL TESTING MODE: fetching all bills from {TEST_STATE} ({TEST_YEAR}) ***")
    STATES = [TEST_STATE]


def load_json(path, default):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, data):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {path}")


def legi_request(op, params):
    base = "https://api.legiscan.com/"
    params.update({"key": API_KEY, "op": op})
    try:
        resp = requests.get(base, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK":
            logger.error(f"API error {op}: {data.get('message', data)}")
            return None
        return data
    except requests.RequestException as e:
        logger.error(f"Request failed ({op}): {e}")
        return None


def extract_plain_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def main():
    cache = load_json(CACHE_FILE, {})
    existing = load_json(OUTPUT_FILE, [])
    existing_map = {b.get("bill_id"): b for b in existing}
    logger.info(f"Loaded cache entries: {len(cache)}, existing bills: {len(existing)}")

    collected = []
    total_fetched = 0
    years = [TEST_YEAR] if (TESTING_MODE or FULL_TESTING_MODE) else list(range(START_YEAR, END_YEAR + 1))

    for state in STATES:
        for year in years:
            page = 1
            while True:
                if TESTING_MODE and total_fetched >= TEST_MAX_BILLS:
                    logger.info("Reached TEST_MAX_BILLS limit, stopping early.")
                    break
                params = {"state": state, "year": year, "query": QUERY, "page": page}
                logger.info(f"Searching {state} for {year}, page {page}")
                data = legi_request("getSearch", params)
                if not data:
                    break

                results = data.get("searchresult", {})
                summary = results.get("summary", {})
                bills = [v for k, v in results.items() if k != "summary"]
                if not bills:
                    logger.info(f"No bills on page {page} for {state} {year}")
                    break

                logger.info(f"Found {len(bills)} bills on {state} {year} page {page}")
                for bill in bills:
                    if TESTING_MODE and total_fetched >= TEST_MAX_BILLS:
                        break
                    bill_id = str(bill.get("bill_id"))
                    state_code = bill.get("state")
                    bill_num = bill.get("bill_number")
                    logger.info(f"Processing bill {state_code}_{bill_num} (ID: {bill_id})")

                    details_resp = legi_request("getBill", {"id": bill_id})
                    if not details_resp:
                        continue
                    details = details_resp.get("bill", {})
                    sess_year = details.get("session", {}).get("year_start", 0)
                    if sess_year < START_YEAR:
                        continue

                    new_hash = details.get("change_hash")
                    old_hash = cache.get(bill_id)
                    now_iso = datetime.now(timezone.utc).isoformat()

                    # Extract all relevant dates
                    explicit = details.get("last_action_date")
                    status_date = details.get("status_date")
                    last_vote_date = details.get("last_vote_date")
                    last_amendment_date = details.get("last_amendment_date")
                    actions = details.get("actions", [])
                    action_dates = [a.get("action_date") for a in actions if a.get("action_date")]
                    most_recent_action = max(action_dates) if action_dates else None
                    candidates = [d for d in [explicit, status_date, last_vote_date, last_amendment_date, most_recent_action] if d]
                    last_action_date = max(candidates) if candidates else None

                    bill_url = details.get("url")  # Bill detail page URL

                    if new_hash and new_hash == old_hash and bill_id in existing_map:
                        entry = existing_map[bill_id]
                        entry.update({
                            "status": details.get("status"),
                            "session_year": f"{details.get('session', {}).get('year_start', '')}-{details.get('session', {}).get('year_end', '')}",
                            "last_action_date": last_action_date,
                            "status_date": status_date,
                            "last_vote_date": last_vote_date,
                            "last_amendment_date": last_amendment_date,
                            "actions": actions,
                            "bill_url": bill_url,
                            "lastUpdatedAt": now_iso
                        })
                        logger.info(f"Reused cache; updated status={entry['status']}, last_action_date={entry['last_action_date']}")
                    else:
                        plain_text = None
                        texts = details.get("texts", [])
                        if texts:
                            doc_id = texts[0].get("doc_id")
                            text_resp = legi_request("getBillText", {"id": doc_id})
                            if text_resp and "text" in text_resp:
                                raw_b64 = text_resp["text"].get("doc", "")
                                try:
                                    decoded = base64.b64decode(raw_b64)
                                    html = decoded.decode("utf-8", errors="ignore")
                                    plain_text = extract_plain_text(html)
                                except Exception as e:
                                    logger.error(f"Failed decoding HTML for {bill_id}: {e}")

                        entry = {
                            "bill_id": bill_id,
                            "state": state_code,
                            "bill_number": bill_num,
                            "session_year": f"{details.get('session', {}).get('year_start', '')}-{details.get('session', {}).get('year_end', '')}",
                            "title": details.get("title"),
                            "description": details.get("description"),
                            "status": details.get("status"),
                            "sponsors": [s.get("name") for s in details.get("sponsors", [])],
                            "text": plain_text,
                            "last_action_date": last_action_date,
                            "status_date": status_date,
                            "last_vote_date": last_vote_date,
                            "last_amendment_date": last_amendment_date,
                            "actions": actions,
                            "bill_url": bill_url,
                            "change_hash": new_hash,
                            "lastUpdatedAt": now_iso
                        }
                        cache[bill_id] = new_hash
                        logger.info(
                            f"Entry data: title='{entry['title']}', sponsors={len(entry['sponsors'])}, "
                            f"status={entry['status']}, last_action_date={entry['last_action_date']}"
                        )

                    collected.append(entry)
                    total_fetched += 1
                    time.sleep(RATE_LIMIT)

                if page >= summary.get("page_total", 1):
                    break
                page += 1
                time.sleep(RATE_LIMIT)
            if TESTING_MODE and total_fetched >= TEST_MAX_BILLS:
                break
        if TESTING_MODE and total_fetched >= TEST_MAX_BILLS:
            break

    dedup = {e["bill_id"]: e for e in collected}
    all_bills = list(dedup.values())
    save_json(OUTPUT_FILE, all_bills)
    save_json(CACHE_FILE, cache)
    logger.info(f"Completed run, saved {len(all_bills)} bills to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
