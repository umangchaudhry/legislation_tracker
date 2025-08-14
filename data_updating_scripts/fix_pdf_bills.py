import os
import json
import base64
import logging
import sys
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import re

# Load environment variables
load_dotenv()
API_KEY = os.getenv("LEGISCAN_API_KEY")

# Files
INPUT_FILE = "data/known_bills.json"
OUTPUT_FILE = "data/known_bills_fixed.json"
BACKUP_FILE = "data/known_bills_backup.json"

# Rate limiting
import time
RATE_LIMIT = 0.2  # seconds between API requests

# Logging configuration
LOG_FILE = "data_updating_scripts/logs/fix_pdf_bills.log"
os.makedirs("data_updating_scripts/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)


def is_pdf_content(text):
    """Check if the text content is an unprocessed PDF."""
    if not text:
        return False
    # Check for PDF header signatures
    pdf_signatures = ["%PDF-1.3", "%PDF-1.4", "%PDF-1.5", "%PDF-1.6", "%PDF-1.7", "%PDF1.3", "%PDF1.4", "%PDF1.5", "%PDF1.6", "%PDF1.7"]
    text_start = text[:20] if len(text) >= 20 else text
    return any(text_start.startswith(sig) for sig in pdf_signatures)


def extract_text_from_pdf_bytes(pdf_bytes):
    """Extract text from PDF bytes using PyPDF2."""
    try:
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_content = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
        
        full_text = "\n".join(text_content)
        
        # Clean up the extracted text
        # Remove excessive whitespace while preserving paragraph breaks
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        full_text = re.sub(r' {2,}', ' ', full_text)
        full_text = full_text.strip()
        
        return full_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return None


def legi_request(op, params):
    """Make a request to the LegiScan API."""
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


def fix_pdf_bill(bill):
    """Fix a single bill with unprocessed PDF content."""
    bill_id = bill.get("bill_id")
    state = bill.get("state")
    bill_num = bill.get("bill_number")
    
    logger.info(f"Fixing PDF content for {state} {bill_num} (ID: {bill_id})")
    
    # First, try to get the bill details again
    details_resp = legi_request("getBill", {"id": bill_id})
    if not details_resp:
        logger.warning(f"Could not fetch bill details for {bill_id}")
        return None
    
    details = details_resp.get("bill", {})
    texts = details.get("texts", [])
    
    if not texts:
        logger.warning(f"No text documents available for {bill_id}")
        return None
    
    # Try to get the text document
    doc_id = texts[0].get("doc_id")
    text_resp = legi_request("getBillText", {"id": doc_id})
    
    if not text_resp or "text" not in text_resp:
        logger.warning(f"Could not fetch text for {bill_id}")
        return None
    
    raw_b64 = text_resp["text"].get("doc", "")
    if not raw_b64:
        logger.warning(f"No document content for {bill_id}")
        return None
    
    try:
        # Decode the base64 content
        decoded = base64.b64decode(raw_b64)
        
        # Check if it's a PDF by looking at the magic bytes
        if decoded[:4] == b'%PDF':
            # It's a PDF, extract text
            extracted_text = extract_text_from_pdf_bytes(decoded)
            if extracted_text and len(extracted_text.strip()) > 100:  # Ensure we got meaningful text
                logger.info(f"Successfully extracted {len(extracted_text)} characters from PDF for {bill_id}")
                return extracted_text
            else:
                logger.warning(f"Extracted text too short or empty for {bill_id}")
                return None
        else:
            # Try to decode as HTML (shouldn't happen for these cases, but just in case)
            try:
                from bs4 import BeautifulSoup
                html = decoded.decode("utf-8", errors="ignore")
                soup = BeautifulSoup(html, "html.parser")
                plain_text = soup.get_text(separator="\n", strip=True)
                if plain_text and len(plain_text.strip()) > 100:
                    logger.info(f"Successfully extracted HTML text for {bill_id}")
                    return plain_text
            except:
                pass
                
        logger.warning(f"Could not process document for {bill_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error processing document for {bill_id}: {e}")
        return None


def main():
    # Load the bills
    logger.info(f"Loading bills from {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'r') as f:
            bills = json.load(f)
    except Exception as e:
        logger.error(f"Could not load bills file: {e}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(bills)} bills")
    
    # Create a backup
    logger.info(f"Creating backup at {BACKUP_FILE}")
    with open(BACKUP_FILE, 'w') as f:
        json.dump(bills, f, indent=2)
    
    # Find bills with unprocessed PDF content
    pdf_bills = []
    for i, bill in enumerate(bills):
        if is_pdf_content(bill.get("text")):
            pdf_bills.append(i)
    
    logger.info(f"Found {len(pdf_bills)} bills with unprocessed PDF content")
    
    # Process each PDF bill
    fixed_count = 0
    failed_count = 0
    
    for idx, bill_idx in enumerate(pdf_bills):
        bill = bills[bill_idx]
        logger.info(f"Processing {idx + 1}/{len(pdf_bills)}: {bill.get('state')} {bill.get('bill_number')}")
        
        # Try to fix the PDF content
        fixed_text = fix_pdf_bill(bill)
        
        if fixed_text:
            # Update the bill with the fixed text
            bills[bill_idx]["text"] = fixed_text
            bills[bill_idx]["lastUpdatedAt"] = datetime.now(timezone.utc).isoformat()
            bills[bill_idx]["text_fixed"] = True  # Mark that we fixed this
            fixed_count += 1
            logger.info(f"Successfully fixed bill {bill.get('bill_id')}")
        else:
            # Mark that we tried but failed
            bills[bill_idx]["text_extraction_failed"] = True
            bills[bill_idx]["lastUpdatedAt"] = datetime.now(timezone.utc).isoformat()
            failed_count += 1
            logger.warning(f"Failed to fix bill {bill.get('bill_id')}")
        
        # Rate limiting
        time.sleep(RATE_LIMIT)
        
        # Save progress every 50 bills
        if (idx + 1) % 50 == 0:
            logger.info(f"Saving progress... ({idx + 1}/{len(pdf_bills)} processed)")
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(bills, f, indent=2)
    
    # Save final results
    logger.info(f"Saving final results to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(bills, f, indent=2)
    
    logger.info(f"Processing complete!")
    logger.info(f"Successfully fixed: {fixed_count} bills")
    logger.info(f"Failed to fix: {failed_count} bills")
    logger.info(f"Output saved to: {OUTPUT_FILE}")
    
    # Optionally, overwrite the original file
    if fixed_count > 0:
        response = input(f"\nDo you want to overwrite {INPUT_FILE} with the fixed data? (y/n): ")
        if response.lower() == 'y':
            with open(INPUT_FILE, 'w') as f:
                json.dump(bills, f, indent=2)
            logger.info(f"Original file {INPUT_FILE} has been updated")


if __name__ == "__main__":
    main()