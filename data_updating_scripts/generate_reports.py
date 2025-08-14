"""
generate_reports.py
--------------------

Generates detailed Markdown reports for AI-related bills from `known_bills_visualize.json`
using the latest LangChain pipeline syntax.

Now includes resume functionality - can be safely stopped and restarted.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import dotenv


dotenv.load_dotenv()

# Create logs directory if it doesn't exist
os.makedirs("data_updating_scripts/logs", exist_ok=True)

# Latest LangChain imports
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("data_updating_scripts/logs/generate_reports.log")],
)

logger = logging.getLogger(__name__)


@dataclass
class BillReport:
    """Stores a bill ID and its generated detailed report."""
    bill_id: str
    report_markdown: str


# Prompt template
DETAILED_REPORT_PROMPT = ChatPromptTemplate.from_template(
    """You are a seasoned legislative analyst adept at interpreting and
    summarising bills related to artificial intelligence. Using the bill
    information provided as JSON, produce a detailed report in Markdown
    format for stakeholders.

    Include:
    - Bill's title, number, and state
    - Status and key dates
    - URL to the bill on legiscan
    - Sponsors and scope
    - Goals and intent
    - Key provisions, regulatory approaches, implementation & enforcement
    - Unique aspects or notable features

    Format:
    - Use Markdown headings and bullet points
    - Paraphrase content
    - Do not invent facts
    - If bill text is truncated in source JSON, note this at the end

    Bill JSON:
    ```json
    {bill_json}
    ```

    Now craft the detailed report.
    """
)


def _ensure_llm() -> ChatOpenAI:
    """Initialise ChatOpenAI with latest settings."""
    if ChatOpenAI is None:
        raise RuntimeError(
            "The 'langchain' and 'openai' packages are required. Install them via 'pip install langchain openai'."
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("The OPENAI_API_KEY environment variable is not set.")
    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    logger.debug("Initialising ChatOpenAI with model %s", model_name)
    return ChatOpenAI(model=model_name, temperature=0)


def create_detailed_report(
    bill: Dict[str, Any], *, llm: Optional[ChatOpenAI] = None
) -> BillReport:
    """Generate a detailed report for a single bill using latest LangChain syntax."""
    if llm is None:
        llm = _ensure_llm()

    bill_json = json.dumps(bill, ensure_ascii=False, indent=2)

    # Latest syntax: prompt | llm
    chain = DETAILED_REPORT_PROMPT | llm
    result = chain.invoke({"bill_json": bill_json})

    # result can be AIMessage; get text
    report_text = getattr(result, "content", str(result))

    return BillReport(bill_id=str(bill.get("bill_id")), report_markdown=report_text)


def load_existing_reports(output_path: str) -> Dict[str, str]:
    """Load existing reports from file if it exists."""
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                reports_list = json.load(f)
                # Convert list to dict for easy lookup
                reports_dict = {
                    report["bill_id"]: report["report_markdown"]
                    for report in reports_list
                    if "bill_id" in report and "report_markdown" in report
                }
                logger.info(f"Loaded {len(reports_dict)} existing reports from {output_path}")
                return reports_dict
        except Exception as e:
            logger.warning(f"Could not load existing reports: {e}")
            return {}
    return {}


def save_reports_to_file(reports_dict: Dict[str, str], output_path: str) -> None:
    """Save reports dictionary to a JSON file."""
    # Convert dict back to list format for consistency
    out_list = [
        {"bill_id": bill_id, "report_markdown": report_markdown}
        for bill_id, report_markdown in reports_dict.items()
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d reports to %s", len(out_list), output_path)


def create_reports_with_resume(
    bills: List[Dict[str, Any]], 
    output_path: str,
    *, 
    llm: Optional[ChatOpenAI] = None,
    save_interval: int = 10
) -> Dict[str, str]:
    """
    Generate detailed reports for multiple bills with resume capability.
    
    Args:
        bills: List of bill dictionaries
        output_path: Path to save reports
        llm: Optional LLM instance
        save_interval: Save progress every N bills
    
    Returns:
        Dictionary of bill_id -> report_markdown
    """
    if not bills:
        return {}
    
    if llm is None:
        llm = _ensure_llm()
    
    # Load existing reports
    reports_dict = load_existing_reports(output_path)
    
    # Track progress
    total_bills = len(bills)
    processed = 0
    skipped = 0
    errors = 0
    
    logger.info(f"Starting report generation for {total_bills} bills")
    
    for i, bill in enumerate(bills, 1):
        bill_id = str(bill.get("bill_id"))
        
        # Skip if already processed
        if bill_id in reports_dict and reports_dict[bill_id] and not reports_dict[bill_id].startswith("ERROR:"):
            logger.info(f"Skipping bill {bill_id} - already processed ({i}/{total_bills})")
            skipped += 1
            continue
        
        logger.info(f"Processing {i}/{total_bills}: Bill ID {bill_id}")
        
        try:
            report = create_detailed_report(bill, llm=llm)
            reports_dict[bill_id] = report.report_markdown
            processed += 1
            
        except Exception as exc:
            logger.exception(
                "Failed to generate report for bill %s: %s", bill_id, exc
            )
            reports_dict[bill_id] = f"ERROR: Failed to generate report - {str(exc)}"
            errors += 1
        
        # Save progress periodically
        if i % save_interval == 0:
            save_reports_to_file(reports_dict, output_path)
            logger.info(f"Progress: {i}/{total_bills} - Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
        
        # Rate limiting to avoid API throttling
        if bill_id not in reports_dict or reports_dict[bill_id].startswith("ERROR:"):
            time.sleep(1)  # 1 second delay between API calls
    
    # Final save
    save_reports_to_file(reports_dict, output_path)
    
    logger.info(f"Report generation complete!")
    logger.info(f"Total bills: {total_bills}")
    logger.info(f"Successfully processed: {processed}")
    logger.info(f"Skipped (already done): {skipped}")
    logger.info(f"Errors: {errors}")
    
    return reports_dict


def read_bills_from_file(path: str) -> List[Dict[str, Any]]:
    """Read bill records from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list of bills in {path}, got {type(data)}")
        return data


def generate_reports_from_files(
    input_path: str = "data/known_bills_visualize.json",
    output_path: str = "data/bill_reports.json",
) -> None:
    """Read bills, generate reports with resume capability, and write them to disk."""
    bills = read_bills_from_file(input_path)
    create_reports_with_resume(bills, output_path)


def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Generate detailed AI legislation reports from bill data with resume capability."
    )
    parser.add_argument("--input", default="data/known_bills_visualize.json", help="Path to input JSON file")
    parser.add_argument("--output", default="data/bill_reports.json", help="Path to output JSON file")
    parser.add_argument("--save-interval", type=int, default=10, help="Save progress every N bills (default: 10)")
    args = parser.parse_args()
    
    try:
        bills = read_bills_from_file(args.input)
        create_reports_with_resume(bills, args.output, save_interval=args.save_interval)
        print(f"✅ Report generation completed successfully!")
        print(f"   Reports saved to: {args.output}")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Error: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()