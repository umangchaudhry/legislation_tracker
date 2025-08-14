#!/usr/bin/env python3
"""
Script to migrate IAPP categories for bills with missing or invalid subcategories.

This script reads bills from known_bills_fixed.json, analyzes bills with missing IAPP categories
using OpenAI API, and saves the results to known_bills_visualize.json.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys
import os
import re
import pandas as pd

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from config import ConfigManager
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Create logs directory if it doesn't exist
os.makedirs("data_updating_scripts/logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("data_updating_scripts/logs/migrate_iapp_categories.log")]
)
logger = logging.getLogger(__name__)

# Exact subcategory lists for validation
EXACT_SUBCATEGORIES = {
    "Governance": ["Program and documentation", "Assessments", "Training", "Responsible individual"],
    "Transparency": ["General notice", "Labeling/notification", "Explanation/incident reporting", "Developer documentation"],
    "Assurance": ["Registration", "Third-party review"],
    "Individual Rights": ["Opt out/appeal", "Nondiscrimination"]
}

# Fallback categories for failed API calls
FALLBACK_CATEGORIES = {
    "Governance": ["Program and documentation"],
    "Transparency": ["General notice"],
    "Assurance": ["Registration"],
    "Individual Rights": ["Opt out/appeal"]
}

class IAPPCategoriesMigrator:
    """Migrates IAPP categories for bills with missing or invalid subcategories."""

    def __init__(self):
        """Initialize the migrator with configuration."""
        self.config = ConfigManager()
        self.input_file = "data/known_bills_fixed.json"
        self.output_file = "data/known_bills_visualize.json"

        if not self.config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.llm = ChatOpenAI(
            model=self.config.OPENAI_LLM_MODEL,
            temperature=0.1,
            max_tokens=1000
        )

        self.iapp_prompt = self._create_iapp_prompt()
        self.chain = self.iapp_prompt | self.llm | StrOutputParser()
        logger.info(f"Initialized IAPPCategoriesMigrator with model: {self.config.OPENAI_LLM_MODEL}")

    def _create_iapp_prompt(self):
        """Create the IAPP analysis prompt with relaxed subcategory matching."""
        prompt_text = """
Analyze the following AI-related bill content using the IAPP (International Association of Privacy Professionals) framework for AI governance categorization.

Your response must be ONLY a JSON object in this exact format with nothing else before or after:
{{"iapp_categories": {{"Governance": ["subcategory1", "subcategory2"], "Transparency": [], "Assurance": [], "Individual Rights": []}}}}

Use these four main categories and their EXACT subcategories (no variations allowed):

**Governance:**
- Program and documentation
- Assessments
- Training
- Responsible individual

**Transparency:**
- General notice
- Labeling/notification
- Explanation/incident reporting
- Developer documentation

**Assurance:**
- Registration
- Third-party review

**Individual Rights:**
- Opt out/appeal
- Nondiscrimination

Guidelines for categorization:
- Select ALL applicable subcategories that the bill directly addresses or substantially discusses
- If a category has no applicable subcategories, try to label it anyway based on surrounding context
- Be specific – prioritize subcategories that are clearly supported, but use judgment if AI or governance themes are present
- Focus on what the bill addresses or emphasizes, even if it doesn’t explicitly mandate requirements
- If the bill discusses AI, automation, decision systems, digital governance, or national technology strategy, categorize it as best as possible
- Avoid returning no categories when possible assuming that the bill is AI governance related, unless it truly could not be categorized into any of the four categories.

Bill content to analyze: {context}
"""
        return ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("human", "Analyze this bill for IAPP categories:")
        ])

    def dataframe_to_documents(self, df):
        """Convert DataFrame to list of Document objects."""
        documents = []
        for _, row in df.iterrows():
            if 'text' in row and pd.notna(row['text']) and row['text'].strip():
                doc = Document(
                    page_content=row['text'],
                    metadata={
                        'bill_key': f"{row.get('state', 'Unknown')}_{row.get('bill_number', 'Unknown')}",
                        'state': row.get('state', 'Unknown'),
                        'bill_number': row.get('bill_number', 'Unknown'),
                        'title': row.get('title', 'No title')
                    }
                )
                documents.append(doc)
        return documents

    def is_valid_iapp_categories(self, iapp_data):
        """Check if IAPP categories are valid (have proper subcategories)."""
        if not iapp_data or not isinstance(iapp_data, dict):
            return False
        for category in ["Governance", "Transparency", "Assurance", "Individual Rights"]:
            if category not in iapp_data or not isinstance(iapp_data[category], list):
                return False
        return True

    def validate_exact_subcategories(self, iapp_data):
        """Validate that all subcategories match the exact required list."""
        if not self.is_valid_iapp_categories(iapp_data):
            return False
        for category, subcategories in iapp_data.items():
            if category in EXACT_SUBCATEGORIES:
                for subcategory in subcategories:
                    if subcategory not in EXACT_SUBCATEGORIES[category]:
                        logger.warning(f"Invalid subcategory '{subcategory}' for category '{category}'")
                        return False
        return True
    
    def truncate_text(self, text: str, max_tokens: int = 120000) -> str:
        """Truncate long text to fit within the model's context window."""
        max_chars = max_tokens * 4  # Approximate: 1 token ≈ 4 chars
        if len(text) > max_chars:
            logger.warning(f"Truncating input text from {len(text)} to {max_chars} characters")
            return text[:max_chars]
        return text


    def analyze_iapp_categories_new(self, bill: Dict) -> Optional[Dict]:
        """Generate IAPP categories for a single bill with exact validation."""
        try:
            bill_number = bill.get('bill_number', 'Unknown')
            bill_text = self.truncate_text(bill.get('text', ''))

            if not bill_text:
                logger.warning(f"No text found for bill {bill_number}")
                return FALLBACK_CATEGORIES

            df = pd.DataFrame([bill])
            docs = self.dataframe_to_documents(df)

            if not docs:
                logger.warning(f"No document created for bill {bill_number}")
                return FALLBACK_CATEGORIES

            response = self.chain.invoke({"context": docs})
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                iapp_categories = result.get('iapp_categories', {})

                if self.validate_exact_subcategories(iapp_categories):
                    logger.info(f"Generated valid IAPP categories for {bill_number}")
                    return iapp_categories
                else:
                    logger.warning(f"Generated categories failed validation for {bill_number}, retrying...")
                    response = self.chain.invoke({"context": docs})
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                        iapp_categories = result.get('iapp_categories', {})
                        if self.validate_exact_subcategories(iapp_categories):
                            logger.info(f"Retry successful for {bill_number}")
                            return iapp_categories

            logger.warning(f"Failed to parse response for {bill_number}, attempting heuristic fallback")

            if 'ai' in bill_text.lower() or 'artificial intelligence' in bill_text.lower():
                logger.info(f"Heuristic triggered: found 'AI' mention in {bill_number}, using fallback categories")
                return FALLBACK_CATEGORIES

            return {
                "Governance": [],
                "Transparency": [],
                "Assurance": [],
                "Individual Rights": []
            }

        except Exception as e:
            logger.error(f"Error generating IAPP categories for bill {bill.get('bill_number', 'Unknown')}: {e}")
            return FALLBACK_CATEGORIES

    def load_bills(self) -> List[Dict]:
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                bills = json.load(f)
            logger.info(f"Loaded {len(bills)} bills from {self.input_file}")
            return bills
        except FileNotFoundError:
            logger.error(f"File not found: {self.input_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            raise

    def save_bills(self, bills: List[Dict]) -> None:
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(bills, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(bills)} bills to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving bills: {e}")
            raise

    def migrate_all_bills(self) -> None:
        bills = self.load_bills()
        total_bills = len(bills)
        processed = 0
        skipped = 0
        errors = 0

        logger.info(f"Starting IAPP categories regeneration for {total_bills} bills")

        for i, bill in enumerate(bills, 1):
            bill_key = f"{bill.get('state', 'Unknown')}_{bill.get('bill_number', 'Unknown')}"

            if not bill.get('text') or not isinstance(bill['text'], str) or not bill['text'].strip():
                logger.info(f"Skipping {bill_key} - no text content")
                skipped += 1
                continue

            logger.info(f"Processing {i}/{total_bills}: {bill_key}")
            new_iapp_categories = self.analyze_iapp_categories_new(bill)
            bill['iapp_categories'] = new_iapp_categories

            logger.info(f"📊 IAPP categories for {bill_key}: {json.dumps(new_iapp_categories, indent=2)}")

            if all(not v for v in new_iapp_categories.values()):
                logger.warning(f"⚠️ Empty IAPP categories for {bill_key} despite having text.")

            if new_iapp_categories == FALLBACK_CATEGORIES:
                errors += 1
                logger.warning(f"Used fallback categories for {bill_key}")
            else:
                processed += 1
                logger.info(f"✅ Generated new categories for {bill_key}")

            if i % 10 == 0:
                self.save_bills(bills)
                logger.info(f"Progress: {i}/{total_bills} processed, {processed} successful, {errors} errors")

            time.sleep(1)

        self.save_bills(bills)
        logger.info(f"IAPP categories regeneration complete!")
        logger.info(f"Total bills: {total_bills}")
        logger.info(f"Successfully processed: {processed}")
        logger.info(f"Skipped (no text): {skipped}")
        logger.info(f"Errors (used fallback): {errors}")
        logger.info(f"Results saved to: {self.output_file}")

def main():
    try:
        migrator = IAPPCategoriesMigrator()
        migrator.migrate_all_bills()
        print("✅ IAPP categories migration completed successfully!")
        print(f" Results saved to: {migrator.output_file}")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
