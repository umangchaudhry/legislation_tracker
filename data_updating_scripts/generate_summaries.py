#!/usr/bin/env python3
"""
Script to generate summaries for all bills in known_bills_visualize.json.

This script reads all bills from known_bills_visualize.json, generates summaries using OpenAI API,
and saves them to data/bill_summaries.json to avoid repeated API calls.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys
import os

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from config import ConfigManager
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from PROMPTS.bill_summary_prompt import BILL_SUMMARY_PROMPT

# Create logs directory if it doesn't exist
os.makedirs("data_updating_scripts/logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("data_updating_scripts/logs/generate_summaries.log")]
)
logger = logging.getLogger(__name__)

class BillSummaryGenerator:
    """Generates summaries for all bills in known_bills_visualize.json."""
    
    def __init__(self):
        """Initialize the summary generator with configuration."""
        self.config = ConfigManager()
        self.known_bills_file = Path("data/known_bills_visualize.json")
        self.summaries_file = Path("data/bill_summaries.json")
        
        # Initialize OpenAI LLM
        if not self.config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(
            model=self.config.OPENAI_LLM_MODEL,
            temperature=0.1,
            max_tokens=1000
        )
        
        # Create the prompt template
        self.prompt_template = PromptTemplate(
            template=BILL_SUMMARY_PROMPT,
            input_variables=["bill_number", "bill_title", "state", "bill_text"]
        )
        
        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        
        logger.info(f"Initialized BillSummaryGenerator with model: {self.config.OPENAI_LLM_MODEL}")
    
    def load_known_bills(self) -> List[Dict]:
        """Load bills from known_bills_visualize.json."""
        try:
            with open(self.known_bills_file, 'r', encoding='utf-8') as f:
                bills = json.load(f)
            logger.info(f"Loaded {len(bills)} bills from {self.known_bills_file}")
            return bills
        except FileNotFoundError:
            logger.error(f"File not found: {self.known_bills_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            raise
    
    def load_existing_summaries(self) -> Dict:
        """Load existing summaries if available."""
        if self.summaries_file.exists():
            try:
                with open(self.summaries_file, 'r', encoding='utf-8') as f:
                    summaries = json.load(f)
                logger.info(f"Loaded {len(summaries)} existing summaries")
                return summaries
            except Exception as e:
                logger.warning(f"Could not load existing summaries: {e}")
                return {}
        return {}
    
    def save_summaries(self, summaries: Dict) -> None:
        """Save summaries to JSON file."""
        try:
            with open(self.summaries_file, 'w', encoding='utf-8') as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(summaries)} summaries to {self.summaries_file}")
        except Exception as e:
            logger.error(f"Error saving summaries: {e}")
            raise
    
    def generate_summary(self, bill: Dict) -> Optional[str]:
        """Generate summary for a single bill."""
        try:
            bill_number = bill.get('bill_number', 'Unknown')
            bill_title = bill.get('title', 'No title')
            state = bill.get('state', 'Unknown')
            bill_text = bill.get('text', '')
            
            if not bill_text:
                logger.warning(f"No text found for bill {bill_number}")
                return "ERROR: No bill text available"
            
            # Prepare the input for the chain
            chain_input = {
                "bill_number": bill_number,
                "bill_title": bill_title,
                "state": state,
                "bill_text": bill_text[:8000]  # Limit text length to avoid token limits
            }
            
            # Generate summary using the chain
            summary = self.chain.invoke(chain_input)
            
            logger.info(f"Generated summary for {bill_number}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary for bill {bill.get('bill_number', 'Unknown')}: {e}")
            return f"ERROR: {str(e)}"
    
    def generate_all_summaries(self) -> None:
        """Generate summaries for all bills."""
        # Load bills and existing summaries
        bills = self.load_known_bills()
        existing_summaries = self.load_existing_summaries()
        
        # Track progress
        total_bills = len(bills)
        processed = 0
        errors = 0
        
        logger.info(f"Starting summary generation for {total_bills} bills")
        
        for i, bill in enumerate(bills, 1):
            bill_key = f"{bill.get('state', 'Unknown')}_{bill.get('bill_number', 'Unknown')}"
            
            # Skip if already processed successfully
            if bill_key in existing_summaries and not existing_summaries[bill_key].get('summary', '').startswith('ERROR:'):
                logger.info(f"Skipping {bill_key} - already processed")
                processed += 1
                continue
            
            logger.info(f"Processing {i}/{total_bills}: {bill_key}")
            
            # Generate summary
            summary = self.generate_summary(bill)
            
            # Store result
            existing_summaries[bill_key] = {
                'bill_number': bill.get('bill_number', 'Unknown'),
                'title': bill.get('title', 'No title'),
                'summary': summary
            }
            
            if summary.startswith('ERROR:'):
                errors += 1
            else:
                processed += 1
            
            # Save progress every 10 bills
            if i % 10 == 0:
                self.save_summaries(existing_summaries)
                logger.info(f"Progress: {i}/{total_bills} processed, {errors} errors")
            
            # Rate limiting
            time.sleep(1)  # 1 second delay between API calls
        
        # Final save
        self.save_summaries(existing_summaries)
        
        logger.info(f"Summary generation complete!")
        logger.info(f"Total bills: {total_bills}")
        logger.info(f"Successfully processed: {processed}")
        logger.info(f"Errors: {errors}")
        logger.info(f"Summaries saved to: {self.summaries_file}")

def main():
    """Main function to run the summary generation."""
    try:
        generator = BillSummaryGenerator()
        generator.generate_all_summaries()
        print("✅ Summary generation completed successfully!")
        print(f" Summaries saved to: {generator.summaries_file}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 