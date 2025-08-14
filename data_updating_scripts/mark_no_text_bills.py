#!/usr/bin/env python3
"""
Script to mark bills without text as having None IAPP categories.

This script reads known_bills_visualize.json, identifies bills without text,
and sets their IAPP categories to None. The file is modified in-place.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Create logs directory if it doesn't exist
os.makedirs("data_updating_scripts/logs", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(), logging.FileHandler("data_updating_scripts/logs/mark_no_text_bills.log")]
)
logger = logging.getLogger(__name__)


class NoTextBillMarker:
    """Class to mark bills without text as having None IAPP categories."""
    
    def __init__(self):
        self.visualize_file = "data/known_bills_visualize.json"
    
    def load_bills(self) -> List[Dict]:
        """Load bills from known_bills_visualize.json."""
        try:
            with open(self.visualize_file, 'r', encoding='utf-8') as f:
                bills = json.load(f)
            logger.info(f"Loaded {len(bills)} bills from {self.visualize_file}")
            return bills
        except Exception as e:
            logger.error(f"Error loading bills: {e}")
            return []
    
    def save_bills(self, bills: List[Dict]) -> None:
        """Save bills back to known_bills_visualize.json."""
        try:
            with open(self.visualize_file, 'w', encoding='utf-8') as f:
                json.dump(bills, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(bills)} bills to {self.visualize_file}")
        except Exception as e:
            logger.error(f"Error saving bills: {e}")
    
    def has_text(self, bill: Dict) -> bool:
        text = bill.get('text')
        return isinstance(text, str) and len(text.strip()) > 50
    
    def mark_no_text_bills(self) -> None:
        """Mark bills without text as having None IAPP categories."""
        # Load bills
        bills = self.load_bills()
        if not bills:
            logger.error("No bills loaded. Exiting.")
            return
        
        # Track progress
        total_bills = len(bills)
        no_text_count = 0
        already_none_count = 0
        
        logger.info(f"Processing {total_bills} bills to mark no-text bills")
        
        for i, bill in enumerate(bills, 1):
            bill_key = f"{bill.get('state', 'Unknown')}_{bill.get('bill_number', 'Unknown')}"
            
            # Check if bill has text
            if not self.has_text(bill):
                no_text_count += 1
                
                # Check if IAPP categories are already None
                current_iapp = bill.get('iapp_categories')
                if current_iapp is None:
                    already_none_count += 1
                    logger.debug(f"Bill {bill_key} already has None IAPP categories")
                else:
                    # Set IAPP categories to None
                    bill['iapp_categories'] = None
                    logger.info(f"Marked bill {bill_key} as having None IAPP categories (no text)")
            
            # Log progress every 100 bills
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{total_bills} processed")
        
        # Save the modified bills
        self.save_bills(bills)
        
        # Summary
        logger.info(f"Processing complete!")
        logger.info(f"Total bills processed: {total_bills}")
        logger.info(f"Bills without text: {no_text_count}")
        logger.info(f"Already had None categories: {already_none_count}")
        logger.info(f"Newly marked as None: {no_text_count - already_none_count}")


def main():
    """Main function to run the no-text bill marker."""
    logger.info("Starting no-text bill marking process")
    
    marker = NoTextBillMarker()
    marker.mark_no_text_bills()
    
    logger.info("No-text bill marking process completed")


if __name__ == "__main__":
    main() 