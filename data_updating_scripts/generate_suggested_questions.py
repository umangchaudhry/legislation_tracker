#!/usr/bin/env python3
"""
Script to generate suggested questions for all bills in known_bills_visualize.json.

This script reads all bills from known_bills_visualize.json, generates 5 suggested questions using OpenAI API,
and saves them to data/bill_suggested_questions.json to avoid repeated API calls.
"""

import json
import logging
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import sys
import os

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from config import ConfigManager
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

# Create logs directory if it doesn't exist
os.makedirs("data_updating_scripts/logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("data_updating_scripts/logs/generate_suggested_questions.log")]
)
logger = logging.getLogger(__name__)

class SuggestedQuestionsGenerator:
    """Generates suggested questions for all bills in known_bills_visualize.json."""
    
    def __init__(self):
        """Initialize the questions generator with configuration."""
        self.config = ConfigManager()
        self.known_bills_file = Path("data/known_bills_visualize.json")
        self.questions_file = Path("data/bill_suggested_questions.json")
        
        # Initialize OpenAI LLM
        if not self.config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(
            model=self.config.OPENAI_LLM_MODEL,
            temperature=0.3,
            max_tokens=500
        )
        
        # Load the system prompt from markdown file
        prompt_path = "data_updating_scripts/PROMPTS/suggested_questions_prompt.md"
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"The specified file was not found: {prompt_path}")
        
        with open(prompt_path, "r") as file:
            system_prompt = file.read()
        
        # Create the prompt and chain
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Generate 5 specific questions about this bill based on its content."),
            ]
        )
        
        self.question_generation_chain = create_stuff_documents_chain(
            self.llm, self.prompt
        )
        
        # Fallback questions
        self.fallback_questions = [
            "What are the key definitions in this bill?",
            "What are the enforcement mechanisms?",
            "Who does this bill apply to?",
            "What are the compliance requirements?",
            "What penalties are specified?"
        ]
        
        logger.info(f"Initialized SuggestedQuestionsGenerator with model: {self.config.OPENAI_LLM_MODEL}")
    
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
    
    def load_existing_questions(self) -> Dict:
        """Load existing questions if available."""
        if self.questions_file.exists():
            try:
                with open(self.questions_file, 'r', encoding='utf-8') as f:
                    questions = json.load(f)
                logger.info(f"Loaded {len(questions)} existing question sets")
                return questions
            except Exception as e:
                logger.warning(f"Could not load existing questions: {e}")
                return {}
        return {}
    
    def save_questions(self, questions: Dict) -> None:
        """Save questions to JSON file."""
        try:
            with open(self.questions_file, 'w', encoding='utf-8') as f:
                json.dump(questions, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(questions)} question sets to {self.questions_file}")
        except Exception as e:
            logger.error(f"Error saving questions: {e}")
            raise
    
    def parse_questions_response(self, response: str) -> List[str]:
        """Parse the LLM response into individual questions."""
        questions = []
        if isinstance(response, str):
            # Split by lines and clean up
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            # Filter out any numbering or bullet points
            for line in lines:
                # Remove common prefixes like "1.", "2.", "3.", "4.", "5.", "•", "-", "*", etc.
                clean_line = line
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '•', '-', '*')):
                    clean_line = line[2:].strip()
                elif line.startswith(('1)', '2)', '3)', '4)', '5)')):
                    clean_line = line[2:].strip()
                
                if clean_line and clean_line.endswith('?'):
                    questions.append(clean_line)
        
        # Ensure we have exactly 5 questions
        if len(questions) < 5:
            # Use fallback questions to fill up to 5
            questions.extend(self.fallback_questions[len(questions):])
        
        return questions[:5]  # Return only the first 5
    
    def generate_questions(self, bill: Dict) -> Optional[List[str]]:
        """Generate suggested questions for a single bill."""
        try:
            bill_number = bill.get('bill_number', 'Unknown')
            bill_title = bill.get('title', 'No title')
            bill_text = bill.get('text', '')
            
            if not bill_text:
                logger.warning(f"No text found for bill {bill_number}")
                return self.fallback_questions
            
            # Convert bill to document format
            df = pd.DataFrame([bill])
            docs = self.dataframe_to_documents(df)
            
            if not docs:
                logger.warning(f"No document created for bill {bill_number}")
                return self.fallback_questions
            
            # Generate questions using the chain
            response = self.question_generation_chain.invoke({"context": docs})
            
            # Parse the response into questions
            questions = self.parse_questions_response(response)
            
            logger.info(f"Generated {len(questions)} questions for {bill_number}")
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions for bill {bill.get('bill_number', 'Unknown')}: {e}")
            return self.fallback_questions
    
    def generate_all_questions(self) -> None:
        """Generate suggested questions for all bills."""
        # Load bills and existing questions
        bills = self.load_known_bills()
        existing_questions = self.load_existing_questions()
        
        # Track progress
        total_bills = len(bills)
        processed = 0
        errors = 0
        
        logger.info(f"Starting question generation for {total_bills} bills")
        
        for i, bill in enumerate(bills, 1):
            bill_key = f"{bill.get('state', 'Unknown')}_{bill.get('bill_number', 'Unknown')}"
            
            # Skip if already processed successfully
            if bill_key in existing_questions and len(existing_questions[bill_key].get('suggested_questions', [])) == 5:
                logger.info(f"Skipping {bill_key} - already processed")
                processed += 1
                continue
            
            logger.info(f"Processing {i}/{total_bills}: {bill_key}")
            
            # Generate questions
            questions = self.generate_questions(bill)
            
            # Store result
            existing_questions[bill_key] = {
                'bill_number': bill.get('bill_number', 'Unknown'),
                'title': bill.get('title', 'No title'),
                'suggested_questions': questions
            }
            
            if questions == self.fallback_questions:
                errors += 1
            else:
                processed += 1
            
            # Save progress every 10 bills
            if i % 10 == 0:
                self.save_questions(existing_questions)
                logger.info(f"Progress: {i}/{total_bills} processed, {errors} errors")
            
            # Rate limiting
            time.sleep(1)  # 1 second delay between API calls
        
        # Final save
        self.save_questions(existing_questions)
        
        logger.info(f"Question generation complete!")
        logger.info(f"Total bills: {total_bills}")
        logger.info(f"Successfully processed: {processed}")
        logger.info(f"Errors: {errors}")
        logger.info(f"Questions saved to: {self.questions_file}")

def main():
    """Main function to run the question generation."""
    try:
        generator = SuggestedQuestionsGenerator()
        generator.generate_all_questions()
        print("✅ Suggested questions generation completed successfully!")
        print(f" Questions saved to: {generator.questions_file}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 