#!/usr/bin/env python3
# scripts/create_eu_ai_act_vectorstore.py

"""
Script to create and save a vectorstore from the EU AI Act PDF.
This creates a FAISS vectorstore that can be loaded quickly in the main app.
"""

import os
import logging
from pathlib import Path
import pickle
from typing import Optional
import dotenv

# PDF processing
import PyPDF2

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
dotenv.load_dotenv()

# Create logs directory if it doesn't exist
os.makedirs("data_updating_scripts/logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("data_updating_scripts/logs/eu_vectorstore.log")],
)

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            logger.info(f"Processing {len(pdf_reader.pages)} pages from {pdf_path}")
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                    continue
            
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
            
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        raise e

def create_eu_ai_act_documents(text_content: str) -> list:
    """Convert EU AI Act text to Document objects with metadata."""
    try:
        # Initialize text splitter with appropriate settings for legal documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for legal text
            chunk_overlap=200,  # More overlap for context preservation
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Create initial document
        doc = Document(
            page_content=text_content,
            metadata={
                'source': 'EU AI Act',
                'document_type': 'regulation',
                'jurisdiction': 'European Union',
                'title': 'Regulation (EU) 2024/1689 on Artificial Intelligence (AI Act)'
            }
        )
        
        # Split into chunks
        splits = text_splitter.split_documents([doc])
        
        # Add chunk-specific metadata
        for i, split in enumerate(splits):
            split.metadata.update({
                'chunk_id': i,
                'total_chunks': len(splits)
            })
        
        logger.info(f"Created {len(splits)} document chunks")
        return splits
        
    except Exception as e:
        logger.error(f"Error creating documents: {e}")
        raise e

def create_and_save_eu_vectorstore(
    pdf_path: str = "data_updating_scripts/eu-ai-act.pdf", 
    vectorstore_path: str = "data/eu_ai_act_vectorstore",
    openai_api_key: Optional[str] = None
) -> bool:
    """
    Create FAISS vectorstore from EU AI Act PDF and save it locally.
    
    Args:
        pdf_path: Path to the EU AI Act PDF file
        vectorstore_path: Directory to save the vectorstore
        openai_api_key: OpenAI API key (if not provided, uses environment variable)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if PDF exists
        if not Path(pdf_path).exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        # Get API key
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found")
            return False
        
        logger.info("Starting EU AI Act vectorstore creation...")
        
        # Extract text from PDF
        logger.info("Extracting text from PDF...")
        text_content = extract_text_from_pdf(pdf_path)
        
        if not text_content or len(text_content) < 1000:
            logger.error("Insufficient text extracted from PDF")
            return False
        
        # Create documents
        logger.info("Creating document chunks...")
        documents = create_eu_ai_act_documents(text_content)
        
        if not documents:
            logger.error("No documents created")
            return False
        
        # Initialize embeddings
        logger.info("Initializing embeddings...")
        embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-3-small"
        )
        
        # Create vectorstore
        logger.info("Creating FAISS vectorstore...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Create directory if it doesn't exist
        Path(vectorstore_path).mkdir(exist_ok=True)
        
        # Save vectorstore
        logger.info(f"Saving vectorstore to {vectorstore_path}...")
        vectorstore.save_local(vectorstore_path)
        
        # Save metadata
        metadata = {
            'pdf_path': pdf_path,
            'total_chunks': len(documents),
            'text_length': len(text_content),
            'embedding_model': 'text-embedding-3-small',
            'chunk_size': 1500,
            'chunk_overlap': 200
        }
        
        metadata_path = Path(vectorstore_path) / "metadata.pickle"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"✅ EU AI Act vectorstore created successfully!")
        logger.info(f"   - Total chunks: {len(documents)}")
        logger.info(f"   - Text length: {len(text_content):,} characters")
        logger.info(f"   - Saved to: {vectorstore_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating EU AI Act vectorstore: {e}")
        return False

def load_eu_vectorstore(
    vectorstore_path: str = "eu_ai_act_vectorstore",
    openai_api_key: Optional[str] = None
) -> Optional[FAISS]:
    """
    Load the EU AI Act vectorstore from disk.
    
    Args:
        vectorstore_path: Path to the saved vectorstore
        openai_api_key: OpenAI API key
    
    Returns:
        FAISS vectorstore or None if failed
    """
    try:
        if not Path(vectorstore_path).exists():
            logger.error(f"Vectorstore not found: {vectorstore_path}")
            return None
        
        # Get API key
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found")
            return None
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-3-small"
        )
        
        # Load vectorstore
        vectorstore = FAISS.load_local(
            vectorstore_path, 
            embeddings,
            allow_dangerous_deserialization=True  # Required for loading pickled objects
        )
        
        logger.info(f"✅ EU AI Act vectorstore loaded from {vectorstore_path}")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error loading EU AI Act vectorstore: {e}")
        return None

def get_vectorstore_info(vectorstore_path: str = "data/eu_ai_act_vectorstore") -> dict:
    """Get information about the saved vectorstore."""
    try:
        metadata_path = Path(vectorstore_path) / "metadata.pickle"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            return metadata
        else:
            return {"error": "Metadata not found"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Create the vectorstore
    success = create_and_save_eu_vectorstore()
    
    if success:
        # Display info
        info = get_vectorstore_info()
        print("\n" + "="*50)
        print("EU AI Act Vectorstore Information:")
        print("="*50)
        for key, value in info.items():
            if key != 'error':
                print(f"{key}: {value}")
        print("="*50)
    else:
        print("❌ Failed to create EU AI Act vectorstore")
        exit(1)