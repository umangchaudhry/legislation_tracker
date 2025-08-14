# ===============================================
# VAILL AI Governance Bills Tracker
# ===============================================
# Table of Contents (functions):
#    1. get_qa_llm()
#    2. get_embeddings()
#    3. get_text_splitter()
#    4. create_bill_documents()
#    5. create_vectorstore_from_bills()
#    6. compare_bills_with_rag()
#    7. answer_bill_question()
#    8. load_eu_ai_act_vectorstore()
#    9. get_eu_vectorstore_info()
#   10. compare_bill_with_eu_ai_act()
#   11. load_bill_reports()
#   12. get_bill_report()
#   13. load_bill_summaries()
#   14. load_bill_suggested_questions()
#   15. get_bill_suggested_questions()
#   16. get_bill_summary()
#   17. load_and_process_data()
#   18. load_openai_api_key()
#   19. display_bill_details()
#   20. get_last_updated_date()
#   21. extract_iapp_subcategories()
#   22. format_date()
#   23. load_us_states_geojson()
#   24. _bill_label()
#   25. _group_to_ul()
#   26. create_bill_options()
# -----------------------------------------------
# Section markers: search for '==== SECTION' lines to jump around.
# ===============================================

# ==== SECTION: Original file begins below (unchanged) ====
#!/usr/bin/env python3
# scripts/app.py

"""
Streamlit visualization for the AI Governance Bills Tracker.

Displays an interactive dashboard of AI-related bills from known_bills_visualize.json, including
a table, map, filters, Q&A, plan comparison, summary generation, and CSV download functionality.
"""

import streamlit as st
import pandas as pd
import time
from streamlit_folium import st_folium
import folium
import json
from pathlib import Path
import os
import dotenv
import io
import logging
from datetime import datetime, date, timedelta
from constants import IAPP_CATEGORIES
import requests
import html
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import pickle
import os

dotenv.load_dotenv()

# Create logs directory if it doesn't exist
os.makedirs("app_logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app_logs/visualize.log")],
)

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="VAILL AI Governance Legislation Tracker",
    page_icon="⚖️"
)

# Custom CSS for clean, section-based layout
st.markdown("""
<style>
    /* Main app background */
    .main .block-container {
        background-color: #ffffff;
        padding-top: 2rem;
    }
    .stApp {
        background-color: #f8fafc;
    }
    /* Sidebar styling - make it white with dark text */
    .css-1d391kg {
        background-color: #ffffff !important;
    }
    .css-1y4p8pa {
        background-color: #ffffff !important;
    }
    .css-k1ih6n {
        background-color: #ffffff !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
    }
    section[data-testid="stSidebar"] > div {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #2c3e50 !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #2c3e50 !important;
    }
    /* Data table styling - aggressive override for dark theme */
    .stDataFrame {
        background-color: #ffffff !important;
    }
    .stDataFrame > div {
        background-color: #ffffff !important;
    }
    .stDataFrame table {
        background-color: #ffffff !important;
    }
    .stDataFrame thead th {
        background-color: #f8fafc !important;
        color: #2c3e50 !important;
        border: 1px solid #e1e8ed !important;
    }
    .stDataFrame tbody td {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        border: 1px solid #e1e8ed !important;
    }
    .stDataFrame tbody tr:hover {
        background-color: #f8fafc !important;
    }
    /* Override any dark backgrounds in dataframe */
    .element-container .stDataFrame {
        background: white !important;
    }
    div[data-testid="stDataFrame"] {
        background: white !important;
    }
    div[data-testid="stDataFrame"] > div {
        background: white !important;
    }
    div[data-testid="stDataFrame"] table {
        background: white !important;
    }
    div[data-testid="stDataFrame"] thead {
        background: #f8fafc !important;
    }
    div[data-testid="stDataFrame"] tbody {
        background: white !important;
    }
    div[data-testid="stDataFrame"] th {
        background-color: #f8fafc !important;
        color: #2c3e50 !important;
    }
    div[data-testid="stDataFrame"] td {
        background-color: white !important;
        color: #2c3e50 !important;
    }
    /* More aggressive table overrides */
    .stDataFrame .dataframe {
        background-color: white !important;
    }
    .stDataFrame .dataframe th {
        background-color: #f8fafc !important;
        color: #2c3e50 !important;
    }
    .stDataFrame .dataframe td {
        background-color: white !important;
        color: #2c3e50 !important;
    }
    .stDataFrame .dataframe tbody tr:nth-child(even) {
        background-color: #fafbfc !important;
    }
    .stDataFrame .dataframe tbody tr:nth-child(odd) {
        background-color: white !important;
    }
    /* Override st-emotion CSS */
    .css-81oif8, .css-1n76uvr, .css-1d391kg .css-12w0qpk, 
    .css-1d391kg .css-81oif8, .css-1d391kg .css-1n76uvr {
        background-color: white !important;
        color: #2c3e50 !important;
    }
    /* Sidebar form elements */
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        border: 1px solid #e1e8ed !important;
    }
    section[data-testid="stSidebar"] .stMultiSelect > div > div {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        border: 1px solid #e1e8ed !important;
    }
    section[data-testid="stSidebar"] .stRadio > div {
        background-color: transparent !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #2c3e50 !important;
    }
    /* Dropdown styling */
    section[data-testid="stSidebar"] .css-1wa3eu0-placeholder,
    section[data-testid="stSidebar"] .css-12w0qpk-placeholder {
        color: #64748b !important;
    }
    section[data-testid="stSidebar"] .css-1uccc91-singleValue {
        color: #2c3e50 !important;
    }
    /* Override any select/multiselect dark backgrounds */
    section[data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: white !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: white !important;
        color: #2c3e50 !important;
        border: 1px solid #e1e8ed !important;
    }
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
        background-color: white !important;
    }
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {
        background-color: white !important;
    }
    /* Override dark control backgrounds */
    section[data-testid="stSidebar"] .css-2b097c-container {
        background-color: white !important;
    }
    section[data-testid="stSidebar"] .css-1d3z3hw-control {
        background-color: white !important;
        border: 1px solid #e1e8ed !important;
    }
    section[data-testid="stSidebar"] .css-1pahdxg-control {
        background-color: white !important;
        border: 1px solid #e1e8ed !important;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .hero-section {
        text-align: center;
        padding: 3rem 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e1e8ed;
        margin-bottom: 2rem;
    }
    .hero-section h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
        line-height: 1.2;
    }
    .hero-section .subtitle {
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 1.5rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
    .what-is-section {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e1e8ed;
        margin-bottom: 2rem;
        padding: 2rem;
    }
    .what-is-section h2 {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .what-is-section p {
        font-size: 1.1rem;
        color: #4a5568;
        line-height: 1.7;
        margin: 0;
    }
    .section-container {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e1e8ed;
        margin-bottom: 2rem;
    }
    .section-header {
        background: #f8fafc;
        padding: 1.5rem 2rem 1rem;
        border-bottom: 1px solid #e1e8ed;
        border-radius: 10px 10px 0 0;
    }
    .section-header h2 {
        margin: 0 0 0.5rem 0;
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .section-header p {
        margin: 0;
        color: #64748b;
        font-size: 1rem;
    }
    .section-content {
        padding: 2rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e1e8ed;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .metric-card p {
        margin: 0;
        color: #64748b;
        font-size: 0.9rem;
    }
    .tool-description {
        background: #e3f2fd;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 6px 6px 0;
        margin-bottom: 1.5rem;
    }
    .tool-description p {
        margin: 0;
        color: #1565c0;
    }
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin-bottom: 2rem;
    }
    .info-card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e1e8ed;
        padding: 2rem;
    }
    .info-card h3 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-card p, .info-card li {
        color: #4a5568;
        line-height: 1.6;
    }
    .info-card ul {
        list-style: none;
        padding: 0;
    }
    .info-card li {
        margin-bottom: 1rem;
        padding-left: 0;
    }
    .info-card li strong {
        color: #2c3e50;
    }
    /* Tab styling for better visibility */
    .stTabs [data-baseweb="tab-list"] {
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.25rem;
        border: 1px solid #e1e8ed;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        color: #4a5568 !important;
        border-radius: 6px !important;
        margin: 0 0.25rem !important;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #e2e8f0 !important;
        color: #2d3748 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white !important;
        color: #2d3748 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    /* Additional form element styling */
    .stButton > button {
        background-color: #667eea !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover {
        background-color: #5a67d8 !important;
    }
    /* Force all sidebar elements to have white background and dark text */
    section[data-testid="stSidebar"] [role="option"] {
        background-color: white !important;
        color: #2c3e50 !important;
    }
    section[data-testid="stSidebar"] [class*="css-"] {
        background-color: white !important;
        color: #2c3e50 !important;
    }
    /* Force data editor to be white */
    .stDataEditor {
        background-color: white !important;
    }
    .stDataEditor > div {
        background-color: white !important;
    }
    .stDataEditor table {
        background-color: white !important;
    }
    .stDataEditor th, .stDataEditor td {
        background-color: white !important;
        color: #2c3e50 !important;
    }
    .stDataEditor th {
        background-color: #f8fafc !important;
    }
    /* Fix form inputs - make them white with dark text */
    .stTextInput > div > div > input {
        background-color: white !important;
        color: #2c3e50 !important;
        border: 1px solid #e1e8ed !important;
    }
    .stTextArea > div > div > textarea {
        background-color: white !important;
        color: #2c3e50 !important;
        border: 1px solid #e1e8ed !important;
    }
    .stFileUploader {
        background-color: white !important;
    }
    .stFileUploader > div {
        background-color: white !important;
        border: 1px solid #e1e8ed !important;
    }
    .stFileUploader section {
        background-color: white !important;
        color: #2c3e50 !important;
    }
    /* Target all input elements */
    input[type="text"], 
    input[type="password"], 
    input[type="email"], 
    input[type="number"],
    textarea {
        background-color: white !important;
        color: #2c3e50 !important;
        border: 1px solid #e1e8ed !important;
    }
    /* Fix any remaining form elements */
    .stForm {
        background-color: white !important;
    }
    .stForm input, .stForm textarea, .stForm select {
        background-color: white !important;
        color: #2c3e50 !important;
        border: 1px solid #e1e8ed !important;
    }
    /* File uploader specific fixes */
    [data-testid="stFileUploader"] {
        background-color: white !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: white !important;
        color: #2c3e50 !important;
        border: 2px dashed #e1e8ed !important;
    }
    [data-testid="stFileUploader"] section:hover {
        background-color: #f8fafc !important;
        border-color: #667eea !important;
    }
    /* Text input container fixes */
    [data-testid="stTextInput"] {
        background-color: white !important;
    }
    [data-testid="stTextInput"] input {
        background-color: white !important;
        color: #2c3e50 !important;
        border: 1px solid #e1e8ed !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 1px #667eea !important;
    }
    /* Comprehensive text color fixes for all light backgrounds */
    .stApp, .main, .block-container, 
    .stTabs, .stMarkdown, .stInfo, .stSuccess, .stWarning {
        color: #2c3e50 !important;
    }
    .stApp p, .stApp span, .stApp div, .stApp label {
        color: #2c3e50 !important;
    }
    /* Fix selectbox and dropdown text visibility */
    .stSelectbox [data-baseweb="select"] {
        background-color: white !important;
        color: #2c3e50 !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        background-color: white !important;
        color: #2c3e50 !important;
        border: 1px solid #e1e8ed !important;
    }
    .stSelectbox [data-baseweb="select"] .css-1uccc91-singleValue {
        color: #2c3e50 !important;
    }
    .stSelectbox [data-baseweb="select"] .css-1wa3eu0-placeholder {
        color: #64748b !important;
    }
    /* Fix multiselect */
    .stMultiSelect [data-baseweb="select"] {
        background-color: white !important;
        color: #2c3e50 !important;
    }
    .stMultiSelect [data-baseweb="select"] > div {
        background-color: white !important;
        color: #2c3e50 !important;
        border: 1px solid #e1e8ed !important;
    }
    /* Fix all dropdown menus */
    [data-baseweb="popover"] {
        background-color: white !important;
    }
    [data-baseweb="popover"] [data-baseweb="menu"] {
        background-color: white !important;
    }
    [data-baseweb="popover"] [role="option"] {
        background-color: white !important;
        color: #2c3e50 !important;
    }
    [data-baseweb="popover"] [role="option"]:hover {
        background-color: #f8fafc !important;
        color: #2c3e50 !important;
    }
    /* Fix info/alert boxes */
    .stAlert {
        color: #2c3e50 !important;
    }
    .stAlert p, .stAlert span, .stAlert div {
        color: #2c3e50 !important;
    }
    /* Fix any remaining light text */
    .stApp * {
        color: #2c3e50 !important;
    }
    /* Override specific light text elements */
    .css-1cpxqw2, .css-16idsys, .css-10trblm {
        color: #2c3e50 !important;
    }
    /* Fix button text to be white on colored backgrounds */
    .stButton > button {
        background-color: #667eea !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }
    .stDownloadButton > button {
        background-color: #667eea !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }
    /* Exception for buttons - keep white text on colored backgrounds */
    .stButton > button, 
    .stDownloadButton > button,
    .main-header,
    .main-header * {
        color: white !important;
    }
    /* Fix expander headers */
    .streamlit-expanderHeader {
        color: #2c3e50 !important;
    }
    /* Fix checkbox and radio text */
    .stCheckbox label, .stRadio label {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1>Tracking and Analyzing State-Level AI Governance Legislation</h1>
    <p class="subtitle">A resource from the Vanderbilt AI Law Lab (VAILL) to help policymakers, researchers, and the public stay informed about the evolving landscape of AI regulation in the United States. <span style="font-size: 0.9rem; opacity: 0.8;
</div>
""", unsafe_allow_html=True)

# What is the Tracker Section
st.markdown("""
<div class="what-is-section">
    <h2>What is the AI Governance Legislation Tracker?</h2>
    <p>This tracker is a centralized, user-friendly platform for monitoring artificial intelligence (AI) governance legislation across the United States. As AI technology rapidly advances, it's becoming increasingly important to understand how different states are approaching its regulation. This tool aims to simplify the process of finding and comparing various state-level AI governance bills, their current statuses, and their key provisions. <span style="font-size: 0.9rem; opacity: 0.8;
</div>
""", unsafe_allow_html=True)

# Tool descriptions
TOOL_DESCRIPTIONS = {
    "bills_table": {
        "name": "Bills Explorer",
        "description": "Navigate and filter AI governance legislation with powerful search tools. View details on bill numbers, titles, status, scope, and last action dates in an easy-to-read table format. Export your filtered results as CSV for further analysis."
    },
    "bills_map": {
        "name": "Geospatial Insights",
        "description": "Visualize the geographic distribution of AI governance bills across states with an interactive map. Circle size represents bill volume, helping you identify legislative hotspots and regional trends at a glance."
    },
    "ai_toolkit": {
        "name": "AI Analysis Toolkit",
        "description": "Comprehensive AI-powered analysis suite for legislative research. Choose from multiple analysis types: Q&A for specific bill insights, comparative analysis across multiple bills, executive summary generation, and EU AI Act comparisons."
    }
}

# AI Toolkit Analysis Types
AI_ANALYSIS_TYPES = {
    "qa": {
        "name": "Legislative Q&A",
        "description": "Get instant answers to your questions about specific AI governance bills. Simply select a bill and ask any question to receive AI-powered insights based on the bill's text and analysis."
    },
    "comparison": {
        "name": "Bill Comparison",
        "description": "Analyze how AI governance approaches differ across multiple bills. Select a focus bill and comparison bills to identify similarities, differences, and unique approaches to regulation on specific topics."
    },
    "summary": {
        "name": "Legislative Report",
        "description": "Review comprehensive reports of selected bills with AI assistance. Reports cover key aspects including scope, enforcement mechanisms, and potential impacts, downloadable as markdown."
    },
    "eu_comparison": {
        "name": "EU AI Act Comparison",
        "description": "Compare US AI governance bills against the EU AI Act. Analyze similarities, differences, and regulatory approaches between US legislation and the comprehensive EU framework."
    }
}

@st.cache_resource
def get_qa_llm():
    """Initialize and cache the ChatOpenAI instance for Q&A."""
    if ChatOpenAI is None:
        raise RuntimeError("langchain_openai package required. Install with: pip install langchain langchain_openai")
    
    return ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4o",
        temperature=0
    )


@st.cache_resource
def get_embeddings():
    """Initialize and cache the OpenAI embeddings."""
    return OpenAIEmbeddings(
        api_key=openai_api_key,
        model="text-embedding-3-small"  # More cost-effective than text-embedding-3-large
    )

@st.cache_resource
def get_text_splitter():
    """Initialize and cache the text splitter for chunking documents."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Reasonable chunk size for embeddings
        chunk_overlap=100,  # Some overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

def create_bill_documents(bill_data_list):
    """Convert bill data to Document objects with metadata."""
    documents = []
    
    for bill_data in bill_data_list:
        # Combine relevant text fields from the bill
        text_content = ""
        
        # Add title and basic info
        if bill_data.get('title'):
            text_content += f"Title: {bill_data['title']}\n\n"
        
        if bill_data.get('summary'):
            text_content += f"Summary: {bill_data['summary']}\n\n"
        
        # Add full text if available
        if bill_data.get('full_text'):
            text_content += f"Full Text:\n{bill_data['full_text']}\n\n"
        elif bill_data.get('bill_text'):
            text_content += f"Bill Text:\n{bill_data['bill_text']}\n\n"
        
        # Add other relevant fields
        if bill_data.get('description'):
            text_content += f"Description: {bill_data['description']}\n\n"
        
        # Create metadata
        metadata = {
            'bill_id': bill_data.get('bill_id', 'Unknown'),
            'state': bill_data.get('state', 'Unknown'),
            'bill_number': bill_data.get('bill_number', 'Unknown'),
            'title': bill_data.get('title', 'Unknown'),
            'url': bill_data.get('url', 'Unknown'),
            'status': bill_data.get('status', 'Unknown'),
            'sponsors': str(bill_data.get('sponsors', 'Unknown')),
            'last_action_date': str(bill_data.get('last_action_date', 'Unknown'))
        }
        
        # Create document
        doc = Document(
            page_content=text_content.strip(),
            metadata=metadata
        )
        documents.append(doc)
    
    return documents

def create_vectorstore_from_bills(bill_data_list):
    """Create a FAISS vectorstore from bill data."""
    try:
        # Get embeddings and text splitter
        embeddings = get_embeddings()
        text_splitter = get_text_splitter()
        
        # Create documents
        documents = create_bill_documents(bill_data_list)
        
        if not documents:
            raise ValueError("No documents created from bill data")
        
        # Split documents into chunks
        splits = text_splitter.split_documents(documents)
        
        if not splits:
            raise ValueError("No text chunks created from documents")
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        raise e

def compare_bills_with_rag(focus_bill_data, comparison_bills_data, question):
    """Compare bills using modern LCEL RAG approach with FAISS vectorstore."""
    try:
        # Combine all bills for the vectorstore
        all_bills = [focus_bill_data] + comparison_bills_data
        
        # Create vectorstore
        vectorstore = create_vectorstore_from_bills(all_bills)
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}  # Retrieve top 6 most relevant chunks
        )
        
        # Format bill information for the prompt
        focus_bill_info = f"{focus_bill_data.get('state', 'Unknown')} {focus_bill_data.get('bill_number', 'Unknown')}: {focus_bill_data.get('title', 'Unknown')}"
        
        comparison_bills_info = []
        for bill in comparison_bills_data:
            bill_info = f"{bill.get('state', 'Unknown')} {bill.get('bill_number', 'Unknown')}: {bill.get('title', 'Unknown')}"
            comparison_bills_info.append(bill_info)
        comparison_bills_info_str = "; ".join(comparison_bills_info)
        
        # Create the comparison prompt template using ChatPromptTemplate
        comparison_prompt = ChatPromptTemplate.from_template("""You are a legislative analyst expert at comparing AI governance bills. 
You have been provided with relevant excerpts from multiple bills to answer a comparison question.

IMPORTANT BILL INFORMATION:
- Focus Bill: {focus_bill_info}
- Comparison Bills: {comparison_bills_info}

Your task is to compare these bills based on the user's question, using the relevant excerpts provided below.

Guidelines for your analysis:
- Clearly identify which bill each piece of information comes from
- Highlight similarities and differences between the bills
- Be specific about provisions, definitions, and requirements
- If information is missing from the excerpts, state that clearly
- Structure your response with clear sections for each bill
- Conclude with a summary of key similarities and differences
- Always include the legiscan link to the bills in your response
- Format your answer as markdown

Relevant excerpts from the bills:
{context}

User Question: {input}

Please provide a comprehensive comparison analysis based on the excerpts above.
""")
        
        # Get LLM
        llm = get_qa_llm()
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, comparison_prompt)
        
        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Run the query with bill information
        result = retrieval_chain.invoke({
            "input": question,
            "focus_bill_info": focus_bill_info,
            "comparison_bills_info": comparison_bills_info_str
        })
        
        # Extract answer and sources
        answer = result.get("answer", "No answer generated")
        source_docs = result.get("context", [])
        
        # Add source information to the answer
        if source_docs:
            answer += "\n\n---\n\n**Sources used in this analysis:**\n\n"
            seen_bills = set()
            for doc in source_docs:
                bill_id = f"{doc.metadata.get('state', 'Unknown')} {doc.metadata.get('bill_number', 'Unknown')}"
                if bill_id not in seen_bills:
                    seen_bills.add(bill_id)
                    answer += f"- {bill_id}: {doc.metadata.get('title', 'Unknown')}\n"
        
        return answer
        
    except Exception as e:
        logger.error(f"Error in RAG comparison: {e}")
        return f"Error during comparison analysis: {str(e)}"


def answer_bill_question(bill_data: dict, question: str) -> str:
    """Answer a question about a specific bill using LangChain."""
    try:
        llm = get_qa_llm()
        
        # Create the prompt template
        qa_prompt = ChatPromptTemplate.from_template(
            """You are a legislative analyst expert at interpreting AI governance bills. 
            A user has asked a question about a specific bill. Use the bill information 
            provided as JSON to answer their question accurately and comprehensively.

            Guidelines:
            - Answer based only on the information provided in the bill JSON
            - Be specific and cite relevant sections when possible
            - If the information isn't available in the bill, clearly state that
            - Keep your answer focused and relevant to the question
            - Use clear, accessible language
            - Always include the legiscan link to the bill in your answer
            - Format your answer as markdown

            Bill JSON:
            ```json
            {bill_json}
            ```

            User Question: {question}

            Please provide a detailed answer based on the bill information above.
            """
        )
        
        # Convert timestamps and other non-serializable objects to strings
        serializable_bill_data = {}
        for key, value in bill_data.items():
            try:
                # Handle None/NaN values
                if value is None:
                    serializable_bill_data[key] = None
                elif isinstance(value, (int, float)) and pd.isna(value):
                    serializable_bill_data[key] = None
                elif hasattr(value, 'strftime'):  # Handle datetime/timestamp objects
                    serializable_bill_data[key] = value.strftime('%Y-%m-%d')
                elif isinstance(value, (list, dict, str, int, float, bool)):
                    # These types are JSON serializable
                    serializable_bill_data[key] = value
                else:
                    # Convert anything else to string
                    serializable_bill_data[key] = str(value)
            except Exception:
                # Fallback: convert to string or None
                serializable_bill_data[key] = str(value) if value is not None else None
        
        # Convert bill data to JSON string
        bill_json = json.dumps(serializable_bill_data, ensure_ascii=False, indent=2)
        
        # Create chain and invoke
        chain = qa_prompt | llm
        result = chain.invoke({
            "bill_json": bill_json,
            "question": question
        })
        
        # Extract content from result
        answer = getattr(result, "content", str(result))
        return answer
        
    except Exception as e:
        logger.error(f"Error in Q&A: {e}")
        return f"Error processing question: {str(e)}"

@st.cache_resource
def load_eu_ai_act_vectorstore():
    """Load the EU AI Act vectorstore from disk."""
    vectorstore_path = "data/eu_ai_act_vectorstore"
    try:
        if not Path(vectorstore_path).exists():
            logger.warning(f"EU AI Act vectorstore not found at {vectorstore_path}")
            return None, "Vectorstore not found. Please run the EU AI Act processing script first."
        
        # Initialize embeddings
        embeddings = get_embeddings()
        
        # Load vectorstore
        vectorstore = FAISS.load_local(
            vectorstore_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        logger.info(f"✅ EU AI Act vectorstore loaded successfully")
        return vectorstore, None
        
    except Exception as e:
        error_msg = f"Error loading EU AI Act vectorstore: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

@st.cache_data
def get_eu_vectorstore_info():
    """Get information about the EU AI Act vectorstore."""
    try:
        metadata_path = Path("data/eu_ai_act_vectorstore") / "metadata.pickle"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            return metadata
        else:
            return {"error": "Metadata not found"}
    except Exception as e:
        return {"error": str(e)}

# Add this function before the existing comparison functions

def compare_bill_with_eu_ai_act(bill_data, question):
    """Compare a US bill with the EU AI Act using RAG approach."""
    try:
        # Load EU AI Act vectorstore
        eu_vectorstore, error = load_eu_ai_act_vectorstore()
        if eu_vectorstore is None:
            return f"Error loading EU AI Act data: {error}"
        
        # Create US bill vectorstore
        us_vectorstore = create_vectorstore_from_bills([bill_data])
        
        # Create retrievers
        eu_retriever = eu_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Get top 4 relevant EU sections
        )
        
        us_retriever = us_vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}  # Get top 3 relevant US bill sections
        )
        
        # Get relevant documents from both sources
        eu_docs = eu_retriever.get_relevant_documents(question)
        us_docs = us_retriever.get_relevant_documents(question)
        
        # Format bill information
        bill_info = f"{bill_data.get('state', 'Unknown')} {bill_data.get('bill_number', 'Unknown')}: {bill_data.get('title', 'Unknown')}"
        
        # Create the comparison prompt
        comparison_prompt = ChatPromptTemplate.from_template("""You are a legal analyst expert at comparing AI governance frameworks between the US and EU. 
You have been provided with relevant excerpts from a US bill and the EU AI Act to answer a comparison question.

IMPORTANT CONTEXT:
- US Bill: {bill_info}
- EU Framework: Regulation (EU) 2024/1689 on Artificial Intelligence (AI Act)

Your task is to compare these regulatory frameworks based on the user's question, using the relevant excerpts provided below.

Guidelines for your analysis:
- Clearly distinguish between US and EU approaches
- Highlight similarities and differences in regulatory philosophy
- Be specific about provisions, definitions, and requirements from each framework
- Note any gaps or areas not covered by either framework
- Consider the different legal systems and enforcement mechanisms
- Structure your response with clear sections for each framework
- Conclude with a summary of key similarities, differences, and implications
- Always include the legiscan link to the US bill in your response
- Format your answer as markdown

Relevant excerpts from the US Bill:
{us_context}

Relevant excerpts from the EU AI Act:
{eu_context}

User Question: {input}

Please provide a comprehensive comparison analysis based on the excerpts above.
""")
        
        # Prepare context
        us_context = "\n\n".join([doc.page_content for doc in us_docs])
        eu_context = "\n\n".join([doc.page_content for doc in eu_docs])
        
        # Get LLM and create chain
        llm = get_qa_llm()
        chain = comparison_prompt | llm
        
        # Run the comparison
        result = chain.invoke({
            "input": question,
            "bill_info": bill_info,
            "us_context": us_context,
            "eu_context": eu_context
        })
        
        # Extract content from result
        answer = getattr(result, "content", str(result))
        
        # Add source information
        answer += "\n\n---\n\n**Sources used in this analysis:**\n\n"
        answer += f"**US Bill:** {bill_info}\n"
        answer += f"**EU Framework:** EU AI Act (Regulation 2024/1689)\n"
        
        return answer
        
    except Exception as e:
        logger.error(f"Error in EU comparison: {e}")
        return f"Error during EU comparison analysis: {str(e)}"
   

@st.cache_data
def load_bill_reports() -> dict:
    """Load pre-generated bill reports from JSON file."""
    reports_file = Path("data/bill_reports.json")
    try:
        if reports_file.exists():
            with open(reports_file, 'r', encoding='utf-8') as f:
                reports_data = json.load(f)
            # Convert to dict with bill_id as key
            reports = {report['bill_id']: report['report_markdown'] for report in reports_data}
            logger.info(f"Loaded {len(reports)} pre-generated reports")
            return reports
        else:
            logger.warning(f"Reports file not found: {reports_file}")
            return {}
    except Exception as e:
        logger.error(f"Error loading reports: {e}")
        return {}

def get_bill_report(bill_data, reports_cache):
    """Get report for a bill from cache or return message."""
    bill_id = str(bill_data.get('bill_id', ''))
    
    if bill_id in reports_cache:
        return reports_cache[bill_id]
    else:
        return "No pre-generated report available for this bill."

reports_cache = load_bill_reports()

@st.cache_data
def load_bill_summaries() -> dict:
    """Load pre-generated bill summaries from JSON file."""
    summaries_file = Path("data/bill_summaries.json")
    try:
        if summaries_file.exists():
            with open(summaries_file, 'r', encoding='utf-8') as f:
                summaries = json.load(f)
            logger.info(f"Loaded {len(summaries)} pre-generated summaries")
            return summaries
        else:
            logger.warning(f"Summaries file not found: {summaries_file}")
            return {}
    except Exception as e:
        logger.error(f"Error loading summaries: {e}")
        return {}

@st.cache_data
def load_bill_suggested_questions() -> dict:
    """Load pre-generated suggested questions from JSON file."""
    questions_file = Path("data/bill_suggested_questions.json")
    try:
        if questions_file.exists():
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            logger.info(f"Loaded {len(questions)} pre-generated question sets")
            return questions
        else:
            logger.warning(f"Questions file not found: {questions_file}")
            return {}
    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        return {}

def get_bill_suggested_questions(bill_data, questions_cache):
    """Get suggested questions for a bill from cache or return fallback."""
    bill_key = f"{bill_data.get('state', 'Unknown')}_{bill_data.get('bill_number', 'Unknown')}"
    
    if bill_key in questions_cache:
        questions = questions_cache[bill_key].get('suggested_questions', [])
        if len(questions) == 5:
            return questions
    
    # Fallback to static example questions
    return [
        "What are the key definitions in this bill?",
        "What are the enforcement mechanisms?",
        "Who does this bill apply to?",
        "What are the compliance requirements?",
        "What penalties are specified?"
    ]

def get_bill_summary(bill_data, summaries_cache):
    """Get summary for a bill from cache or return error message."""
    bill_key = f"{bill_data.get('state', 'Unknown')}_{bill_data.get('bill_number', 'Unknown')}"
    
    if bill_key in summaries_cache:
        summary = summaries_cache[bill_key].get('summary', '')
        if summary.startswith('ERROR:'):
            return f"Summary generation failed: {summary}"
        return summary
    else:
        return "No pre-generated summary available. Run the summary generation script first."

@st.cache_data
def load_and_process_data() -> pd.DataFrame:
    start_time = time.time()
    json_path = Path("data/known_bills_visualize.json")

    if not json_path.exists():
        logger.warning(f"Data file not found: {json_path}")
        return None

    try:
        with json_path.open("r", encoding='utf-8') as f:
            bills_data = json.load(f)
        logger.info(f"Loaded {len(bills_data)} bills from {json_path}")

        df = pd.DataFrame(bills_data)
        # Convert dates
        if "last_action_date" in df.columns:
            df["last_action_date"] = pd.to_datetime(
                df["last_action_date"], errors="coerce"
            )
        if "lastUpdatedAt" in df.columns:
            df["lastUpdatedAt"] = pd.to_datetime(df["lastUpdatedAt"], errors="coerce")

        logger.info(f"DataFrame created in {time.time() - start_time:.2f} seconds")
        return df
    except Exception as e:
        logger.error(f"Error loading {json_path}: {e}")
        return None
    
# Load OpenAI API key from Streamlit secrets or environment variable
def load_openai_api_key():
    # First, try Streamlit secrets (for deployed environments)
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Fallback to environment variable (for local dev)
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            logger.info("Loaded OpenAI API key from environment variable.")
            return api_key
        # Fallback to user input (for local dev without env var)
        else:
            st.warning("OpenAI API key not found in secrets or environment variables.")
            api_key = st.text_input("Enter your OpenAI API key:", type="password")
            if api_key:
                logger.info("OpenAI API key provided via user input.")
                return api_key
            else:
                st.error("Please provide an OpenAI API key to continue.")
                st.stop()

# Load the key
openai_api_key = load_openai_api_key()  

def display_bill_details(bill_data, summaries_cache):
    """Display bill details and summary in a formatted way."""
    st.markdown("#### Bill Details")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**State:** {bill_data.get('state', 'N/A')}")
        st.write(f"**Bill Number:** {bill_data.get('bill_number', 'N/A')}")
        st.write(f"**Status:** {bill_data.get('status', 'N/A')}")
        
        # Format last action date
        if 'last_action_date' in bill_data and pd.notna(bill_data['last_action_date']):
            if isinstance(bill_data['last_action_date'], str):
                st.write(f"**Last Action Date:** {bill_data['last_action_date']}")
            else:
                st.write(f"**Last Action Date:** {bill_data['last_action_date'].strftime('%Y-%m-%d')}")
        else:
            st.write(f"**Last Action Date:** N/A")
    
    with col2:
        # Extract IAPP categories for display
        if 'iapp_categories' in bill_data and isinstance(bill_data['iapp_categories'], dict):
            all_subcategories = []
            for category, subcategories in bill_data['iapp_categories'].items():
                if isinstance(subcategories, list):
                    all_subcategories.extend(subcategories)
            
            if all_subcategories:
                iapp_display = ", ".join(all_subcategories[:3])  # Show first 3
                if len(all_subcategories) > 3:
                    iapp_display += f" + {len(all_subcategories) - 3} more"
            else:
                iapp_display = "None"
        else:
            iapp_display = "N/A"
        
        st.write(f"**Categories:** {iapp_display}")
        
        # Show sponsors if available
        sponsors = bill_data.get('sponsors', 'N/A')
        if isinstance(sponsors, list):
            sponsors_display = ", ".join(sponsors[:2])  # Show first 2 sponsors
            if len(sponsors) > 2:
                sponsors_display += f" + {len(sponsors) - 2} more"
        else:
            sponsors_display = str(sponsors) if sponsors else "N/A"
        
        st.write(f"**Sponsors:** {sponsors_display}")
    
    # Show full title
    st.write(f"**Title:** {bill_data.get('title', 'N/A')}")
    
    # Display pre-generated summary
    st.markdown("#### Bill Summary")
    summary = get_bill_summary(bill_data, summaries_cache)
    
    if summary.startswith('Summary generation failed') or summary.startswith('No pre-generated summary'):
        st.warning(summary)
    else:
        st.info(summary)

df = load_and_process_data()
summaries_cache = load_bill_summaries()
questions_cache = load_bill_suggested_questions()

if df is None:
    st.write("No data available. Ensure 'known_bills_visualize.json' is populated.")
    st.stop()

# Sidebar for filters with improved styling
with st.sidebar:
    # Add logo to the sidebar
    logo_path = "vaill_logo.png"
    try:
        st.image(logo_path, use_container_width=True) 
    except FileNotFoundError:
        st.warning("Logo image 'vaill_logo.png' not found.")
    
    st.markdown("### Filter Controls")

    # Date Filter Section
    date_df = (
        df.dropna(subset=["last_action_date"])
        if "last_action_date" in df.columns
        else pd.DataFrame()
    )

    if not date_df.empty:
        current_date = datetime.now().date()
        df_dates = df[df["last_action_date"].notna()]["last_action_date"]
        min_year = df_dates.min().year
        max_year = min(df_dates.max().year, current_date.year)
        
        st.markdown("#### Date Range")
        
        filter_type = st.radio(
            "Filter by:",
            options=["No Date Filter", "Year Only", "Year & Month"],
            index=0,
            help="Choose how to filter bills by their last action date"
        )
        
        # Initialize filtered_df
        filtered_df = df.copy()
        
        if filter_type == "Year Only":
            available_years = sorted(df_dates.dt.year.unique())
            available_years = [year for year in available_years if year <= current_date.year]
            
            if available_years:
                selected_years = st.multiselect(
                    "Select Years:",
                    options=available_years,
                    default=[max(available_years)],
                    help="Select one or more years to filter bills"
                )
                
                if selected_years:
                    mask = df["last_action_date"].dt.year.isin(selected_years)
                    filtered_df = df[mask].copy()
                    
                    if len(selected_years) == 1:
                        year_range = f"{selected_years[0]}"
                    else:
                        year_range = f"{min(selected_years)}-{max(selected_years)}"
                    st.success(f"Filtering: {year_range}")
            else:
                st.warning("No years available for filtering")
                
        elif filter_type == "Year & Month":
            available_years = list(range(min_year, max_year + 1))
            
            col1, col2 = st.columns(2)
            
            with col1:
                start_year = st.selectbox(
                    "From Year:",
                    options=available_years,
                    index=max(0, len(available_years) - 2) if len(available_years) > 1 else 0,
                    key="start_year_select"
                )
                
            with col2:
                end_year_options = list(range(start_year, max_year + 1))
                end_year = st.selectbox(
                    "To Year:",
                    options=end_year_options,
                    index=len(end_year_options) - 1,
                    key="end_year_select"
                )
            
            months = {
                1: "January", 2: "February", 3: "March", 4: "April",
                5: "May", 6: "June", 7: "July", 8: "August", 
                9: "September", 10: "October", 11: "November", 12: "December"
            }
            
            col3, col4 = st.columns(2)
            
            with col3:
                start_month = st.selectbox(
                    "From Month:",
                    options=list(months.keys()),
                    format_func=lambda x: months[x],
                    index=0,
                    key="start_month_select"
                )
                
            with col4:
                max_end_month = 12
                if end_year == current_date.year:
                    max_end_month = current_date.month
                    
                end_month_options = list(range(1, max_end_month + 1))
                if end_month_options:
                    end_month = st.selectbox(
                        "To Month:",
                        options=end_month_options,
                        format_func=lambda x: months[x],
                        index=len(end_month_options) - 1,
                        key="end_month_select"
                    )
                else:
                    end_month = 1
                    st.warning("Invalid month range")
            
            try:
                start_date = date(start_year, start_month, 1)
                
                if end_month == 12:
                    last_day = date(end_year + 1, 1, 1) - timedelta(days=1)
                else:
                    last_day = date(end_year, end_month + 1, 1) - timedelta(days=1)
                
                end_date = min(last_day, current_date)
                
                if start_date <= end_date:
                    mask = (df["last_action_date"].dt.date >= start_date) & (
                        df["last_action_date"].dt.date <= end_date
                    )
                    filtered_df = df[mask].copy()
                    
                    st.success(f"Filtering: {start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}")
                else:
                    st.error("Start date must be before end date")
                    
            except ValueError as e:
                st.error(f"Invalid date range: {e}")
            
        if filter_type != "No Date Filter" and not filtered_df.empty:
            date_stats = filtered_df["last_action_date"].dropna()
            if not date_stats.empty:
                st.info(f"{len(date_stats)} bills with dates in range")
            
    else:
        filtered_df = df.copy()
        st.warning("No date information available for filtering")
        
    st.markdown("#### Bill Type")
    bill_type_filter = st.radio(
        "Show bills:",
        options=["All Bills", "State Bills Only", "Federal Bills Only"],
        index=0
    )

    # Apply the filter
    if bill_type_filter == "State Bills Only":
        filtered_df = filtered_df[filtered_df["state"] != "US"]
    elif bill_type_filter == "Federal Bills Only":
        filtered_df = filtered_df[filtered_df["state"] == "US"]

    # IAPP Categories Filter
    if "iapp_categories" in filtered_df.columns:
        st.markdown("#### Categories")
        
        all_iapp_categories = set()
        filtered_df["iapp_categories"].apply(
            lambda x: all_iapp_categories.update(x.keys()) if isinstance(x, dict) else None
        )
        
        if all_iapp_categories:
            for category in sorted(all_iapp_categories):
                if category in IAPP_CATEGORIES:
                    all_subcategories = set()
                    filtered_df["iapp_categories"].apply(
                        lambda x: all_subcategories.update(x.get(category, [])) 
                        if isinstance(x, dict) and category in x else None
                    )
                    
                    if all_subcategories:
                        subcategory_options = sorted(all_subcategories)
                        selected_subcategories = st.multiselect(
                            f"{category}",
                            options=subcategory_options,
                            default=[],
                            key=f"iapp_{category.lower().replace(' ', '_')}"
                        )
                        
                        if selected_subcategories:
                            filtered_df = filtered_df[
                                filtered_df["iapp_categories"].apply(
                                    lambda x: (
                                        any(subcat in x.get(category, []) for subcat in selected_subcategories)
                                        if isinstance(x, dict) and category in x
                                        else False
                                    )
                                )
                            ]

# Main content with tab-based layout
(tab1, tab2, tab3) = st.tabs([
    TOOL_DESCRIPTIONS["bills_table"]["name"], 
    TOOL_DESCRIPTIONS["bills_map"]["name"], 
    TOOL_DESCRIPTIONS["ai_toolkit"]["name"]
])

# TAB 1: BILLS EXPLORER
with tab1:
    st.markdown(f'<div class="tool-description"><p>{TOOL_DESCRIPTIONS["bills_table"]["description"]}</p></div>', unsafe_allow_html=True)
    
    if filtered_df.empty:
        st.warning("No bills match the selected filters.")
    else:
        # Separate federal and state bills
        federal_bills = filtered_df[filtered_df["state"] == "US"]
        state_bills = filtered_df[filtered_df["state"] != "US"]
        
        # Helper function for last updated date
        def get_last_updated_date():
            if "lastUpdatedAt" in filtered_df.columns and not filtered_df.empty:
                valid_dates = filtered_df[filtered_df["lastUpdatedAt"].notna()]
                if not valid_dates.empty:
                    most_recent = valid_dates["lastUpdatedAt"].max()
                    return most_recent.strftime("%Y-%m-%d") if pd.notna(most_recent) else "N/A"
            return "N/A"
        
        # Metrics section
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><h2>Database Overview</h2><p>Current statistics for filtered bill dataset</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-content">', unsafe_allow_html=True)
        
        if bill_type_filter == "Federal Bills Only":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card"><h3>{}</h3><p>Federal Bills</p></div>'.format(len(federal_bills)), unsafe_allow_html=True)
            with col2:
                current_year = datetime.now().year
                this_year_bills = len(filtered_df[filtered_df["last_action_date"].dt.year == current_year]) if "last_action_date" in filtered_df.columns else 0
                st.markdown('<div class="metric-card"><h3>{}</h3><p>Bills This Year</p></div>'.format(this_year_bills), unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card"><h3>{}</h3><p>Last Updated</p></div>'.format(get_last_updated_date()), unsafe_allow_html=True)
                
        elif bill_type_filter == "State Bills Only":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card"><h3>{}</h3><p>State Bills</p></div>'.format(len(state_bills)), unsafe_allow_html=True)
            with col2:
                if not state_bills.empty:
                    state_counts = state_bills["state"].value_counts()
                    most_active = state_counts.index[0] if not state_counts.empty else "N/A"
                    count = state_counts.iloc[0] if not state_counts.empty else 0
                    display_text = f"{most_active} ({count})" if most_active != "N/A" else "N/A"
                    st.markdown('<div class="metric-card"><h3>{}</h3><p>Most Active State</p></div>'.format(display_text), unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-card"><h3>N/A</h3><p>Most Active State</p></div>', unsafe_allow_html=True)
            with col3:
                    st.markdown('<div class="metric-card"><h3>{}</h3><p>Last Updated</p></div>'.format(get_last_updated_date()), unsafe_allow_html=True)
                
        else:  # All Bills
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card"><h3>{}</h3><p>Federal Bills</p></div>'.format(len(federal_bills)), unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card"><h3>{}</h3><p>State Bills</p></div>'.format(len(state_bills)), unsafe_allow_html=True)
            with col3:
                if not state_bills.empty:
                    state_counts = state_bills["state"].value_counts()
                    most_active = state_counts.index[0] if not state_counts.empty else "N/A"
                    count = state_counts.iloc[0] if not state_counts.empty else 0
                    display_text = f"{most_active} ({count})" if most_active != "N/A" else "N/A"
                    st.markdown('<div class="metric-card"><h3>{}</h3><p>Most Active State</p></div>'.format(display_text), unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-card"><h3>N/A</h3><p>Most Active State</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-card"><h3>{}</h3><p>Last Updated</p></div>'.format(get_last_updated_date()), unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Bills table section
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><h2>Legislation Database</h2><p>Comprehensive listing of AI governance legislation</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-content">', unsafe_allow_html=True)
        
        if bill_type_filter == "State Bills Only":
            st.write(f"Showing {len(filtered_df)} state bills")
        elif bill_type_filter == "Federal Bills Only":
            st.write(f"Showing {len(filtered_df)} federal bills")
        else:
            federal_count = len(filtered_df[filtered_df["state"] == "US"])
            state_count = len(filtered_df[filtered_df["state"] != "US"])
            st.write(f"Showing {len(filtered_df)} bills ({federal_count} federal, {state_count} state)")
        display_df = filtered_df.copy()

        # Process IAPP Categories for display
        def extract_iapp_subcategories(iapp_categories):
            """Extract all subcategories from IAPP categories dict and format for display."""
            if not isinstance(iapp_categories, dict) or not iapp_categories:
                return "None"
            
            all_subcategories = []
            for category, subcategories in iapp_categories.items():
                if isinstance(subcategories, list):
                    all_subcategories.extend(subcategories)
            
            if not all_subcategories:
                return "None"
            
            subcategories_str = ", ".join(all_subcategories)
            return subcategories_str

        # Add IAPP categories column if the field exists
        if "iapp_categories" in display_df.columns:
            display_df["iapp_categories_display"] = display_df["iapp_categories"].apply(extract_iapp_subcategories)
        else:
            display_df["iapp_categories_display"] = "None"

        # Define column mappings for display
        column_mapping = {
            "state": "State",
            "bill_number": "Bill Number",
            "title": "Title",
            "status": "Status",
            "iapp_categories_display": "Categories",
            "last_action_date": "Last Action Date",
            "sponsors": "Sponsors"
        }

        # Format dates for display with validation
        if "last_action_date" in display_df.columns:
            current_date = datetime.now()
            
            def format_date(row):
                if pd.isna(row["last_action_date"]):
                    return "N/A"
                elif row["last_action_date"] > current_date:
                    return f"{row['last_action_date'].strftime('%Y-%m-%d')} (FUTURE DATE - INVALID)"
                else:
                    return row['last_action_date'].strftime('%Y-%m-%d')
            
            display_df["formatted_last_action_date"] = display_df.apply(format_date, axis=1)
            display_df["last_action_date"] = display_df["formatted_last_action_date"]
            display_df = display_df.drop(columns=["formatted_last_action_date"])

        # Create a column with proper links if bill_url exists
        if "bill_url" in display_df.columns and "bill_number" in display_df.columns:
            display_df["View"] = display_df.apply(
                lambda row: (
                    f"{row['bill_url']}"
                    if pd.notna(row["bill_url"]) and row["bill_url"]
                    else ""
                ),
                axis=1,
            )

        # Select and rename columns for display
        display_columns = [
            col for col in column_mapping.keys() if col in display_df.columns
        ]
        if "View" in display_df.columns:
            display_columns.append("View")

        display_df = display_df[display_columns].copy()
        display_df = display_df.rename(columns=column_mapping)

        # Display the dataframe with clickable links and IAPP categories truncation
        column_config = {
            "View": st.column_config.LinkColumn("Link"),
            "Categories": st.column_config.TextColumn(
                "Categories",
                help="AI governance subcategories from IAPP framework",
                max_chars=50,
                width="medium"
            )
        }
        
        st.data_editor(
            display_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            disabled=True,
            height=500,
        )

        # Add CSV download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Table as CSV",
            data=csv,
            file_name="filtered_bills.csv",
            mime="text/csv",
        )
        
        st.markdown('</div></div>', unsafe_allow_html=True)

# How to Use This Tracker and About the Data sections
st.markdown("""
<div class="info-grid">
    <div class="info-card">
        <h3>How to Use This Tracker</h3>
        <ul>
            <li><strong>For Policymakers:</strong> Quickly compare legislative approaches from other states to inform drafting and decision-making.</li>
            <li><strong>For Researchers:</strong> Access a centralized database of AI-related bills to analyze trends and export data for academic study.</li>
            <li><strong>For the Public:</strong> Stay informed about how your state is regulating AI technology and understand proposed laws.</li>
        </ul>
    </div>
    <div class="info-card">
        <h3>About the Data</h3>
        <p>The data in this tracker is compiled from state legislative records, primarily utilizing the Legiscan API. The tracker focuses on state-level legislation, with federal bills included for context but separated in counts and views. Bill statuses are simplified into "Signed Into Law", "Active", and "Inactive" for clarity.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# TAB 2: GEOSPATIAL INSIGHTS

# ---- helper: cached GeoJSON loader ----
@st.cache_data(show_spinner=False)
def load_us_states_geojson():
    url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json"
    try:
        return requests.get(url, timeout=10).json()
    except Exception:
        with open("us-states.json", "r") as f:
            return json.load(f)

with tab2:
    st.markdown(f'<div class="tool-description"><p>{TOOL_DESCRIPTIONS["bills_map"]["description"]}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>Geographic Distribution Map</h2><p>Interactive visualization of AI governance bills across US states</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-content">', unsafe_allow_html=True)

    # --- Year selector (expects values like "2025-2026") ---
    year_options = sorted([str(y) for y in filtered_df['session_year'].dropna().unique()])
    if year_options:
        selected_year = st.selectbox("Session year", year_options, index=len(year_options)-1)
        df_year = filtered_df[filtered_df['session_year'] == selected_year].copy()
    else:
        selected_year = None
        df_year = filtered_df.copy()

    # --- Counts by state (expects 2-letter postal abbreviations) ---
    tmp = df_year.copy()
    tmp['state'] = tmp['state'].astype(str).str.upper()

    counts_df = (
        tmp['state']
        .value_counts()
        .rename_axis('state')
        .reset_index(name='bills')
    )
    state_to_count = dict(zip(counts_df['state'], counts_df['bills']))

    # --- Build per-state popup HTML (scrollable) ---
    title_col = 'title' if 'title' in tmp.columns else ('bill_title' if 'bill_title' in tmp.columns else None)
    bn_col = 'bill_number' if 'bill_number' in tmp.columns else ('number' if 'number' in tmp.columns else None)

    def _bill_label(row):
        bn = str(row.get(bn_col, '') or '').strip() if bn_col else ''
        tt = str(row.get(title_col, '') or '').strip() if title_col else ''
        bn = html.escape(bn)
        tt = html.escape(tt)
        if bn and tt:
            return f"<li><b>{bn}</b>: {tt}</li>"
        elif bn:
            return f"<li><b>{bn}</b></li>"
        elif tt:
            return f"<li>{tt}</li>"
        return "<li>(unnamed bill)</li>"

    def _group_to_ul(g: pd.DataFrame) -> str:
        # Be robust to pandas versions: 'state' may or may not be present in g
        if 'state' in g.columns:
            g = g.drop(columns='state')
        return "<ul>" + "".join(_bill_label(r) for _, r in g.iterrows()) + "</ul>"

    if not tmp.empty:
        # Try pandas >= 2.2 signature first (supports include_groups)
        try:
            bills_by_state = (
                tmp.groupby('state', group_keys=False)
                   .apply(_group_to_ul, include_groups=False)
                   .to_dict()
            )
        except TypeError:
            # Older pandas: no include_groups; still safe due to the 'state' check in _group_to_ul
            bills_by_state = (
                tmp.groupby('state', group_keys=False)
                   .apply(_group_to_ul)
                   .to_dict()
            )
    else:
        bills_by_state = {}

    # --- GeoJSON & properties (also build scrollable popup HTML) ---
    us_states = load_us_states_geojson()

    for feat in us_states.get('features', []):
        abbr = (feat.get('id') or "").upper()
        props = feat.setdefault('properties', {})
        props['bills'] = int(state_to_count.get(abbr, 0))

        state_name = html.escape(props.get('name', ''))
        total = props['bills']
        list_html = bills_by_state.get(abbr, "<i>No bills for the selected session.</i>")

        count_label = "bill" if total == 1 else "bills"
        props['popup_html'] = (
            f"<div style='max-width:420px;'>"
            f"<h4 style='margin:0 0 8px 0;'>{state_name} — {total} {count_label}"
            + (f" ({html.escape(selected_year)})" if selected_year else "")
            + "</h4>"
            f"<div style='max-height:280px; overflow-y:auto; padding-right:6px;'>{list_html}</div>"
            f"</div>"
        )

    # --- Map ---
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles="OpenStreetMap")

    # Choropleth (one-tone Blues)
    folium.Choropleth(
        geo_data=us_states,
        name="Bills choropleth",
        data=counts_df,
        columns=["state", "bills"],
        key_on="feature.id",
        fill_color="Blues",
        fill_opacity=0.85,
        line_opacity=0.2,          # thin internal boundaries from choropleth layer
        line_color="#ffffff",
        nan_fill_color="#f0f0f0",
        legend_name=f"AI governance bills by state ({selected_year})" if selected_year else "AI governance bills by state",
    ).add_to(m)

    # Outline layer so borders are clearly visible
    folium.GeoJson(
        us_states,
        name="State boundaries",
        style_function=lambda x: {
            "fillOpacity": 0,
            "color": "#666666",
            "weight": 1.2
        },
        highlight_function=lambda x: {"weight": 2, "color": "#333333", "fillOpacity": 0},
        tooltip=folium.features.GeoJsonTooltip(
            fields=["name", "bills"],
            aliases=["State", "Bills"],
            sticky=True,
            localize=True,
        )
    ).add_to(m)

    # Transparent layer for clickable popups (scrollable content)
    folium.GeoJson(
        us_states,
        name="State popups",
        style_function=lambda x: {"color": "#00000000", "fillOpacity": 0, "weight": 0},
        popup=folium.features.GeoJsonPopup(
            fields=["popup_html"],
            labels=False,
            localize=True,
            parse_html=True,
            max_width=500,   # width cap; vertical scroll inside content
        ),
    ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)

    st_folium(m, width=1200, height=700)

    st.markdown('</div></div>', unsafe_allow_html=True)


# TAB 3: AI ANALYSIS TOOLKIT
with tab3:
    st.markdown(f'<div class="tool-description"><p>{TOOL_DESCRIPTIONS["ai_toolkit"]["description"]}</p></div>', unsafe_allow_html=True)

    # Analysis type selector
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>Analysis Type Selection</h2><p>Choose your preferred AI-powered analysis method</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-content">', unsafe_allow_html=True)
    
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        options=list(AI_ANALYSIS_TYPES.keys()),
        format_func=lambda x: AI_ANALYSIS_TYPES[x]["name"],
        index=0,
        key="analysis_type_selector"
    )
    
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Create bill options helper function
    def create_bill_options():
        if "bill_number" in filtered_df.columns and "state" in filtered_df.columns and "title" in filtered_df.columns:
            bill_options = [
                (
                    f"{row['state']}_{row['bill_number']}",
                    f"{row['state']}_{row['bill_number']}: {row['title'][:50] + '...' if len(row['title']) > 50 else row['title']}"
                )
                for _, row in filtered_df.iterrows()
                if pd.notna(row["title"]) and row["title"].strip()
            ]
            bill_options = sorted(bill_options, key=lambda x: x[0])
            bill_keys = [option[0] for option in bill_options]
            bill_labels = [option[1] for option in bill_options]
            return bill_keys, bill_labels
        return [], []

    bill_keys, bill_labels = create_bill_options()

    if not bill_keys:
        st.warning("No relevant bills with titles available for analysis.")
    else:
        # Analysis interface section
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><h2>{}</h2><p>{}</p></div>'.format(
            AI_ANALYSIS_TYPES[analysis_type]["name"], 
            AI_ANALYSIS_TYPES[analysis_type]["description"]
        ), unsafe_allow_html=True)
        st.markdown('<div class="section-content">', unsafe_allow_html=True)
        
        # Different UI based on analysis type
        if analysis_type == "qa":
            # Legislative Q&A Interface
            selected_option = st.selectbox(
                "Select a bill to query:",
                options=bill_labels,
                index=None,  
                placeholder="-- Select or type a bill --",
                key="toolkit_qa_bill"
            )
            
            # Handle bill selection
            if selected_option is None:
                selected_bill = None
            else:
                selected_bill_index = bill_labels.index(selected_option)
                selected_bill = bill_keys[selected_bill_index]
            
            # Display bill details and summary when a bill is selected
            if selected_bill:
                # Extract the selected bill's data
                state, bill_number = selected_bill.split("_", 1)
                bill_mask = (df["state"] == state) & (df["bill_number"] == bill_number)
                bill_df = df[bill_mask].copy()
                
                if not bill_df.empty:
                    bill_data = bill_df.iloc[0].to_dict()
                    display_bill_details(bill_data, summaries_cache)
                    
                    st.markdown("---")
                    st.markdown("#### Example Questions You Can Ask:")
                    st.markdown("""
                    • What are the key definitions in this bill?
                    • What are the enforcement mechanisms?
                    • Who does this bill apply to?
                    • What are the compliance requirements?
                    • What penalties are specified?
                    • What are the transparency requirements?
                    • How does this bill define AI systems?
                    • What are the implementation timelines?
                    """)
                    
                    st.markdown("#### Suggested Questions for This Bill:")
                    
                    try:
                        filtered_bill_mask = (filtered_df["state"] == state) & (filtered_df["bill_number"] == bill_number)
                        filtered_bill_df = filtered_df[filtered_bill_mask].copy()
                        
                        if not filtered_bill_df.empty:
                            bill_data = filtered_bill_df.iloc[0].to_dict()
                            suggested_questions = get_bill_suggested_questions(bill_data, questions_cache)
                            
                            for i, question in enumerate(suggested_questions):
                                if st.button(question, key=f"toolkit_suggested_q_{i}_{selected_bill}", use_container_width=True):
                                    st.session_state.toolkit_qa_input = question
                                    st.rerun()
                        else:
                            st.info("Using example questions above")
                            
                    except Exception as e:
                        logger.error(f"Error loading suggested questions: {e}")
                        st.info("Using example questions above")
            
            st.markdown("---")
            st.markdown("#### Ask a Question")
            
            user_input = st.text_input(
                "Ask a question about this bill:", 
                key="toolkit_qa_input"
            )

            if st.button("Ask", key="toolkit_qa_button"):
                if not selected_bill:
                    st.error("Please select a bill first.")
                elif not user_input.strip():
                    st.error("Please enter a question.")
                else:
                    # Get the selected bill data
                    state, bill_number = selected_bill.split("_", 1)
                    bill_mask = (filtered_df["state"] == state) & (filtered_df["bill_number"] == bill_number)
                    bill_df = filtered_df[bill_mask].copy()
                    
                    if not bill_df.empty:
                        bill_data = bill_df.iloc[0].to_dict()
                        
                        with st.spinner("Analyzing bill and generating answer..."):
                            try:
                                answer = answer_bill_question(bill_data, user_input)
                                
                                st.markdown("#### Answer")
                                st.markdown(answer)
                                
                            except Exception as e:
                                st.error(f"Failed to generate answer: {str(e)}")
                                logger.error(f"Q&A error: {e}")
                    else:
                        st.error("Could not find the selected bill data.")

        
        elif analysis_type == "comparison":
            # Bill Comparison Interface
            bill_options = [
                (
                    f"{row['state']}_{row['bill_number']}",
                    f"{row['state']}_{row['bill_number']}: {row['title'][:50] + '...' if len(row['title']) > 50 else row['title']}"
                )
                for _, row in filtered_df.iterrows()
                if pd.notna(row["title"]) and row["title"].strip()
            ]
            bill_options = sorted(bill_options, key=lambda x: x[0])
            bill_keys = [option[0] for option in bill_options]
            bill_labels = [option[1] for option in bill_options]
            
            if not bill_keys:
                st.warning("No relevant bills with titles available for comparison.")
            else:
                focus_bill_option = st.selectbox(
                    "Select a focus bill:",
                    options=bill_labels,
                    index=None,
                    placeholder="-- Select or type a bill --",
                    key="toolkit_focus_bill",
                    help="Type to search through available bills"
                )
                
                if focus_bill_option is None:
                    focus_bill = None
                else:
                    focus_bill_index = bill_labels.index(focus_bill_option)
                    focus_bill = bill_keys[focus_bill_index]

                if focus_bill is not None:
                    comparison_keys = [key for key in bill_keys if key != focus_bill]
                    comparison_labels = [label for key, label in zip(bill_keys, bill_labels) if key != focus_bill]

                    comparison_bills_options = st.multiselect(
                        "Select bills to compare:",
                        options=comparison_labels,
                        placeholder="-- Select or type bills to compare --",
                        key="toolkit_comparison_bills"
                    )
                    
                    comparison_bills = []
                    for selected_label in comparison_bills_options:
                        if selected_label in comparison_labels:
                            comp_index = comparison_labels.index(selected_label)
                            comparison_bills.append(comparison_keys[comp_index])
                else:
                    comparison_bills = []

                # Display bill details when bills are selected
                if focus_bill is not None:
                    # Display focus bill details
                    focus_state, focus_bill_number = focus_bill.split("_", 1)
                    focus_mask = (filtered_df["state"] == focus_state) & (filtered_df["bill_number"] == focus_bill_number)
                    focus_bill_df = filtered_df[focus_mask].copy()
                    
                    if not focus_bill_df.empty:
                        focus_bill_data = focus_bill_df.iloc[0].to_dict()
                        st.markdown("---")
                        st.markdown("#### 📋 Focus Bill Details")
                        display_bill_details(focus_bill_data, summaries_cache)
                    
                    # Display comparison bills if selected
                    if comparison_bills:
                        st.markdown("---")
                        st.markdown("#### 📋 Comparison Bills Details")
                        
                        # Create columns for comparison bills
                        if len(comparison_bills) > 0:
                            cols = st.columns(len(comparison_bills))
                            
                            for i, comp_bill in enumerate(comparison_bills):
                                with cols[i]:
                                    comp_state, comp_bill_number = comp_bill.split("_", 1)
                                    comp_mask = (filtered_df["state"] == comp_state) & (filtered_df["bill_number"] == comp_bill_number)
                                    comp_bill_df = filtered_df[comp_mask].copy()
                                    
                                    if not comp_bill_df.empty:
                                        comp_bill_data = comp_bill_df.iloc[0].to_dict()
                                        st.markdown(f"**{comp_bill}**")
                                        display_bill_details(comp_bill_data, summaries_cache)

                st.markdown("---")
                st.markdown("#### Example Comparison Questions:")
                st.markdown("""
                • How do these bills define AI systems differently?
                • What are the key differences in enforcement mechanisms?
                • Which bill has stricter compliance requirements?
                • How do the penalty structures compare?
                • What are the similarities in scope and coverage?
                • How do the implementation timelines differ?
                • Which bill provides more detailed privacy protections?
                • How do the exemptions and exceptions compare?
                """)

                comparison_question = st.text_input(
                    "Ask a comparison question:", key="toolkit_comparison_input"
                )

                if st.button("Compare", key="toolkit_compare_button"):
                    if not focus_bill:
                        st.error("Please select a focus bill first.")
                    elif not comparison_bills:
                        st.error("Please select at least one bill to compare against.")
                    elif not comparison_question.strip():
                        st.error("Please enter a comparison question.")
                    else:
                        # Get the selected bills data
                        focus_state, focus_bill_number = focus_bill.split("_", 1)
                        focus_mask = (filtered_df["state"] == focus_state) & (filtered_df["bill_number"] == focus_bill_number)
                        focus_bill_df = filtered_df[focus_mask].copy()
                        
                        comparison_bills_data = []
                        for comp_bill in comparison_bills:
                            comp_state, comp_bill_number = comp_bill.split("_", 1)
                            comp_mask = (filtered_df["state"] == comp_state) & (filtered_df["bill_number"] == comp_bill_number)
                            comp_bill_df = filtered_df[comp_mask].copy()
                            
                            if not comp_bill_df.empty:
                                comparison_bills_data.append(comp_bill_df.iloc[0].to_dict())
                        
                        if not focus_bill_df.empty and comparison_bills_data:
                            focus_bill_data = focus_bill_df.iloc[0].to_dict()
                            
                            with st.spinner("Creating vectorstore and analyzing bills for comparison..."):
                                try:
                                    answer = compare_bills_with_rag(
                                        focus_bill_data, 
                                        comparison_bills_data, 
                                        comparison_question
                                    )
                                    
                                    st.markdown("#### Comparison Analysis")
                                    st.markdown(answer)
                                    
                                except Exception as e:
                                    st.error(f"Failed to generate comparison: {str(e)}")
                                    logger.error(f"Comparison error: {e}")
                        else:
                            st.error("Could not find the selected bills data.")
                
        elif analysis_type == "summary":
            # Executive Summary Interface
            selected_option = st.selectbox(
                "Select a bill to read its report:",
                options=bill_labels,
                index=None,
                placeholder="-- Select or type a bill --",
                key="toolkit_summary_bill"
            )
            
            # Handle bill selection
            if selected_option is None:
                selected_bill = None
            else:
                selected_bill_index = bill_labels.index(selected_option)
                selected_bill = bill_keys[selected_bill_index]
            
            # Display bill report when a bill is selected
            if selected_bill:
                # Extract the selected bill's data
                state, bill_number = selected_bill.split("_", 1)
                bill_mask = (filtered_df["state"] == state) & (filtered_df["bill_number"] == bill_number)
                bill_df = filtered_df[bill_mask].copy()
                
                if not bill_df.empty:
                    bill_data = bill_df.iloc[0].to_dict()
                    
                    # Get and display the report
                    report = get_bill_report(bill_data, reports_cache)
                    
                    if report.startswith('No pre-generated report'):
                        st.warning(report)
                    else:
                        st.markdown(report, unsafe_allow_html=True)
                        
                        # Add download button for the report
                        pdf_filename = f"{state}_{bill_number}_report.md"
                        st.download_button(
                            label="Download Report as Markdown",
                            data=report,
                            file_name=pdf_filename,
                            mime="text/markdown",
                            key="download_report"
                        )

        elif analysis_type == "eu_comparison":
            # EU AI Act Comparison Interface
            
            # First, check if EU vectorstore is available
            eu_vectorstore, eu_error = load_eu_ai_act_vectorstore()
            
            if eu_vectorstore is None:
                st.error("EU AI Act vectorstore not available.")
                st.info("""
                To use the EU AI Act comparison feature:
                1. Ensure `eu-ai-act.pdf` is in the project directory
                2. Run the EU AI Act processing script: `python create_eu_ai_act_vectorstore.py`
                3. Refresh this page
                """)
                st.code("python create_eu_ai_act_vectorstore.py")
                
                if eu_error:
                    st.error(f"Error details: {eu_error}")
            else:
                # Show EU AI Act info
                eu_info = get_eu_vectorstore_info()
                if "error" not in eu_info:
                    st.success(f"✅ EU AI Act loaded")
                
                selected_option = st.selectbox(
                    "Select a US bill to compare with the EU AI Act:",
                    options=bill_labels,
                    index=None,
                    placeholder="-- Select or type a bill --",
                    key="toolkit_eu_comparison_bill"
                )
                
                # Handle bill selection
                if selected_option is None:
                    selected_bill = None
                else:
                    selected_bill_index = bill_labels.index(selected_option)
                    selected_bill = bill_keys[selected_bill_index]
                
                # Display bill details when a bill is selected
                if selected_bill:
                    # Extract the selected bill's data
                    state, bill_number = selected_bill.split("_", 1)
                    bill_mask = (filtered_df["state"] == state) & (filtered_df["bill_number"] == bill_number)
                    bill_df = filtered_df[bill_mask].copy()
                    
                    if not bill_df.empty:
                        bill_data = bill_df.iloc[0].to_dict()
                        st.markdown("---")
                        st.markdown("#### 📋 US Bill Details")
                        display_bill_details(bill_data, summaries_cache)
                        
                        # Show EU AI Act summary
                        st.markdown("---")
                        st.markdown("#### 🇪🇺 EU AI Act Overview")
                        st.info("""
                        The EU AI Act (Regulation 2024/1689) is comprehensive legislation that regulates AI systems based on their risk level. 
                        It establishes prohibited AI practices, high-risk AI systems requirements, transparency obligations, and governance structures. 
                        The Act applies to providers and deployers of AI systems in the EU market.
                        """)
                
                st.markdown("---")
                st.markdown("#### Example EU Comparison Questions:")
                st.markdown("""
                • How does this US bill's definition of AI compare to the EU AI Act's definition?
                • What are the key differences in risk assessment approaches?
                • How do enforcement mechanisms compare between the two frameworks?
                • Which framework has stricter requirements for high-risk AI systems?
                • How do transparency and documentation requirements compare?
                • What are the differences in prohibited AI practices?
                • How do the two frameworks approach algorithmic impact assessments?
                • What are the similarities and differences in governance structures?
                • How do compliance timelines compare?
                • Which framework provides better protection for fundamental rights?
                """)
                
                eu_comparison_question = st.text_input(
                    "Ask a comparison question:", 
                    key="toolkit_eu_comparison_input"
                )
                
                if st.button("Compare with EU AI Act", key="toolkit_eu_compare_button"):
                    if not selected_bill:
                        st.error("Please select a US bill first.")
                    elif not eu_comparison_question.strip():
                        st.error("Please enter a comparison question.")
                    else:
                        # Get the selected bill data
                        state, bill_number = selected_bill.split("_", 1)
                        bill_mask = (filtered_df["state"] == state) & (filtered_df["bill_number"] == bill_number)
                        bill_df = filtered_df[bill_mask].copy()
                        
                        if not bill_df.empty:
                            bill_data = bill_df.iloc[0].to_dict()
                            
                            with st.spinner("Analyzing US bill and EU AI Act for comparison..."):
                                try:
                                    answer = compare_bill_with_eu_ai_act(
                                        bill_data, 
                                        eu_comparison_question
                                    )
                                    
                                    st.markdown("#### US vs EU AI Governance Comparison")
                                    st.markdown(answer)
                                    
                                except Exception as e:
                                    st.error(f"Failed to generate EU comparison: {str(e)}")
                                    logger.error(f"EU comparison error: {e}")
                        else:
                            st.error("Could not find the selected bill data.")  