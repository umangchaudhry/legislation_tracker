# Vanderbilt AI Governance Legislation Tracker

## Quick Navigation
- [Goal](#goal)  
- [Background](#background)  
- [Data](#data)  
- [Models](#models)  
- [Repo Structure](#repo-structure)  
- [Contact](#contact-info)  

---

## Goal
The Vanderbilt AI Governance Legislation Tracker is a web application designed to monitor and provide accessible information about artificial intelligence (AI) governance legislation across the United States. This repository contains the source code, database, and documentation for the tracker. The final deliverable is a streamlit web application that allows users to search, filter, and explore AI governance bills at the state level as well as the US House and Senate. It can be accessed by running `streamlit_app.py` locally after setting up the required dependencies (see [Repo Structure](#repo-structure)).

The goal is to create a centralized, user-friendly platform to help policymakers, researchers, and the public stay informed about the evolving landscape of AI regulation in the U.S. as well as abroad with specific focus on the EU AI Act.

---

## Background
As AI technology advances rapidly, understanding how states regulate it is increasingly critical. I'm a law school professor at Vanderbilt University, co-directing the Vanderbilt AI Law Lab (VAILL), and I created this tracker to experiment with designing a tool that simplifies access to state-level AI governance legislation. This project started with a small list of bills but aims to grow into a comprehensive resource. It offers features like search functionality, status filtering, and educational content about AI governance provisions.

This README provides an overview of the project. To navigate the repository, start with the [Repo Structure](#repo-structure) section for instructions on running the app or exploring the code. The [Data](#data) section details the legislative data used.

---

## Data
The project uses a manually curated dataset of AI governance legislation from across the U.S., stored in `/data/known_bills_visualize.json`. The data includes fields like state, bill number, scope, key provisions, summary, and current status.  

Other files in the `/data` folder include cached bill information, summaries, suggested questions, and backups. These support the frontend experience and reduce repeated API calls.  

**Data Privacy Concerns**: The data is sourced from publicly available legislative records from Legiscan.com, so there are no privacy concerns regarding personal information. Scripts have been provided to occasionally update the database.  

---

## Models
The tracker relies on models provided by OpenAI for analysis and QA, primarily GPT-4o at the time of this update.  

---

## Repo Structure
The repository is organized as follows:

```
ai-legislation-tracker/
│
├── constants.py                 # Shared constants for the app  
├── requirements.txt             # Project dependencies  
├── streamlit_app.py             # Main Streamlit frontend  
├── update_data.py               # Entry point for updating legislative data  
├── vaill_logo.png               # Branding/logo for UI  
│
├── data/                        # Legislative data store
│   ├── bill_cache.json          # Bill Cache that ensures optimized bill pulling
│   ├── bill_reports.json        # Bill Reports
│   ├── bill_summaries.json      # Bill Summaries
│   ├── bill_suggested_questions.json # Personalized suggested questions per bill
│   ├── known_bills.json         # Unmodified initial data pull
│   ├── known_bills_backup.json  # Backup for pre-processing errors
│   ├── known_bills_fixed.json   # Fixed text fields handling PDFs, HTMLs etc. 
│   └── known_bills_visualize.json # Final processed bill data for tracker
│
├── data_updating_scripts/       # Scripts for processing and refreshing data
│   ├── config.py
│   ├── eu-ai-act.pdf            # Reference text for EU AI Act
│   ├── eu_vectorstore.py        # Vectorized EU AI Act for semantic search
│   ├── fix_pdf_bills.py         # Handles bills pulled as PDFs
│   ├── generate_reports.py      
│   ├── generate_summaries.py
│   ├── generate_suggested_questions.py
│   ├── get_data.py
│   ├── known_bills_status.py
│   ├── mark_no_text_bills.py
│   ├── migrate_iapp_categories.py
│   └── PROMPTS/                 # Prompt templates
│       ├── bill_summary_prompt.py
│       └── suggested_questions_prompt.md
```

### Notes on Updating Scripts
- **Dynamic Processing**: The updating scripts are written to avoid re-processing bills that have already been handled or re-pulling data that is already cached.  
- **Individual Runs**: Each script inside `data_updating_scripts/` can be run independently if targeted updates are needed (e.g., regenerating summaries, fixing PDFs, or refreshing bill statuses).  
- **Main Update Pipeline**: `update_data.py` is the central entry point that coordinates updates across the dataset.  

---

## Contact Info
- **Principal Investigator**: [Dr. Mark Williams](https://github.com/willimj3), Law School Professor and Co-Director, Vanderbilt AI Law Lab (VAILL).  
  - Email: [mark.williams@vanderbilt.edu](mailto:mark.williams@vanderbilt.edu)  
- **Project Lead**: [Umang Chaudhry](https://github.com/umangchaudhry), Senior Data Scientist, Vanderbilt Data Science Institute  
  - Email: [umang.chaudhry@vanderbilt.edu](mailto:umang.chaudhry@vanderbilt.edu)  
- **Project Manager**: [Isabella Urquia](https://github.com/isabellaurquia), Undergraduate Student, Vanderbilt University  
  - Email: [isabella.m.urquia@vanderbilt.edu](mailto:isabella.m.urquia@vanderbilt.edu)  
- **Team Contributors**:  
  - [Isabella Urquia](https://github.com/isabellaurquia), Undergraduate Student, Vanderbilt University  
  - [Timeo Williams](https://github.com/timeowilliams), Masters Student, Vanderbilt University  
  - [Harshita Ahuja](https://github.com/Harshitaahuja23), DS Masters Student, Vanderbilt University  
  - [Margot Zhao](https://github.com/MargotZhao), DS Masters Student, Vanderbilt University  

For inquiries, please reach out to the Project Lead or Principal Investigator.  
