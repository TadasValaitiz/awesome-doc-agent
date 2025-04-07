# AI Document Assistant

An AI-powered document processing and data cleaning assistant built with LangGraph and Streamlit.

## Features

- 🔍 Automatic detection of data quality issues
- 🧹 Clean missing values, outliers, and duplicates
- 📊 Generate data quality reports
- ⬇️ Download your cleaned dataset

## Architecture

This application consists of two main components:

1. **LangGraph Backend**: AI agent system handling document processing and analysis
2. **Streamlit Frontend**: User interface for document upload and interaction

### Document Processing Workflow

The document analysis and cleaning process follows this workflow:

![Document Processing Workflow](diagram.png)

## Cloud Deployments

### LangGraph Cloud

The application uses LangGraph for agent orchestration and is deployed at:
```
https://awesome-doc-agent-4aea3ef58f0f58c9b577c1f02420ef02.us.langgraph.app
```

### Streamlit Cloud

The user interface is deployed on Streamlit Cloud at:
```
https://awesome-doc-agent.streamlit.app/
```

## Local Development

### Prerequisites

- Python 3.10+
- pip

### Backend Setup

```bash
cd server
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Frontend Setup

```bash
cd app
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Available Agents

This system includes several specialized agents:
- Document Analysis Agent
- Enrichment Agent
- Generic Chat Agent

## Authentication

The application uses Firebase for authentication with support for:
- Email/Password Login
- Anonymous Guest Access
