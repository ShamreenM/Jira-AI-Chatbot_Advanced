# Jira-AI-Chatbot_Advanced

# 🤖 JIRA AI Chatbot (RAG + Hybrid + MultiQuery + CrossEncoder + OpenAI + Streamlit)

## Problem Statement
In large projects, thousands of Jira tickets are created over time. Searching for relevant issues, root causes, or past resolutions becomes difficult and time-consuming using traditional keyword search. Teams often spend significant effort manually analyzing tickets.

## Solution

To address this challenge, built an Jira AI chatbot using a Retrieval-Augmented Generation approach. Instead of relying only on the LLM’s knowledge, the system retrieves relevant Jira ticket information from a vector database and uses that context to generate accurate responses. Integrated semantic search with metadata filtering using ChromaDB and OpenAI embeddings to retrieve issue details such as status, priority, creation date, and resolution.
Designed structured prompt engineering to ensure accurate extraction of JIRA metadata fields and prevent hallucinations.

An AI-powered JIRA assistant built using:
- Pandas (Preprocessing)
- LangChain Orchestration Framework
- Sentence Transformer for embediing vector (Hugging Face)
- Open AI
- ChromaDB (Vector Database)
- Streamlit (Frontend)
- Refer Jira_API_Chatbot.ipynb for above steps perfomed
But model ready code is available in JiraAIChatbot.py
---
## Architecture

<img width="1536" height="1024" alt="ChatGPT Image Mar 31, 2026, 03_10_11 PM" src="https://github.com/user-attachments/assets/5feb5564-baf9-488e-9904-633eba27b543" />


---
## 🚀 Features

- Natural language Querying
- **Hybrid Search**: Dense + BM25 semantic search
- **Multi-Query Expansion**: Intelligent query generation
- **Cross-Encoder Re-ranking**: Advanced result ranking
- **Query Type Detection** : Smart routing for different query types 
- Intelligent Retrieval from Jira Dataset(vector based retrieval)
- Contextual Answers (analyze -> summarize -> present readable format)
- Faster decision making
---

## 📋 Supported Query Types

- 🎫 **Ticket Lookup**: `SRCTREEWIN-12345 status`
- 🧠 **Analysis**: `Why are bugs delayed?`
- 📅 **Date Queries**: `Issues raised on 15/Mar/2024`
- 🔍 **Semantic Search**: `High priority open bugs`
- 📊 **Status Queries**: `All stories in progress`

# 🛠️ Setup Instructions

Follow these steps to run the project locally:

---

Chroma (Vector DB) for embeddings are available here - https://drive.google.com/file/d/1QYxliFEJbuowOrCZWKq09_GLPdsfYI7Y/view?usp=sharing. Download and keep it in same folder as below.

1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Create .env File
```bash
OPENAPI_API_KEY=your_api_key_here
```

5️⃣ Run the Application
```bash
streamlit run app.py
```

App will open in your browser at:

http://localhost:8501

# Sample Questions to ask bot

1. Any issue asking to Please upload the image to Google Drive and share the link?
2. What is the reason for stories getting forwarded for last 2 sprints
3. What is the current status of issue SRCTREEWIN-14037
4. Is this issue Displaying all changes between hash1 and hash2 doesn't raised?'
