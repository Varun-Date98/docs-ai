# 🧠 DocsAI — AI Meeting Assistant MVP
*A lightweight, Retrieval-Augmented Generation (RAG) agent for understanding meeting notes and documents.*

---

## 🚀 Overview

DocsAI is an **AI-powered meeting assistant** that ingests user-uploaded files (e.g., meeting notes, pitch deck) and enables **conversational Q&A** over their content.  

It’s designed as a **minimal viable product (MVP)** to demonstrate backend architecture and agentic reasoning patterns for building features similar to *Zoom Docs* and *Zoom AI Companion*.

Users can:
- Upload documents asynchronously (`.txt`, `.ppt`, `.md`).
- Automatically process and chunk documents into embeddings.
- Store embeddings in a persistent **Chroma vector database**.
- Chat with an **LLM-powered retrieval agent** that answers questions using the document context.

---

## Demo

![Demo Video](assets/recording.gif)
---

## 🧩 Key Features

- ⚙️ **Asynchronous File Ingestion**  
  - Non-blocking uploads via FastAPI background tasks.  
  - Each file is chunked, embedded, and added to the vector database.  

- 💬 **Conversational Q&A Interface**  
  - Built using Streamlit’s `st.chat_input()`.  
  - Uses OpenAI’s GPT models for contextual responses.  

- 🧠 **Retrieval-Augmented Generation (RAG)**  
  - Embeds text using `text-embedding-3-small`.  
  - Retrieves top-3 context chunks via Chroma.  
  - Formats responses using LangChain’s `ChatPromptTemplate` pipeline.  

- 🪶 **Lightweight & Modular**  
  - FastAPI backend + Streamlit frontend.  
  - Simple startup with one command.  
  - Clear separation of ingestion, retrieval, and response layers.  

---

## 🧱 Architecture

```plaintext
+---------------------+        +------------------------+
|  Streamlit Frontend | -----> |  FastAPI Backend        |
| (Chat + Upload UI)  |        | (Handles tasks & chat)  |
+---------------------+        +-----------+-------------+
                                           |
                                           v
                                 +------------------------+
                                 | Background Task Queue  |
                                 |  (process_file)        |
                                 +-----------+------------+
                                             |
                                             v
                                 +------------------------+
                                 |  Chroma Vector Store   |
                                 | (Persistent embeddings)|
                                 +-----------+------------+
                                             |
                                             v
                                 +------------------------+
                                 |   OpenAI LLM API       |
                                 |   (GPT-4o-mini )       |
                                 +------------------------+
```

---


```
## 🧠 Pipeline Flow

1. **User Uploads a File**
   - Streamlit sends the file → FastAPI `/upload`
   - FastAPI immediately responds `202 Accepted` with `task_id`

2. **Background Processing**
   - File is read → text extracted (`pdfplumber`, `python-pptx`, or plain text)
   - Text split into overlapping chunks via `RecursiveCharacterTextSplitter`
   - Embeddings created using `OpenAIEmbeddings`
   - Chroma adds new documents incrementally (no full rebuild)

3. **Chat Interaction**
   - Frontend sends query → `/chat`
   - Backend retrieves top-k relevant chunks
   - Constructs a contextual prompt:
     ```
     You are a meeting assistant agent. Answer using the provided context...
     ```
   - Sends to `ChatOpenAI`
   - Streams response back to frontend in real time

---

## 💡 Example Questions

**Input Document:** `Meeting Notes.txt`

| User Question | Model Answer |
|----------------|---------------|
| What was the customer churn rate in Q3? | The churn rate in Q3 was **7.2%**, up from 5.9% last quarter. |
| What was the key finding from Josh’s cohort analysis? | Users who set up integrations within 24 hours retained **2× longer**. |
| What issues did Sarah mention during the meeting? | Sarah raised **latency issues** in the speech-to-text pipeline. |

---

## 🧰 Tech Stack

| Layer               | Tools / Libraries                     |
|---------------------|---------------------------------------|
| Frontend            | Streamlit, streamlit-float            |
| Backend             | FastAPI, Uvicorn                      |
| AI / Embeddings     | LangChain, OpenAI API                 |
| Vector Store        | Chroma                                |
| Async Tasks         | FastAPI BackgroundTasks               |
| Parsing             | python-pptx, PyPDF2, python-multipart |

---

## ⚙️ Installation

```bash
# Create venv
python -m venv venv
source venv/bin/activate  or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI key
export OPENAI_API_KEY="sk-..."
export API_URL="http://localhost:8080"
```

---

## ▶️ Run Locally

### 1️⃣ Start backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8080
```

### 2️⃣ Start frontend
```bash
streamlit run frontend/app.py
```

---

## 🧩 Example API Calls

### Upload a file
```bash
curl -X POST "http://localhost:8080/upload"      -F "file=@sample_docs/Meeting_Notes.txt"
```

### Chat with your docs
```bash
curl -X POST "http://localhost:8080/chat"      -H "Content-Type: application/json"      -d '{"message": "What was the churn rate in Q3?"}'
```

---

## 🧠 Future Enhancements

| Area | Improvement |
|-------|-------------|
| **Agent Reasoning** | Add a router chain that decides between retrieval, summarization, and analysis modes. |
| **Observability** | Add metrics for response time, retrieval precision, and latency. |
| **Authentication** | Support per-user vector collections and secure uploads. |
| **Scalability** | Deploy to AWS Lambda or ECS with S3-based storage. |
| **UI Polish** | Replace Streamlit with React + FastAPI WebSocket backend for real-time streaming. |

---

## 💬 Why This Project Matters

This MVP demonstrates **end-to-end ownership of an AI product feature** — from ingestion to chat UX:
- Multi-modal data ingestion
- Asynchronous processing
- Real-time AI interaction
- Prompt observability
- Modular RAG design

---

## 🧾 License
MIT License © 2025 [Your Name]
