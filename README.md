##  Overview

This project is a fully functional **Life Insurance Support Agent** capable of answering questions about policies, claims, and eligibility. It features a **persistent conversational memory**, retrieving answers from a configurable Knowledge Base (RAG), and supports **Real-time Voice Interaction**.

The system is optimized for speed (3-4s latency) using asynchronous processing and `gpt-4o-mini`.

### Key Features

- ** LangGraph Agent:** Uses a state graph to manage conversation flow and knowledge retrieval.
- ** Low Latency:** Optimized with `asyncio`, in-memory audio processing, and targeted prompting.
- ** Persistent Memory:** Remembers user context across turns using **AsyncSqliteSaver**.
- ** Multi-Modal Interface:**
  - **Web UI:** Speak to the agent via browser (Voice-to-Voice).
  - **Voice CLI:** A "Matrix-style" terminal interface with real-time recording.
  - **Chat CLI:** Standard text-based terminal chat.
- ** Configurable Knowledge Base:** Easy-to-update JSON file for insurance domain knowledge.

---

##  Architecture

The system follows a client-server architecture:

1.  **Client (Web/CLI)** captures audio and sends it via **WebSockets**.
2.  **FastAPI** receives the audio and processes it in memory (no disk I/O).
3.  **OpenAI Whisper** transcribes the audio to text.
4.  **LangGraph** retrieves relevant context from `knowledge_base.json` and calls the LLM.
5.  **SQLite** stores the conversation history automatically.
6.  **Response** is sent back to the client for Text-to-Speech (TTS).

---

##  Project Structure

```text
├── backend/
│   ├── __init__.py
│   ├── main.py             # FastAPI Server & WebSocket Handler
│   ├── langgraph_agent.py  # The "Brain" (LangGraph Logic)
│   ├── cli_chat.py         # Text CLI Client
│   ├── voice_cli.py        # Voice CLI Client
│   └── utils.py            # Audio utilities
├── frontend/
│   ├── templates/
│   │   └── index.html      # Web UI HTML
│   └── static/
│       └── app.js          # Web UI Logic
├── knowledge_base.json     # Domain knowledge source
├── chat_history.db         # Auto-generated SQLite memory
├── .env                    # API Keys
└── requirements.txt        # Dependencies
```

---

##  Tech Stack

| Component         | Technology                  |
| ----------------- | --------------------------- |
| Backend Framework | FastAPI                     |
| Model Runtime     | OpenAI API                  |
| Agent Framework   | LangChain + LangGraph       |
| Graph Storage     | langgraph-checkpoint-sqlite |
| Web Server        | Uvicorn                     |
| Config            | python-dotenv               |
| Templates         | Jinja2                      |

---

##  Installation

### **1. Clone the Repository**

```bash
https://github.com/shehab0911/langgraph-life-insurance-agent-in-Voice-and-Chat-CLI.git
cd langgraph-life-insurance-agent-in-Voice-and-Chat-CLI.git
```

---

### **2. Create Virtual Environment**

#### **Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

#### **Mac/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

##  Environment Variables

Create a `.env` file in project root:

```
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

Optional:

```
DEBUG=True
```

---

##  Usage

You can run the system in three different modes.

### 1\. Start the Backend Server (Required)

This must be running for any interface to work.

```bash
uvicorn backend.main:app --reload
```

_Server will run at `http://127.0.0.1:8000`_

### 2\. Web Voice Interface

1.  Open your browser to [http://127.0.0.1:8000](http://127.0.0.1:8000).
2.  Click the **Microphone 🎤** button.
3.  Speak your question (e.g., _"What is the difference between term and whole life?"_).
4.  The agent will reply with text and voice.

### 3\. Production Voice CLI

A beautiful terminal-based voice client.

```bash
python -m backend.voice_cli
```

- Select **Record** with arrow keys.
- Speak, and watch the AI process in real-time.

### 4\. Text Chat CLI

A simple text-only debugging tool.

```bash
python -m backend.cli_chat
```

---

##  LangGraph Agent Architecture

Your agent is built using **LangGraph** with SQLite checkpointing:

```
User Input
   ↓
State Checkpoint
   ↓
LLM Node (reasoning)
   ↓
Tool Node (optional execution)
   ↓
Response Stream
```

### **Graph Components**

| Node             | Purpose                                        |
| ---------------- | ---------------------------------------------- |
| **AgentNode**    | Handles LLM reasoning and action generation    |
| **ToolNode**     | Executes required tools (search, DB ops, etc.) |
| **Router**       | Decides next step depending on agent output    |
| **Memory Store** | Saves previous steps                           |

---

### Voice CLI UI

Here’s a sample interaction with the Life Insurance Support Agent:

![Life Insurance Voice Agent UI](voice_cli.png)

---

##  Latency Optimization

To meet the \<4 second latency requirement, several optimizations were implemented:

1.  **Async Processing:** The backend uses `asyncio` to handle Whisper and LangGraph without blocking the server.
2.  **Memory-Only Audio:** Audio files are processed in RAM (`io.BytesIO`), avoiding slow disk writes.
3.  **Prompt Engineering:** The system prompt strictly enforces concise, 1-2 sentence answers to reduce LLM generation time.
4.  **Language Hinting:** Whisper is forced to `language="en"` to skip the detection phase.

---

##  Customization

**Updating Knowledge:**
Edit `knowledge_base.json` to add new insurance topics. The agent automatically detects keywords and retrieves the info.

**Changing Personality:**
Edit `SYSTEM_PROMPT` in `backend/langgraph_agent.py` to change how the agent behaves (e.g., make it more formal or verbose).

---

##  Future Improvements

- Convert the UI into a more mesmerizing, modern design
- Add Whisper speech-to-text
- Add TTS response generation
- Expand toolset (email automation, browser tooling, RAG search)
- Add authentication
- Vercel / Render

---

##  License

This project is licensed under the **MIT License**.

---
