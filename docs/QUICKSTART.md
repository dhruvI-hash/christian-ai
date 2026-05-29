# Christianity-Focused AI Assistant — Quickstart Guide

This guide will help you get the full-stack Christianity-focused AI Assistant monorepo up and running on your local machine.

---

## 🏗️ Architecture & Requirements

- **Backend**: Python 3.11+ (FastAPI, LangChain agents)
- **Frontend**: Next.js 14 (React, TypeScript, TailwindCSS)
- **Vector DBs**: ChromaDB (dense search, runs locally) and Qdrant (dense + sparse search)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)

---

## 🛠️ Step 1: Environment Configuration

1. Locate the `.env.example` file in the project root: [env.example](file:///c:/Users/kashy/OneDrive/Desktop/Dhruvi/assignment1/christianity-ai/.env.example).
2. Copy `.env.example` and rename it to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Open your new `.env` file and insert your API keys:
   - **`OPENAI_API_KEY`**: Required for LangChain/OpenAI models (`gpt-4o`, `gpt-4o-mini`, DALL-E 3).
   - **`GEMINI_API_KEY`**: Optional - for Google Gemini integration.
   - **`QDRANT_URL` & `QDRANT_API_KEY`**: Optional - if using Qdrant Cloud or external servers. (ChromaDB runs locally automatically).

---

## 🐍 Step 2: Backend Setup

The backend serves the FastAPI server, runs the guardrails, manages scripture verification tools, handles vector search (RAG), and calls DALL-E for image generation.

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate on Windows (PowerShell)
   .venv\Scripts\Activate.ps1

   # Activate on macOS/Linux
   source .venv/bin/activate
   ```
3. Install dependencies into this venv (not global Python):
   ```bash
   pip install -r requirements.txt
   ```
4. **Crucial Windows Compatibility Step**: To avoid conflicts with packages compiled under older NumPy compilers (such as `pyarrow` used by ChromaDB/SentenceTransformers), run the following command to pin NumPy to version 1:
   ```bash
   pip install "numpy<2"
   ```
5. Run the FastAPI server:
   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```
6. Verify backend is running by opening:
   - API Info: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
   - Interactive Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## ⚛️ Step 3: Frontend Setup

The frontend provides the premium interactive interface, featuring collapsible navigation, denomination alignment selectors, live LLM routers, a Markdown chat workspace, and a RAG administrator slide-out panel.

1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Start the Next.js development server:
   ```bash
   npm run dev
   ```
   > [!NOTE]
   > Next.js will run on **[http://localhost:3000](http://localhost:3000)** by default. If port `3000` is already in use, it will automatically route to **[http://localhost:3002](http://localhost:3002)**. The backend CORS system is configured to support both ports seamlessly.

---

## ✝️ Step 4: Theological Assistant Workflows

Open your browser to the local frontend port (e.g., **`http://localhost:3002`**) and explore the premium features:

### 1. RAG Management (Knowledge Ingestion)
- Click the **"RAG Admin"** button at the top-right of the dashboard to slide out the management panel.
- Choose **ChromaDB** or **Qdrant** as your active target database.
- Drag and drop scriptural documents (PDF, DOCX, TXT, or Markdown) into the ingestion box.
- Write a scripture test query directly into the Search panel to verify semantic matching accuracy.

### 2. Denominational Personalization
- Expand the sidebar and select from **Catholic**, **Protestant**, **Orthodox**, **Evangelical**, or **Non-denominational**.
- Ask a denominational-sensitive question (e.g., *"What is the significance of the Eucharist?"* or *"Explain sola scriptura"*).
- Watch how the AI formats answers matching the selected denomination's catechism or traditions.

### 3. Scripture Citation & Verification
- Enter any query requesting Bible verses (e.g., *"Give me verses about comfort in grief"*).
- The main agent will automatically trigger the `scripture_verify_tool` in the background, matching requested references against the ingested canon to block hallucinations.
- Verified verses show up decorated with a premium **"Verified"** gold badge linking straight to BibleGateway.

### 4. Multimodal Sacred Image Generation
- Request an image in the chat (e.g., *"Generate a stained glass window depicting the Resurrection"*).
- The sub-agent passes the concept through strict theological safety layers before generating a stunning watercolor, realistic, painterly, or minimalist representation via DALL-E 3.

---

## 🛠️ Troubleshooting & Support

- **CORS Errors**: If you encounter network errors, ensure the backend is running at `http://127.0.0.1:8000` and the frontend origin (e.g., `http://localhost:3002`) is correctly whitelisted in [main.py](file:///c:/Users/kashy/OneDrive/Desktop/Dhruvi/assignment1/christianity-ai/backend/main.py#L55-L68).
- **Wrong Python environment**: Always activate `backend/.venv` before running the server. In VS Code/Cursor, set the interpreter to `backend/.venv/Scripts/python.exe`.
- **Global package conflicts**: Use the project venv only (`pip install -r requirements.txt` inside `backend/.venv`), not your system Anaconda environment.
- **Ollama Local LLM**: If selecting Ollama models, ensure your local Ollama server is running (`ollama serve`) and the relevant models (`qwen3.5:4b`, `granite3.1:3b`) are pulled.
