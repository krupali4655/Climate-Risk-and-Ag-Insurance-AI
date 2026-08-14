# 🌱 Climate Risk & Ag-Insurance AI Agent

An enterprise-grade, modular AI application built to assist farmers, underwriters, and policymakers with agricultural risk assessment, government policy navigation, real-time weather tracking, and deterministic insurance claim calculations.

---

## 🚀 Key Features & Architecture

This project progresses through four advanced AI engineering phases:

1. **Conversational Memory (Phase 1):** Built using Streamlit session states and Groq’s high-speed inference engine (`llama-3.3-70b-versatile`) to maintain multi-turn context.
2. **Enterprise RAG Pipeline (Phase 2):** Ingests official government agricultural policies (e.g., PMFBY and RWBCIS guidelines) via PDF chunkers, generates local vector embeddings using `FastEmbed`, and stores them persistently in **ChromaDB** for rapid semantic search.
3. **Agentic Tool Calling (Phase 3):** Empowers the LLM to autonomously reason, decide when to query external tools, and handle multi-source integration.
4. **Decoupled Modular Architecture (Phase 4):** Separates core execution logic and tools into an independent backend module (`ag_server.py`), simulating a microservice/Model Context Protocol (MCP) design pattern.

---

## 🛠️ Tech Stack

* **Language:** Python 3.11+
* **Frontend/UI:** Streamlit
* **LLM Provider:** Groq API (`llama-3.3-70b-versatile`)
* **Orchestration & RAG:** LangChain, ChromaDB, FastEmbed, PyPDF
* **External APIs:** Open-Meteo API (Real-time weather data)
* **Security & Configuration:** Python-Dotenv

---

## 📁 Project Structure

```text
concept1/
│
├── .venv/                  # Virtual environment
├── data/                   # Government policy PDFs (PMFBY, RWBCIS)
├── chroma_db/              # Persistent vector store database
├── .env                    # Secret environment variables (Ignored by Git)
├── .gitignore              # Git ignore configurations
├── ingest.py               # Offline data preparation and indexing script
├── ag_server.py            # Modular backend server containing tools
├── app.py                  # Main Streamlit client application
└── README.md               # Project documentation

```

---

## ⚙️ Installation & Setup Guide

Follow these steps to set up and run the project locally on your machine:

### 1. Clone the Repository & Navigate to Folder

```bash
git clone <your-repository-url>
cd concept1

```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install streamlit groq langchain langchain-community langchain-text-splitters fastembed chromadb pypdf requests python-dotenv

```

### 4. Configure Your Environment Variables

Create a file named `.env` in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_actual_groq_api_key_here

```

### 5. Add Data & Run Ingestion

1. Create a folder named `data` in the root directory.
2. Place your agricultural policy PDFs inside the `data/` folder.
3. Run the ingestion script to build the local vector database:

```bash
python ingest.py

```

### 6. Launch the Application

Run the Streamlit frontend client:

```bash
python -m streamlit run app.py

```

---

## 💡 Example Prompts to Try

* **Policy Navigation (RAG):** *"According to the guidelines, what is the exact maximum premium payable by farmers for Kharif crops?"*
* **Live Weather Tool:** *"I have a farm in New Delhi. What is the current temperature and precipitation there right now?"*
* **Deterministic Math Tool:** *"Calculate the crop insurance claim where threshold yield is 3000, actual yield is 2100, and the sum insured is 75000."*

```

```