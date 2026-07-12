# 🧠 RepoMind

**Ask questions about any public GitHub repository in plain English.**

RepoMind is a Retrieval-Augmented Generation (RAG) chatbot. Point it at a GitHub
repo, and it will clone the code, embed it into a vector database, and let you
ask natural-language questions like *"where is the authentication logic?"* or
*"what does the `parse_config` function do?"* — with answers grounded in the
actual source, citing the exact file and line.

100% free to run: local embeddings (sentence-transformers), local vector store
(ChromaDB), and Groq's free-tier LLM API for generation.

---

## How it works

1. **Clone** — the target repo is cloned locally (shallow clone).
2. **Chunk** — code/doc files are split into overlapping ~60-line chunks.
3. **Embed** — each chunk is turned into a vector using a local embedding model
   (`all-MiniLM-L6-v2`).
4. **Store** — vectors are saved in a local ChromaDB collection.
5. **Retrieve** — when you ask a question, it's embedded too, and the most
   similar chunks are pulled from ChromaDB.
6. **Generate** — those chunks + your question are sent to a free Groq LLM,
   which answers using only that retrieved context.

---

## Step-by-step setup

### 1. Prerequisites
- Python 3.9+
- Git installed
- A free Groq API key: https://console.groq.com/keys

### 2. Install dependencies
```bash
cd RepoMind
pip install -r requirements.txt
```

### 3. Add your API key
Copy the example env file and fill in your key:
```bash
cp .env.example .env
# then edit .env and paste your GROQ_API_KEY
```

### 4. Run the app
```bash
streamlit run app.py
```
This opens the app in your browser at `http://localhost:8501`.

---

## Usage (input → output)

| Step | Input | Output |
|---|---|---|
| Index a repo | Paste a GitHub URL (e.g. `https://github.com/pallets/flask`) into the sidebar and click **Index Repo** | Status message confirming how many chunks were indexed |
| Ask a question | Type a question in the chat box, e.g. *"How does routing work in this repo?"* | An AI-generated answer, plus an expandable "Sources" list showing which files/lines it came from |

You can also run indexing manually from the command line without the UI:
```bash
python ingest.py https://github.com/user/repo
```

---

## Project structure
```
RepoMind/
├── app.py            # Streamlit chat UI
├── ingest.py          # Clone → chunk → embed → store pipeline
├── requirements.txt
├── .env.example
└── README.md
```

---

## Deploying for free
1. Push this project to a public GitHub repo.
2. Go to https://streamlit.io/cloud and connect your repo.
3. Add `GROQ_API_KEY` as a secret in the Streamlit Cloud dashboard.
4. Deploy — you'll get a live shareable URL.

---

## Why this project
Most people can *use* an AI chatbot. This project shows you can *build* the
retrieval pipeline underneath one — chunking, embeddings, vector search, and
grounded generation — the same architecture powering most real-world AI
products today.
