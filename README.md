# 🧠 RepoMind

**Ask questions about any public GitHub repository in plain English.**

RepoMind is a Retrieval-Augmented Generation (RAG) chatbot. Point it at a GitHub
repo, and it will clone the code, embed it into a vector database, and let you
ask natural-language questions like *"where is the authentication logic?"* or
*"what does the `parse_config` function do?"* — with answers grounded in the
actual source, citing the exact file and line, plus the actual retrieved code.

100% free to run: local embeddings (sentence-transformers), local vector store
(ChromaDB), and Groq's free-tier LLM API for generation.

---

## Features

- 🔍 **Grounded Q&A** — every answer cites the file(s) and line number(s) it came from
- 🧩 **Smart chunking** — Python files are split along function/class boundaries (AST-based) instead of raw line counts, so a chunk is never a function cut in half; other file types fall back to overlapping line-based chunks
- 📝 **Auto repo summary** — right after indexing, RepoMind generates a short plain-English summary of what the project does and how it's structured
- 💬 **Conversation memory** — follow-up questions like *"can you show an example?"* are automatically rewritten into a self-contained search query using recent chat history
- 📄 **View retrieved code** — expand any answer to see the exact source snippet(s) used to generate it, not just the file/line reference
- 📊 **Indexing stats** — file-type breakdown and a truncation warning if a repo is too large to index in full
- 💾 **Download conversation** — export your full Q&A session (including the repo summary) as a `.txt` transcript
- 🔑 **Flexible API key handling** — reads `GROQ_API_KEY` from Streamlit Cloud secrets, a local `.env`, or a manual sidebar input, in that order

---

## How it works

1. **Clone** — the target repo is cloned locally (shallow clone).
2. **Chunk** — Python files are split by function/class using the `ast` module
   (large functions/classes are further sub-split); all other supported file
   types are split into overlapping ~60-line chunks.
3. **Embed** — each chunk is turned into a vector using a local embedding model
   (`all-MiniLM-L6-v2`).
4. **Store** — vectors are saved in a local ChromaDB collection (capped at a
   max chunk count so free-tier CPU embedding stays fast even on large repos).
5. **Retrieve** — when you ask a question, it's rewritten using recent chat
   history (if any), embedded, and the most similar chunks are pulled from ChromaDB.
6. **Generate** — those chunks + your question + recent conversation context
   are sent to a free Groq LLM, which answers using only that retrieved context.

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
(You can also paste the key directly into the sidebar at runtime if you'd
rather not use a `.env` file — RepoMind checks Streamlit secrets, then
`.env`, then the sidebar input.)

### 4. Run the app
```bash
streamlit run app.py
```
This opens the app in your browser at `http://localhost:8501`.

---

## Usage (input → output)

| Step | Input | Output |
|---|---|---|
| Index a repo | Paste a GitHub URL (e.g. `https://github.com/pallets/flask`) into the sidebar and click **Index Repo** | Chunk/file count, file-type breakdown, a truncation notice if the repo is very large, and an auto-generated repo summary |
| Ask a question | Type a question in the chat box, e.g. *"How does routing work in this repo?"* | An AI-generated answer (aware of the recent conversation), source chips for each file/line used, and an expandable panel showing the actual retrieved code |
| Clear / restart | Click **Clear chat** in the sidebar | Resets the conversation without re-indexing the repo |
| Save a session | Click **Download conversation** in the sidebar | A `.txt` transcript of the summary + full Q&A history |

You can also run indexing manually from the command line without the UI:
```bash
python ingest.py https://github.com/user/repo
```

---

## Project structure
```
RepoMind/
├── app.py            # Streamlit chat UI
├── ingest.py          # Clone → chunk (AST-aware) → embed → store pipeline
├── requirements.txt
├── .env.example
└── README.md
```

---

## Deploying for free

### 1. Push the project to a public GitHub repo
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/RepoMind.git
git push -u origin main
```

If you're updating an already-connected repo after making local changes:
```bash
git add .
git commit -m "Update app.py and ingest.py"
git push
```

> Make sure `.env` is listed in `.gitignore` before your first commit so your
> Groq API key is never pushed to a public repo.

### 2. Deploy on Streamlit Cloud
1. Go to https://streamlit.io/cloud and connect your GitHub repo.
2. Add `GROQ_API_KEY` as a secret in the Streamlit Cloud dashboard.
3. Deploy — you'll get a live shareable URL.

---

## Why this project
Most people can *use* an AI chatbot. This project shows you can *build* the
retrieval pipeline underneath one — chunking, embeddings, vector search, and
grounded generation — the same architecture powering most real-world AI
products today.
## Why this project
Most people can *use* an AI chatbot. This project shows you can *build* the
retrieval pipeline underneath one — chunking, embeddings, vector search, and
grounded generation — the same architecture powering most real-world AI
products today.
