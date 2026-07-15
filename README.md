<div align="center">

🧠 RepoMind

Ask questions about any public GitHub repository in plain English.

**RepoMind** is a Retrieval-Augmented Generation (RAG) chatbot for codebases. Point it at
any public GitHub repository and it clones the code, intelligently chunks it, embeds it
into a local vector database, and lets you have a real conversation about it —
*"where is the authentication logic?"*, *"what does `parse_config` do?"*, *"can you show
an example?"* — with every answer citing the exact file, line number, and source snippet
it came from.

It's designed to run **entirely on free infrastructure**: local embeddings
(`sentence-transformers`), a local vector store (ChromaDB), and Groq's free-tier LLM API
for generation — no paid API keys, no cloud vector DB, no GPU required.

---

## ✨ Features

| | |
|---|---|
| 🔍 **Grounded, cited answers** | Every response points to the exact file(s) and line(s) it's based on — no hallucinated code |
| 🧩 **AST-aware chunking** | Python files are split along function/class boundaries, not arbitrary line counts, so a chunk is never a function cut in half |
| 📝 **Auto-generated repo summary** | Get a plain-English "what this project does" summary the moment indexing finishes |
| 💬 **Conversational memory** | Follow-ups like *"can you show an example?"* are rewritten into self-contained queries using recent chat history |
| 📄 **Inspectable retrieval** | Expand any answer to see the actual retrieved code, not just a file reference |
| 📊 **Transparent indexing** | Live file-type breakdown and truncation warnings for very large repos |
| 💾 **Exportable sessions** | Download a full Q&A transcript as a `.txt` file |
| 🔑 **Zero-friction API keys** | Reads `GROQ_API_KEY` from Streamlit secrets, `.env`, or a sidebar prompt — whichever is available |
| 💸 **$0 to run** | Local embeddings + local vector store + Groq free tier |

---

## 🏗 How it works

```
GitHub URL
    │
    ▼
①  Clone            shallow clone of the target repo
    │
    ▼
②  Chunk            Python → AST-based split (function/class boundaries)
    │                other files → overlapping ~60-line chunks
    ▼
③  Embed            all-MiniLM-L6-v2 (local, no API calls)
    │
    ▼
④  Store            persisted in a local ChromaDB collection
    │
    ▼
⑤  Retrieve         question → rewritten with chat history → embedded → top-k chunks
    │
    ▼
⑥  Generate         chunks + question + history → Groq LLM → cited, grounded answer
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Vector store | [ChromaDB](https://www.trychroma.com/) (local, persistent) |
| LLM inference | [Groq](https://groq.com/) (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) |
| Repo handling | [GitPython](https://gitpython.readthedocs.io/) |

---

## 🚀 Quickstart

### Prerequisites
- Python 3.9+
- Git
- A free Groq API key → [console.groq.com/keys](https://console.groq.com/keys)

### Install & run
```bash
git clone https://github.com/<your-username>/RepoMind.git
cd RepoMind
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your GROQ_API_KEY

streamlit run app.py
```
The app opens at `http://localhost:8501`. No key handy yet? You can also paste it
directly into the sidebar at runtime.

---

## 📖 Usage

| Step | You do | RepoMind gives you |
|---|---|---|
| **Index a repo** | Paste a GitHub URL (e.g. `https://github.com/pallets/flask`) → click **Index Repo** | Chunk/file counts, a file-type breakdown, a truncation notice on very large repos, and an auto-generated project summary |
| **Ask a question** | Type it in the chat box | A cited, grounded answer + source chips + an expandable panel with the actual retrieved code |
| **Ask a follow-up** | *"can you show an example?"* | Automatically resolved against recent chat context before retrieval |
| **Reset** | Click **Clear chat** | Fresh conversation, same indexed repo |
| **Save your work** | Click **Download conversation** | A `.txt` transcript of the summary + full Q&A history |

Prefer the CLI? You can index a repo without the UI:
```bash
python ingest.py https://github.com/user/repo
```

---

📁 Project Structure

RepoMind/
├── app.py             # Streamlit chat UI — retrieval, generation, conversation state
├── ingest.py           # Clone → chunk (AST-aware) → embed → store pipeline
├── requirements.txt
├── .env.example
└── README.md

---

## ☁️ Deployment

Deploy your own instance for free on Streamlit Community Cloud.

**1. Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/RepoMind.git
git push -u origin main
```

Pushing later changes:
```bash
git add .
git commit -m "Update app.py and ingest.py"
git push
```

> ⚠️ Add `.env` to `.gitignore` **before your first commit** so your Groq API key never
> ends up in a public repo.

**2. Deploy on Streamlit Cloud**
1. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and connect your GitHub repo.
2. Add `GROQ_API_KEY` as a secret in the app's settings.
3. Deploy — you'll get a live, shareable URL.

---

## 🗺 Roadmap

- [ ] Support for private repos (via GitHub token)
- [ ] Multi-repo comparison ("how does auth differ between these two projects?")
- [ ] Persistent index cache across sessions (skip re-cloning unchanged repos)
- [ ] Support for additional LLM providers

*(Have an idea? Open an issue or a PR.)*

---

## 🤝 Contributing

Contributions are welcome. Fork the repo, create a feature branch, and open a pull
request:
```bash
git checkout -b feature/your-feature-name
git commit -m "Add: your feature"
git push origin feature/your-feature-name
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

## 💡 Why this project

Most people can *use* an AI chatbot. This project demonstrates the ability to *build*
the retrieval pipeline underneath one — AST-aware chunking, local embeddings, vector
search, query rewriting for multi-turn conversations, and grounded generation — the same
core architecture behind most production RAG systems today, running entirely on free
infrastructure.

<div align="center">

Built with 🧠 by [Your Name](https://github.com/<your-username>)

</div>
