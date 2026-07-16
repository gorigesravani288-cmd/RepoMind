<div align="center">

# 🧠 RepoMind

**Ask any public GitHub repository questions in plain English — and get answers grounded in the actual source code.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/vector%20store-ChromaDB-6E56CF)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20(free%20tier)-F55036)](https://groq.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

[Features](#-features) • [How it works](#-how-it-works) • [Quickstart](#-quickstart) • [Usage](#-usage) • [Architecture Diagrams](#-architecture-diagrams) • [Deployment](#-deployment) • [Roadmap](#-roadmap)

</div>

---

## Overview

**RepoMind** is a Retrieval-Augmented Generation (RAG) chatbot for codebases. Point it at
any public GitHub repository and it clones the code, intelligently chunks it, embeds it
into a local vector database, and lets you have a real conversation about it —
*"where is the authentication logic?"*, *"what does `parse_config` do?"*, *"can you show
an example?"* — with every answer citing the exact file, line number, and source snippet
it came from. Ask it to **diagram the architecture** and it generates a visual module
dependency graph on the spot, built from the repo's own import relationships.

It's designed to run **entirely on free infrastructure**: local embeddings
(`sentence-transformers`), a local vector store (ChromaDB), Groq's free-tier LLM API
for generation, and a client-side JS diagram renderer (Mermaid.js) — no paid API keys,
no cloud vector DB, no GPU, and no system-level packages like Graphviz required.

---

## ✨ Features

| | |
|---|---|
| 🔍 **Grounded, cited answers** | Every response points to the exact file(s) and line(s) it's based on — no hallucinated code |
| 🧩 **AST-aware chunking** | Python files are split along function/class boundaries, not arbitrary line counts, so a chunk is never a function cut in half |
| 📝 **Auto-generated repo summary** | Get a plain-English "what this project does" summary the moment indexing finishes |
| 📊 **Auto-generated architecture diagrams** | Ask *"diagram the architecture"* to see a visual module dependency graph, grouped by folder, built from real import relationships — no Graphviz or system installs needed |
| 💬 **Example question chips** | First-time visitors see one-click example prompts instead of a blank chat box, so it's obvious what to try |
| 💬 **Conversational memory** | Follow-ups like *"can you show an example?"* are rewritten into self-contained queries using recent chat history |
| 📄 **Inspectable retrieval** | Expand any answer to see the actual retrieved code, not just a file reference |
| 📊 **Transparent indexing** | Live file-type breakdown and truncation warnings for very large repos |
| 💾 **Exportable sessions** | Download a full Q&A transcript as a `.txt` file |
| 🔑 **Zero-friction API keys** | Reads `GROQ_API_KEY` from Streamlit secrets, `.env`, or a sidebar prompt — whichever is available |
| 💸 **$0 to run** | Local embeddings + local vector store + Groq free tier + client-side diagram rendering |

---

## 🏗 How it works

1. **Clone** — shallow clone of the target repo from the GitHub URL you provide.
2. **Chunk** — Python files are split by AST along function/class boundaries; all
   other supported files are split into overlapping ~60-line chunks.
3. **Embed** — each chunk is embedded locally with `all-MiniLM-L6-v2` (no API calls).
4. **Store** — embeddings are persisted in a local ChromaDB collection; the cloned
   repo's temp directory is deleted right after, since everything needed later
   already lives in ChromaDB.
5. **Retrieve** — your question is rewritten using recent chat history, embedded, and
   used to pull the top-k most relevant chunks.
6. **Generate** — those chunks, your question, and the conversation history are sent
   to a Groq LLM, which returns a cited, grounded answer.
7. **Diagram (on request)** — asking to "diagram the architecture" skips the LLM
   entirely: RepoMind re-reads the already-stored chunks straight from ChromaDB,
   parses each Python file's imports, and renders a Mermaid.js flowchart of how
   the repo's own modules depend on each other.

---

## 📊 Architecture Diagrams

Ask RepoMind to **"diagram the architecture"** (or just click the example chip) and it
builds a live module dependency graph directly from the repo's own code — no separate
crawl of the filesystem, and no dependency on Graphviz or any other system-level
graphing library, which makes it work identically on free hosting with zero extra config.

**How it's built:**
- Import relationships are extracted from the same chunk text already stored in
  ChromaDB during indexing — the diagram works even though the original cloned repo
  is deleted from disk right after ingestion.
- Only **internal** imports (between the repo's own files) are shown; standard-library
  and third-party imports (`os`, `requests`, `streamlit`, etc.) are filtered out so the
  diagram reflects actual architecture, not a list of dependencies.
- Files are grouped into folder-based subgraphs so architectural boundaries are visible
  at a glance, and `examples/`, `tests/`, and `docs/` are excluded by default to keep
  the focus on core source code.
- Large repos are automatically trimmed to the ~35 most-connected files, since an
  unfiltered 100+ node graph is unreadable — isolated single-file utilities are dropped
  first as the least architecturally informative.
- Rendering happens entirely client-side via [Mermaid.js](https://mermaid.js.org/)
  loaded from a CDN, so it needs zero system installs and deploys identically on
  Streamlit Community Cloud.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Vector store | [ChromaDB](https://www.trychroma.com/) (local, persistent) |
| LLM inference | [Groq](https://groq.com/) (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) |
| Architecture diagrams | [Mermaid.js](https://mermaid.js.org/) (client-side, CDN-loaded, no system deps) |
| Repo handling | [GitPython](https://gitpython.readthedocs.io/) |

---

## 🚀 Quickstart

### Prerequisites
- Python 3.9+
- Git
- A free Groq API key → [console.groq.com/keys](https://console.groq.com/keys)

### Install & run
```bash
git clone https://github.com/<your-gorigesravani288-cmd>/RepoMind.git
cd RepoMind
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your GROQ_API_KEY

streamlit run app.py
```
If `streamlit` isn't recognized as a command on Windows, run it via:
```bash
python -m streamlit run app.py
```
The app opens at `http://localhost:8501`. No key handy yet? You can also paste it
directly into the sidebar at runtime — note that indexing and diagrams work even
*without* a Groq key, since only Q&A and the auto-summary need it.

---

## 📖 Usage

| Step | You do | RepoMind gives you |
|---|---|---|
| **Index a repo** | Paste a GitHub URL (e.g. `https://github.com/pallets/flask`) → click **Index Repo** | Chunk/file counts, a file-type breakdown, a truncation notice on very large repos, and an auto-generated project summary |
| **Get started fast** | Click one of the example question chips shown on first load | An instant answer with no need to think of a question yourself |
| **Ask a question** | Type it in the chat box | A cited, grounded answer + source chips + an expandable panel with the actual retrieved code |
| **Visualize the codebase** | Ask *"diagram the architecture"* | A Mermaid flowchart showing internal module dependencies, grouped by folder |
| **Ask a follow-up** | *"can you show an example?"* | Automatically resolved against recent chat context before retrieval |
| **Reset** | Click **Clear chat** | Fresh conversation, same indexed repo |
| **Save your work** | Click **Download conversation** | A `.txt` transcript of the summary + full Q&A history |

Prefer the CLI? You can index a repo without the UI:
```bash
python ingest.py https://github.com/user/repo
```

---

## 📁 Project Structure

- **`app.py`** — Streamlit chat UI: retrieval, generation, conversation state, example question chips, diagram request routing
- **`ingest.py`** — Clone → chunk (AST-aware) → embed → store pipeline
- **`diagram.py`** — Rebuilds internal import relationships from stored chunks and renders them as a Mermaid.js diagram
- **`requirements.txt`** — Python dependencies
- **`.env.example`** — template for your `GROQ_API_KEY`
- **`README.md`** — this file

---

## ☁️ Deployment

Deploy your own instance for free on Streamlit Community Cloud.

**1. Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-gorigesravani288-cmd>/RepoMind.git
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
3. Deploy — you'll get a live, shareable URL. The diagram feature needs no extra
   configuration, since Mermaid.js loads from a CDN rather than requiring a system
   package install.

---

## 🗺 Roadmap

- [ ] Support for private repos (via GitHub token)
- [ ] Multi-repo comparison ("how does auth differ between these two projects?")
- [ ] Persistent index cache across sessions (skip re-cloning unchanged repos)
- [ ] Support for additional LLM providers
- [ ] Diagram support for non-Python languages (JS/TS import graphs)

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
search, query rewriting for multi-turn conversations, grounded generation, and a
zero-dependency architecture-visualization feature built entirely from data already
captured during indexing — the same core architecture behind most production RAG
systems today, running entirely on free infrastructure.

<div align="center">

Built with 🧠 by [Gorige Sravani](https://github.com/<your-gorigesravani288-cmd>)

</div>
