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

**RepoMind** is a Retrieval-Augmented Generation (RAG) chatbot for codebases — and a
genuinely conversational assistant, not just a lookup tool. Point it at any public
GitHub repository and it clones the code, intelligently chunks it, embeds it into a
local vector database, and lets you have a real conversation about it — *"where is the
authentication logic?"*, *"what does `parse_config` do?"*, *"any suggestions to improve
this repo?"* — with grounded answers citing the exact file, line number, and source
snippet they came from. It also handles ordinary conversation naturally (greetings,
thanks, small talk) and answers general programming questions even when they're not
about the indexed repo at all. Ask it to **diagram the architecture** and it generates
a real, downloadable module dependency graph on the spot, built from the repo's own
import relationships.

It's designed to run **entirely on free infrastructure**: local embeddings
(`sentence-transformers`), a local vector store (ChromaDB), Groq's free-tier LLM API
for generation, and a client-side JS diagram renderer (Mermaid.js) — no paid API keys,
no cloud vector DB, no GPU, and no system-level packages like Graphviz required.

---

## ✨ Features

| | |
|---|---|
| 🔍 **Grounded, cited answers** | Every repo-specific response points to the exact file(s) and line(s) it's based on — no hallucinated code |
| 💬 **Genuinely conversational** | Casual messages ("ok", "thanks", "hi") get instant, natural replies — never a robotic "couldn't find that in context" |
| 🧠 **General-purpose, not repo-locked** | Ask something unrelated to the indexed repo and RepoMind still answers helpfully using its own knowledge, instead of refusing |
| 💡 **Real suggestions on request** | Ask "any suggestions?" or "what addons could I add?" and get genuine, thoughtful ideas — not a canned refusal |
| 🧩 **AST-aware chunking** | Python files are split along function/class boundaries, not arbitrary line counts, so a chunk is never a function cut in half |
| 📝 **Auto-generated repo summary** | Get a plain-English "what this project does" summary the moment indexing finishes |
| 📊 **Auto-generated architecture diagrams** | Ask *"show me the architecture of this repo"* to see a visual module dependency graph, grouped by folder, built from real import relationships — no Graphviz or system installs needed |
| 🔄 **Conversational memory** | Follow-ups like *"can you show an example?"* are rewritten into self-contained queries using recent chat history before retrieval |
| 📄 **Inspectable retrieval** | Expand any answer to see the actual retrieved code, not just a file reference |
| 📊 **Transparent indexing** | Live file-type breakdown and truncation warnings for very large repos |
| 💾 **Real combined export** | Download the full conversation as a single HTML file with actual embedded diagram images (not just text) — each diagram includes its own download and copy-to-clipboard icons |
| 🔑 **Zero-friction API keys** | Reads `GROQ_API_KEY` from Streamlit secrets, `.env`, or a collapsed sidebar field — indexing and diagrams work even *without* a key |
| ⚡ **Fast by design** | Chat runs on Groq's `llama-3.1-8b-instant` for low-latency, real-time-feeling responses |
| 💸 **$0 to run** | Local embeddings + local vector store + Groq free tier + client-side diagram rendering |

---

## 🏗 How it works

1. **Clone** — shallow clone of the target repo from the GitHub URL you provide.
2. **Chunk** — Python files are split by AST along function/class boundaries; all
   other supported files (`.js`, `.md`, `.json`, `.yaml`, etc.) are split into
   overlapping ~60-line chunks.
3. **Embed** — each chunk is embedded locally with `all-MiniLM-L6-v2` (no API calls).
4. **Store** — embeddings are persisted in a local ChromaDB collection; the cloned
   repo's temp directory is deleted right after, since everything needed later
   already lives in ChromaDB.
5. **Route** — each message is checked first: casual conversation gets an instant
   canned reply, a diagram request routes to the diagram builder, and everything
   else proceeds to retrieval.
6. **Retrieve** — your question is rewritten using recent chat history, embedded, and
   used to pull the top-k most relevant chunks.
7. **Generate** — those chunks, your question, and the conversation history are sent
   to Groq's `llama-3.1-8b-instant`, which returns a warm, cited answer when the repo
   covers it — and a genuinely helpful general answer or suggestion when it doesn't.
8. **Diagram (on request)** — skips the LLM entirely: RepoMind re-reads the
   already-stored chunks straight from ChromaDB, parses each Python file's imports,
   and renders a Mermaid.js flowchart of how the repo's own modules depend on each
   other.

---

## 📊 Architecture Diagrams

Ask RepoMind to **"show me the architecture of this repo"** and it builds a live module
dependency graph directly from the repo's own code — no separate crawl of the
filesystem, and no dependency on Graphviz or any other system-level graphing library,
which makes it work identically on free hosting with zero extra config.

**How it's built:**
- Import relationships are extracted from the same chunk text already stored in
  ChromaDB during indexing — the diagram works even though the original cloned repo
  is deleted from disk right after ingestion. Chunks are re-sorted by their original
  line position before parsing, so the reconstructed source is syntactically valid.
- Only **internal** imports (between the repo's own files) are shown; standard-library
  and third-party imports (`os`, `requests`, `streamlit`, etc.) are filtered out so the
  diagram reflects actual architecture, not a list of dependencies.
- Files are grouped into folder-based subgraphs so architectural boundaries are visible
  at a glance, and `examples/`, `tests/`, `docs/`, and similar non-core folders are
  excluded by default to keep the focus on core source code.
- Large repos are automatically trimmed to the ~35 most-connected files, since an
  unfiltered 100+ node graph is unreadable — isolated single-file utilities are dropped
  first as the least architecturally informative.
- Works correctly on both Windows and Linux-indexed repos, regardless of which OS's
  path separator (`\` vs `/`) was used when the repo was originally indexed.
- Rendering happens client-side via [Mermaid.js](https://mermaid.js.org/) loaded from
  a CDN for interactive in-app viewing, with straight-line edges and generous spacing
  for readability.
- For exports, a real PNG is independently generated **server-side** via the free
  [mermaid.ink](https://mermaid.ink) rendering API, so the diagram can be embedded as
  an actual image in downloaded conversations — not just left as code.
- Not every repo has a meaningful diagram: projects with little to no Python source
  (workflow/config-based tools, pure documentation repos, etc.) get a clear, honest
  message instead of an empty or misleading graph.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Vector store | [ChromaDB](https://www.trychroma.com/) (local, persistent) |
| LLM inference | [Groq](https://groq.com/) (`llama-3.1-8b-instant` for fast, real-time chat) |
| Architecture diagrams | [Mermaid.js](https://mermaid.js.org/) (client-side, CDN-loaded) + [mermaid.ink](https://mermaid.ink) (server-side PNG export) |
| Repo handling | [GitPython](https://gitpython.readthedocs.io/) |

---

## 🚀 Quickstart

### Prerequisites
- Python 3.9+
- Git
- A free Groq API key → [console.groq.com/keys](https://console.groq.com/keys) *(optional — indexing and diagrams work without one; only chat Q&A needs it)*

### Install & run
```bash
git clone https://github.com/gorigesravani288-cmd/RepoMind.git
cd RepoMind
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your GROQ_API_KEY
```

```bash
streamlit run app.py
```
If `streamlit` isn't recognized as a command on Windows, run it via:
```bash
python -m streamlit run app.py
```
The app opens at `http://localhost:8501`. No key handy yet? Paste it directly into the
collapsed **"Add API key"** section in the sidebar at runtime.

---

## 📖 Usage

| Step | You do | RepoMind gives you |
|---|---|---|
| **Index a repo** | Paste a GitHub URL (e.g. `https://github.com/pallets/flask`) → click **Index Repo** | Chunk/file counts, a file-type breakdown, a truncation notice on very large repos, and an auto-generated project summary |
| **Just chat** | Type "hi" or "thanks" | An instant, natural reply — no API call, no delay |
| **Ask about the repo** | Type a real question in the chat box | A cited, grounded answer + source chips + an expandable panel with the actual retrieved code |
| **Ask anything else** | Ask a general question, or for suggestions/addon ideas | A genuinely helpful answer using RepoMind's own knowledge, clearly distinguished from repo-grounded facts |
| **Visualize the codebase** | Ask *"show me the architecture of this repo"* | A Mermaid flowchart showing internal module dependencies, grouped by folder |
| **Ask a follow-up** | *"can you show an example?"* | Automatically resolved against recent chat context before retrieval |
| **Reset** | Click **Clear chat** | Fresh conversation, same indexed repo |
| **Save your work** | Click **Download conversation** | One combined `.html` file — full Q&A text plus any diagrams as real embedded images, each with its own download/copy icons |

Prefer the CLI? You can index a repo without the UI:
```bash
python ingest.py https://github.com/user/repo
```

---

## 📁 Project Structure

- **`app.py`** — Streamlit chat UI: routing (casual chat / diagram / RAG), retrieval,
  generation, conversation state, and the combined HTML export
- **`ingest.py`** — Clone → chunk (AST-aware) → embed → store pipeline
- **`diagram.py`** — Rebuilds internal import relationships from stored chunks, renders
  them as a Mermaid.js diagram, and generates a server-side PNG for export
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
git remote add origin https://github.com/gorigesravani288-cmd/RepoMind.git
git push -u origin main
```

Pushing later changes:
```bash
git add .
git commit -m "Update app.py and diagram.py"
git push
```

> ⚠️ Add `.env` to `.gitignore` **before your first commit** so your Groq API key never
> ends up in a public repo.

**2. Deploy on Streamlit Cloud**
1. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and connect your GitHub repo.
2. Add `GROQ_API_KEY` as a secret in the app's settings — visitors will never see this
   key or be asked for one; the sidebar just shows a small "🔑 API key loaded ✅" note.
3. Deploy — you'll get a live, shareable URL. The diagram feature needs no extra
   configuration, since Mermaid.js loads from a CDN rather than requiring a system
   package install.

---

## 🗺 Roadmap

- [ ] Clickable example question chips for first-time visitors
- [ ] Support for private repos (via GitHub token)
- [ ] Multi-repo comparison ("how does auth differ between these two projects?")
- [ ] Persistent index cache across sessions (skip re-cloning unchanged repos)
- [ ] Diagram support for non-Python languages (JS/TS import graphs)
- [ ] Optional voice input/output

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
search, query rewriting for multi-turn conversations, grounded generation with graceful
fallback to general knowledge, and a zero-dependency architecture-visualization feature
built entirely from data already captured during indexing — the same core architecture
behind most production RAG systems today, running entirely on free infrastructure.

<div align="center">

Built with 🧠 by [Gorige Sravani](https://github.com/gorigesravani288-cmd)

</div>
