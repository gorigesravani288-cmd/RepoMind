"""
app.py
------
Streamlit front-end for RepoMind.

INPUT (from the user, via the UI):
  1. A GitHub repo URL to index (one-time, or whenever they switch repos)
  2. Natural-language questions about that repo

OUTPUT (shown in the UI):
  - Indexing status/progress
  - AI-generated answers, each with the source file(s) and line numbers
    the answer was grounded in

EXECUTION:
  streamlit run app.py
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

from ingest import ingest_repo, DB_PATH, COLLECTION_NAME

load_dotenv()

st.set_page_config(page_title="RepoMind", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Overall page */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    /* Header */
    .rm-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.1rem;
    }
    .rm-header .emoji {
        font-size: 2.2rem;
    }
    .rm-header h1 {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #FF4B4B, #FF8A5B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .rm-subtitle {
        color: #8a8f98;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.15);
    }
    .rm-step-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #FF6B4A;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }
    .rm-repo-badge {
        background: rgba(255, 107, 74, 0.08);
        border: 1px solid rgba(255, 107, 74, 0.25);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        font-size: 0.82rem;
        line-height: 1.4;
        word-break: break-all;
    }
    .rm-empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #8a8f98;
    }
    .rm-empty-state .icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    /* Source chips */
    .rm-source-chip {
        display: inline-block;
        background: rgba(120, 120, 120, 0.12);
        border-radius: 6px;
        padding: 2px 8px;
        margin: 2px 4px 2px 0;
        font-family: monospace;
        font-size: 0.78rem;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="rm-header">
    <span class="emoji">🧠</span>
    <h1>RepoMind</h1>
</div>
<div class="rm-subtitle">Ask questions about any public GitHub repository in plain English — grounded in the real source code.</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: index a repo + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="rm-step-label">Step 1 · Index a repo</div>', unsafe_allow_html=True)
    repo_url = st.text_input(
        "GitHub repo URL",
        placeholder="https://github.com/user/repo",
        label_visibility="collapsed",
    )
    index_clicked = st.button("🔍  Index Repo", type="primary", use_container_width=True)

    if index_clicked:
        if not repo_url.strip():
            st.warning("Enter a repo URL first.")
        else:
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                with st.spinner("Cloning, chunking, and embedding the repo..."):
                    try:
                        n_chunks = ingest_repo(repo_url.strip())
                        st.session_state["indexed_repo"] = repo_url.strip()
                        st.session_state["messages"] = []
                        st.session_state["last_index_count"] = n_chunks
                    except Exception as e:
                        st.error(f"Failed to index repo: {e}")
            if "indexed_repo" in st.session_state and st.session_state.get("indexed_repo") == repo_url.strip():
                st.toast(f"Indexed {st.session_state.get('last_index_count', 0)} chunks ✅", icon="🎉")

    if "indexed_repo" in st.session_state:
        st.markdown(
            f'<div class="rm-repo-badge">📦 <b>Indexed:</b><br>{st.session_state["indexed_repo"]}'
            f'<br><span style="color:#8a8f98;">{st.session_state.get("last_index_count", "?")} chunks</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("🗑️  Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    st.markdown('<div class="rm-step-label">Step 2 · API key</div>', unsafe_allow_html=True)
    # Check Streamlit Cloud secrets first, then local .env, then manual input
    groq_key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
    if groq_key or os.getenv("GROQ_API_KEY"):
        groq_key = groq_key or os.getenv("GROQ_API_KEY")
        st.markdown("🔑 Key loaded from secrets ✅")
    else:
        groq_key = st.text_input("GROQ_API_KEY", type="password", placeholder="gsk_...")

    st.markdown("---")
    st.caption("Built with Streamlit · ChromaDB · Groq · Sentence-Transformers")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query: str, k: int = 5):
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    embedder = get_embedder()
    query_emb = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=k)
    hits = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        hits.append({"text": doc, "file": meta["file"], "start_line": meta["start_line"]})
    return hits


def ask_llm(question: str, context_chunks, api_key: str) -> str:
    context_str = "\n\n".join(
        f"[{c['file']} : line {c['start_line']}]\n{c['text']}" for c in context_chunks
    )
    prompt = f"""You are RepoMind, an assistant that answers questions about a codebase
using ONLY the provided context.

Rules:
- Answer the question directly. Do NOT restate the question or describe what you're about to do.
- Do NOT add filler commentary like "this indicates its importance" or "this suggests" unless it's a genuine, specific insight backed by the context.
- If the context has enough information, give a clear, specific, useful answer in 2-5 sentences.
- If the context does NOT have enough information to answer well, say so in one honest sentence, and briefly suggest what the person could ask instead (e.g. a more specific file or topic).
- Always mention which file(s) your answer is based on, but only once, naturally in the sentence — not as a separate disclaimer.

Context:
{context_str}

Question: {question}

Answer:"""

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "indexed_repo" not in st.session_state:
    st.markdown("""
    <div class="rm-empty-state">
        <div class="icon">📂</div>
        <b>No repo indexed yet</b><br>
        Paste a public GitHub URL in the sidebar and click <b>Index Repo</b> to get started.
    </div>
    """, unsafe_allow_html=True)
else:
    if not st.session_state["messages"]:
        st.info(f"💬 Ready! Ask anything about **{st.session_state['indexed_repo'].split('/')[-1]}**.")

for msg in st.session_state["messages"]:
    avatar = "🙋" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            chips = "".join(
                f'<span class="rm-source-chip">{s["file"]}:{s["start_line"]}</span>'
                for s in msg["sources"]
            )
            st.markdown(chips, unsafe_allow_html=True)

question = st.chat_input("Ask something about the indexed repo...")

if question:
    if "indexed_repo" not in st.session_state:
        st.warning("Index a repo first using the sidebar.")
    elif not groq_key:
        st.warning("Add your GROQ_API_KEY in the sidebar (or .env file).")
    else:
        st.session_state["messages"].append({"role": "user", "content": question, "sources": None})
        with st.chat_message("user", avatar="🙋"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Retrieving context and generating answer..."):
                hits = retrieve(question)
                answer = ask_llm(question, hits, groq_key)
                st.markdown(answer)
                chips = "".join(
                    f'<span class="rm-source-chip">{h["file"]}:{h["start_line"]}</span>' for h in hits
                )
                st.markdown(chips, unsafe_allow_html=True)
        st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": hits})
