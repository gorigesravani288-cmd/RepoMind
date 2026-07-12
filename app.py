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
import streamlit as st
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

from ingest import ingest_repo, DB_PATH, COLLECTION_NAME

load_dotenv()

st.set_page_config(page_title="RepoMind", page_icon="🧠", layout="wide")
st.title("🧠 RepoMind")
st.caption("Ask questions about any public GitHub repository in plain English.")

# ---------- Sidebar: index a repo ----------
with st.sidebar:
    st.header("1. Index a repo")
    repo_url = st.text_input("GitHub repo URL", placeholder="https://github.com/user/repo")
    if st.button("Index Repo", type="primary"):
        if not repo_url.strip():
            st.warning("Enter a repo URL first.")
        else:
            with st.spinner("Cloning, chunking, and embedding the repo... this can take a minute."):
                try:
                    n_chunks = ingest_repo(repo_url.strip())
                    st.session_state["indexed_repo"] = repo_url.strip()
                    st.session_state["messages"] = []
                    st.success(f"Indexed {n_chunks} chunks from {repo_url}")
                except Exception as e:
                    st.error(f"Failed to index repo: {e}")

    if "indexed_repo" in st.session_state:
        st.info(f"Currently indexed:\n{st.session_state['indexed_repo']}")

    st.header("2. API key")
    groq_key = os.getenv("GROQ_API_KEY") or st.text_input("GROQ_API_KEY", type="password")

# ---------- Helpers ----------
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
using ONLY the provided context. If the answer isn't in the context, say so honestly.
Always mention which file(s) your answer is based on.

Context:
{context_str}

Question: {question}

Answer:"""

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about the indexed repo...")

if question:
    if "indexed_repo" not in st.session_state:
        st.warning("Index a repo first using the sidebar.")
    elif not groq_key:
        st.warning("Add your GROQ_API_KEY in the sidebar (or .env file).")
    else:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                hits = retrieve(question)
                answer = ask_llm(question, hits, groq_key)
                st.markdown(answer)
                with st.expander("Sources"):
                    for h in hits:
                        st.markdown(f"`{h['file']}` (line {h['start_line']})")
        st.session_state["messages"].append({"role": "assistant", "content": answer})
