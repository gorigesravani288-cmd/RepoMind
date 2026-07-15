"""
app.py
------
Streamlit front-end for RepoMind.

INPUT (from the user, via the UI):
  1. A GitHub repo URL to index (one-time, or whenever they switch repos)
  2. Natural-language questions about that repo

OUTPUT (shown in the UI):
  - Indexing status/progress + file-type breakdown
  - An auto-generated repo summary right after indexing
  - AI-generated answers (with conversation memory), each showing the
    source file(s)/line numbers AND the actual retrieved code snippet

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

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 900px; }
    .rm-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.1rem; }
    .rm-header .emoji { font-size: 2.2rem; }
    .rm-header h1 {
        font-size: 2rem; font-weight: 800; margin: 0;
        background: linear-gradient(90deg, #FF4B4B, #FF8A5B);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .rm-subtitle { color: #8a8f98; font-size: 0.95rem; margin-bottom: 1.5rem; }
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.15); }
    .rm-step-label {
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.06em;
        text-transform: uppercase; color: #FF6B4A; margin-top: 1rem; margin-bottom: 0.3rem;
    }
    .rm-repo-badge {
        background: rgba(255, 107, 74, 0.08); border: 1px solid rgba(255, 107, 74, 0.25);
        border-radius: 10px; padding: 0.6rem 0.8rem; font-size: 0.82rem; line-height: 1.4;
        word-break: break-all;
    }
    .rm-stats {
        font-size: 0.78rem; color: #8a8f98; margin-top: 0.4rem; line-height: 1.5;
    }
    .rm-empty-state { text-align: center; padding: 3rem 1rem; color: #8a8f98; }
    .rm-empty-state .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .rm-source-chip {
        display: inline-block; background: rgba(120, 120, 120, 0.12); border-radius: 6px;
        padding: 2px 8px; margin: 2px 4px 2px 0; font-family: monospace; font-size: 0.78rem;
    }
    .rm-summary-box {
        background: rgba(120, 120, 120, 0.06); border-left: 3px solid #FF6B4A;
        border-radius: 6px; padding: 0.8rem 1rem; margin: 0.8rem 0 1.2rem 0; font-size: 0.9rem;
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
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({"text": doc, "file": meta["file"], "start_line": meta["start_line"], "distance": dist})
    return hits


def rewrite_query(question: str, history, api_key: str) -> str:
    """Turn a follow-up question like 'can you show an example?' into a
    fully self-contained search query using recent conversation history."""
    if not history:
        return question

    recent = history[-4:]
    history_str = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    prompt = f"""Given this recent conversation and a new follow-up question, rewrite the
follow-up into a single, fully self-contained search query that includes the missing
context (e.g. replace "it"/"that"/"this" with the actual topic). Output ONLY the rewritten
query, nothing else.

Conversation:
{history_str}

Follow-up question: {question}

Rewritten standalone query:"""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        rewritten = response.choices[0].message.content.strip().strip('"')
        return rewritten if rewritten else question
    except Exception:
        return question


def get_groq_key():
    """Check Streamlit Cloud secrets first, then local .env, then manual sidebar input."""
    key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
    key = key or os.getenv("GROQ_API_KEY")
    if not key:
        key = st.session_state.get("manual_groq_key", "")
    return key


def ask_llm(question: str, context_chunks, api_key: str, history=None) -> str:
    context_str = "\n\n".join(
        f"[{c['file']} : line {c['start_line']}]\n{c['text']}" for c in context_chunks
    )
    history_str = ""
    if history:
        recent = history[-4:]
        history_str = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    prompt = f"""You are RepoMind, an assistant that answers questions about a codebase
using ONLY the provided context.

Rules:
- Answer the question directly. Do NOT restate the question or describe what you're about to do.
- Do NOT add filler commentary like "this indicates its importance" unless it's a genuine, specific insight backed by the context.
- If the context has enough information, give a clear, specific, useful answer in 2-5 sentences.
- If the context does NOT have enough information, say so in one honest sentence and suggest what to ask instead.
- Always mention which file(s) your answer is based on, naturally in the sentence.
- Use the recent conversation (if any) to resolve references like "it" or "that function".

Recent conversation:
{history_str if history_str else "(none yet)"}

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


def generate_summary(api_key: str) -> str:
    """Auto-summarize the repo right after indexing, using a broad retrieval."""
    hits = retrieve("What is this project, what does it do, and how is it structured?", k=8)
    context_str = "\n\n".join(f"[{c['file']}]\n{c['text'][:500]}" for c in hits)
    prompt = f"""Based on the following code/doc excerpts from a repository, write a short,
plain-English summary (3-4 sentences) covering: what the project does, and its overall structure.
Do not add filler or meta-commentary. Be specific and concrete.

Excerpts:
{context_str}

Summary:"""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Sidebar: index a repo + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="rm-step-label">Step 1 · API key</div>', unsafe_allow_html=True)
    _preloaded_key = (st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None) or os.getenv("GROQ_API_KEY")
    if _preloaded_key:
        st.markdown("🔑 Key loaded from secrets ✅")
    else:
        st.text_input("GROQ_API_KEY", type="password", placeholder="gsk_...", key="manual_groq_key")

    st.markdown('<div class="rm-step-label">Step 2 · Index a repo</div>', unsafe_allow_html=True)
    repo_url = st.text_input(
        "GitHub repo URL", placeholder="https://github.com/user/repo", label_visibility="collapsed",
    )
    index_clicked = st.button("🔍  Index Repo", type="primary", use_container_width=True)
    st.caption("⏱️ Small repos: ~30-60s. Larger repos: a few minutes (free-tier CPU).")

    if index_clicked:
        if not repo_url.strip():
            st.warning("Enter a repo URL first.")
        elif st.session_state.get("indexed_repo") == repo_url.strip():
            st.info("This repo is already indexed — no need to re-index. Just start chatting below!")
        else:
            groq_key_for_summary = get_groq_key()
            with st.spinner("Cloning, chunking, and embedding the repo... larger repos take longer."):
                try:
                    n_chunks, stats = ingest_repo(repo_url.strip())
                    st.session_state["indexed_repo"] = repo_url.strip()
                    st.session_state["messages"] = []
                    st.session_state["last_index_count"] = n_chunks
                    st.session_state["last_index_stats"] = stats
                    st.session_state["repo_summary"] = None
                except Exception as e:
                    st.error(f"Failed to index repo: {e}")
                    st.stop()
            if groq_key_for_summary:
                with st.spinner("Generating repo summary..."):
                    try:
                        st.session_state["repo_summary"] = generate_summary(groq_key_for_summary)
                    except Exception:
                        st.session_state["repo_summary"] = None
            st.toast(f"Indexed {n_chunks} chunks ✅", icon="🎉")
            if stats.get("truncated"):
                st.info(
                    "📏 This is a large repo — indexed the first ~1,500 code chunks to keep things "
                    "fast on free hosting. Answers will cover most of the codebase, but very "
                    "obscure files may not be included."
                )

    if "indexed_repo" in st.session_state:
        stats = st.session_state.get("last_index_stats", {})
        file_breakdown = ", ".join(
            f"{count} {ext}" for ext, count in sorted(stats.get("files", {}).items(), key=lambda x: -x[1])[:6]
        )
        truncated_note = " ⚠️ (large repo, partially indexed)" if stats.get("truncated") else ""
        st.markdown(
            f'<div class="rm-repo-badge">📦 <b>Indexed:</b><br>{st.session_state["indexed_repo"]}'
            f'<div class="rm-stats">{st.session_state.get("last_index_count", "?")} chunks · '
            f'{stats.get("total_files", "?")} files{truncated_note}<br>{file_breakdown}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("🗑️  Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        if st.session_state.get("messages"):
            transcript_lines = [f"RepoMind conversation — {st.session_state['indexed_repo']}", "=" * 50, ""]
            if st.session_state.get("repo_summary"):
                transcript_lines += ["Repo Summary:", st.session_state["repo_summary"], "", "-" * 50, ""]
            for m in st.session_state["messages"]:
                role_label = "You" if m["role"] == "user" else "RepoMind"
                transcript_lines.append(f"{role_label}: {m['content']}")
                if m.get("sources"):
                    srcs = ", ".join(f"{s['file']}:{s['start_line']}" for s in m["sources"])
                    transcript_lines.append(f"  Sources: {srcs}")
                transcript_lines.append("")
            transcript_text = "\n".join(transcript_lines)

            st.download_button(
                "💾  Download conversation",
                data=transcript_text,
                file_name=f"repomind_{st.session_state['indexed_repo'].split('/')[-1]}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.markdown("---")
    st.caption("Built with Streamlit · ChromaDB · Groq · Sentence-Transformers")

groq_key = get_groq_key()

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
    if st.session_state.get("repo_summary"):
        st.markdown(
            f'<div class="rm-summary-box">📝 <b>What this repo does:</b><br>{st.session_state["repo_summary"]}</div>',
            unsafe_allow_html=True,
        )
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
            with st.expander("View retrieved code"):
                for s in msg["sources"]:
                    st.caption(f"{s['file']} (line {s['start_line']})")
                    st.code(s["text"], language="python")

question = st.chat_input("Ask something about the indexed repo...")

if question:
    if "indexed_repo" not in st.session_state:
        st.warning("Index a repo first using the sidebar.")
    elif not groq_key:
        st.warning("Add your GROQ_API_KEY in the sidebar.")
    else:
        st.session_state["messages"].append({"role": "user", "content": question, "sources": None})
        with st.chat_message("user", avatar="🙋"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Retrieving context and generating answer..."):
                history_so_far = st.session_state["messages"][:-1]
                search_query = rewrite_query(question, history_so_far, groq_key)
                hits = retrieve(search_query)
                answer = ask_llm(question, hits, groq_key, history=history_so_far)
                st.markdown(answer)
                chips = "".join(
                    f'<span class="rm-source-chip">{h["file"]}:{h["start_line"]}</span>' for h in hits
                )
                st.markdown(chips, unsafe_allow_html=True)
                with st.expander("View retrieved code"):
                    for h in hits:
                        st.caption(f"{h['file']} (line {h['start_line']})")
                        st.code(h["text"], language="python")
        st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": hits})