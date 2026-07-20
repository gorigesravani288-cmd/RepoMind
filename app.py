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
import base64
import json
import uuid
import streamlit as st
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit.components.v1 as components
from diagram import build_repo_module_graph, trim_to_most_connected, graph_to_mermaid, render_mermaid, mermaid_to_png_bytes

from ingest import ingest_repo, DB_PATH, COLLECTION_NAME

load_dotenv()

st.set_page_config(page_title="RepoMind", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------------------
# Session persistence: survives a page refresh, which normally wipes
# st.session_state entirely (that's plain Streamlit behavior, not a bug in
# this app specifically -- a browser refresh starts a brand new session).
#
# How it works: a random session ID is put in the URL the first time someone
# visits. On every refresh, the SAME URL (with the same ID) is requested, so
# we can look up and restore the saved conversation from a small local JSON
# file keyed by that ID.
#
# Known limitation: the underlying vector index (ChromaDB) uses ONE global
# collection for whichever repo was indexed most recently (see ingest.py) --
# it isn't per-session. So this restores your chat text and repo name
# correctly, but if a different repo was indexed in the meantime (e.g. by
# you in another tab), the restored chat would reference a repo whose index
# has since been overwritten. Fine for normal single-user use; worth knowing
# if you ever open multiple repos in parallel tabs.
# ---------------------------------------------------------------------------
SESSIONS_DIR = "./sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)


RECENT_REPOS_FILE = os.path.join(SESSIONS_DIR, "recent_repos.json")
MAX_RECENT_REPOS = 8


def load_recent_repos() -> list:
    """Returns the list of previously indexed repo URLs, most recent first.
    Shared across all visitors (it's just a convenience list of public repo
    URLs, nothing sensitive) -- returns [] on any error rather than crashing
    the sidebar."""
    try:
        if not os.path.exists(RECENT_REPOS_FILE):
            return []
        with open(RECENT_REPOS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def add_recent_repo(repo_url: str):
    """Adds a repo to the recent list (most recent first, deduplicated,
    capped at MAX_RECENT_REPOS). Fails silently -- this is a nice-to-have,
    never something that should block indexing if disk I/O has an issue."""
    try:
        recents = load_recent_repos()
        recents = [r for r in recents if r != repo_url]  # move to front if already there
        recents.insert(0, repo_url)
        recents = recents[:MAX_RECENT_REPOS]
        with open(RECENT_REPOS_FILE, "w", encoding="utf-8") as f:
            json.dump(recents, f)
    except Exception:
        pass


def _session_file(session_id: str) -> str:
    # session_id is our own uuid4, so this is safe from path traversal
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def save_session_state():
    """Persists the parts of session_state needed to restore a conversation
    after a refresh. Fails silently (never blocks the UI) if disk I/O has
    any issue -- persistence is a nice-to-have, not something that should
    ever break the actual chat."""
    try:
        session_id = st.session_state.get("_session_id")
        if not session_id:
            return
        data = {
            "indexed_repo": st.session_state.get("indexed_repo"),
            "repo_summary": st.session_state.get("repo_summary"),
            "messages": st.session_state.get("messages", []),
        }
        with open(_session_file(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def load_session_state(session_id: str) -> bool:
    """Restores a previously saved conversation into session_state.
    Returns True if something was actually restored, False otherwise
    (no file, corrupted file, etc. -- all handled gracefully, just starts
    fresh in that case rather than erroring)."""
    try:
        path = _session_file(session_id)
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("indexed_repo"):
            st.session_state["indexed_repo"] = data["indexed_repo"]
        if data.get("repo_summary"):
            st.session_state["repo_summary"] = data["repo_summary"]
        st.session_state["messages"] = data.get("messages", [])
        return True
    except Exception:
        return False


# Get or create this browser tab's session ID from the URL.
_url_sid = st.query_params.get("sid")
if _url_sid:
    st.session_state["_session_id"] = _url_sid
elif "_session_id" not in st.session_state:
    st.session_state["_session_id"] = uuid.uuid4().hex
    st.query_params["sid"] = st.session_state["_session_id"]

# If this is a fresh session_state (i.e. right after a refresh -- "messages"
# won't exist yet) but the URL has a known session ID, restore it.
if "messages" not in st.session_state:
    load_session_state(st.session_state["_session_id"])

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


def _safe_get_secret(key: str):
    """Reads st.secrets safely. Streamlit Cloud always has a secrets store,
    but locally (no secrets.toml file) accessing st.secrets.get() raises
    StreamlitSecretNotFoundError instead of just returning None. This wraps
    that so local development never crashes -- it just falls back to .env."""
    try:
        return st.secrets.get(key, None)
    except Exception:
        return None


def get_groq_key():
    """Check Streamlit Cloud secrets first, then local .env, then manual sidebar input."""
    key = _safe_get_secret("GROQ_API_KEY")
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

    prompt = f"""You are RepoMind, a friendly, knowledgeable coding assistant. You're currently
helping someone explore a specific codebase, but you are NOT limited to only answering
questions about that repo -- you're a capable, general conversational assistant who happens
to also have this repo's code available as extra context when it's relevant.

Rules:
- FIRST, judge the tone and intent of the message. If it's casual, informal, small talk, a typo'd greeting, slang, or clearly not a real question (e.g. "hlo bro", "sup", "lol nice") -- even if it doesn't match a common exact phrase -- respond briefly and warmly in matching casual tone, like a real person would. Do NOT treat casual messages as incomplete or unclear technical questions needing clarification, and do NOT say things like "your question isn't quite clear" to a greeting or casual remark.
- If it's a genuine question or request (about the repo, general tech, or otherwise), treat it as a professional/technical query and answer accordingly with real substance.
- Answer the question directly. Do NOT restate the question or describe what you're about to do.
- Do NOT add filler commentary like "this indicates its importance" unless it's a genuine, specific insight backed by the context.
- If the question is about THIS repo and the provided context covers it, give a clear, specific, useful answer in 2-5 sentences, and naturally mention which file(s) it's based on.
- If the question is about this repo but the context doesn't fully cover it, briefly say so, then still help using your own general knowledge -- suggestions, best practices, addon/feature ideas, or a general explanation. Signal the shift naturally (e.g. "the repo doesn't show this directly, but generally...").
- If the question is general programming/tech/advice and isn't really about this specific repo at all, just answer it normally and helpfully using your own knowledge -- you don't need repo context to answer general questions, and you should never refuse or deflect a reasonable question just because the repo's code doesn't mention it.
- If asked for suggestions, improvement ideas, or addon ideas, answer directly and helpfully, combining anything relevant from the context with your own good judgment.
- Keep a warm, natural, conversational tone throughout -- like chatting with a helpful, knowledgeable teammate, not querying a lookup tool.
- Use the recent conversation (if any) to resolve references like "it" or "that function".

Recent conversation:
{history_str if history_str else "(none yet)"}

Context from the repo (use it when relevant to the question; ignore it when the question is general and doesn't need it):
{context_str}

Question: {question}

Answer:"""

    try:
        client = Groq(api_key=api_key, timeout=25.0)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Groq can time out, rate-limit, or reject a request for various
        # reasons -- never let that crash the whole app with a raw
        # traceback. Show a clear message AND the real underlying reason
        # (not just the exception type name) so problems are diagnosable
        # directly from the chat, without needing to dig through the
        # terminal/server logs every time.
        err_text = str(e)
        err_lower = err_text.lower()
        if "timeout" in err_lower or "timed out" in err_lower:
            return ("⏱️ The response took too long and timed out. This can happen when Groq's "
                    "free tier is busy — please try asking again in a moment.")
        if "rate limit" in err_lower or "429" in err_lower:
            return ("⏳ Hit a rate limit on the free API tier. Please wait a few seconds and try again.")
        return (f"⚠️ Something went wrong reaching the AI model ({type(e).__name__}).\n\n"
                f"Details: {err_text}\n\nPlease try asking again.")


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
    client = Groq(api_key=api_key, timeout=25.0)
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar: index a repo + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    _preloaded_key = _safe_get_secret("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    st.markdown('<div class="rm-step-label">Step 1 · Index a repo</div>', unsafe_allow_html=True)

    # Clicking a recent repo pre-fills the URL box via session_state, since a
    # text_input's value can only be set before it's instantiated, not after.
    _prefill = st.session_state.pop("_prefill_repo_url", "")
    repo_url = st.text_input(
        "GitHub repo URL", value=_prefill, placeholder="https://github.com/user/repo",
        label_visibility="collapsed", key="repo_url_box",
    )
    index_clicked = st.button("🔍  Index Repo", type="primary", use_container_width=True)
    st.caption("⏱️ Small repos: ~30-60s. Larger repos: a few minutes (free-tier CPU).")

    _recent_repos = load_recent_repos()
    if _recent_repos:
        with st.expander(f"🕘 Recent repos ({len(_recent_repos)})"):
            for r in _recent_repos:
                short_name = r.rstrip("/").split("/")[-1]
                if st.button(f"📎 {short_name}", key=f"recent_{r}", use_container_width=True,
                             help=r):
                    st.session_state["_prefill_repo_url"] = r
                    st.rerun()
            st.caption("Clicking a repo fills in the URL above — click **Index Repo** to re-index it.")

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
                    add_recent_repo(repo_url.strip())
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
            repo_name = st.session_state["indexed_repo"].split("/")[-1]

            # Build ONE combined HTML file: readable text + real embedded
            # images (as base64 data URIs) side by side, all in a single
            # downloadable file that opens directly in any browser.
            html_parts = [
                "<html><head><meta charset='utf-8'>",
                f"<title>RepoMind — {repo_name}</title>",
                "<style>",
                "body{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:900px;",
                "margin:40px auto;padding:0 20px;line-height:1.5;color:#1a1a1a;}",
                "h1{font-size:1.4rem;} .msg{margin:18px 0;padding:14px 18px;border-radius:10px;}",
                ".user{background:#eef2ff;} .assistant{background:#f7f7f8;}",
                ".role{font-weight:700;font-size:0.8rem;text-transform:uppercase;",
                "color:#666;margin-bottom:6px;}",
                "img{max-width:100%;display:block;border:1px solid #ddd;border-radius:8px;margin-top:10px;}",
                ".sources{font-size:0.8rem;color:#555;margin-top:8px;}",
                ".diagram-wrap{position:relative;display:inline-block;max-width:100%;margin-top:10px;}",
                ".diagram-toolbar{position:absolute;top:8px;right:8px;display:flex;gap:6px;}",
                ".icon-btn{background:rgba(255,255,255,0.92);border:1px solid #ccc;border-radius:6px;",
                "padding:5px 9px;cursor:pointer;font-size:0.95rem;text-decoration:none;color:#333;",
                "box-shadow:0 1px 3px rgba(0,0,0,0.15);}",
                ".icon-btn:hover{background:#fff;}",
                "</style></head><body>",
                f"<h1>🧠 RepoMind conversation — {repo_name}</h1>",
            ]
            if st.session_state.get("repo_summary"):
                html_parts.append(f"<p><em>{st.session_state['repo_summary']}</em></p><hr>")

            for m in st.session_state["messages"]:
                role_label = "You" if m["role"] == "user" else "RepoMind"
                css_class = "user" if m["role"] == "user" else "assistant"
                html_parts.append(f'<div class="msg {css_class}"><div class="role">{role_label}</div>')
                html_parts.append(f"<div>{m['content']}</div>")

                if m.get("diagram_png_b64"):
                    b64 = m["diagram_png_b64"]
                    html_parts.append(
                        f'<div class="diagram-wrap">'
                        f'<img src="data:image/png;base64,{b64}" alt="Architecture diagram">'
                        f'<div class="diagram-toolbar">'
                        f'<a class="icon-btn" href="data:image/png;base64,{b64}" '
                        f'download="repo_architecture_diagram.png" title="Download image">⬇️</a>'
                        f'<button class="icon-btn" onclick="copyDiagramImage(\'{b64}\', this)" '
                        f'title="Copy image">📋</button>'
                        f'</div></div>'
                    )
                elif m.get("mermaid"):
                    # Image render failed at generation time (e.g. no internet)
                    # -- fall back to including the raw diagram code instead
                    # of silently losing it.
                    html_parts.append(
                        "<p><em>Diagram image unavailable — paste this code into "
                        '<a href="https://mermaid.live">mermaid.live</a> to view it:</em></p>'
                        f"<pre>{m['mermaid']}</pre>"
                    )

                if m.get("sources"):
                    srcs = ", ".join(f"{s['file']}:{s['start_line']}" for s in m["sources"])
                    html_parts.append(f'<div class="sources">Sources: {srcs}</div>')
                html_parts.append("</div>")

            html_parts.append("""
            <script>
            async function copyDiagramImage(b64, btn) {
                try {
                    const byteChars = atob(b64);
                    const byteNumbers = new Array(byteChars.length);
                    for (let i = 0; i < byteChars.length; i++) {
                        byteNumbers[i] = byteChars.charCodeAt(i);
                    }
                    const byteArray = new Uint8Array(byteNumbers);
                    const blob = new Blob([byteArray], {type: "image/png"});
                    await navigator.clipboard.write([new ClipboardItem({"image/png": blob})]);
                    const original = btn.innerHTML;
                    btn.innerHTML = "✅";
                    setTimeout(() => { btn.innerHTML = original; }, 1500);
                } catch (e) {
                    alert("Copy isn't supported in this browser/context. Please use the download button instead.");
                }
            }
            </script>
            """)
            html_parts.append("</body></html>")
            combined_html = "\n".join(html_parts)

            st.download_button(
                "💾  Download conversation (text + diagrams)",
                data=combined_html,
                file_name=f"repomind_{repo_name}.html",
                mime="text/html",
                use_container_width=True,
            )
            st.caption("Opens in any browser — includes both your Q&A and any diagram images together.")

    st.markdown("---")

    if _preloaded_key:
        st.caption("🔑 API key loaded ✅")
    else:
        with st.expander("🔑 Add API key (needed for Q&A, not for indexing/diagrams)"):
            st.text_input("GROQ_API_KEY", type="password", placeholder="gsk_...", key="manual_groq_key")
            st.caption("Get a free key at console.groq.com/keys")

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

for i, msg in enumerate(st.session_state["messages"]):
    avatar = "🙋" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "user":
            col_text, col_edit = st.columns([20, 1])
            with col_text:
                st.markdown(msg["content"])
            with col_edit:
                if st.button("✏️", key=f"edit_btn_{i}", help="Edit and resend this message"):
                    st.session_state["_editing_index"] = i
                    st.rerun()
        else:
            st.markdown(msg["content"])
        if msg.get("mermaid"):
            render_mermaid(msg["mermaid"])
            with st.expander("View diagram code (paste into mermaid.live to view/edit)"):
                st.code(msg["mermaid"], language="text")
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

# If a message is being edited, show an inline edit box instead of (or above)
# the normal chat input, prefilled with the original text.
edited_question = None
if st.session_state.get("_editing_index") is not None:
    edit_idx = st.session_state["_editing_index"]
    original_text = st.session_state["messages"][edit_idx]["content"]
    st.info("✏️ Editing your message — this will remove it and everything after it, then re-ask with your edit.")
    new_text = st.text_area("Edit your message:", value=original_text, key="_edit_textarea")
    col_save, col_cancel = st.columns([1, 1])
    with col_save:
        if st.button("✅ Save & resend", use_container_width=True):
            # Remove this message and everything after it, then treat the
            # edited text exactly like a freshly typed question.
            st.session_state["messages"] = st.session_state["messages"][:edit_idx]
            st.session_state["_editing_index"] = None
            edited_question = new_text
    with col_cancel:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state["_editing_index"] = None
            st.rerun()

question = st.chat_input("Ask something about the indexed repo...")
if not question and edited_question:
    question = edited_question

CASUAL_RESPONSES = {
    "ok": "👍 Got it! Let me know if you have more questions about this repo.",
    "okay": "👍 Got it! Let me know if you have more questions about this repo.",
    "k": "👍 Let me know if you'd like to know more about this repo.",
    "cool": "😊 Glad that helped! Feel free to ask anything else about the repo.",
    "nice": "😊 Glad that helped! Feel free to ask anything else about the repo.",
    "great": "😊 Glad that helped! Feel free to ask anything else about the repo.",
    "thanks": "You're welcome! 🙌 Ask away if there's anything else about this repo you'd like to know.",
    "thank you": "You're welcome! 🙌 Ask away if there's anything else about this repo you'd like to know.",
    "thankyou": "You're welcome! 🙌 Ask away if there's anything else about this repo you'd like to know.",
    "ty": "You're welcome! 🙌",
    "hi": "Hey! 👋 I'm ready to help — ask me anything about the indexed repo, or try \"show me the architecture of this repo\".",
    "hello": "Hey! 👋 I'm ready to help — ask me anything about the indexed repo, or try \"show me the architecture of this repo\".",
    "hlo": "Hey! 👋 I'm ready to help — ask me anything about the indexed repo, or try \"show me the architecture of this repo\".",
    "helo": "Hey! 👋 I'm ready to help — ask me anything about the indexed repo, or try \"show me the architecture of this repo\".",
    "hii": "Hey! 👋 I'm ready to help — ask me anything about the indexed repo, or try \"show me the architecture of this repo\".",
    "heya": "Hey! 👋 What would you like to know about this repo?",
    "hey": "Hey! 👋 What would you like to know about this repo?",
    "yo": "Hey! 👋 What would you like to know about this repo?",
    "sup": "Hey! 👋 What would you like to know about this repo?",
    "bye": "Bye! 👋 Come back anytime you want to explore another repo.",
    "goodbye": "Bye! 👋 Come back anytime you want to explore another repo.",
    "yes": "Got it 👍 — what would you like to know?",
    "no": "No worries — let me know if there's anything else I can help with.",
    "sure": "👍 Go ahead and ask whenever you're ready.",
    "yep": "👍 Go ahead and ask whenever you're ready.",
    "nope": "No worries — let me know if there's anything else I can help with.",
    "good": "😊 Glad to hear it! Ask me anything else about the repo.",
    "good job": "Thank you! 😊 Happy to help with anything else about this repo.",
    "wow": "😄 Right? Let me know if you'd like to dig into anything specific.",
    "haha": "😄 Glad you liked that! Anything else you'd like to explore in the repo?",
}

# Greeting words that, when combined with a casual address word (e.g. "hi bro",
# "hlo dude"), should still count as a simple greeting rather than falling
# through to the full RAG pipeline and confusing the model.
GREETING_WORDS = {"hi", "hello", "hlo", "helo", "hii", "hey", "heya", "yo", "sup"}
CASUAL_ADDRESS_WORDS = {"bro", "dude", "man", "buddy", "mate", "friend", "there", "bud"}


def is_casual_chat(text: str) -> str | None:
    """Checks if a message is casual conversation (greeting, thanks, filler)
    rather than a real question about the repo. Returns a friendly canned
    reply if so, or None if it looks like a genuine question that should go
    through the normal RAG pipeline instead.

    Two layers: an exact-match check (e.g. "thanks", "ok"), and a slightly
    looser check for a greeting word followed only by casual address words
    (e.g. "hi bro", "hlo dude") -- common real phrasing that a strict
    exact-match alone would miss, sending it into the RAG pipeline and
    producing a confusing "I need more context" reply to what was really
    just a hello.

    This deliberately stays conservative -- it does NOT try to catch every
    possible casual phrase, since being overly aggressive risks swallowing
    real short questions like "why" or "how". When in doubt, this returns
    None and lets the real pipeline (which is much smarter) handle it."""
    normalized = text.strip().lower().strip("!.,?")
    if normalized in CASUAL_RESPONSES:
        return CASUAL_RESPONSES[normalized]

    words = normalized.split()
    if words and words[0] in GREETING_WORDS and all(w in CASUAL_ADDRESS_WORDS for w in words[1:]):
        return CASUAL_RESPONSES.get(words[0], CASUAL_RESPONSES["hi"])

    return None


if question:
    _casual_reply = is_casual_chat(question) if "indexed_repo" in st.session_state else None
    if "indexed_repo" not in st.session_state:
        st.warning("Index a repo first using the sidebar.")
    elif _casual_reply:
        # Pure conversational filler ("ok", "thanks", "hi") gets an instant
        # canned reply -- no API call needed, no cost, no latency, and
        # crucially: never a robotic "I couldn't find that in the context"
        # response to something that was never a real question.
        st.session_state["messages"].append({"role": "user", "content": question, "sources": None})
        with st.chat_message("user", avatar="🙋"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(_casual_reply)
        st.session_state["messages"].append({"role": "assistant", "content": _casual_reply, "sources": None})
    elif "architecture" in question.lower() or "diagram" in question.lower():
        # Diagram requests are handled separately from the normal RAG flow --
        # this needs no Groq API key at all, since it's pure local analysis
        # of already-indexed chunk data.
        st.session_state["messages"].append({"role": "user", "content": question, "sources": None})
        with st.chat_message("user", avatar="🙋"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Analyzing internal import relationships..."):
                try:
                    graph = build_repo_module_graph(DB_PATH, COLLECTION_NAME)
                    graph, was_trimmed = trim_to_most_connected(graph, max_nodes=35)
                    mermaid_code = graph_to_mermaid(graph)
                except Exception:
                    mermaid_code = ""
                    was_trimmed = False

            if mermaid_code:
                notes = []
                if was_trimmed:
                    notes.append("showing the 35 most-connected files")
                notes.append("examples/tests/docs excluded by default")
                st.caption(f"📏 {' · '.join(notes)}.")
                st.markdown("Here's how the modules in this repo depend on each other:")
                render_mermaid(mermaid_code)
                with st.expander("View diagram code (paste into mermaid.live to view/edit)"):
                    st.code(mermaid_code, language="text")
                reply = "Generated a module dependency diagram above, based on internal imports between the repo's Python files."

                # Fetch a real PNG once now (not at download time) so the combined
                # export button doesn't need to re-render on every click.
                png_bytes = mermaid_to_png_bytes(mermaid_code)
                diagram_png_b64 = base64.b64encode(png_bytes).decode("ascii") if png_bytes else None
            else:
                st.warning(
                    "Couldn't find clear internal import relationships to diagram for this repo — "
                    "it may not be a Python-heavy codebase, or its files may not import each other directly."
                )
                reply = "No diagram could be generated for this repo."
                diagram_png_b64 = None
        st.session_state["messages"].append(
            {"role": "assistant", "content": reply, "sources": None,
             "mermaid": mermaid_code or None, "diagram_png_b64": diagram_png_b64}
        )
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

# Persist current state so a page refresh can restore it (see the session
# persistence block near the top of this file for how restoration works).
save_session_state()