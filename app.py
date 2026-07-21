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
import time
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
import Auth

from ingest import ingest_repo, DB_PATH, COLLECTION_NAME, get_user_collection_name

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


MAX_RECENT_REPOS = 8
# How long an unused session file is kept before cleanup sweeps it away.
# Anonymous (pre-login) sessions are just a random tab ID with no real
# account behind them, so they're swept aggressively. Username-keyed
# sessions belong to a real account and are kept much longer, since a
# user might not log back in for weeks and shouldn't lose their state.
ANON_SESSION_MAX_AGE_DAYS = 3
USER_SESSION_MAX_AGE_DAYS = 180


def _safe_key(key: str) -> str:
    # key is either our own uuid4 (pre-login) or a validated username
    # (Auth.py already restricts usernames to a safe alnum/_/- character
    # set), so this is safe from path traversal either way.
    return "".join(c for c in key if c.isalnum() or c in ("_", "-"))


def _recent_repos_file(username: str) -> str:
    return os.path.join(SESSIONS_DIR, f"recent_repos_{_safe_key(username)}.json")


def load_recent_repos(username: str) -> list:
    """Returns this user's previously indexed repo URLs, most recent first.
    Per-user (keyed by username) so one account's repo history is never
    visible to another -- returns [] on any error rather than crashing
    the sidebar."""
    try:
        path = _recent_repos_file(username)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def add_recent_repo(username: str, repo_url: str):
    """Adds a repo to this user's recent list (most recent first,
    deduplicated, capped at MAX_RECENT_REPOS). Fails silently -- this is a
    nice-to-have, never something that should block indexing if disk I/O
    has an issue."""
    try:
        recents = load_recent_repos(username)
        recents = [r for r in recents if r != repo_url]  # move to front if already there
        recents.insert(0, repo_url)
        recents = recents[:MAX_RECENT_REPOS]
        with open(_recent_repos_file(username), "w", encoding="utf-8") as f:
            json.dump(recents, f)
    except Exception:
        pass


def _repo_history_file(username: str) -> str:
    return os.path.join(SESSIONS_DIR, f"repo_history_{_safe_key(username)}.json")


def load_repo_history(username: str) -> dict:
    """Returns {repo_url: {messages, repo_summary, last_index_count,
    last_index_stats, collection_name}} for every repo this user has ever
    indexed -- this is what lets clicking a "recent repo" jump straight
    back into that repo's own chat instead of re-cloning and re-embedding
    it from scratch. Returns {} on any error."""
    try:
        path = _repo_history_file(username)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_repo_snapshot(username: str, repo_url: str):
    """Saves the CURRENT session's state (messages, summary, stats) as
    this repo's entry in the user's repo history, so switching to another
    repo and back doesn't lose the conversation. Called after every
    message and after indexing -- fails silently, same as save_session_state."""
    try:
        history = load_repo_history(username)
        history[repo_url] = {
            "messages": st.session_state.get("messages", []),
            "repo_summary": st.session_state.get("repo_summary"),
            "last_index_count": st.session_state.get("last_index_count"),
            "last_index_stats": st.session_state.get("last_index_stats"),
            "collection_name": get_user_collection_name(username, repo_url),
        }
        with open(_repo_history_file(username), "w", encoding="utf-8") as f:
            json.dump(history, f)
    except Exception:
        pass



    return os.path.join(SESSIONS_DIR, f"{_safe_key(key)}.json")


def cleanup_old_sessions():
    """Sweeps stale session/recent-repo files out of ./sessions so the
    directory doesn't grow forever. Every anonymous visit before login
    creates a random-UUID session file that's otherwise never cleaned up;
    this deletes those (and any file) once they've been untouched past
    their max age. Username-keyed files (real accounts) get a much longer
    grace period than anonymous tab sessions. Runs a lightweight, best-
    effort sweep -- any error on any individual file is skipped, never
    raised, since this must never be able to break app startup."""
    try:
        now = time.time()
        for fname in os.listdir(SESSIONS_DIR):
            if not fname.endswith(".json") or fname in ("users.json", "login_attempts.json"):
                continue
            if fname.startswith("recent_repos_"):
                continue  # tied to account lifetime, not activity -- leave alone
            path = os.path.join(SESSIONS_DIR, fname)
            try:
                age_days = (now - os.path.getmtime(path)) / 86400
                stem = fname[:-5]
                # A hex uuid4 (32 hex chars) is an anonymous tab session;
                # anything else is a username-keyed account session.
                is_anon = len(stem) == 32 and all(c in "0123456789abcdef" for c in stem)
                max_age = ANON_SESSION_MAX_AGE_DAYS if is_anon else USER_SESSION_MAX_AGE_DAYS
                if age_days > max_age:
                    os.remove(path)
            except Exception:
                continue
    except Exception:
        pass


# Extension -> language name understood by st.code()'s syntax highlighter.
# Used so retrieved/displayed code snippets are actually highlighted as
# their real language instead of always being shown as Python.
LANGUAGE_BY_EXTENSION = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".jsx": "jsx", ".java": "java", ".go": "go", ".rb": "ruby", ".c": "c",
    ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".md": "markdown", ".txt": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".rs": "rust",
    ".php": "php",
}


def language_for_file(file_path: str) -> str:
    """Maps a file's extension to the language name st.code() expects, so a
    retrieved snippet from a .js or .md file isn't syntax-highlighted as if
    it were Python. Falls back to plain "text" for anything unrecognized
    (extension-less files like README, LICENSE, Dockerfile, etc.)."""
    _, ext = os.path.splitext(file_path)
    return LANGUAGE_BY_EXTENSION.get(ext.lower(), "text")


def save_session_state():
    """Persists session state to TWO locations once logged in:
      1. Keyed by the browser tab's random session ID -- so a plain page
         refresh (same tab, same URL) restores you without needing to log
         in again.
      2. Keyed by username -- so logging in again from ANY device/browser
         restores your own repo and chat history, not just whatever this
         particular tab last had.
    Before login, only (1) applies (there's no username yet).
    Fails silently (never blocks the UI) if disk I/O has any issue --
    persistence is a nice-to-have, not something that should ever break
    the actual chat."""
    try:
        data = {
            "indexed_repo": st.session_state.get("indexed_repo"),
            "repo_summary": st.session_state.get("repo_summary"),
            "messages": st.session_state.get("messages", []),
            "last_index_count": st.session_state.get("last_index_count"),
            "last_index_stats": st.session_state.get("last_index_stats"),
            "logged_in_user": st.session_state.get("logged_in_user"),
        }
        sid = st.session_state.get("_session_id")
        if sid:
            with open(_session_file(sid), "w", encoding="utf-8") as f:
                json.dump(data, f)

        username = st.session_state.get("logged_in_user")
        if username:
            with open(_session_file(username), "w", encoding="utf-8") as f:
                json.dump(data, f)
            repo_url = st.session_state.get("indexed_repo")
            if repo_url:
                save_repo_snapshot(username, repo_url)
    except Exception:
        pass


def load_session_state(session_id: str) -> bool:
    """Restores a previously saved conversation into session_state.
    Returns True if something was actually restored, False otherwise
    (no file, corrupted file, etc. -- all handled gracefully, just starts
    fresh in that case rather than erroring).

    IMPORTANT CAVEAT: this restores what THIS browser session last saw, but
    ingest.py uses a single global ChromaDB collection -- if a different
    repo was indexed on the server since this session was saved (by you in
    another tab, or anyone else), the restored chat/stats may no longer
    match what's actually in the vector index. st.session_state["_restored"]
    is set to True whenever a restore happens, so the UI can show a gentle
    "re-index if things look off" hint rather than presenting stale data
    as if it were freshly confirmed."""
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
        if data.get("last_index_count") is not None:
            st.session_state["last_index_count"] = data["last_index_count"]
        if data.get("last_index_stats") is not None:
            st.session_state["last_index_stats"] = data["last_index_stats"]
        if data.get("logged_in_user"):
            st.session_state["logged_in_user"] = data["logged_in_user"]
        st.session_state["_restored"] = True
        return True
    except Exception:
        return False


@st.cache_resource
def _run_session_cleanup_once():
    """st.cache_resource makes this execute exactly once per running app
    process (not once per user, not once per rerun) -- a cheap, safe way to
    sweep stale session files on startup without doing a directory listing
    on every single interaction."""
    cleanup_old_sessions()
    return True


_run_session_cleanup_once()

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
    # If that revealed a logged-in user, also pull their canonical
    # username-keyed data -- it may be newer than what this particular
    # browser tab last saved (e.g. updated from another device).
    if st.session_state.get("logged_in_user"):
        load_session_state(st.session_state["logged_in_user"])

# ---------------------------------------------------------------------------
# Authentication gate: real per-user signup/login (see Auth.py for details
# and security notes). Nothing below this block runs until logged in.
# ---------------------------------------------------------------------------
if not st.session_state.get("logged_in_user"):
    st.markdown("<h1 style='text-align:center;'>🧠 RepoMind</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:gray;'>Sign in to start exploring codebases.</p>",
        unsafe_allow_html=True,
    )
    _, _center, _ = st.columns([1, 1.2, 1])
    with _center:
        login_tab, signup_tab = st.tabs(["🔑 Log in", "✍️ Sign up"])

        with login_tab:
            with st.form("login_form"):
                li_username = st.text_input("Username", key="li_username")
                li_password = st.text_input("Password", type="password", key="li_password")
                if st.form_submit_button("Log in", use_container_width=True, type="primary"):
                    if not li_username or not li_password:
                        st.error("Please enter both a username and password.")
                    else:
                        ok, msg = Auth.login(li_username, li_password)
                        if ok:
                            st.session_state["logged_in_user"] = li_username.strip()
                            # Pull in THIS user's own previously saved repo/chat
                            # state (if any) -- keyed by username now, not the
                            # random browser-tab ID, so it follows their account.
                            if not load_session_state(li_username.strip()):
                                st.session_state["messages"] = []
                            save_session_state()
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

        with signup_tab:
            with st.form("signup_form"):
                su_username = st.text_input("Choose a username", key="su_username")
                su_password = st.text_input("Choose a password", type="password", key="su_password")
                su_password2 = st.text_input("Confirm password", type="password", key="su_password2")
                if st.form_submit_button("Create account", use_container_width=True, type="primary"):
                    if not su_username or not su_password:
                        st.error("Please fill in all fields.")
                    elif su_password != su_password2:
                        st.error("Passwords don't match.")
                    else:
                        ok, msg = Auth.signup(su_username, su_password)
                        if ok:
                            st.success(msg + " Switch to the Log in tab above.")
                        else:
                            st.error(msg)
    st.stop()  # nothing below this line renders until logged in

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


def current_collection_name() -> str:
    """Every (user, repo) pair lives in its own ChromaDB collection -- this
    is what makes 'per-user sessions' real data isolation between users,
    AND what lets a user keep multiple previously-indexed repos around at
    once instead of each new index wiping out the last one."""
    username = st.session_state.get("logged_in_user")
    repo_url = st.session_state.get("indexed_repo")
    return get_user_collection_name(username, repo_url) if username else COLLECTION_NAME


def retrieve(query: str, k: int = 5):
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(current_collection_name())
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


def _build_qa_prompt(question: str, context_chunks, history=None) -> str:
    context_str = "\n\n".join(
        f"[{c['file']} : line {c['start_line']}]\n{c['text']}" for c in context_chunks
    )
    history_str = ""
    if history:
        recent = history[-4:]
        history_str = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    return f"""You are RepoMind, a friendly, knowledgeable coding assistant. You're currently
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


def _friendly_groq_error(e: Exception) -> str:
    # Groq can time out, rate-limit, or reject a request for various
    # reasons -- never let that crash the whole app with a raw traceback.
    # Show a clear message AND the real underlying reason (not just the
    # exception type name) so problems are diagnosable directly from the
    # chat, without needing to dig through the terminal/server logs.
    err_text = str(e)
    err_lower = err_text.lower()
    if "timeout" in err_lower or "timed out" in err_lower:
        return ("⏱️ The response took too long and timed out. This can happen when Groq's "
                "free tier is busy — please try asking again in a moment.")
    if "rate limit" in err_lower or "429" in err_lower:
        return "⏳ Hit a rate limit on the free API tier. Please wait a few seconds and try again."
    return (f"⚠️ Something went wrong reaching the AI model ({type(e).__name__}).\n\n"
            f"Details: {err_text}\n\nPlease try asking again.")


def ask_llm(question: str, context_chunks, api_key: str, history=None) -> str:
    prompt = _build_qa_prompt(question, context_chunks, history=history)
    try:
        client = Groq(api_key=api_key, timeout=25.0)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content
    except Exception as e:
        return _friendly_groq_error(e)


def ask_llm_stream(question: str, context_chunks, api_key: str, history=None):
    """Generator version of ask_llm, for use with st.write_stream() so the
    answer appears token-by-token instead of all at once after a multi-
    second wait -- matches the README's "fast, real-time-feeling" pitch,
    since Groq is fast enough that the gap was mostly us buffering the
    whole response before showing any of it.

    Yields text chunks (deltas). If the request fails before any tokens
    arrive, yields the same friendly error message ask_llm would have
    returned, as a single chunk -- st.write_stream renders a generator's
    yields exactly like it would a single string, so callers don't need
    to branch on success vs. failure."""
    prompt = _build_qa_prompt(question, context_chunks, history=history)
    try:
        client = Groq(api_key=api_key, timeout=25.0)
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            stream=True,
        )
        got_any = False
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                got_any = True
                yield delta
        if not got_any:
            yield "⚠️ No response came back — please try asking again."
    except Exception as e:
        yield _friendly_groq_error(e)


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

    _recent_repos = load_recent_repos(st.session_state["logged_in_user"])
    if _recent_repos:
        with st.expander(f"🕘 Recent repos ({len(_recent_repos)})"):
            for r in _recent_repos:
                short_name = r.rstrip("/").split("/")[-1]
                if st.button(f"📎 {short_name}", key=f"recent_{r}", use_container_width=True,
                             help=r):
                    _username = st.session_state["logged_in_user"]
                    _collection_name = get_user_collection_name(_username, r)
                    _history = load_repo_history(_username).get(r)
                    _collection_still_present = False
                    if _history:
                        try:
                            chromadb.PersistentClient(path=DB_PATH).get_collection(_collection_name)
                            _collection_still_present = True
                        except Exception:
                            _collection_still_present = False

                    if _history and _collection_still_present:
                        # Jump straight back into this repo -- its vector
                        # data is still in ChromaDB (each repo gets its own
                        # collection, so indexing something else never
                        # touched it) and its chat is restored from disk.
                        # No re-cloning, no re-embedding, no round trip
                        # through "Index Repo".
                        st.session_state["indexed_repo"] = r
                        st.session_state["messages"] = _history.get("messages", [])
                        st.session_state["repo_summary"] = _history.get("repo_summary")
                        st.session_state["last_index_count"] = _history.get("last_index_count")
                        st.session_state["last_index_stats"] = _history.get("last_index_stats")
                        st.session_state["_restored"] = False
                        save_session_state()
                        st.rerun()
                    else:
                        # We've never indexed this repo before on this
                        # deployment (or its collection was cleared some
                        # other way) -- fall back to prefilling the URL so
                        # the user can re-index it deliberately.
                        st.session_state["_prefill_repo_url"] = r
                        st.session_state["_recent_repo_missing"] = short_name
                        st.rerun()
            st.caption("Click a repo to jump straight back into it — already-indexed repos open instantly.")

    if st.session_state.pop("_recent_repo_missing", None):
        st.info("That repo's index wasn't found — URL filled in above, click **Index Repo** to rebuild it.")

    if index_clicked:
        if not repo_url.strip():
            st.warning("Enter a repo URL first.")
        elif st.session_state.get("indexed_repo") == repo_url.strip():
            st.info("This repo is already indexed — no need to re-index. Just start chatting below!")
        else:
            groq_key_for_summary = get_groq_key()
            with st.spinner("Cloning, chunking, and embedding the repo... larger repos take longer."):
                try:
                    _new_collection_name = get_user_collection_name(
                        st.session_state["logged_in_user"], repo_url.strip()
                    )
                    n_chunks, stats = ingest_repo(repo_url.strip(), collection_name=_new_collection_name)
                    st.session_state["indexed_repo"] = repo_url.strip()
                    st.session_state["messages"] = []
                    st.session_state["last_index_count"] = n_chunks
                    st.session_state["last_index_stats"] = stats
                    st.session_state["repo_summary"] = None
                    add_recent_repo(st.session_state["logged_in_user"], repo_url.strip())
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
        if st.session_state.get("last_index_count") is None:
            st.caption(
                "⚠️ This looks like a restored session with no confirmed chunk count. "
                "The live search index may have changed since then — click **Index Repo** "
                "again above to be sure you're querying this exact repo."
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

    st.caption(f"👤 Logged in as **{st.session_state['logged_in_user']}**")

    with st.expander("🔒 Change password"):
        with st.form("change_password_form", clear_on_submit=True):
            cp_old = st.text_input("Current password", type="password", key="cp_old")
            cp_new = st.text_input("New password", type="password", key="cp_new")
            cp_confirm = st.text_input("Confirm new password", type="password", key="cp_confirm")
            cp_submitted = st.form_submit_button("Update password", use_container_width=True)
            if cp_submitted:
                if cp_new != cp_confirm:
                    st.error("New passwords don't match.")
                else:
                    ok, msg = Auth.change_password(st.session_state["logged_in_user"], cp_old, cp_new)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    if st.button("🚪 Log out", use_container_width=True):
        # Clear this browser tab's view completely -- the user's own saved
        # data (under their username file) is left untouched on disk, so
        # it's still there next time they log in from anywhere.
        st.session_state["logged_in_user"] = None
        st.session_state["indexed_repo"] = None
        st.session_state["repo_summary"] = None
        st.session_state["messages"] = []
        st.session_state["last_index_count"] = None
        st.session_state["last_index_stats"] = None
        save_session_state()
        st.rerun()

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
                    st.code(s["text"], language=language_for_file(s["file"]))

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
                    graph = build_repo_module_graph(DB_PATH, current_collection_name())
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
            history_so_far = st.session_state["messages"][:-1]
            with st.spinner("Retrieving relevant context..."):
                search_query = rewrite_query(question, history_so_far, groq_key)
                hits = retrieve(search_query)
            # Streamed so the answer appears as it's generated, instead of a
            # multi-second silent wait followed by the whole thing at once.
            answer = st.write_stream(ask_llm_stream(question, hits, groq_key, history=history_so_far))
            chips = "".join(
                f'<span class="rm-source-chip">{h["file"]}:{h["start_line"]}</span>' for h in hits
            )
            st.markdown(chips, unsafe_allow_html=True)
            with st.expander("View retrieved code"):
                for h in hits:
                    st.caption(f"{h['file']} (line {h['start_line']})")
                    st.code(h["text"], language=language_for_file(h["file"]))
        st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": hits})

# Persist current state so a page refresh can restore it (see the session
# persistence block near the top of this file for how restoration works).
save_session_state()