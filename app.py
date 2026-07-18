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


import re

# Vocabulary of casual/filler words and common slang abbreviations. A message
# is treated as casual if EVERY word in it belongs to this set -- this is
# what lets combinations like "ok tq" or "u r sweet" get caught, since it
# doesn't rely on matching one exact pre-written phrase the way a fixed
# regex list does.
_CASUAL_WORDS = {
    # thanks / acknowledgment
    "ok", "okay", "kk", "k", "thanks", "thank", "thankyou", "thx", "tq", "ty",
    "np", "welcome", "appreciate", "appreciated", "cool", "nice", "great",
    "good", "awesome", "sweet", "perfect", "superb", "excellent", "amazing",
    "wow", "sounds", "got", "it", "alright", "sure",
    # yes/no/filler
    "yes", "yep", "yeah", "yup", "no", "nope", "nah",
    # greetings/farewell
    "hi", "hello", "hey", "yo", "sup", "wassup", "bye", "goodbye", "morning",
    "evening", "night",
    # you / are (for "u r sweet", "you are great", etc.)
    "u", "ur", "you", "youre", "r", "are", "so", "very", "much", "this",
    # laughter / mood
    "lol", "haha", "hehe", "lmao", "nvm", "nevermind", "never", "mind",
    "love", "loved", "well", "done", "job", "one", "work", "lot", "a", "there",
}


def is_casual_message(text: str) -> bool:
    """Detects casual/conversational messages (thanks, compliments,
    greetings, slang acknowledgments) so they get a natural reply instead
    of being run through code retrieval, which would otherwise find nothing
    relevant and produce an unhelpful, oddly-cited response. A message
    counts as casual only if EVERY word in it is in the casual vocabulary
    AND it's short (<=6 words) -- this keeps real short questions like
    "why?" or "is this thread safe" going to retrieval as normal, since
    "thread"/"safe"/"why" aren't in the casual vocabulary."""
    cleaned = text.strip().lower()
    words = re.findall(r"[a-z']+", cleaned)
    if not words or len(words) > 6:
        return False
    return all(w in _CASUAL_WORDS for w in words)


def casual_reply(question: str, api_key: str) -> str:
    """Generates a short, natural reply to casual messages without touching
    retrieval/ChromaDB at all. Falls back to a canned reply if no API key
    is available, so this never hard-fails."""
    if not api_key:
        return "👍 Let me know if you'd like to ask anything else about the repo!"
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": (
                    f'The user said: "{question}" in a chat about a codebase. '
                    "Reply the way a warm, genuine friend would -- brief (1 sentence), "
                    "casual, a little personality is welcome. "
                    "as a normal conversational reply -- do not mention code, "
                    "files, or retrieval."
                ),
            }],
            temperature=0.5,
            max_tokens=40,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "👍 Let me know if you'd like to ask anything else about the repo!"


_HELP_PATTERNS = re.compile(
    r"(what can (you|i) (do|ask)|help me|how does this work|"
    r"what (are|is) (your|the) (features|capabilities)|what should i ask)",
    re.IGNORECASE,
)


def is_help_question(text: str) -> bool:
    """Detects meta-questions about what RepoMind can do, so they get a
    direct capabilities summary instead of being run through code retrieval
    (which would find nothing relevant and refuse)."""
    return bool(_HELP_PATTERNS.search(text.strip()))


def help_reply(repo_name: str) -> str:
    return (
        f"Happy to help! Here's what you can ask me about **{repo_name}**:\n\n"
        "- **Factual questions** — \"What does `parse_config` do?\", \"Where is auth handled?\"\n"
        "- **Open-ended analysis** — \"What are the pros and cons of this structure?\", "
        "\"Any suggestions to improve this?\"\n"
        "- **Architecture** — say \"diagram the architecture\" for a visual module map\n"
        "- **Follow-ups** — I remember recent context, so \"can you show an example?\" works too\n\n"
        "Just ask naturally — no special syntax needed!"
    )


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

    prompt = f"""You are RepoMind. You help people understand a codebase the way a
knowledgeable friend would -- not a search engine reciting facts, but someone who
genuinely enjoys explaining things clearly. Think of how ChatGPT or Claude explain
a concept: plain language first, technical precision woven in naturally, and a
relatable analogy or everyday comparison when it actually helps something click
(e.g. "this function is basically the bouncer at the door -- it checks who's
allowed in before anything else happens"). Skip the analogy if it wouldn't add
anything; forcing one in just to have one reads as gimmicky.

This repo could be written in ANY language or tool -- Python, JavaScript, Go, Rust,
Java, PHP, an n8n automation workflow (JSON node graphs), a Zapier/Make config, a
Terraform/IaC setup, or anything else. Read whatever is actually in the context and
explain it on its own terms -- don't assume Python, and don't force concepts from
one ecosystem onto another (e.g. describe n8n nodes/triggers as what they are,
not as if they were Python functions).

Rules:
- Answer the question directly. Do NOT restate the question or describe what you're about to do.
- Explain like you're talking to a smart friend who's new to this specific codebase -- not a beginner to programming, just new to THIS code/workflow. Assume competence, not expertise here.
- Do NOT add filler commentary like "this indicates its importance" unless it's a genuine, specific insight backed by the context.
- For factual questions (what does X do, where is Y defined), give a clear, specific answer in 2-5 sentences -- plain language, technical terms only where they earn their place.
- For open-ended questions (advantages/disadvantages, suggestions, improvements, opinions, "what do you think"), reason about and analyze the retrieved code even if the answer isn't explicitly written out -- this is expected, not a failure. Use your general software engineering knowledge together with what's in the context to give a thoughtful, honest answer, including trade-offs where relevant.
- When you notice a real drawback, limitation, or risk while answering ANY question -- a missing error handler, a hardcoded value, an outdated pattern, a fragile automation step, tight coupling, etc. -- mention it briefly and suggest a concrete improvement, even if the person didn't explicitly ask for a review. Keep this to 1-2 sentences so it doesn't overwhelm the main answer; skip it entirely if nothing notable stands out (don't invent a drawback just to have one).
- Only say you don't have enough information if the context is genuinely unrelated to the question (e.g. asking about a file/feature that doesn't exist in this repo at all). Never refuse an analysis or opinion question just because the answer wasn't spelled out verbatim.
- Always mention which file(s) your answer is based on, naturally in the sentence, when your answer draws on specific code.
- Use the recent conversation (if any) to resolve references like "it" or "that function".
- Warm, conversational tone throughout -- this should feel like a helpful friend explaining something they know well, genuinely glad to help, never robotic or clinical.

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
    _preloaded_key = _safe_get_secret("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    st.markdown('<div class="rm-step-label">Step 1 · Index a repo</div>', unsafe_allow_html=True)
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
                "img{max-width:100%;border:1px solid #ddd;border-radius:8px;margin-top:10px;}",
                ".sources{font-size:0.8rem;color:#555;margin-top:8px;}",
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
                    html_parts.append(
                        f'<img src="data:image/png;base64,{m["diagram_png_b64"]}" alt="Architecture diagram">'
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

for msg in st.session_state["messages"]:
    avatar = "🙋" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
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

question = st.chat_input("Ask something about the indexed repo...")

if question:
    if "indexed_repo" not in st.session_state:
        st.warning("Index a repo first using the sidebar.")
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
    elif is_casual_message(question):
        st.session_state["messages"].append({"role": "user", "content": question, "sources": None})
        with st.chat_message("user", avatar="🙋"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🧠"):
            reply = casual_reply(question, groq_key)
            st.markdown(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply, "sources": None})
    elif is_help_question(question):
        st.session_state["messages"].append({"role": "user", "content": question, "sources": None})
        with st.chat_message("user", avatar="🙋"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🧠"):
            repo_name = st.session_state["indexed_repo"].split("/")[-1]
            reply = help_reply(repo_name)
            st.markdown(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply, "sources": None})
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