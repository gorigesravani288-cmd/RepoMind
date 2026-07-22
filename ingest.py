"""
ingest.py
---------
Clones a GitHub repository, splits its code/docs into chunks,
embeds each chunk, and stores the embeddings in a local ChromaDB
collection so app.py can retrieve relevant context at query time.

INPUT:  a public GitHub repo URL (e.g. https://github.com/user/repo)
OUTPUT: a persistent ChromaDB store at ./chroma_db containing the
        embedded chunks of that repo, ready to be queried.
"""

import os
import ast
import shutil
import tempfile
import hashlib
import git
import chromadb
from sentence_transformers import SentenceTransformer

# File types worth indexing. Everything else (images, binaries, .git, etc.) is skipped.
INCLUDE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb", ".c", ".cpp",
    ".h", ".hpp", ".md", ".txt", ".json", ".yaml", ".yml", ".rs", ".php",
}
SKIP_DIRS = {
    ".git", ".github", ".vscode", ".idea", "node_modules",
    "__pycache__", "venv", ".venv", "dist", "build",
    "vendor", "vendors", "third_party", "coverage", ".pytest_cache",
    ".mypy_cache", ".tox", "site-packages", "target", "out",
}
# Generated/lock files that are huge, machine-written, and never useful to search.
SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "pipfile.lock", "cargo.lock", "gemfile.lock", "composer.lock",
}
SKIP_FILE_SUFFIXES = (".min.js", ".min.css", ".map")
# Common extension-less files worth indexing (case-insensitive match on filename)
INCLUDE_FILENAMES = {"readme", "license", "contributing", "changelog", "makefile", "dockerfile"}

CHUNK_LINES = 60      # lines per chunk
CHUNK_OVERLAP = 10    # overlapping lines between consecutive chunks
MAX_FILE_LINES = 3000   # skip pathologically huge single files (usually generated/data)
MAX_TOTAL_CHUNKS = 4000  # cap total chunks so free-tier CPU embedding stays fast on huge repos.
                         # Combined with the priority ordering below (core source before
                         # tests/docs/examples), this is enough headroom for the interconnected
                         # core of most repos -- including large ones like Django -- to fit
                         # before the cap is hit.
DB_PATH = "./chroma_db"
COLLECTION_NAME = "repo_chunks"


def clone_repo(repo_url: str, dest: str) -> None:
    print(f"[1/4] Cloning {repo_url} ...")
    # single_branch avoids fetching refs for every other branch, which matters
    # on large, long-lived repos with many branches.
    git.Repo.clone_from(repo_url, dest, depth=1, single_branch=True)


def chunk_python_by_ast(source_text: str):
    """
    Split Python source into chunks along function/class boundaries instead of
    raw line counts, so each chunk is a complete, meaningful unit of code
    (never cuts a function in half). Falls back to None if the file has a
    syntax error or no top-level defs, so the caller can use line-based chunking.
    """
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None

    lines = source_text.splitlines()
    chunks = []
    covered_lines = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            segment_lines = lines[start - 1:end]
            segment = "\n".join(segment_lines).strip()
            if not segment:
                continue

            # If a single function/class is huge, sub-split it so embeddings
            # stay focused (very large classes would otherwise dilute the vector).
            if len(segment_lines) > CHUNK_LINES:
                step = CHUNK_LINES - CHUNK_OVERLAP
                for sub_start in range(0, len(segment_lines), step):
                    sub_lines = segment_lines[sub_start:sub_start + CHUNK_LINES]
                    sub_text = "\n".join(sub_lines).strip()
                    if sub_text:
                        chunks.append({"text": sub_text, "start_line": start + sub_start})
            else:
                chunks.append({"text": segment, "start_line": start})

            covered_lines.update(range(start, end + 1))

    # Capture top-level code not inside any function/class (imports, constants,
    # module-level logic) as its own chunk so nothing gets lost.
    leftover_lines = [
        (i + 1, line) for i, line in enumerate(lines)
        if (i + 1) not in covered_lines and line.strip()
    ]
    if leftover_lines:
        leftover_text = "\n".join(l for _, l in leftover_lines).strip()
        if leftover_text:
            chunks.append({"text": leftover_text, "start_line": leftover_lines[0][0]})

    return chunks if chunks else None


# Folders that are useful to have SOME coverage of, but shouldn't consume
# the chunk budget before the real, interconnected source code does on a
# large repo. This does NOT exclude them (unlike diagram.py, which fully
# excludes these from the architecture diagram) -- it just means they're
# indexed LAST, so questions about tests/docs still work on repos small
# enough to fit everything, while large repos (Django, the Linux kernel,
# etc.) spend their limited budget on core source first.
LOW_PRIORITY_DIRS = {
    "tests", "test", "docs", "doc", "examples", "example",
    ".github", "benchmarks", "benchmark", "scripts", "tools", "js_tests",
}


def _file_priority(rel_path: str) -> int:
    """Returns 0 (high priority, indexed first) or 1 (low priority, indexed
    last) based on whether any folder in the path is a known low-priority
    directory. Checking every path segment (not just the top-level one)
    catches cases like "django/tests/..." where the low-priority folder
    isn't at the repo root."""
    parts = rel_path.replace("\\", "/").split("/")
    for part in parts[:-1]:  # exclude the filename itself
        if part.lower() in LOW_PRIORITY_DIRS:
            return 1
    return 0


def collect_chunks(repo_dir: str):
    """Walk the repo and split eligible files into overlapping line-based chunks.
    Also returns a stats dict: {"files": {ext_or_name: count}, "total_files": N,
    "truncated": bool} — truncated is True if MAX_TOTAL_CHUNKS was hit on a very
    large repo, so the app can inform the user rather than silently cutting off.

    IMPORTANT: this runs in TWO PHASES rather than chunking as it walks.
    Phase 1 collects every eligible file path across the whole repo tree.
    Phase 2 sorts them (core source first, tests/docs/examples last via
    _file_priority) and only THEN chunks them in that order, stopping at
    MAX_TOTAL_CHUNKS. Without this, a single-pass walk-and-chunk approach
    could exhaust the chunk budget on non-essential files before ever
    reaching a large repo's actual interconnected core -- silently leaving
    features like the architecture diagram with nothing meaningful to work
    with, even though the repo genuinely has plenty of real internal
    imports.
    """
    file_stats = {}
    indexed_files = set()
    truncated = False

    # --- Phase 1: collect every eligible file path (no chunking yet) ---
    eligible_files = []  # list of (rel_path, fpath, ext, name_no_ext)
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            fname_lower = fname.lower()
            if fname_lower in SKIP_FILENAMES or fname_lower.endswith(SKIP_FILE_SUFFIXES):
                continue

            ext = os.path.splitext(fname)[1]
            name_no_ext = os.path.splitext(fname)[0].lower()
            if ext not in INCLUDE_EXTENSIONS and name_no_ext not in INCLUDE_FILENAMES:
                continue

            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, repo_dir)
            eligible_files.append((rel_path, fpath, ext, name_no_ext))

    # --- Phase 2: sort so core source is chunked before low-priority dirs ---
    eligible_files.sort(key=lambda item: (_file_priority(item[0]), item[0]))

    # --- Phase 3: chunk in priority order, stopping at the budget ---
    chunks = []
    for rel_path, fpath, ext, name_no_ext in eligible_files:
        if len(chunks) >= MAX_TOTAL_CHUNKS:
            truncated = True
            break

        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            continue

        if len(lines) > MAX_FILE_LINES:
            continue  # skip pathologically huge single files (usually generated/data dumps)
        if not lines:
            continue  # skip empty files -- nothing to chunk or embed

        label = ext if ext else name_no_ext
        file_stats[label] = file_stats.get(label, 0) + 1
        indexed_files.add(rel_path)

        file_text = "".join(lines)
        file_chunks = None
        if ext == ".py":
            file_chunks = chunk_python_by_ast(file_text)

        if file_chunks is not None:
            for c in file_chunks:
                chunks.append({"text": c["text"], "file": rel_path, "start_line": c["start_line"]})
        else:
            # Line-based fallback (non-Python files, or Python with a syntax error)
            step = CHUNK_LINES - CHUNK_OVERLAP
            for start in range(0, len(lines), step):
                chunk_lines = lines[start:start + CHUNK_LINES]
                text = "".join(chunk_lines).strip()
                if not text:
                    continue
                chunks.append({
                        "text": text,
                        "file": rel_path,
                        "start_line": start + 1,
                    })

    stats = {"files": file_stats, "total_files": len(indexed_files), "truncated": truncated}
    return chunks, stats


def get_user_collection_name(username: str) -> str:
    """Derives a safe, per-user ChromaDB collection name so each logged-in
    user's indexed repo and chat data are genuinely isolated from every
    other user -- not just separated in the UI. ChromaDB collection names
    only allow letters, digits, underscores, and hyphens, so this strips
    anything else out defensively (auth.py already restricts usernames to
    a safe character set at signup, but this stays defensive in case that
    ever changes)."""
    safe = "".join(c for c in username if c.isalnum() or c in ("_", "-"))
    return f"{COLLECTION_NAME}_{safe}" if safe else COLLECTION_NAME


def get_user_repo_collection_name(username: str, repo_url: str) -> str:
    """Derives a collection name unique to this (user, repo) PAIR, not just
    the user. This is what allows a user to have several previously-indexed
    repos coexisting at once -- switching back to one they visited before
    (e.g. clicking it in "Recent repos") can reuse its existing vector data
    instead of re-cloning and re-embedding it from scratch every time."""
    user_part = get_user_collection_name(username)
    repo_hash = hashlib.md5(repo_url.strip().lower().encode("utf-8")).hexdigest()[:10]
    return f"{user_part}_{repo_hash}"


def collection_exists(db_path: str, collection_name: str) -> bool:
    """Checks whether a collection already has data, without raising if it
    doesn't -- ChromaDB's get_collection() throws an exception for a
    missing collection rather than returning None, so this wraps that into
    a simple boolean."""
    try:
        client = chromadb.PersistentClient(path=db_path)
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


def embed_and_store(chunks, repo_url: str, collection_name: str = COLLECTION_NAME):
    print(f"[2/4] Loading embedding model ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"[3/4] Embedding {len(chunks)} chunks ...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    print(f"[4/4] Writing to ChromaDB at {DB_PATH} (collection: {collection_name}) ...")
    client = chromadb.PersistentClient(path=DB_PATH)
    # Fresh collection per ingest run so old repo data doesn't mix in.
    # Using a per-user collection_name (see get_user_collection_name) means
    # this only clears THIS user's previous repo, never another user's data.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    ids = [f"{c['file']}::{c['start_line']}::{i}" for i, c in enumerate(chunks)]
    metadatas = [{"file": c["file"], "start_line": c["start_line"], "repo": repo_url} for c in chunks]

    # Write in batches so ChromaDB doesn't choke on one huge payload for large repos.
    write_batch = 500
    for start in range(0, len(chunks), write_batch):
        end = start + write_batch
        collection.add(
            ids=ids[start:end],
            embeddings=[e.tolist() for e in embeddings[start:end]],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
    print(f"Done. Indexed {len(chunks)} chunks from {repo_url}.")


def ingest_repo(repo_url: str, collection_name: str = COLLECTION_NAME):
    tmp_dir = tempfile.mkdtemp()
    try:
        clone_repo(repo_url, tmp_dir)
        chunks, stats = collect_chunks(tmp_dir)
        if not chunks:
            raise ValueError("No indexable files found in this repo.")
        embed_and_store(chunks, repo_url, collection_name=collection_name)
        return len(chunks), stats
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <github_repo_url>")
        sys.exit(1)
    n, stats = ingest_repo(sys.argv[1])
    print(f"Indexed {n} chunks across {stats['total_files']} files: {stats['files']}")