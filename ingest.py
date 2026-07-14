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
}
# Common extension-less files worth indexing (case-insensitive match on filename)
INCLUDE_FILENAMES = {"readme", "license", "contributing", "changelog", "makefile", "dockerfile"}

CHUNK_LINES = 60      # lines per chunk
CHUNK_OVERLAP = 10    # overlapping lines between consecutive chunks
DB_PATH = "./chroma_db"
COLLECTION_NAME = "repo_chunks"


def clone_repo(repo_url: str, dest: str) -> None:
    print(f"[1/4] Cloning {repo_url} ...")
    git.Repo.clone_from(repo_url, dest, depth=1)


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


def collect_chunks(repo_dir: str):
    """Walk the repo and split eligible files into overlapping line-based chunks.
    Also returns a stats dict: {"files": {ext_or_name: count}, "total_files": N}
    """
    chunks = []
    file_stats = {}
    indexed_files = set()

    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            ext = os.path.splitext(fname)[1]
            name_no_ext = os.path.splitext(fname)[0].lower()
            if ext not in INCLUDE_EXTENSIONS and name_no_ext not in INCLUDE_FILENAMES:
                continue
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, repo_dir)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except Exception:
                continue

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

    stats = {"files": file_stats, "total_files": len(indexed_files)}
    return chunks, stats


def embed_and_store(chunks, repo_url: str):
    print(f"[2/4] Loading embedding model ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"[3/4] Embedding {len(chunks)} chunks ...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    print(f"[4/4] Writing to ChromaDB at {DB_PATH} ...")
    client = chromadb.PersistentClient(path=DB_PATH)
    # Fresh collection per ingest run so old repo data doesn't mix in.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids = [f"{c['file']}::{c['start_line']}::{i}" for i, c in enumerate(chunks)]
    metadatas = [{"file": c["file"], "start_line": c["start_line"], "repo": repo_url} for c in chunks]

    collection.add(
        ids=ids,
        embeddings=[e.tolist() for e in embeddings],
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Done. Indexed {len(chunks)} chunks from {repo_url}.")


def ingest_repo(repo_url: str):
    tmp_dir = tempfile.mkdtemp()
    try:
        clone_repo(repo_url, tmp_dir)
        chunks, stats = collect_chunks(tmp_dir)
        if not chunks:
            raise ValueError("No indexable files found in this repo.")
        embed_and_store(chunks, repo_url)
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