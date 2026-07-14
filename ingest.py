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