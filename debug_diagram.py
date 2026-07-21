"""
debug_diagram.py
-----------------
Run this locally (after you've already indexed a repo) to see EXACTLY what's
stored in ChromaDB and why the diagram builder isn't finding relationships.

Usage:
    python debug_diagram.py
"""

import chromadb
from ingest import DB_PATH, COLLECTION_NAME

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)
all_data = collection.get(include=["documents", "metadatas"])

# 1. Show a sample of the raw "file" field exactly as stored -- this reveals
#    whether paths use "/" or "\" as separators.
py_files = sorted(set(
    m["file"] for m in all_data["metadatas"] if m["file"].endswith(".py")
))
print(f"Found {len(py_files)} indexed .py files. First 10:")
for f in py_files[:10]:
    print(f"   repr: {f!r}")

# 2. Show what module name our current matching logic would extract
print("\n--- Module name extraction test (old buggy '/' split) ---")
for f in py_files[:10]:
    old_way = f.rsplit("/", 1)[-1][:-3]
    print(f"   {f!r}  ->  {old_way!r}")

import re
print("\n--- Module name extraction test (fixed, handles '/' AND '\\\\') ---")
for f in py_files[:10]:
    fixed_way = re.split(r"[\\/]", f)[-1][:-3]
    print(f"   {f!r}  ->  {fixed_way!r}")

# 3. Try parsing one real file's combined+sorted chunks and show its imports
print("\n--- Import extraction test on first file ---")
import ast
if py_files:
    target = py_files[0]
    chunks = [
        (m["start_line"], d) for d, m in zip(all_data["documents"], all_data["metadatas"])
        if m["file"] == target
    ]
    chunks.sort(key=lambda pair: pair[0])
    combined = "\n".join(doc for _, doc in chunks)
    print(f"File: {target}")
    try:
        tree = ast.parse(combined)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        print(f"Parsed OK. Imports found: {imports}")
    except SyntaxError as e:
        print(f"SyntaxError while parsing combined chunks: {e}")
        print("First 300 chars of combined text:")
        print(combined[:300])