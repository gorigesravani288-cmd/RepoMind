"""
diagram.py
----------
Generates a Mermaid.js flowchart showing how the Python modules/files in an
indexed repo depend on each other, based on their import statements.

IMPORTANT: ingest.py deletes the cloned repo from disk right after indexing
(see the `finally: shutil.rmtree(...)` in ingest_repo). So this file does NOT
read from the filesystem at all -- it rebuilds everything from the chunk
text + metadata that's already permanently stored in ChromaDB. This means
diagram generation works at any point after indexing, with zero extra
system dependencies (no Graphviz install needed).

For readability, the diagram:
  - Groups files into folder-based subgraphs (e.g. "src/flask" as one
    cluster), so you can see architectural boundaries, not just a flat list.
  - Uses short "folder/file.py" labels instead of full paths.
  - Excludes non-core noise (examples/, tests/, docs/) by default, since
    those clutter "what is the actual architecture" with tutorial/test code.
  - Trims to the most-connected files if the repo is still large after that.

INPUT:  the same DB_PATH / COLLECTION_NAME used by app.py and ingest.py
OUTPUT: a Mermaid diagram rendered inline in the Streamlit app
"""

import ast
import re
import chromadb

# Folders that are usually not "the architecture" -- tutorials, tests, docs.
# Excluded by default to keep the diagram focused on core source code.
EXCLUDE_TOP_DIRS = {
    "examples", "example", "tests", "test", "docs", "doc",
    ".github", "benchmarks", "benchmark", "scripts", "tools",
}


def _path_parts(path: str):
    """Splits a path into its segments, handling both '/' and '\\' separators
    (ingest.py's os.path.relpath() produces OS-native separators, so a repo
    indexed on Windows uses '\\' while Streamlit Cloud/Linux would use '/')."""
    return [p for p in re.split(r"[\\/]", path) if p]


def _basename_no_ext(path: str) -> str:
    """Extracts just the filename (no folders, no .py extension) from a path."""
    parts = _path_parts(path)
    name = parts[-1] if parts else path
    return name[:-3] if name.endswith(".py") else name


def _top_level_group(path: str) -> str:
    """The first folder in the path, used to cluster files visually.
    Root-level files (no folder) go in a "(root)" group."""
    parts = _path_parts(path)
    return parts[0] if len(parts) > 1 else "(root)"


def _short_label(path: str) -> str:
    """A short, readable label: 'parent_folder/file.py' instead of the full
    path, so diagram nodes stay compact while still being disambiguated from
    same-named files in other folders (e.g. two different __init__.py)."""
    parts = _path_parts(path)
    return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else path)


def _extract_imports_from_text(source_text: str):
    """Safely parse import statements out of a chunk of Python source.
    Returns [] on ANY error -- a single malformed/partial chunk must never
    crash diagram generation for the whole repo."""
    try:
        tree = ast.parse(source_text)
    except Exception:
        return []
    imports = []
    try:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
    except Exception:
        return []
    return imports


def build_repo_module_graph(db_path: str, collection_name: str, exclude_noise: bool = True):
    """
    Reads every already-indexed chunk for the current repo from ChromaDB,
    groups chunk text back together by source file, and extracts import
    relationships between the repo's OWN Python files.

    External/stdlib imports (streamlit, os, requests, etc.) are intentionally
    ignored so the diagram stays focused on internal architecture.

    Returns: dict {file_path: set(other_file_paths_it_imports)}
             Returns {} (empty, not an error) if anything goes wrong or the
             repo has no Python files -- caller checks for this and shows a
             friendly message instead of a broken diagram.
    """
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(collection_name)
        all_data = collection.get(include=["documents", "metadatas"])
    except Exception:
        return {}

    # Group chunk text by file (ChromaDB does NOT return chunks in original
    # file order, so we must keep each chunk's start_line and sort before
    # joining -- otherwise ast.parse() gets scrambled/invalid source and
    # silently fails on every file).
    file_chunks = {}
    for doc, meta in zip(all_data.get("documents", []), all_data.get("metadatas", [])):
        f = meta.get("file", "")
        if not f.endswith(".py"):
            continue
        if exclude_noise and _top_level_group(f).lower() in EXCLUDE_TOP_DIRS:
            continue
        start_line = meta.get("start_line", 0)
        file_chunks.setdefault(f, []).append((start_line, doc))

    if not file_chunks:
        return {}

    # Sort each file's chunks back into their original line order.
    file_texts = {
        f: [doc for _, doc in sorted(chunk_list, key=lambda pair: pair[0])]
        for f, chunk_list in file_chunks.items()
    }

    # Map "module name" (filename without extension) -> real file path,
    # so an import like "from utils import x" can be matched to utils.py.
    # "__init__" is deliberately excluded: nearly every package has one, so
    # mapping it by bare filename alone would wrongly wire together unrelated
    # packages that happen to each have their own __init__.py.
    module_to_file = {
        _basename_no_ext(f): f for f in file_texts if _basename_no_ext(f) != "__init__"
    }

    graph = {f: set() for f in file_texts}
    for f, chunks in file_texts.items():
        combined_text = "\n".join(chunks)
        imports = _extract_imports_from_text(combined_text)
        for imp in imports:
            target_file = module_to_file.get(imp)
            if target_file and target_file != f:
                graph[f].add(target_file)

    return graph


def trim_to_most_connected(graph: dict, max_nodes: int = 35):
    """Large repos can have 50-100+ files, which renders as an unreadable
    tangle. This keeps only the most-connected files (by total in+out edges),
    which are almost always the actual architectural core of the project --
    isolated single-file utilities add clutter without insight.

    Returns (trimmed_graph, was_trimmed: bool)."""
    degree = {f: 0 for f in graph}
    for f, targets in graph.items():
        degree[f] += len(targets)
        for t in targets:
            degree[t] = degree.get(t, 0) + 1

    # Drop nodes with zero connections entirely -- an isolated file adds no
    # information to an architecture diagram.
    connected = {f: d for f, d in degree.items() if d > 0}
    if len(connected) <= max_nodes:
        trimmed_files = set(connected.keys())
        was_trimmed = False
    else:
        top_files = sorted(connected.items(), key=lambda pair: -pair[1])[:max_nodes]
        trimmed_files = {f for f, _ in top_files}
        was_trimmed = True

    trimmed_graph = {
        f: {t for t in targets if t in trimmed_files}
        for f, targets in graph.items() if f in trimmed_files
    }
    return trimmed_graph, was_trimmed


def graph_to_mermaid(graph: dict) -> str:
    """Converts the graph dict into Mermaid flowchart syntax, with files
    grouped into folder-based subgraphs for readability.
    Returns "" (empty string, not an error) if there are no internal edges
    to show -- caller should treat this as 'nothing to diagram', not a crash."""
    if not graph or not any(graph.values()):
        return ""

    def safe_id(name: str) -> str:
        # Mermaid node/subgraph IDs can't contain slashes, dots, dashes, spaces,
        # or parentheses safely.
        return re.sub(r"[\\/.\-\s()]", "_", name)

    # Collect every node that appears anywhere (as a source or a target).
    all_nodes = set(graph.keys())
    for targets in graph.values():
        all_nodes.update(targets)

    groups = {}
    for node in all_nodes:
        groups.setdefault(_top_level_group(node), []).append(node)

    lines = ["graph TD"]
    for group_name, nodes in sorted(groups.items()):
        gid = safe_id(group_name) or "root"
        lines.append(f'    subgraph {gid}["📁 {group_name}"]')
        for n in sorted(nodes):
            lines.append(f'        {safe_id(n)}["{_short_label(n)}"]')
        lines.append("    end")

    for f, targets in graph.items():
        for t in targets:
            lines.append(f'    {safe_id(f)} --> {safe_id(t)}')

    return "\n".join(lines)


def render_mermaid(mermaid_code: str, height: int = 550):
    """Renders Mermaid code inline in Streamlit via an embedded HTML component.
    Import is done lazily inside the function so this module can be imported
    even in contexts without Streamlit available (e.g. quick local testing).

    useMaxWidth:false + an explicit scrollable wrapper is important: without
    it, Mermaid auto-scales large diagrams down to fit the container, which
    squishes anything with more than a handful of nodes into an unreadable
    smear. Scrolling (instead of scaling) keeps every node legible.
    scrolling=False on components.html avoids a redundant second scrollbar
    stacked on top of this div's own scrollbar."""
    import streamlit.components.v1 as components

    html = f"""
    <div style="width:100%; overflow-x:auto; overflow-y:auto; max-height:{height-20}px;
                border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:12px;">
      <div class="mermaid">
      {mermaid_code}
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
      mermaid.initialize({{
        startOnLoad: true,
        theme: "neutral",
        flowchart: {{
          useMaxWidth: false,
          htmlLabels: true,
          curve: "linear",
          nodeSpacing: 45,
          rankSpacing: 80
        }}
      }});
    </script>
    """
    components.html(html, height=height, scrolling=False)