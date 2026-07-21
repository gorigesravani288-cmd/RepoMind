if st.session_state.get("messages"):
            import html as _html

            def _esc(text):
                return _html.escape(str(text))

            repo_name = st.session_state["indexed_repo"].split("/")[-1]
            html_parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>RepoMind conversation — {_esc(repo_name)}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 900px;
          margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; border-bottom: 2px solid #eee; padding-bottom: 0.5rem; }}
  .msg {{ margin: 1.2rem 0; padding: 0.8rem 1rem; border-radius: 8px; }}
  .user {{ background: #f0f4ff; }}
  .assistant {{ background: #f7f7f7; }}
  .role {{ font-weight: 600; margin-bottom: 0.3rem; }}
  .sources {{ font-size: 0.85rem; color: #555; margin-top: 0.5rem; }}
  .mermaid {{ background: #fff; padding: 1rem; border-radius: 8px; margin-top: 0.8rem; }}
</style></head><body>
<h1>🧠 RepoMind conversation — {_esc(repo_name)}</h1>
"""]

            if st.session_state.get("repo_summary"):
                html_parts.append(
                    f'<div class="msg assistant"><div class="role">Repo Summary</div>'
                    f'<div>{_esc(st.session_state["repo_summary"])}</div></div>'
                )

            for m in st.session_state["messages"]:
                role_label = "You" if m["role"] == "user" else "RepoMind"
                css_class = "user" if m["role"] == "user" else "assistant"
                html_parts.append(f'<div class="msg {css_class}"><div class="role">{role_label}</div>')
                html_parts.append(f'<div>{_esc(m["content"])}</div>')
                if m.get("mermaid"):
                    # Rendered live by the same Mermaid.js script the app itself uses --
                    # this is what makes the diagram appear as an actual visual, in the
                    # same file as the text, when the downloaded file is opened.
                    html_parts.append(f'<div class="mermaid">{_esc(m["mermaid"])}</div>')
                if m.get("sources"):
                    srcs = ", ".join(f"{s['file']}:{s['start_line']}" for s in m["sources"])
                    html_parts.append(f'<div class="sources">Sources: {_esc(srcs)}</div>')
                html_parts.append("</div>")

            html_parts.append(
                '<script>mermaid.initialize({startOnLoad:true, theme:"neutral"});</script>'
                "</body></html>"
            )
            html_export = "\n".join(html_parts)

            st.download_button(
                "💾  Download conversation (with diagrams)",
                data=html_export,
                file_name=f"repomind_{repo_name}.html",
                mime="text/html",
                use_container_width=True,
            )