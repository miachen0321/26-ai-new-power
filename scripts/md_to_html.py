"""Convert a Markdown paper (with pandoc-style YAML front matter) to HTML.

Usage:
    python3 scripts/md_to_html.py <input.md> <output.html>

The output links the shared paper stylesheet `report/paper.css`, mirrors the
pandoc title-block structure (header#title-block-header with h1.title,
p.subtitle, p.author, p.date) so existing CSS classes still apply, and keeps
relative image paths intact.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import markdown

CSS_REL_PATH = "paper.css"

BODY_EXTRA_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "smarty"]


def strip_front_matter(text: str) -> tuple[str, dict[str, str]]:
    """Split YAML-style front matter; returns (body, metadata) with both keys.

    Both quoted and bare values are accepted."""
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    meta: dict[str, str] = {}
    if not match:
        return text, meta
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return text[match.end():], meta


def render(body_markdown: str, meta: dict[str, str]) -> str:
    body_html = markdown.markdown(body_markdown, extensions=BODY_EXTRA_EXTENSIONS)
    title = html.escape(meta.get("title", ""))
    subtitle = html.escape(meta.get("subtitle", ""))
    author = html.escape(meta.get("author", ""))
    date = html.escape(meta.get("date", ""))
    lang = html.escape(meta.get("lang", "zh-TW"))
    return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="author" content="{author}" />
  <meta name="dcterms.date" content="{date}" />
  <title>{title}</title>
  <link rel="stylesheet" href="{CSS_REL_PATH}" />
</head>
<body>
<header id="title-block-header">
  <h1 class="title">{title}</h1>
  <p class="subtitle">{subtitle}</p>
  <p class="author">{author}</p>
  <p class="date">{date}</p>
</header>
{body_html}
</body>
</html>
"""


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: python3 {sys.argv[0]} <input.md> <output.html>")
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    body, meta = strip_front_matter(source.read_text(encoding="utf-8"))
    destination.write_text(render(body, meta), encoding="utf-8")
    print(f"Converted {source} -> {destination}")


if __name__ == "__main__":
    main()
