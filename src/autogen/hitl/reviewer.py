"""HITL reviewer: render anonymized tables for user confirmation before cloud codegen."""

import html
import sys
import webbrowser
from pathlib import Path

from src import config as cfg
from src.autogen.models import ExtractedTable
from src.logger import get_logger

_BANNER = "This data will be sent to OpenRouter API. Verify no sensitive values are visible."
_log = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

_STYLE = """
body { font-family: sans-serif; max-width: 1400px; margin: 2rem auto; }
.banner { background: #fff3cd; border: 1px solid #ffc107; padding: 1rem; margin-bottom: 1.5rem; font-weight: bold; }
details { margin-bottom: 1rem; border: 1px solid #ccc; padding: 0.5rem; }
summary { cursor: pointer; font-weight: bold; }
.compare { display: flex; gap: 1rem; align-items: flex-start; }
.panel { flex: 1; min-width: 0; overflow-x: auto; }
.panel-label { font-weight: bold; margin-bottom: 0.3rem; }
table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
th, td { border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: left; }
th { background: #f5f5f5; }
td.changed { background: #fff3cd; }
.tabs { display: flex; flex-wrap: wrap; }
.tabs > input[type=radio] { display: none; }
.tabs > label { padding: 0.5rem 1rem; border: 1px solid #ccc; border-bottom: none; cursor: pointer; background: #f5f5f5; }
.tabs > .tab { display: none; order: 99; width: 100%; border: 1px solid #ccc; padding: 0.5rem; }
.tabs > input:checked + label { background: #fff; font-weight: bold; }
.tabs > input:checked + label + .tab { display: block; }
.tabs.outer > label { background: #e9ecef; }
.tabs.outer > input:checked + label { background: #fff3cd; }
.level-hint { font-size: 0.85rem; color: #6c757d; margin: 0.25rem 0 0.5rem; }
"""


def _render_table(rows: list[list[str]], compare_rows: list[list[str]] | None = None) -> str:
    """Build a <table> from rows; tint cells differing from compare_rows as 'changed'."""
    if not rows:
        return "<p><em>empty</em></p>"
    header = "".join(f"<th>{html.escape(c)}</th>" for c in rows[0])
    data_rows = ""
    for i, row in enumerate(rows[1:], start=1):
        base = compare_rows[i] if compare_rows and i < len(compare_rows) else None
        cells = ""
        for j, cell in enumerate(row):
            changed = base is not None and (j >= len(base) or cell != base[j])
            cls = " class='changed'" if changed else ""
            cells += f"<td{cls}>{html.escape(cell)}</td>"
        data_rows += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{header}</tr></thead><tbody>{data_rows}</tbody></table>"


def _render_group(originals: list[ExtractedTable], anons: list[ExtractedTable]) -> str:
    """Numbered side-by-side comparisons for one PDF's tables."""
    parts: list[str] = []
    for n, (original, anon) in enumerate(zip(originals, anons), start=1):
        title = html.escape(anon.name or original.name or "")
        label = f"Table {n}" + (f" &mdash; {title}" if title else "")
        left = _render_table(original.rows)
        right = _render_table(anon.rows, compare_rows=original.rows)
        body = (
            "<div class='compare'>"
            f"<div class='panel'><div class='panel-label'>Non-Anonymized</div>{left}</div>"
            f"<div class='panel'><div class='panel-label'>Anonymized</div>{right}</div>"
            "</div>"
        )
        parts.append(f"<details open><summary>{label}</summary>{body}</details>")
    return "\n".join(parts) or "<p><em>no tables</em></p>"


Group = tuple[str, list[tuple[str, list[ExtractedTable], list[ExtractedTable]]]]


def render_preview_html(groups: list[Group]) -> str:
    """Return self-contained HTML with two-tier CSS-only tabs.

    Level 1 = extractor; level 2 = attachment. Each ``group`` is ``(extractor, attachments)`` where
    each attachment is ``(attachment_id, originals, anons)``. The whole preview is approved as one.
    """
    outer: list[str] = []
    for i, (extractor, attachments) in enumerate(groups):
        ext_checked = " checked" if i == 0 else ""
        inner: list[str] = []
        for j, (label, originals, anons) in enumerate(attachments):
            att_checked = " checked" if j == 0 else ""
            inner.append(
                f"<input type='radio' name='att{i}' id='att{i}_{j}'{att_checked}>"
                f"<label for='att{i}_{j}'>{html.escape(label)}</label>"
                f"<div class='tab'>{_render_group(originals, anons)}</div>"
            )
        inner_html = (
            "<div class='level-hint'>Attachment comparison &mdash; original vs anonymized</div>"
            f"<div class='tabs'>{''.join(inner) or '<p><em>no attachments</em></p>'}</div>"
        )
        outer.append(
            f"<input type='radio' name='ext' id='ext{i}'{ext_checked}>"
            f"<label for='ext{i}'>{html.escape(extractor)}</label>"
            f"<div class='tab'>{inner_html}</div>"
        )
    body = (
        "<div class='level-hint'>Extractor</div>"
        f"<div class='tabs outer'>{''.join(outer) or '<p><em>no extractors</em></p>'}</div>"
    )
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>HITL Preview</title><style>{_STYLE}</style></head>"
        f"<body><div class='banner'>{_BANNER}</div>{body}</body></html>"
    )


def review_and_confirm(groups: list[Group], preview_path: str) -> bool:
    """Render the comparison, open in browser, prompt; return True iff confirmed."""
    if not sys.stdin.isatty():
        raise RuntimeError(
            "HITL review requires an interactive terminal. "
            "Run with codegen_backend='local' for non-interactive use."
        )

    path = Path(preview_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_preview_html(groups), encoding="utf-8")
    _log.info("HITL preview written: %s", preview_path)

    webbrowser.open(path.absolute().as_uri())
    print(f"\nPreview written to: {preview_path}")
    print("Review the tables above and confirm no sensitive data is visible.\n")

    answer = input("Proceed? [y/N]: ").strip().lower()
    approved = answer == "y"
    _log.info("HITL %s", "approved" if approved else "declined")
    return approved
