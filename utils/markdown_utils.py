"""
DocFlow — Markdown Utilities
"""
from __future__ import annotations

import re


def tables_to_markdown(rows: list[list[str]]) -> str:
    """Convert a 2D list to a Markdown table."""
    if not rows:
        return ""

    # Normalise rows to strings
    str_rows = [[str(cell).replace("|", "\\|") for cell in row] for row in rows]

    # Column widths
    col_count = max(len(row) for row in str_rows)
    str_rows = [row + [""] * (col_count - len(row)) for row in str_rows]
    widths = [max(len(row[c]) for row in str_rows) for c in range(col_count)]

    def fmt_row(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    lines = [
        fmt_row(str_rows[0]),
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    for row in str_rows[1:]:
        lines.append(fmt_row(row))

    return "\n".join(lines)


def clean_markdown(text: str) -> str:
    """Remove excessive whitespace and normalise Markdown."""
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing spaces
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()
