"""
DocFlow — XLSX / Excel Parser
Converts spreadsheets to Markdown tables and structured JSON.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from utils.markdown_utils import tables_to_markdown


class XLSXParser:
    async def parse(self, file_path: Path, options: Any) -> dict:
        logger.info(f"[XLSXParser] Parsing: {file_path.name}")
        try:
            import pandas as pd
        except ImportError:
            return {"markdown": "", "text": "", "structured": {},
                    "tables": [], "images": [], "metadata": {}, "hyperlinks": []}

        ext = file_path.suffix.lower()
        if ext == ".csv":
            sheets = {"Sheet1": pd.read_csv(str(file_path))}
        else:
            xl = pd.ExcelFile(str(file_path))
            sheets = {name: xl.parse(name) for name in xl.sheet_names}

        md_parts: list[str] = []
        tables_data: list[dict] = []

        for sheet_name, df in sheets.items():
            # Clean NaN
            df = df.fillna("").astype(str)
            md_parts.append(f"\n## Sheet: {sheet_name}\n")

            rows = [list(df.columns)] + df.values.tolist()
            md_table = tables_to_markdown(rows)
            md_parts.append(md_table)

            tables_data.append({
                "sheet": sheet_name,
                "rows": len(df),
                "columns": list(df.columns),
                "data": df.to_dict(orient="records"),
                "markdown": md_table,
            })

        metadata = {
            "sheet_count": len(sheets),
            "sheet_names": list(sheets.keys()),
        }

        full_md = "\n\n".join(md_parts)
        return {
            "markdown": full_md,
            "text": full_md,
            "structured": {s: tables_data[i] for i, s in enumerate(sheets)},
            "tables": tables_data,
            "images": [],
            "metadata": metadata,
            "hyperlinks": [],
        }


class CSVParser(XLSXParser):
    """CSV is a special case of XLSX parsing."""
    pass
