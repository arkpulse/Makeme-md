"""
DocFlow — Command-Line Interface
Convert documents from the terminal.

Usage:
    docflow convert report.pdf
    docflow convert ./docs/ --recursive --output ./output/
    docflow search "machine learning" --top-k 5
    docflow serve --port 8000
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("Install CLI extras: pip install typer rich")
    sys.exit(1)

app = typer.Typer(
    name="docflow",
    help="🧠 DocFlow — Universal AI Document Intelligence Platform",
    add_completion=False,
)
console = Console()


@app.command()
def convert(
    path: str = typer.Argument(..., help="File or folder to convert"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Recurse into folders"),
    ocr: bool = typer.Option(True, "--ocr/--no-ocr", help="Enable OCR"),
    summarize: bool = typer.Option(False, "--summarize", help="Generate AI summaries"),
    embed: bool = typer.Option(False, "--embed", help="Generate embeddings"),
    chunk_strategy: str = typer.Option("recursive", "--chunk-strategy", help="Chunking strategy"),
    chunk_size: int = typer.Option(512, "--chunk-size", help="Chunk size in characters"),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format: markdown|json|both"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Convert one or more documents to Markdown/JSON."""
    sys.path.insert(0, str(Path(__file__).parent))
    from core.pipeline import DocFlowPipeline, ProcessingOptions

    input_path = Path(path)
    output_dir = Path(output) if output else input_path.parent / "docflow_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    opts = ProcessingOptions(
        enable_ocr=ocr,
        enable_summarization=summarize,
        enable_embeddings=embed,
        store_in_vectordb=False,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
    )

    pipeline = DocFlowPipeline()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Processing...", total=None)

        if input_path.is_file():
            results = [asyncio.run(pipeline.process(input_path, opts))]
        else:
            results = asyncio.run(pipeline.process_folder(input_path, opts, recursive=recursive))

        progress.update(task, completed=True)

    # Output results
    success_count = 0
    for result in results:
        if not result.success:
            console.print(f"[red]✗ {result.file_name}: {result.errors}[/red]")
            continue

        success_count += 1

        if format in ("markdown", "both"):
            md_path = output_dir / f"{Path(result.file_name).stem}.md"
            md_path.write_text(result.markdown, encoding="utf-8")

        if format in ("json", "both"):
            json_path = output_dir / f"{Path(result.file_name).stem}.json"
            json_path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")

        if verbose:
            console.print(f"[green]✓ {result.file_name}[/green] → {len(result.chunks)} chunks, {result.processing_time_s:.2f}s")
        else:
            console.print(f"[green]✓[/green] {result.file_name}")

    console.print(f"\n[bold]Done:[/bold] {success_count}/{len(results)} files → [cyan]{output_dir}[/cyan]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    vector_store: str = typer.Option("chroma", "--store", help="Vector store backend"),
):
    """Perform semantic search over indexed documents."""
    sys.path.insert(0, str(Path(__file__).parent))
    from embeddings.engine import EmbeddingEngine
    from vectorstores.base import get_vector_store

    async def _search():
        engine = EmbeddingEngine()
        store = get_vector_store(vector_store)
        embedding = await engine.embed(query)
        return await store.search(embedding, top_k=top_k)

    results = asyncio.run(_search())

    if not results:
        console.print("[yellow]No results found. Index some documents first with `docflow convert --embed`[/yellow]")
        return

    table = Table(title=f'Search: "{query}"', show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", style="cyan", width=8)
    table.add_column("Source", style="green", width=20)
    table.add_column("Text", style="white")

    for i, hit in enumerate(results, 1):
        table.add_row(
            str(i),
            f"{hit.get('score', 0):.3f}",
            hit.get("source", "unknown"),
            hit.get("text", "")[:200] + "..." if len(hit.get("text", "")) > 200 else hit.get("text", ""),
        )

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    workers: int = typer.Option(1, "--workers"),
    reload: bool = typer.Option(False, "--reload"),
):
    """Start the DocFlow API server."""
    import uvicorn
    console.print(f"[cyan]🚀 DocFlow API starting on http://{host}:{port}[/cyan]")
    console.print(f"[dim]Docs: http://localhost:{port}/docs[/dim]")
    uvicorn.run("api.main:app", host=host, port=port, workers=workers, reload=reload)


@app.command()
def ui():
    """Launch the Streamlit dashboard."""
    import subprocess
    console.print("[cyan]🌐 DocFlow UI starting on http://localhost:8501[/cyan]")
    subprocess.run(["streamlit", "run", "frontend/app.py"])


def main():
    app()


if __name__ == "__main__":
    main()
