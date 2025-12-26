"""
CLI for sf-knowledge-tools.

Commands:
  ingest  - Add PDF documents to the knowledge base
  query   - Search the knowledge base
  export  - Generate markdown from retrieved content
  status  - Show knowledge base statistics
"""

import click
from pathlib import Path
from typing import Optional
import yaml

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def get_config() -> dict:
    """Load configuration from knowledge.yml."""
    config_paths = [
        Path.cwd() / "config" / "knowledge.yml",
        Path(__file__).parent.parent / "config" / "knowledge.yml",
        Path.home() / ".sf-knowledge" / "knowledge.yml"
    ]

    for path in config_paths:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)

    # Return defaults
    return {
        'storage': {
            'database_path': './data/knowledge.db',
            'pdf_dir': './pdfs/'
        },
        'embeddings': {
            'model': 'BAAI/bge-large-en-v1.5',
            'dimensions': 1024,
            'batch_size': 32
        },
        'chunking': {
            'target_size': 1000,
            'max_size': 1500,
            'overlap': 100
        },
        'search': {
            'default_k': 5,
            'hybrid_weight': 0.7
        }
    }


def get_db_path(config: dict) -> Path:
    """Get database path from config."""
    db_path = config.get('storage', {}).get('database_path', './data/knowledge.db')
    return Path(db_path).expanduser()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """
    📚 sf-knowledge-tools - Local PDF knowledge extraction and RAG.

    Process Salesforce documentation PDFs, extract knowledge,
    and export as markdown for sf-skills.
    """
    pass


@main.command()
@click.argument('pdf_path', type=click.Path(exists=True, path_type=Path))
@click.option('--category', '-c', help='Category (apex, flow, lwc, integration, etc.)')
@click.option('--tags', '-t', multiple=True, help='Tags for the document')
@click.option('--title', help='Override document title')
def ingest(pdf_path: Path, category: str, tags: tuple, title: str):
    """
    📄 Ingest a PDF document into the knowledge base.

    Extracts text, creates semantic chunks, and generates embeddings.

    Example:
        sf-knowledge ingest salesforce-apex-guide.pdf --category apex
    """
    import hashlib
    import uuid
    import json

    from .ingester.pdf_extractor import PDFExtractor
    from .chunker.semantic_chunker import SemanticChunker
    from .embedder.embedding_client import EmbeddingClient, EmbeddingConfig
    from .storage.vector_store import VectorStore

    config = get_config()
    db_path = get_db_path(config)

    console.print(f"\n[bold blue]📄 Ingesting:[/] {pdf_path.name}")

    # Initialize components
    vector_store = VectorStore(db_path)

    # Check for duplicates
    with open(pdf_path, 'rb') as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()

    existing = vector_store.get_document_by_hash(content_hash)
    if existing:
        console.print(f"[yellow]⚠️  Document already exists:[/] {existing['title']}")
        console.print(f"   ID: {existing['id']}")
        return

    # Generate document ID
    doc_id = str(uuid.uuid4())[:8]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:

        # Extract PDF
        task1 = progress.add_task("Extracting PDF...", total=None)
        with PDFExtractor(pdf_path) as extractor:
            metadata = extractor.get_metadata()
            pages = list(extractor.extract_all())
            toc_map = extractor.get_toc_structure()
        progress.update(task1, completed=True, total=1)

        # Add document to database
        task2 = progress.add_task("Registering document...", total=1)
        doc_title = title or metadata.title or pdf_path.stem
        vector_store.add_document(
            doc_id=doc_id,
            filename=pdf_path.name,
            filepath=str(pdf_path.absolute()),
            title=doc_title,
            content_hash=content_hash,
            page_count=metadata.page_count,
            file_size=pdf_path.stat().st_size,
            category=category,
            tags=list(tags) if tags else None
        )
        progress.update(task2, advance=1)

        # Chunk the document
        task3 = progress.add_task("Chunking content...", total=1)
        chunker = SemanticChunker(
            target_tokens=config['chunking']['target_size'],
            max_tokens=config['chunking']['max_size'],
            overlap_tokens=config['chunking']['overlap']
        )
        chunks = chunker.chunk_document(pages, doc_id, toc_map)
        progress.update(task3, advance=1)

        # Generate embeddings
        task4 = progress.add_task(f"Generating embeddings ({len(chunks)} chunks)...", total=len(chunks))
        embedder = EmbeddingClient(EmbeddingConfig(
            model_name=config['embeddings']['model'],
            batch_size=config['embeddings']['batch_size']
        ))

        # Get embeddings in batch
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_documents(texts, show_progress=False)
        progress.update(task4, completed=len(chunks))

        # Store chunks with embeddings
        task5 = progress.add_task("Storing chunks...", total=len(chunks))
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_store.add_chunk(
                chunk_id=chunk.id,
                document_id=doc_id,
                text=chunk.text,
                embedding=embedding.tolist(),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                chapter=chunk.chapter,
                section=chunk.section,
                subsection=chunk.subsection,
                content_type=chunk.content_type,
                topic_tags=chunk.topic_tags,
                prev_chunk_id=chunk.prev_chunk_id,
                next_chunk_id=chunk.next_chunk_id
            )
            progress.update(task5, advance=1)

        # Update document status
        vector_store.update_document_status(doc_id, 'completed')

    # Show summary
    console.print(f"\n[bold green]✅ Successfully ingested:[/] {doc_title}")

    table = Table(show_header=False, box=None)
    table.add_column("Field", style="dim")
    table.add_column("Value")
    table.add_row("Document ID", doc_id)
    table.add_row("Pages", str(metadata.page_count))
    table.add_row("Chunks", str(len(chunks)))
    table.add_row("Category", category or "unset")
    table.add_row("Tags", ", ".join(tags) if tags else "none")

    console.print(table)


@main.command()
@click.argument('query_text')
@click.option('--skill', '-s', help='Filter by skill (apex, flow, lwc, etc.)')
@click.option('--category', '-c', help='Filter by document category')
@click.option('--limit', '-k', default=5, help='Number of results (default: 5)')
@click.option('--show-context', is_flag=True, help='Include surrounding chunks')
@click.option('--format', '-f', 'output_format', type=click.Choice(['rich', 'json', 'markdown']), default='rich')
def query(query_text: str, skill: str, category: str, limit: int, show_context: bool, output_format: str):
    """
    🔍 Search the knowledge base.

    Returns semantically similar chunks with citations.

    Example:
        sf-knowledge query "bulkification patterns" --skill apex
    """
    import json as json_module

    from .query.rag_engine import create_rag_engine

    config = get_config()
    db_path = get_db_path(config)

    console.print(f"\n[bold blue]🔍 Searching:[/] {query_text}\n")

    # Create RAG engine
    engine = create_rag_engine(
        db_path=db_path,
        model_name=config['embeddings']['model'],
        default_k=limit,
        hybrid_weight=config['search']['hybrid_weight']
    )

    # Execute query
    with console.status("[bold green]Searching..."):
        result = engine.query(
            query_text=query_text,
            k=limit,
            category_filter=category,
            include_context=show_context
        )

    if not result.chunks:
        console.print("[yellow]No results found.[/]")
        return

    # Log query for analytics
    engine.log_query(result, skill)

    if output_format == 'json':
        output = {
            'query': result.query,
            'total_matches': result.total_matches,
            'search_time_ms': result.search_time_ms,
            'chunks': result.chunks,
            'sources': result.get_unique_sources()
        }
        console.print(json_module.dumps(output, indent=2))
        return

    if output_format == 'markdown':
        console.print(f"# Results for: {query_text}\n")
        for i, chunk in enumerate(result.chunks, 1):
            console.print(f"## Result {i}")
            console.print(f"**Source:** {chunk.get('chapter', 'Unknown')} | p. {chunk.get('page_start', '?')}\n")
            console.print(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
            console.print()
        return

    # Rich format (default)
    console.print(f"[dim]Found {result.total_matches} results in {result.search_time_ms:.1f}ms[/]\n")

    for i, chunk in enumerate(result.chunks, 1):
        is_context = chunk.get('is_context', False)
        score = chunk.get('score', 0)

        # Header with source info
        chapter = chunk.get('chapter', 'Unknown')
        section = chunk.get('section', '')
        page = chunk.get('page_start', '?')

        location = f"Chapter: {chapter}" if chapter else ""
        if section:
            location += f" > {section}"
        location += f" | p. {page}"

        if is_context:
            header = f"[dim]📎 Context chunk[/]"
        else:
            header = f"[bold]📌 Result {i}[/] [dim](score: {score:.3f})[/]"

        panel_content = chunk['text'][:800]
        if len(chunk['text']) > 800:
            panel_content += "..."

        panel = Panel(
            panel_content,
            title=header,
            subtitle=f"[dim]{location}[/]",
            border_style="blue" if not is_context else "dim"
        )
        console.print(panel)
        console.print()

    # Show sources
    sources = result.get_unique_sources()
    if sources:
        console.print("[bold]📚 Sources:[/]")
        for source in sources:
            console.print(f"  • {source}")


@main.command()
@click.argument('topic')
@click.option('--skill', '-s', required=True, help='Target skill (apex, flow, lwc, etc.)')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output file path')
@click.option('--limit', '-k', default=10, help='Number of chunks to include (default: 10)')
@click.option('--template', '-t', help='Custom Jinja2 template name')
def export(topic: str, skill: str, output: Path, limit: int, template: str):
    """
    📝 Export knowledge as markdown.

    Retrieves relevant chunks and generates a formatted markdown document.

    Example:
        sf-knowledge export "Apex Bulkification" --skill sf-apex -o patterns.md
    """
    from .export.markdown_generator import MarkdownGenerator
    from .query.rag_engine import create_rag_engine

    config = get_config()
    db_path = get_db_path(config)

    console.print(f"\n[bold blue]📝 Exporting:[/] {topic}")
    console.print(f"[dim]Skill: {skill}[/]\n")

    # Create RAG engine and retrieve content
    engine = create_rag_engine(
        db_path=db_path,
        model_name=config['embeddings']['model']
    )

    with console.status("[bold green]Retrieving content..."):
        result = engine.search_by_topic(topic, skill=skill, k=limit)

    if not result.chunks:
        console.print("[yellow]No content found for this topic.[/]")
        return

    # Generate markdown
    generator = MarkdownGenerator(template_name=template)

    markdown_content = generator.generate(
        topic=topic,
        skill=skill,
        result=result
    )

    # Determine output path
    if not output:
        # Use exports/<skill>/<topic-slug>.md
        slug = topic.lower().replace(' ', '-').replace('/', '-')
        slug = ''.join(c for c in slug if c.isalnum() or c == '-')
        exports_dir = Path.cwd() / "exports" / skill
        exports_dir.mkdir(parents=True, exist_ok=True)
        output = exports_dir / f"{slug}.md"

    # Write output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown_content)

    console.print(f"[bold green]✅ Exported to:[/] {output}")

    # Show preview
    console.print("\n[bold]Preview:[/]")
    preview_lines = markdown_content.split('\n')[:20]
    console.print(Markdown('\n'.join(preview_lines) + '\n...'))

    # Log export
    from .storage.vector_store import VectorStore
    store = VectorStore(db_path)
    store.log_export(
        topic=topic,
        skill=skill,
        output_path=str(output),
        chunk_ids=[c['chunk_id'] for c in result.chunks]
    )


@main.command()
def status():
    """
    📊 Show knowledge base statistics.

    Displays document count, chunk count, and storage info.
    """
    from .storage.vector_store import VectorStore

    config = get_config()
    db_path = get_db_path(config)

    if not db_path.exists():
        console.print("[yellow]⚠️  No knowledge base found.[/]")
        console.print(f"[dim]Expected at: {db_path}[/]")
        console.print("\nRun [bold]sf-knowledge ingest[/] to add your first document.")
        return

    store = VectorStore(db_path)
    stats = store.get_stats()

    console.print("\n[bold]📊 Knowledge Base Status[/]\n")

    # Database info
    table = Table(title="Database", show_header=False, box=None)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Location", str(db_path))
    table.add_row("Size", f"{db_path.stat().st_size / (1024*1024):.1f} MB")
    console.print(table)
    console.print()

    # Content stats
    table2 = Table(title="Content", show_header=False, box=None)
    table2.add_column("Metric", style="dim")
    table2.add_column("Value", style="bold")
    table2.add_row("Documents", str(stats.get('document_count', 0)))
    table2.add_row("Chunks", str(stats.get('chunk_count', 0)))
    table2.add_row("Embeddings", str(stats.get('embedding_count', 0)))
    console.print(table2)
    console.print()

    # Categories
    if stats.get('categories'):
        table3 = Table(title="By Category")
        table3.add_column("Category")
        table3.add_column("Documents", justify="right")
        for cat, count in stats['categories'].items():
            table3.add_row(cat or "uncategorized", str(count))
        console.print(table3)
        console.print()

    # Recent queries
    recent_queries = stats.get('recent_queries', [])
    if recent_queries:
        console.print("[bold]Recent Queries:[/]")
        for q in recent_queries[:5]:
            console.print(f"  • {q['query_text'][:50]}... ({q['result_count']} results)")


@main.command()
@click.argument('document_id')
@click.option('--force', '-f', is_flag=True, help='Skip confirmation')
def remove(document_id: str, force: bool):
    """
    🗑️  Remove a document from the knowledge base.
    """
    from .storage.vector_store import VectorStore

    config = get_config()
    db_path = get_db_path(config)

    store = VectorStore(db_path)
    doc = store.get_document(document_id)

    if not doc:
        console.print(f"[red]Document not found:[/] {document_id}")
        return

    if not force:
        console.print(f"[bold]Document:[/] {doc['title']}")
        console.print(f"[dim]File: {doc['filename']}[/]")
        if not click.confirm("Remove this document and all its chunks?"):
            return

    store.delete_document(document_id)
    console.print(f"[green]✅ Removed:[/] {doc['title']}")


if __name__ == '__main__':
    main()
