# sf-knowledge-tools

Private PDF knowledge extraction and RAG pipeline for sf-skills.

## Overview

This tool processes large PDF documents (Salesforce documentation, design patterns, internal standards) and extracts knowledge into structured markdown files organized by skill.

**Key Features:**
- 100% offline processing (no API costs)
- SQLite + sqlite-vec for vector storage
- Hybrid search (semantic + keyword)
- Export to markdown with citations

## Installation

### Prerequisites (Homebrew)

```bash
# Install Tesseract OCR (for scanned PDFs)
brew install tesseract

# Install uv (if not already installed)
brew install uv
```

### Setup with uv

```bash
# Clone the repo
git clone <your-private-repo-url>
cd sf-knowledge-tools

# Create venv and install dependencies (one command!)
uv sync

# Or install with dev dependencies
uv sync --dev
```

### First-time embedding model download

The first time you run ingestion, the embedding model (~1.3GB) will download automatically from HuggingFace.

## Usage

```bash
# Ingest a PDF
uv run sf-knowledge ingest pdfs/salesforce-apex-guide.pdf --category apex

# Query the knowledge base
uv run sf-knowledge query "bulkification patterns" --skill sf-apex

# Export to markdown
uv run sf-knowledge export "Apex Bulkification" --skill sf-apex

# Check status
uv run sf-knowledge status
```

> **Tip:** `uv run` ensures you're always using the correct virtual environment and dependencies.

## Directory Structure

```
sf-knowledge-tools/
├── pdfs/           # Source PDFs (gitignored)
├── data/           # SQLite database (gitignored)
├── exports/        # Generated markdown (by skill)
│   ├── sf-apex/
│   ├── sf-flow/
│   └── ...
├── knowledge/      # Python library
└── config/         # Configuration
```

## Workflow

1. **Ingest**: Add PDFs to `pdfs/` and run `sf-knowledge ingest`
2. **Query**: Search with `sf-knowledge query "topic"`
3. **Export**: Generate markdown with `sf-knowledge export`
4. **PR**: Copy `exports/sf-apex/*` to sf-skills repo and create PR

## Configuration

Edit `config/knowledge.yml` to customize:
- Embedding model
- Chunk sizes
- Search parameters
- Export paths

## License

Private - Not for distribution
