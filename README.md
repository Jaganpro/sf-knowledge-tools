<p align="center">
  <img src="https://img.shields.io/badge/Salesforce-00A1E0?style=for-the-badge&logo=salesforce&logoColor=white" alt="Salesforce"/>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

<h1 align="center">📚 sf-knowledge-tools</h1>

<p align="center">
  <strong>Local PDF Knowledge Extraction & RAG Pipeline</strong><br>
  <em>Transform Salesforce documentation into searchable, AI-ready knowledge</em>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-usage">Usage</a> •
  <a href="#%EF%B8%8F-architecture">Architecture</a> •
  <a href="#-configuration">Configuration</a>
</p>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔒 100% Offline
No API calls, no cloud dependencies. Your documents stay on your machine. Process sensitive internal documentation with confidence.

### ⚡ Fast & Efficient
PyMuPDF extracts 90% of pages in milliseconds. sqlite-vec provides sub-50ms vector search at scale.

</td>
<td width="50%">

### 🎯 Hybrid Search
Combines semantic understanding (vector similarity) with keyword matching (FTS5) using Reciprocal Rank Fusion.

### 📝 Export Ready
Generate clean markdown with citations, organized by skill. Perfect for PRs to your documentation repos.

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

```bash
# macOS (Homebrew)
brew install tesseract uv

# Or install uv via pip
pip install uv
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Jaganpro/sf-knowledge-tools.git
cd sf-knowledge-tools

# Install dependencies (creates .venv automatically)
uv sync
```

> 💡 **First Run:** The embedding model (~1.3GB) downloads automatically on first use.

---

## 📖 Usage

### Ingest a PDF

```bash
sf-knowledge ingest ~/Documents/salesforce-apex-guide.pdf --category apex
```

<details>
<summary>📋 Example Output</summary>

```
📄 Ingesting: salesforce-apex-guide.pdf
  Extracting PDF...                      ━━━━━━━━━━━━━━━━━━━━ 100%
  Chunking content...                    ━━━━━━━━━━━━━━━━━━━━ 100%
  Generating embeddings (450 chunks)...  ━━━━━━━━━━━━━━━━━━━━ 100%
  Storing chunks...                      ━━━━━━━━━━━━━━━━━━━━ 100%

✅ Successfully ingested: Salesforce Apex Developer Guide
   Document ID  a1b2c3d4
   Pages        234
   Chunks       450
   Category     apex
```

</details>

### Search the Knowledge Base

```bash
sf-knowledge query "How do I handle governor limits in batch Apex?"
```

<details>
<summary>📋 Example Output</summary>

```
🔍 Searching: How do I handle governor limits in batch Apex?

Found 5 results in 45.2ms

╭─────────────────────── 📌 Result 1 (score: 0.892) ───────────────────────╮
│ Governor limits are enforced at runtime. In batch Apex, each execute    │
│ method invocation gets a fresh set of limits. To avoid hitting limits:  │
│                                                                          │
│ 1. Use Database.Stateful to maintain state across batches               │
│ 2. Keep batch size manageable (default 200, reduce if needed)           │
│ 3. Use Database.executeBatch() with scope parameter                     │
╰────────────────── Chapter: Batch Apex | p. 145 ──────────────────────────╯
```

</details>

### Export to Markdown

```bash
sf-knowledge export "Apex Governor Limits" --skill sf-apex
```

```
✅ Exported to: exports/sf-apex/apex-governor-limits.md
```

### Check Status

```bash
sf-knowledge status
```

```
📊 Knowledge Base Status

     Database
 Location  data/knowledge.db
 Size      24.5 MB

     Content
 Documents   3
 Chunks      2,450
 Embeddings  2,450
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         sf-knowledge-tools                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   📄 PDF Input                                                           │
│       │                                                                  │
│       ▼                                                                  │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│   │   PDF Extractor │───▶│ Semantic Chunker│───▶│ Embedding Client│     │
│   │  PyMuPDF + OCR  │    │  ~1000 tokens   │    │  BGE-large-v1.5 │     │
│   └─────────────────┘    └─────────────────┘    └────────┬────────┘     │
│                                                          │               │
│                                                          ▼               │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                     SQLite + sqlite-vec                          │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│   │  │  Documents  │  │   Chunks    │  │  Vector Embeddings      │  │   │
│   │  │   (meta)    │  │   (text)    │  │  (1024-dim, normalized) │  │   │
│   │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│   │   RAG Engine    │───▶│    Exporter     │───▶│  📝 Markdown    │     │
│   │  Hybrid Search  │    │  Jinja2 + Cites │    │   (by skill)    │     │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **PDF Extraction** | PyMuPDF + pdfplumber | Fast text extraction, table handling, OCR fallback |
| **Chunking** | Rule-based | ~1000 tokens, respects headers/code blocks |
| **Embeddings** | BAAI/bge-large-en-v1.5 | 1024-dim vectors, top MTEB retrieval model |
| **Storage** | SQLite + sqlite-vec | Single-file DB with vector similarity search |
| **Search** | Hybrid (Vector + FTS5) | Reciprocal Rank Fusion for best results |
| **Export** | Jinja2 | Templated markdown with citations |

---

## 📁 Project Structure

```
sf-knowledge-tools/
├── 📂 knowledge/              # Core Python library
│   ├── ingester/              # PDF extraction
│   ├── chunker/               # Semantic chunking
│   ├── embedder/              # Embedding generation
│   ├── storage/               # Vector store & schema
│   ├── query/                 # RAG engine
│   ├── export/                # Markdown generation
│   └── cli.py                 # Command-line interface
├── 📂 config/
│   └── knowledge.yml          # Configuration
├── 📂 pdfs/                   # Your source PDFs (gitignored)
├── 📂 data/                   # SQLite database (gitignored)
├── 📂 exports/                # Generated markdown (gitignored)
├── 📄 pyproject.toml          # Dependencies (uv)
└── 📄 LICENSE                 # MIT License
```

---

## ⚙️ Configuration

Edit `config/knowledge.yml` to customize behavior:

```yaml
embeddings:
  model: BAAI/bge-large-en-v1.5    # HuggingFace model
  dimensions: 1024
  batch_size: 32

chunking:
  target_size: 1000                 # Target tokens per chunk
  max_size: 1500                    # Maximum tokens
  overlap: 100                      # Overlap between chunks

search:
  default_k: 5                      # Results to return
  hybrid_weight: 0.7                # Vector vs keyword balance
```

---

## 🔄 Workflow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   1. INGEST  │────▶│   2. QUERY   │────▶│  3. EXPORT   │────▶│    4. PR     │
│              │     │              │     │              │     │              │
│ Add PDFs to  │     │ Search your  │     │ Generate     │     │ Copy to your │
│ knowledge DB │     │ knowledge    │     │ markdown     │     │ docs repo    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ for the Salesforce developer community</sub>
</p>
