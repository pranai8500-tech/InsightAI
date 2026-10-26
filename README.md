# InsightAI — AI-Powered Data Analytics

**InsightAI** is a Streamlit-based data analytics prototype for uploading tabular datasets, reviewing data quality, exploring data interactively, and generating AI-assisted insights — all in one workspace.

> **Status**: Active prototype (Phases 1–3 complete, Phase 4 in progress). See [Roadmap](#roadmap).

---

## Problem Statement

Data analysts and business users often struggle with:

- **Manual data profiling** — tedious column-by-column inspection
- **Data quality blind spots** — missing values, duplicates, and anomalies go unnoticed
- **Time-consuming cleaning** — repetitive, error-prone transformations
- **AI integration gaps** — LLMs can't access local data securely or execute verified calculations
- **Tool fragmentation** — separate tools for profiling, cleaning, exploration, and AI chat

**InsightAI** consolidates these workflows into a single, privacy-conscious Streamlit application with a validated analytical engine and AI-assisted interpretation.

---

## Key Features

### 📊 Dashboard (Phase 3)
- Dynamic KPI cards generated from dataset profile (record count, missing %, duplicates, numeric summaries, currency totals)
- 6 automatic charts: category comparison, time series, risk by dimension, second measure, distribution, correlation heatmap
- Evidence-based insights generated from verified calculations (not AI hallucination)

### 🔎 Data Preview
- Dataset overview: rows, columns, memory, missing cells, duplicates
- Configurable row preview (5–200 rows)
- Column types, non-null counts, unique values, memory usage

### 🩺 Data Quality
- Missing values by column (count + percentage)
- Duplicate rows and entirely empty columns
- Duplicate column name detection
- Potential date columns with format detection and parse success rate
- Inconsistent text categories (case/whitespace variations)
- Suspicious numeric values: negatives, excessive zeros, IQR outliers

### 🔍 Data Explorer
- Column statistics (non-null, null, unique, memory)
- Value counts for categorical columns (top 30)
- Histograms for numeric columns (Plotly)
- Multi-condition filter builder (AND logic)
- Download filtered results as CSV

### 🧹 Cleaning Preview (Non-Destructive)
- Trim leading/trailing spaces
- Normalize repeated whitespace
- Remove exact duplicate rows
- Remove entirely empty columns
- Before/after preview (8 rows each)
- Planned changes summary
- Apply to session copy (original unchanged)
- Download cleaned CSV

### 💬 Ask Your Data (AI Chat)
- NVIDIA Nemotron 3 Ultra via OpenAI-compatible API
- Privacy-safe context: only metadata (column names, types, counts, quality warnings) sent to AI
- **No row data, private values, or large samples transmitted**
- Conversation history preserved in session state
- Secure API key via Streamlit secrets or environment variable
- Deterministic fallback for common questions (e.g., "top N by revenue")
- Validated analytical plans executed locally via Pandas/DuckDB

---

## Architecture & Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Upload    │────▶│  Profiling   │────▶│  Semantic Model │
│  (CSV/XLSX) │     │  (data_quality)│    │  (semantic_model)│
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────┐
                    ▼                              ▼              ▼
             ┌─────────────┐              ┌──────────────┐ ┌───────────┐
             │   Dashboard │              │ Data Explorer│ │ Ask Your  │
             │   (KPIs,    │              │  (Filters,   │ │   Data    │
             │  Charts,    │              │   Charts,    │ │   (AI +   │
             │  Insights)  │              │  Filters)    │ │  Engine)  │
             └─────────────┘              └──────────────┘ └───────────┘
```

### Key Design Principles
- **Schema-independent** — no hardcoded column names; works with any tabular dataset
- **Validated execution** — AI generates structured JSON plans; engine validates columns/types before execution
- **Privacy-first** — only schema + aggregates sent to AI; raw data never leaves session
- **Deterministic fallbacks** — common questions (top-N, trends) bypass AI entirely

---

## Supported Formats & Limitations

| Format | Support | Notes |
|--------|---------|-------|
| CSV | ✅ Full | UTF-8, auto-detected delimiter |
| XLSX | ✅ Full | Sheet selector for multi-sheet workbooks |
| XLS | ✅ Legacy | Requires `xlrd>=2.0` |

### Known Limitations
- **Not a production BI tool** — prototype for exploration and prototyping
- **No persistent storage** — data lives in Streamlit session memory
- **AI context limited** — only schema + aggregates sent; no row-level data
- **Domain detection** — keyword-based; confidence varies by dataset
- **Single-table focus** — multi-table joins not yet supported
- **Memory-bound** — large datasets (>100MB) may hit browser/container limits

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit ≥1.36 |
| Data Engine | Pandas ≥2.1, DuckDB ≥1.0 |
| Visualization | Plotly ≥5.20 |
| AI API | OpenAI-compatible (NVIDIA Nemotron 3 Ultra) |
| Testing | pytest, mock |
| Config | TOML (Streamlit secrets) |

---

## Installation (Windows)

### Prerequisites
- **Python 3.10+** (3.10–3.12 tested)
- **Git** (for cloning)
- **NVIDIA API key** (for AI chat feature)

### Quick Start

```powershell
# 1. Clone the repository
git clone https://github.com/<your-username>/InsightAI.git
cd InsightAI

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Configure NVIDIA API key (required for chat)
copy .streamlit\secrets.example.toml .streamlit\secrets.toml
# Edit .streamlit\secrets.toml and replace the placeholder with your key

# 5. Launch
streamlit run app.py
```

Open the URL printed in the terminal (typically `http://localhost:8501`).

### NVIDIA API Key Setup (Required for Chat)

**Option 1: Streamlit Secrets (Recommended)**
```powershell
copy .streamlit\secrets.example.toml .streamlit\secrets.toml
# Edit .streamlit\secrets.toml:
# NVIDIA_API_KEY = "your-actual-api-key-here"
```

**Option 2: Environment Variable**
```powershell
$env:NVIDIA_API_KEY = "your-actual-api-key-here"
streamlit run app.py
```

> **Security**: `.streamlit/secrets.toml` is in `.gitignore` and will never be committed. Never hardcode keys in source code.

### Obtaining a NVIDIA API Key
1. Go to [NVIDIA NGC](https://ngc.nvidia.com/) and sign in
2. Navigate to **API Keys** (or your profile → API Keys)
3. Create a new key with access to Nemotron models
4. Copy the key and add it to your secrets or environment

---

## Running Tests

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests/ -v
```

- **206 tests** covering analytics engine, AI client, data quality, semantic modeling
- All tests use **mocked API responses** — no live API key required
- Run `python -m py_compile app.py modules\*.py` for syntax validation

---

## Project Structure

```text
InsightAI/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── .gitignore                  # Git exclusions
├── .streamlit/
│   ├── config.toml             # Streamlit theme/config
│   ├── secrets.example.toml    # Template (no real keys)
│   └── secrets.toml            # Local only (gitignored)
├── modules/
│   ├── __init__.py
│   ├── ai_client.py            # NVIDIA API client
│   ├── analytics_engine.py     # Validated Pandas/DuckDB engine
│   ├── analysis_planner.py     # AI → JSON plan + validation
│   ├── semantic_model.py       # Column profiling & roles
│   ├── dynamic_kpis.py         # Schema-aware KPI generation
│   ├── dynamic_charts.py       # Chart type selection
│   ├── dashboard_charts.py     # Plotly styling
│   ├── dashboard_kpis.py       # Legacy KPI specs
│   ├── data_loader.py          # CSV/Excel loading
│   ├── data_quality.py         # Profiling & issue detection
│   ├── column_resolver.py      # Fuzzy column matching
│   ├── column_map.py           # Canonical column names
│   ├── cleaning.py             # Cleaning operations
│   └── data_loader.py          # File parsing
├── tests/
│   ├── test_analytics_engine.py   # 122 KB (core logic)
│   ├── test_ai_client.py          # 12 KB
│   └── test_data_quality.py       # 8 KB
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Screenshots

*Placeholders — replace with actual captures*

| Section | Preview |
|---------|---------|
| **Upload & Overview** | `docs/screenshots/01-upload-overview.png` |
| **Data Quality** | `docs/screenshots/02-data-quality.png` |
| **Data Explorer** | `docs/screenshots/03-data-explorer.png` |
| **Dashboard (KPIs + Charts)** | `docs/screenshots/04-dashboard.png` |
| **Ask Your Data (Chat)** | `docs/screenshots/05-chat.png` |

> Create a `docs/screenshots/` folder and add real captures before publishing.

---

## Privacy & Security

| Aspect | Approach |
|--------|----------|
| **API Keys** | Streamlit secrets + env vars only; `.gitignore` protects `secrets.toml` |
| **Data Transmission** | Only column names, types, counts, quality warnings sent to AI |
| **Row Data** | Never transmitted to AI; processed locally via Pandas |
| **Session Data** | In-memory only; no disk persistence unless user exports |
| **Credentials** | Never logged, printed, or exposed in UI |

---

## Known Limitations & Future Work

### Limitations (Current)
| Area | Limitation |
|------|------------|
| Domain detection | Keyword-based; confidence low for non-retail datasets |
| Multi-table | Single table only; no join support |
| Large data | Memory-bound; >100MB may OOM in browser |
| AI planning | Requires API key; no offline mode |
| Visualization | Fixed chart set; no custom chart builder |

### Roadmap
| Phase | Focus |
|-------|-------|
| **Phase 4** | KPI recommendations, custom Plotly dashboards, report exports (PDF/HTML) |
| **Phase 5** | Multi-table joins, forecasting, advanced stats, deployment hardening |
| **Phase 6** | User auth, audit logs, scheduled runs, plugin architecture |

---

## Contributing

```powershell
# 1. Fork & clone
git clone https://github.com/<your-username>/InsightAI.git

# 2. Create branch
git checkout -b feature/your-feature

# 3. Develop & test
.\.venv\Scripts\Activate.ps1
python -m pytest tests/ -v
python -m py_compile app.py modules\*.py

# 4. Commit & PR
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature
```

---

## License

> **License to be determined.** If you are the repository owner, add your chosen license (MIT, Apache-2.0, BSD-3, etc.) and copyright notice here.

---

## Acknowledgments

- **NVIDIA** — Nemotron 3 Ultra via NGC API
- **Streamlit** — Rapid prototyping framework
- **Pandas / Plotly / DuckDB** — Core data stack
- **pytest / unittest.mock** — Testing infrastructure

---

*InsightAI — Turn your data into decisions.*


## 🚀 Live Demo

Try the deployed **InsightAI — AI-Powered Data Analytics Chatbot**:

👉 [**Open InsightAI Chatbot**](https://insightai-ft5wqqi7hxectrfnudytky.streamlit.app/)

Upload your CSV or Excel dataset and explore data insights, visualizations, and AI-powered analytics.
