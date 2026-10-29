# InsightAI — AI-Powered Data Analytics Platform

**Transform raw tabular data into validated analytics, interactive visualizations, and AI-assisted business insights.**

<p align="center">
  <a href="https://insightai-ft5wqqi7hxectrfnudytky.streamlit.app/">
    <img src="https://img.shields.io/badge/Live_Demo-Try_InsightAI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo" />
  </a>
  <a href="https://github.com/pranai8500-tech/InsightAI">
    <img src="https://img.shields.io/badge/GitHub-Source_Code-181717?style=for-the-badge&logo=github" alt="GitHub Repository" />
  </a>
  <img src="https://img.shields.io/badge/Status-Active_Prototype-blue?style=for-the-badge" alt="Active Prototype" />
  <img src="https://img.shields.io/badge/AI-Assisted_Analytics-7C3AED?style=for-the-badge" alt="AI-Assisted Analytics" />
</p>

---

## 1. Executive Summary

InsightAI is a Streamlit-based analytics platform designed to bring data profiling, data quality assessment, exploratory analysis, non-destructive cleaning, interactive dashboards, and AI-assisted data questioning into one workspace.

The application combines a schema-aware analytical workflow with a validated execution engine. Rather than relying entirely on AI-generated answers, it uses local Pandas/DuckDB operations for analytical calculations and uses AI to assist with interpreting questions and planning supported analyses.

**Objective:** Reduce repetitive exploratory work, improve visibility into data quality, and help users move from raw datasets to evidence-based insights more efficiently.

**Project status:** Active prototype. Phases 1–3 are reported complete; Phase 4 is in progress.

**[Launch InsightAI →](https://insightai-ft5wqqi7hxectrfnudytky.streamlit.app/)**

---

## 2. Business Problem

Before meaningful analysis can begin, analysts often need to inspect unfamiliar datasets, identify quality issues, understand column types, prepare data, calculate metrics, and create visualizations.

These steps can introduce repetitive work and make it harder to move quickly from data preparation to business interpretation.

InsightAI addresses five common challenges:

| Challenge | InsightAI approach |
|---|---|
| Manual data profiling | Automated dataset summaries and column-level statistics |
| Limited data quality visibility | Detection of missing values, duplicates, inconsistent categories, and potential anomalies |
| Repetitive cleaning | Previewable, non-destructive cleaning operations |
| Fragmented analysis workflow | Integrated data preview, exploration, dashboards, and AI-assisted questioning |
| Unreliable AI calculations | Structured analytical plans validated before local execution |

The platform is intended to support analysts and business users working with tabular datasets while keeping the analytical process transparent and reviewable.

---

## 3. Core Capabilities

### A. Dataset Profiling & Data Quality

Build an initial understanding of a dataset before analysis.

- Dataset dimensions, memory usage, missing cells, and duplicate rows
- Column data types, non-null counts, unique values, and memory usage
- Missing-value counts and percentages by column
- Detection of entirely empty columns and duplicate column names
- Potential date-column identification and parsing success rates
- Inconsistent text categories caused by case or whitespace differences
- Suspicious numeric values, including negatives, excessive zeros, and IQR-based outliers

**Business value:** Identify data reliability concerns before they affect downstream metrics and decisions.

### B. Interactive Data Explorer

Explore datasets through configurable previews, statistics, filters, and visualizations.

- Configurable row previews from 5 to 200 rows
- Categorical value counts, including the top 30 values
- Numeric histograms using Plotly
- Multi-condition filtering using AND logic
- Column-level descriptive statistics
- CSV export of filtered results

**Business value:** Investigate distributions, isolate relevant records, and understand patterns before developing deeper analyses.

### C. Non-Destructive Data Cleaning

Preview potential transformations before applying them to a session copy.

- Trim leading and trailing whitespace
- Normalize repeated whitespace
- Remove exact duplicate rows
- Remove entirely empty columns
- Compare before-and-after previews
- Review a planned-change summary
- Export the cleaned dataset as CSV

The original uploaded dataset remains unchanged by these cleaning operations.

**Business value:** Make routine data preparation more transparent and reduce the risk of unintended changes.

### D. Dynamic Analytics Dashboard

Generate a dashboard based on the uploaded dataset's profile and available fields.

- Dynamic KPI cards based on dataset characteristics
- Record counts, missing-data indicators, duplicate counts, and numeric summaries
- Automatic chart selection for categories, time series, distributions, risk dimensions, and correlations
- Insights grounded in verified calculations

The dashboard uses dataset-aware logic rather than requiring every dataset to follow a predefined schema.

**Business value:** Accelerate initial exploration and provide a starting point for business-focused analysis.

### E. Ask Your Data — AI-Assisted Analytics

Interact with data through natural-language questions.

- NVIDIA Nemotron 3 Ultra through an OpenAI-compatible API
- Structured JSON analytical plans
- Column and data-type validation before execution
- Local analytical execution using Pandas/DuckDB
- Deterministic fallback handling for supported questions, such as top-N analysis
- Conversation history maintained in Streamlit session state
- API key configuration through Streamlit secrets or environment variables

The AI assists with interpreting questions and preparing analytical plans; the validated engine performs supported calculations.

**Business value:** Make exploratory analysis more accessible while retaining a structured execution process.

---

## 4. System Architecture

InsightAI separates dataset profiling, semantic interpretation, analytical execution, and presentation.

```text
                  ┌──────────────────────┐
                  │   CSV / Excel Upload │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Data Loading &       │
                  │ Quality Profiling    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Semantic Model       │
                  │ Column Roles & Types │
                  └──────────┬───────────┘
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
      ┌─────────────┐ ┌──────────────┐ ┌────────────────┐
      │ Dashboard   │ │ Data Explorer│ │ Ask Your Data  │
      │ KPIs/Charts │ │ Filters/Stats│ │ AI + Planner   │
      └─────────────┘ └──────────────┘ └───────┬────────┘
                                               │
                                               ▼
                                    ┌────────────────────┐
                                    │ Plan Validation    │
                                    │ & Local Execution  │
                                    └─────────┬──────────┘
                                              │
                                              ▼
                                    ┌────────────────────┐
                                    │ Analytical Results │
                                    │ & Interpretation   │
                                    └────────────────────┘
```

### Design Principles

- **Schema independence:** Profile columns and infer useful roles instead of assuming fixed business column names.
- **Validated execution:** Validate AI-generated analytical plans against available columns, types, and supported operations before execution.
- **Calculation-grounded insights:** Base analytical outputs on computations performed by the local engine rather than treating free-form AI responses as verified results.
- **Privacy-conscious context:** Limit AI context to the metadata described in the implementation rather than sending raw row-level records.
- **Deterministic fallbacks:** Handle supported common analytical questions without depending on an AI response.

---

## 5. Privacy & Security

InsightAI is designed to limit the information sent to the AI service.

| Area | Current approach |
|---|---|
| AI context | Column names, types, counts, and quality warnings, as described in the implementation |
| Raw row-level data | Reported as not sent to the AI service; verify against the current code |
| Analytical execution | Supported calculations run locally through Pandas/DuckDB |
| API credentials | Streamlit secrets or environment variables |
| Local secrets | `secrets.toml` should remain excluded through `.gitignore` |
| Data persistence | Uploaded data is maintained in session memory; persistent storage is reported as not implemented |

**Important:** These statements describe the intended application design. Review deployment configuration, dependencies, logging, hosting behavior, and external integrations before using sensitive or regulated data.

Never commit a real API key, credentials file, or private dataset to a public repository.

---

## 6. Technology Stack

| Layer | Technologies |
|---|---|
| Application framework | Streamlit |
| Data manipulation | Pandas, NumPy |
| Analytical execution | Pandas, DuckDB |
| Visualization | Plotly |
| AI integration | NVIDIA Nemotron 3 Ultra via an OpenAI-compatible API |
| Analytical planning | Structured JSON plans and validation |
| Testing | pytest, `unittest.mock` |
| Configuration | TOML, Streamlit secrets, environment variables |
| Development | Python, Git, GitHub |

---

## 7. Supported Data Formats

| Format | Support | Notes |
|---|---|---|
| CSV | Supported | UTF-8 and automatic delimiter detection |
| XLSX | Supported | Sheet selection for multi-sheet workbooks |
| XLS | Legacy support | Requires `xlrd >= 2.0` |

### Current Limitations

- The application is an active prototype, not a production BI platform.
- Uploaded data is not persistently stored by the application.
- The analytical workflow focuses on a single table; multi-table joins are not yet supported.
- Large datasets may exceed available memory or hosting limits.
- Domain detection is keyword-based and may vary in accuracy.
- AI-assisted questions require a configured API key, except for supported deterministic fallbacks.
- Custom dashboard building and advanced reporting remain part of the roadmap.

---

## 8. Installation & Local Setup

### Prerequisites

- Python 3.10 or later; the project documentation reports testing on Python 3.10–3.12
- Git
- An NVIDIA API key for the AI chat feature

### Setup on Windows

Clone the actual repository:

```powershell
git clone https://github.com/pranai8500-tech/InsightAI.git
cd InsightAI
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the application:

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal, typically `http://localhost:8501`.

### Configure the AI API Key

Create a local secrets file at `.streamlit/secrets.toml`, following the format expected by the application and any provided example file.

Example:

```toml
NVIDIA_API_KEY = "your-actual-api-key"
```

Alternatively, set the environment variable in PowerShell:

```powershell
$env:NVIDIA_API_KEY = "your-actual-api-key"
streamlit run app.py
```

Do not commit `secrets.toml` or expose the key in screenshots, logs, source code, or public discussions.

---

## 9. Testing & Quality Assurance

The project documentation reports a test suite covering the analytics engine, AI client, data quality logic, and semantic modeling.

The reported test coverage includes mocked API responses so that supported application logic can be tested without a live API key.

Run the tests locally:

```bash
python -m pytest tests/ -v
```

Run a syntax check:

```bash
python -m py_compile app.py
```

The project notes report **206 tests**. Run the suite in your current environment before presenting this count as a verified current result.

---

## 10. Project Structure

The following is the documented representative structure. Update it if the actual repository differs.

```text
InsightAI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.example.toml
├── modules/
│   ├── ai_client.py
│   ├── analytics_engine.py
│   ├── analysis_planner.py
│   ├── semantic_model.py
│   ├── dynamic_kpis.py
│   ├── dynamic_charts.py
│   ├── dashboard_charts.py
│   ├── dashboard_kpis.py
│   ├── data_loader.py
│   ├── data_quality.py
│   ├── column_resolver.py
│   ├── column_map.py
│   └── cleaning.py
└── tests/
    ├── test_analytics_engine.py
    ├── test_ai_client.py
    └── test_data_quality.py
```

---

## 11. Roadmap

| Phase | Focus | Status |
|---|---|---|
| Phases 1–3 | Core data workflows, profiling, and dashboard capabilities | Reported complete |
| Phase 4 | KPI recommendations, custom Plotly dashboards, PDF/HTML report exports | In progress |
| Phase 5 | Multi-table joins, forecasting, advanced statistics, deployment hardening | Planned |
| Phase 6 | User authentication, audit logs, scheduled runs, plugin architecture | Planned |

---

## 12. Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement the change and add appropriate tests.
4. Run the test suite.
5. Submit a pull request describing the change and its business or technical value.

Example:

```bash
git checkout -b feature/your-feature
python -m pytest tests/ -v
git add .
git commit -m "feat: describe your change"
git push origin feature/your-feature
```

---

## 13. License

Add the appropriate license and copyright information for the InsightAI repository before redistributing the project. Do not assume that a license applies unless it is actually included in the repository.

---

## Live Demo

**[Launch InsightAI — AI-Powered Data Analytics](https://insightai-ft5wqqi7hxectrfnudytky.streamlit.app/)**

Upload a CSV or Excel dataset to explore its structure, inspect data quality, preview cleaning operations, generate visualizations, and try supported AI-assisted analytical questions.

## Author

**Pranai**

- **GitHub:** [pranai8500-tech](https://github.com/pranai8500-tech)
- **Project Repository:** [InsightAI](https://github.com/pranai8500-tech/InsightAI)
- **Data Analyst Portfolio:** [View Portfolio](https://data-analyst-portfolio-ochre.vercel.app/)

*InsightAI — From raw data to evidence-based insights.*
