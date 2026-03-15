# DDR Generator Dashboard

> **AI-powered system that converts raw property inspection and thermal imaging PDFs into structured, client-ready Detailed Diagnostic Reports.**

Built as part of the UrbanRoof Applied AI Builder assignment. The system reads two input documents — a visual inspection report and a thermal imaging report — runs a seven-step AI pipeline, and produces a fully formatted DDR HTML file matching the professional UrbanRoof report format.

---

## Table of Contents

- [Overview](#overview)
- [DDR Report Structure](#ddr-report-structure)
- [System Architecture](#system-architecture)
- [Pipeline — Step by Step](#pipeline--step-by-step)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [Running the System](#running-the-system)
- [Configuration](#configuration)
- [AI Provider](#ai-provider)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Overview

Property inspection generates large volumes of unstructured data across two separate documents — a visual inspection report and a thermal imaging report. Manually consolidating these into a client-ready DDR is time-consuming and error-prone.

This system automates the entire process:

- Extracts all observations from both documents using an LLM
- Deduplicates findings so no issue appears twice
- Cross-references inspection and thermal data to detect conflicts
- Assesses severity of every observation with written reasoning
- Matches extracted images to their relevant findings
- Assembles a self-contained HTML report covering all 7 required DDR sections

The system is designed to generalise — it works on any similar inspection report, not just the sample documents provided.

---

## DDR Report Structure

The generated report contains all seven sections required by the assignment:

| # | Section | Description |
|---|---|---|
| 1 | **Property Issue Summary** | High-level count of all issue types, severity distribution, total observations |
| 2 | **Area-wise Observations** | Each finding with description, probable cause, impact, and matched images from source documents |
| 3 | **Probable Root Cause** | Causes grouped by type, explained in plain client-friendly language |
| 4 | **Severity Assessment** | Every observation scored as Low / Medium / High / Critical with written reasoning |
| 5 | **Recommended Actions** | Repair actions mentioned in the source documents, grouped by type |
| 6 | **Additional Notes** | Conflicts between inspection and thermal data, explicitly documented |
| 7 | **Missing or Unclear Information** | Any field the system could not determine is written as "Not Available" — never guessed |

**Image handling:**
- Images are extracted directly from the source PDFs
- Each image is deduplicated using MD5 hashing
- Images are matched to observations by page proximity
- If no image is available for an observation, the report states "Image Not Available"
- All images are base64-encoded into the HTML so the report is fully self-contained

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Interface                    │
│              Streamlit (ui.py)                       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                 Pipeline Orchestrator                │
│                    app.py                            │
└──┬──────────┬──────────┬──────────┬─────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
┌──────┐  ┌──────┐  ┌────────┐  ┌────────┐
│parser│  │extrac│  │reason- │  │report/ │
│      │  │tion/ │  │  ing/  │  │        │
│PDF   │  │Obser-│  │Conflict│  │DDR     │
│Parser│  │vation│  │Detect- │  │Genera- │
│Image │  │Extra-│  │Severity│  │tor     │
│Extra-│  │ctor  │  │Mapper  │  │        │
│ctor  │  │      │  │        │  │        │
└──────┘  └──────┘  └────────┘  └────────┘
```

---

## Pipeline — Step by Step

The system executes seven steps in sequence. Each step feeds its output into the next.

### Step 1 — Document Parsing
**Module:** `parser/pdf_parser.py`

PyMuPDF (`fitz`) reads both PDFs page by page and extracts all text content. Blank pages (cover pages, disclaimers) are identified and skipped. Returns a `ParsedDocument` object containing all non-empty page text.

### Step 2 — Image Extraction
**Module:** `parser/image_extractor.py`

Every embedded image in both PDFs is extracted. Images are deduplicated using MD5 hashing — if the same image appears across multiple pages (common in inspection reports), only one copy is kept. Returns an `ExtractionResult` with the unique image set and page metadata.

### Step 3 — AI Observation Extraction
**Module:** `extraction/observation_extractor.py`

The full document text is sent to an LLM via an OpenAI-compatible API with a structured extraction prompt. The prompt instructs the model to:
- Extract every distinct observation
- Return a validated JSON array
- Never invent facts not present in the document
- Return `null` for any field not mentioned

The response is validated using **Pydantic** — malformed output is caught and corrected before it enters the pipeline. Observations are then deduplicated by area + issue type fingerprint.

Each observation contains: area, floor, issue type, description, probable cause, impact, leakage timing, recommended action, and source page.

### Step 4 — Image–Observation Mapping
**Module:** `reasoning/observation_image_mapper.py`

Each observation is matched to the most relevant extracted images based on page proximity in the source document. Confidence scoring determines how strongly each image is associated with each observation.

### Step 5 — Conflict Detection
**Module:** `reasoning/conflict_detector.py`

Inspection observations and thermal observations are cross-referenced by area and issue type. Three conflict types are detected:

| Conflict Type | Meaning | Severity |
|---|---|---|
| `THERMAL_ONLY` | Thermal shows an issue the inspection missed | High |
| `INSPECTION_ONLY` | Inspection found an issue not confirmed by thermal | Medium |
| `TYPE_MISMATCH` | Both found an issue in the same area but classified it differently | Low |

All conflicts are documented in the final report — the system never silently resolves them.

### Step 6 — Severity Assessment
**Module:** `reasoning/severity_assessor.py`

Every observation is scored using a multi-factor rule system considering issue type, leakage timing, impact description, and conflict flags. Scores are mapped to four levels:

- `LOW` — monitor, no immediate action
- `MEDIUM` — repairs needed within a reasonable timeframe
- `HIGH` — urgent attention required
- `CRITICAL` — immediate action, potential structural risk

Written reasoning is attached to every assessment.

### Step 7 — Report Generation
**Module:** `report/ddr_generator.py`

All pipeline outputs are assembled into a self-contained HTML report. Images are base64-encoded directly into the HTML so no external files are needed. The report can be opened in any browser and printed to PDF using `Ctrl+P`.

---

## Project Structure

```
ai-ddr-generator/
│
├── parser/                        # Document ingestion
│   ├── __init__.py
│   ├── pdf_parser.py              # Text extraction from PDFs
│   └── image_extractor.py         # Image extraction with deduplication
│
├── extraction/                    # AI-powered data extraction
│   ├── __init__.py
│   └── observation_extractor.py   # LLM extraction + Pydantic validation
│
├── reasoning/                     # Logic and analysis
│   ├── __init__.py
│   ├── conflict_detector.py       # Cross-reference inspection vs thermal
│   ├── severity_assessor.py       # Rule-based severity scoring
│   └── observation_image_mapper.py # Match images to observations
│
├── report/                        # Output generation
│   ├── __init__.py
│   └── ddr_generator.py           # HTML DDR report assembly
│
├── data/                          # Input PDFs (not committed)
│   ├── inspection_report.pdf
│   └── thermal_report.pdf
│
├── images/                        # Extracted images (not committed)
│   ├── inspection/
│   └── thermal/
│
├── output/                        # Generated reports (not committed)
│
├── app.py                         # CLI pipeline orchestrator
├── ui.py                          # Streamlit web interface
├── config.py                      # All constants and configuration
├── requirements.txt               # Python dependencies
├── .env                           # API key (not committed)
├── .gitignore
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Web UI | Streamlit |
| PDF Parsing | PyMuPDF (fitz) |
| AI Provider | Groq (free) — OpenAI-compatible API |
| LLM Model | `llama-3.1-8b-instant` |
| Data Validation | Pydantic v2 |
| HTTP Client | openai Python SDK |
| Report Format | Self-contained HTML |
| Fonts | DM Serif Display + DM Sans |

---

## Setup and Installation

**Prerequisites**
- Python 3.12+
- A free Groq API key from [console.groq.com](https://console.groq.com)

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/ai-ddr-generator.git
cd ai-ddr-generator
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Create the `.env` file**

Create a file named `.env` in the project root:
```
OPENAI_API_KEY=gsk_your_groq_key_here
```

Groq keys start with `gsk_`. The variable name stays as `OPENAI_API_KEY` because the system uses the OpenAI-compatible client.

**5. Place your input PDFs**

Copy your inspection and thermal PDFs into the `data/` folder:
```
data/
├── inspection_report.pdf
└── thermal_report.pdf
```

---

## Running the System

**Option A — Command line (no UI)**
```bash
python app.py
```

The report is saved to `output/DDR_Report_<timestamp>.html`. Open it in any browser.

**Option B — Web UI**
```bash
streamlit run ui.py
```

Opens at `http://localhost:8501`. Upload PDFs via the sidebar, fill in the property details, and click Generate Report.

---

## Configuration

All settings are in `config.py`. No other file needs to be changed.

```python
# AI model — switch provider by changing these two lines
AI_MODEL:    str       = "llama-3.1-8b-instant"   # or "gpt-4o-mini"
AI_BASE_URL: str|None  = "https://api.groq.com/openai/v1"  # None for OpenAI

# Input files
INSPECTION_PDF: Path = DATA_DIR / "inspection_report.pdf"
THERMAL_PDF:    Path = DATA_DIR / "thermal_report.pdf"

# Inspection metadata (appears in report header)
PROPERTY_ADDRESS: str = "Flat No-8/63, Yamuna CHS, Mulund East, Mumbai"
INSPECTED_BY:     str = "Tushar Rahane"
INSPECTION_DATE:  str = "July 24, 2023"
```

**Switching to OpenAI:**
```python
AI_MODEL    = "gpt-4o-mini"
AI_BASE_URL = None  # uses OpenAI default endpoint
# OPENAI_API_KEY in .env should be your OpenAI key (starts with sk-)
```

---

## AI Provider

| Provider | Model | Cost | Speed | Accuracy |
|---|---|---|---|---|
| **Groq** *(default)* | llama-3.1-8b-instant | Free | ~3–5s | Good |
| Groq | llama-3.3-70b-versatile | Free | ~8–12s | Better |
| OpenAI | gpt-4o-mini | Paid | ~10s | High |
| OpenAI | gpt-4o | Paid | ~15s | Highest |

The system uses the OpenAI Python SDK for all providers. Switching requires only changing `AI_MODEL` and `AI_BASE_URL` in `config.py` — no code changes.

---

## Limitations

**Model accuracy** — The free Groq model occasionally misses nuanced observations or misclassifies issue types compared to GPT-4. The architecture makes switching models a one-line change.

**Image placement** — Images are matched to observations based on page proximity in the source document. The system does not visually understand what a photo depicts. An image appearing near a dampness observation will be linked to it — which is usually correct but not always precise.

**No persistent storage** — Each run is stateless. Reports are saved as local HTML files. There is no database, no user accounts, and no report history.

**PDF only** — The system currently accepts PDF input only. Inspection data in Word, Excel, or other formats is not supported.

---

## Future Improvements

**Vision-based image matching** — Pass each extracted image to a multimodal LLM and ask it to describe what it shows. Match image descriptions to observation text semantically rather than by page proximity. This would make image placement genuinely intelligent.

**Broader document support** — Add parsers for Word documents, Excel inspection sheets, and structured JSON from IoT sensors. The extraction pipeline is modular — adding a new parser does not require changing any downstream logic.

**Deployed web application** — Host the system with authentication so any inspector can upload documents and retrieve reports from anywhere, with a database storing report history per user.

**Fine-tuned extraction model** — Fine-tune a smaller model on UrbanRoof's specific inspection vocabulary and DDR format. This would give higher extraction accuracy than a general-purpose LLM at lower cost and latency.

---

## Requirements

```
PyMuPDF==1.27.2
openai==2.28.0
python-dotenv==1.2.2
pydantic==2.12.5
streamlit==1.42.0
tqdm==4.67.3
```

