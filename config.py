"""
config.py

Central configuration for the UrbanRoof DDR Generator pipeline.

All tunable constants live here. No other module should define its
own path constants, AI model settings, or inspection metadata.

To adapt this system to a new inspection job:
    1. Update PROPERTY_ADDRESS, INSPECTED_BY, INSPECTION_DATE.
    2. Place the PDFs in the /data folder with the names defined below.
    3. Run: python app.py

To switch AI models:
    Update AI_MODEL to any model supported by the OpenAI API,
    e.g. "gpt-4o" for higher accuracy or "gpt-3.5-turbo" for lower cost.

Environment variables (set in .env file):
    OPENAI_API_KEY=sk-...   Required. Never hardcode this here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

DATA_DIR: Path = Path("data")
IMAGE_DIR: Path = Path("images")
OUTPUT_DIR: Path = Path("output")

# ---------------------------------------------------------------------------
# Input file paths
# Rename the PDFs in your /data folder to match these names,
# or update these constants to match your actual filenames.
# ---------------------------------------------------------------------------

INSPECTION_PDF: Path = DATA_DIR / "Inspection Report.pdf"
THERMAL_PDF: Path = DATA_DIR / "Thermal Images.pdf"

# ---------------------------------------------------------------------------
# Image extraction output directories
# ---------------------------------------------------------------------------

INSPECTION_IMAGES_DIR: Path = IMAGE_DIR / "inspection"
THERMAL_IMAGES_DIR: Path = IMAGE_DIR / "thermal"

# ---------------------------------------------------------------------------
# AI model configuration
# ---------------------------------------------------------------------------

# Model name passed to the API.
# Groq free tier:  "llama-3.1-8b-instant"  or  "mixtral-8x7b-32768"
# OpenAI paid:     "gpt-4o-mini"           or  "gpt-4o"
AI_MODEL: str = "llama-3.1-8b-instant"
AI_BASE_URL: str | None = "https://api.groq.com/openai/v1"

# Maximum characters of document text sent to the LLM in one call.
# gpt-4o-mini supports ~128k tokens. 40,000 chars ≈ 10,000 tokens.
# Increase if reports are very long and observations are being missed.
MAX_TEXT_LENGTH: int = 40_000

# Maximum number of retry attempts on transient OpenAI API errors.
# Errors retried: RateLimitError (429), InternalServerError (500).
MAX_RETRIES: int = 3

# Base delay in seconds between retries (doubles each attempt).
# Attempt 1 → 1.0s, Attempt 2 → 2.0s, Attempt 3 → 4.0s
RETRY_BASE_DELAY: float = 1.0

# ---------------------------------------------------------------------------
# Inspection metadata
# Update these for each new inspection job.
# These appear in the DDR report header.
# ---------------------------------------------------------------------------

PROPERTY_ADDRESS: str = (
    "Flat No-8/63, Yamuna CHS, Hari Om Nagar, Mulund East, Mumbai - 400081"
)
INSPECTED_BY: str = "Tushar Rahane"
INSPECTION_DATE: str = "July 24, 2023"