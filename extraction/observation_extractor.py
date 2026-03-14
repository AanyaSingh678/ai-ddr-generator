"""
extraction/observation_extractor.py

Converts raw parsed inspection text into structured observations
using an LLM (OpenAI GPT) with Pydantic schema validation.

Pipeline:
    ParsedDocument
        → page text assembled
        → sent to OpenAI with structured extraction prompt
        → response parsed + validated via Pydantic
        → deduplicated by area + issue fingerprint
        → returns ObservationExtractionResult

Why AI instead of regex?
    Inspection reports use natural language, inconsistent formatting,
    and domain-specific phrasing. Regex can match keywords but cannot
    understand context, infer causes, or merge related findings.
    An LLM handles all of this reliably with a well-designed prompt.
"""

import hashlib
import json
import logging
import os
import time
from enum import Enum

from dotenv import load_dotenv
from openai import (
    APIStatusError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel, Field, field_validator

from parser.pdf_parser import ParsedDocument

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums — constrain LLM output to known categories
# ---------------------------------------------------------------------------

class IssueType(str, Enum):
    """
    Known issue categories found in UrbanRoof inspection reports.
    Using an Enum prevents the LLM from inventing new categories.
    """

    DAMPNESS = "dampness"
    SEEPAGE = "seepage"
    CRACK = "crack"
    TILE_GAP = "tile_gap"
    HOLLOWNESS = "hollowness"
    VEGETATION = "vegetation"
    EFFLORESCENCE = "efflorescence"
    PAINT_SPALLING = "paint_spalling"
    STRUCTURAL = "structural"
    PLUMBING = "plumbing"
    UNKNOWN = "unknown"


class SeverityLevel(str, Enum):
    """
    Severity levels aligned with UrbanRoof DDR report format.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LeakageTiming(str, Enum):
    """
    When the issue occurs — extracted directly from report text.
    """

    MONSOON = "monsoon"
    ALL_TIME = "all_time"
    NOT_SURE = "not_sure"
    NOT_MENTIONED = "not_mentioned"


# ---------------------------------------------------------------------------
# Pydantic models — validated data contracts
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    A single structured observation extracted from inspection text.

    Every field that could be missing from the source document uses
    str | None with a None default. The report generator will render
    these as "Not Available" in the final DDR output.
    """

    area: str = Field(
        description=(
            "Specific room or surface. "
            "e.g. 'Hall Ceiling', 'Master Bedroom Bathroom'."
        )
    )
    floor: str | None = Field(
        default=None,
        description=(
            "Which floor. "
            "e.g. 'Ground Floor', '1st Floor'. Null if not mentioned."
        ),
    )
    issue_type: IssueType = Field(
        description="Category of the issue from the predefined list."
    )
    description: str = Field(
        description="Plain English description of exactly what was observed."
    )
    probable_cause: str | None = Field(
        default=None,
        description=(
            "Most likely cause based on the report text. "
            "Null if not stated."
        ),
    )
    impact: str | None = Field(
        default=None,
        description="What damage this is causing or will cause if untreated.",
    )
    leakage_timing: LeakageTiming = Field(
        default=LeakageTiming.NOT_MENTIONED,
        description="When the leakage or issue occurs.",
    )
    recommended_action: str | None = Field(
        default=None,
        description="Any repair action mentioned in the report for this issue.",
    )
    source_page: int | None = Field(
        default=None,
        description="Page number in the source document where this was found.",
    )

    @field_validator("area", "description")
    @classmethod
    def must_not_be_empty(cls, value: str) -> str:
        """Prevents empty strings from passing validation."""
        if not value or not value.strip():
            raise ValueError("Field cannot be empty or whitespace.")
        return value.strip()

    @field_validator("area")
    @classmethod
    def normalize_area(cls, value: str) -> str:
        """Capitalizes area names consistently, e.g. 'hall ceiling' → 'Hall Ceiling'."""
        return value.strip().title()

    @property
    def fingerprint(self) -> str:
        """
        A short hash that uniquely identifies this observation by area + issue.
        Used to detect and remove duplicate observations across pages.

        Two observations with the same area and issue type are considered
        duplicates even if their descriptions differ slightly.
        """
        key = f"{self.area.lower()}|{self.issue_type.value}"
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def __repr__(self) -> str:
        return (
            f"Observation("
            f"area='{self.area}', "
            f"issue='{self.issue_type.value}', "
            f"floor='{self.floor}', "
            f"page={self.source_page})"
        )


class LLMExtractionResponse(BaseModel):
    """
    Validates the full JSON response from the LLM.
    The LLM is instructed to return a list of observations under this key.
    If the LLM returns a different top-level key, Pydantic raises immediately
    with a clear error rather than a cryptic KeyError downstream.
    """

    observations: list[Observation]


class ObservationExtractionResult:
    """
    Final result of an extraction run.

    Holds deduplicated observations and extraction statistics
    for use by downstream pipeline modules (conflict detector,
    severity assessor, DDR report generator).
    """

    def __init__(
        self,
        observations: list[Observation],
        total_raw: int,
        duplicates_removed: int,
        source_name: str,
    ) -> None:
        self.observations = observations
        self.total_raw = total_raw
        self.duplicates_removed = duplicates_removed
        self.source_name = source_name

    @property
    def count(self) -> int:
        """Total number of unique observations."""
        return len(self.observations)

    def get_by_issue_type(self, issue_type: IssueType) -> list[Observation]:
        """Returns all observations of a specific issue type."""
        return [o for o in self.observations if o.issue_type == issue_type]

    def get_by_area(self, area: str) -> list[Observation]:
        """Returns all observations for a specific area (case-insensitive)."""
        return [
            o for o in self.observations
            if o.area.lower() == area.lower()
        ]

    def __repr__(self) -> str:
        return (
            f"ObservationExtractionResult("
            f"source='{self.source_name}', "
            f"count={self.count}, "
            f"duplicates_removed={self.duplicates_removed})"
        )


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are an expert building inspection analyst working for UrbanRoof Pvt. Ltd.

You will be given raw text extracted from a property inspection report.
Your task is to identify every distinct structural or moisture-related observation
and return them as structured JSON.

IMPORTANT RULES:
- Extract ONLY observations that are clearly stated in the text.
- Do NOT invent, infer, or assume any observation not present in the text.
- If a field is not mentioned in the text, use null.
- If the same issue appears multiple times for the same area, extract it ONCE only.
- Use simple, client-friendly language in the description field.
- The "area" field must be specific: "Hall Ceiling", not just "Hall".

ISSUE TYPE must be exactly one of:
dampness | seepage | crack | tile_gap | hollowness | vegetation |
efflorescence | paint_spalling | structural | plumbing | unknown

LEAKAGE TIMING must be exactly one of:
monsoon | all_time | not_sure | not_mentioned

Return ONLY a valid JSON object in this exact format.
No explanation text. No markdown. No code blocks. Raw JSON only.

{{
  "observations": [
    {{
      "area": "Hall Ceiling",
      "floor": "Ground Floor",
      "issue_type": "dampness",
      "description": "Dampness, efflorescence and paint spalling observed at the ceiling.",
      "probable_cause": "Moisture rising from bathroom tile gaps above via capillary action.",
      "impact": "Continued paint spalling and potential structural surface degradation.",
      "leakage_timing": "all_time",
      "recommended_action": "Bathroom grouting treatment and waterproofing.",
      "source_page": null
    }}
  ]
}}

Inspection text to analyse:
\"\"\"
{text}
\"\"\"
"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class ObservationExtractor:
    """
    Extracts structured observations from a ParsedDocument using OpenAI.

    The full document text is sent to the LLM in a single call.
    The response is validated against the Pydantic schema.
    Duplicate observations (same area + issue type) are removed.

    Retry behaviour:
        Transient API errors (rate limits, server errors) are retried
        up to MAX_RETRIES times using exponential backoff.
        Permanent errors (bad API key, invalid request) raise immediately.

    Usage:
        extractor = ObservationExtractor(document)
        result = extractor.extract()
        print(result)
        for obs in result.observations:
            print(obs)

    Environment variables required (.env file):
        OPENAI_API_KEY=sk-...
    """

    # Only these error types are worth retrying.
    # RateLimitError (429) and InternalServerError (500) are temporary.
    # AuthenticationError, BadRequestError, NotFoundError are permanent — never retry.
    RETRYABLE_ERRORS: tuple[type[OpenAIError], ...] = (
        RateLimitError,
        InternalServerError,
    )

    def __init__(self, document: ParsedDocument) -> None:
        import config
        self.MODEL = config.AI_MODEL
        self.MAX_TEXT_LENGTH = config.MAX_TEXT_LENGTH
        self.MAX_RETRIES = config.MAX_RETRIES
        self.RETRY_BASE_DELAY = config.RETRY_BASE_DELAY
        self.document = document
        self.client = self._init_client()

    def _init_client(self) -> OpenAI:
        """
        Initialises the OpenAI client using the API key from the environment.

        Raises:
            EnvironmentError: If OPENAI_API_KEY is not set in the .env file.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file: OPENAI_API_KEY=sk-..."
            )
        return OpenAI(api_key=api_key)

    def _prepare_text(self) -> str:
        """
        Assembles document text from non-empty pages and truncates if needed.

        Skips blank pages (cover page, disclaimer, etc.) to avoid sending
        useless content to the LLM and wasting tokens.

        Returns:
            Cleaned document text ready for the extraction prompt.
        """
        pages = self.document.non_empty_pages
        full_text = "\n".join(page.text for page in pages)

        if len(full_text) > self.MAX_TEXT_LENGTH:
            logger.warning(
                "Document text (%d chars) exceeds limit (%d chars). Truncating.",
                len(full_text),
                self.MAX_TEXT_LENGTH,
            )
            full_text = full_text[: self.MAX_TEXT_LENGTH]

        return full_text

    def _call_llm(self, text: str) -> str:
        """
        Sends the extraction prompt to OpenAI and returns the raw response string.

        Retries up to MAX_RETRIES times on transient errors using exponential
        backoff (1s → 2s → 4s). Permanent errors raise immediately without
        retrying to avoid wasting time and API quota.

        Backoff schedule:
            Attempt 1 fails → wait 1.0s  (1.0 × 2⁰)
            Attempt 2 fails → wait 2.0s  (1.0 × 2¹)
            Attempt 3 fails → wait 4.0s  (1.0 × 2²)
            Attempt 4 fails → raise RuntimeError

        Args:
            text: The prepared document text to analyse.

        Returns:
            Raw string response from the LLM.

        Raises:
            RuntimeError: If all retry attempts are exhausted on transient errors.
            RuntimeError: If a permanent (non-retryable) API error occurs.
        """
        prompt = EXTRACTION_PROMPT.format(text=text)

        logger.info(
            "Sending extraction request to %s (~%d chars).",
            self.MODEL,
            len(prompt),
        )

        last_exception: OpenAIError | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert building inspection analyst. "
                                "You extract structured data from inspection reports. "
                                "Always respond with valid JSON only."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0.0,  # deterministic output — critical for data extraction
                    max_tokens=4096,
                )

                raw_response = response.choices[0].message.content

                logger.debug(
                    "LLM response received (attempt %d/%d, %d chars): %s...",
                    attempt,
                    self.MAX_RETRIES,
                    len(raw_response),
                    raw_response[:200],
                )

                return raw_response

            except self.RETRYABLE_ERRORS as e:
                last_exception = e
                delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))

                if attempt == self.MAX_RETRIES:
                    # Final attempt exhausted — fall through to raise below
                    break

                logger.warning(
                    "Transient OpenAI error on attempt %d/%d (%s): %s. "
                    "Retrying in %.1fs.",
                    attempt,
                    self.MAX_RETRIES,
                    type(e).__name__,
                    str(e),
                    delay,
                )
                time.sleep(delay)

            except APIStatusError as e:
                # Permanent error — bad API key, invalid model, malformed request.
                # Retrying will never help. Raise immediately.
                raise RuntimeError(
                    f"OpenAI API error (non-retryable, HTTP {e.status_code}): {e.message}"
                ) from e

            except OpenAIError as e:
                # Catch-all for any other unexpected OpenAI errors. Do not retry.
                raise RuntimeError(
                    f"Unexpected OpenAI error: {type(e).__name__}: {e}"
                ) from e

        raise RuntimeError(
            f"OpenAI API call failed after {self.MAX_RETRIES} attempts. "
            f"Last error: {type(last_exception).__name__}: {last_exception}"
        )

    def _parse_and_validate(self, raw_response: str) -> list[Observation]:
        """
        Parses the LLM JSON response and validates it against the Pydantic schema.

        Handles the case where the LLM wraps JSON in markdown code fences
        (e.g. ```json ... ```) despite being explicitly told not to.
        Validation via LLMExtractionResponse ensures one malformed entry
        does not silently corrupt the entire result.

        Args:
            raw_response: Raw string returned by the LLM.

        Returns:
            List of validated Observation objects.

        Raises:
            ValueError: If the response is not valid JSON or fails schema validation.
        """
        cleaned = raw_response.strip()

        # Strip markdown code fences if the LLM added them despite instructions
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM returned invalid JSON.\n"
                f"Details: {e}\n"
                f"Raw response (first 500 chars): {raw_response[:500]}"
            ) from e

        validated_response = LLMExtractionResponse.model_validate(data)

        logger.info(
            "Validated %d observations from LLM response.",
            len(validated_response.observations),
        )

        return validated_response.observations

    @staticmethod
    def _deduplicate(
        observations: list[Observation],
    ) -> tuple[list[Observation], int]:
        """
        Removes duplicate observations based on area + issue_type fingerprint.

        UrbanRoof reports mention the same issue in multiple places:
        the summary section, the detailed observation section, and
        sometimes again in the thermal reference section. Only the
        first occurrence of each area + issue combination is kept.

        Args:
            observations: Raw list potentially containing duplicates.

        Returns:
            Tuple of (deduplicated list, number of duplicates removed).
        """
        seen_fingerprints: set[str] = set()
        unique: list[Observation] = []

        for obs in observations:
            fp = obs.fingerprint
            if fp in seen_fingerprints:
                logger.debug(
                    "Duplicate removed: area='%s', issue='%s'.",
                    obs.area,
                    obs.issue_type.value,
                )
                continue
            seen_fingerprints.add(fp)
            unique.append(obs)

        duplicates_removed = len(observations) - len(unique)
        return unique, duplicates_removed

    def extract(self) -> ObservationExtractionResult:
        """
        Runs the full observation extraction pipeline on the parsed document.

        Steps:
            1. Assemble text from non-empty pages, truncate if needed.
            2. Call OpenAI with the extraction prompt (with retry logic).
            3. Parse and validate the JSON response via Pydantic.
            4. Deduplicate by area + issue type fingerprint.
            5. Return structured ObservationExtractionResult.

        Returns:
            ObservationExtractionResult with validated, deduplicated observations.

        Raises:
            EnvironmentError: If OPENAI_API_KEY is missing.
            RuntimeError:     If the LLM call fails after all retries.
            ValueError:       If the LLM response cannot be parsed or validated.
        """
        logger.info(
            "Starting observation extraction: '%s'",
            self.document.source_path.name,
        )

        text = self._prepare_text()

        if not text.strip():
            logger.warning(
                "Document '%s' contains no extractable text. Returning empty result.",
                self.document.source_path.name,
            )
            return ObservationExtractionResult(
                observations=[],
                total_raw=0,
                duplicates_removed=0,
                source_name=self.document.source_path.name,
            )

        raw_response = self._call_llm(text)
        raw_observations = self._parse_and_validate(raw_response)
        unique_observations, duplicates_removed = self._deduplicate(raw_observations)

        logger.info(
            "Extraction complete: %d unique observations, %d duplicates removed.",
            len(unique_observations),
            duplicates_removed,
        )

        return ObservationExtractionResult(
            observations=unique_observations,
            total_raw=len(raw_observations),
            duplicates_removed=duplicates_removed,
            source_name=self.document.source_path.name,
        )

    def __repr__(self) -> str:
        return (
            f"ObservationExtractor("
            f"document='{self.document.source_path.name}', "
            f"model='{self.MODEL}')"
        )