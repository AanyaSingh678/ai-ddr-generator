"""
app.py

Main entry point for the UrbanRoof DDR Generator pipeline.

All configuration (paths, model name, property metadata) lives in config.py.
This file only orchestrates the pipeline steps — it contains no constants.

Pipeline:
    1. Parse inspection and thermal PDFs       → ParsedDocument
    2. Extract unique images from both PDFs    → ExtractionResult
    3. Extract observations using AI           → ObservationExtractionResult
    4. Map observations to images              → MappingResult
    5. Detect conflicts between sources        → ConflictDetectionResult
    6. Assess severity of each observation     → SeverityAssessmentResult
    7. Generate and save DDR report            → DDRReport (HTML)

Usage:
    python app.py

Output:
    output/DDR_Report_<timestamp>.html
"""

import logging
import sys
from pathlib import Path

import config
from parser.image_extractor import ExtractionResult, ImageExtractor
from parser.pdf_parser import PDFParser, ParsedDocument
from extraction.observation_extractor import (
    ObservationExtractionResult,
    ObservationExtractor,
)
from reasoning.conflict_detector import ConflictDetectionResult, ConflictDetector
from reasoning.observation_image_mapper import ObservationImageMapper
from reasoning.severity_assessor import SeverityAssessor
from report.ddr_generator import DDRGenerator


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def step_parse_document(pdf_path: Path) -> ParsedDocument:
    """
    Parses a single PDF and returns a ParsedDocument.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        ParsedDocument with page-level text and heading metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        RuntimeError:      If the PDF is encrypted or corrupted.
    """
    logger.info("Parsing: '%s'", pdf_path.name)
    document = PDFParser(str(pdf_path)).parse()
    logger.info(
        "Parsed '%s': %d pages (%d non-empty).",
        pdf_path.name,
        document.total_pages,
        len(document.non_empty_pages),
    )
    return document


def step_extract_images(pdf_path: Path, output_dir: Path) -> ExtractionResult:
    """
    Extracts unique images from a PDF.

    Args:
        pdf_path:   Path to the PDF file.
        output_dir: Directory to save extracted images.

    Returns:
        ExtractionResult with unique images and deduplication stats.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        RuntimeError:      If the PDF cannot be opened.
    """
    logger.info("Extracting images from: '%s'", pdf_path.name)
    result = ImageExtractor(str(pdf_path), str(output_dir)).extract()
    logger.info(
        "Images from '%s': %d unique, %d duplicates skipped.",
        pdf_path.name,
        result.unique_count,
        result.duplicates_skipped,
    )
    return result


def step_extract_observations(
    document: ParsedDocument,
) -> ObservationExtractionResult:
    """
    Runs AI observation extraction on a parsed document.
    Model and token settings are read from config.py via ObservationExtractor.

    Args:
        document: A ParsedDocument from step_parse_document.

    Returns:
        ObservationExtractionResult with validated, deduplicated observations.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
        RuntimeError:     If the OpenAI API call fails after all retries.
        ValueError:       If the LLM response cannot be parsed.
    """
    logger.info(
        "Extracting observations from: '%s' using model '%s'.",
        document.source_path.name,
        config.AI_MODEL,
    )
    result = ObservationExtractor(document).extract()
    logger.info(
        "Observations from '%s': %d unique (%d duplicates removed).",
        document.source_path.name,
        result.count,
        result.duplicates_removed,
    )
    return result


def step_build_empty_extraction_result(
    source_name: str,
) -> ObservationExtractionResult:
    """
    Creates an empty ObservationExtractionResult for use when the
    thermal document is unavailable.

    ConflictDetector requires two ObservationExtractionResult objects.
    This provides a valid empty result rather than None or a plain list.

    Args:
        source_name: Label for the missing source (used in logging).

    Returns:
        Empty ObservationExtractionResult.
    """
    return ObservationExtractionResult(
        observations=[],
        total_raw=0,
        duplicates_removed=0,
        source_name=source_name,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Runs the complete UrbanRoof DDR generation pipeline.

    Fatal steps (missing inspection PDF, AI failure) exit with code 1.
    Non-fatal steps (missing thermal PDF) log a warning and continue.
    All configuration is read from config.py — nothing is hardcoded here.
    """
    logger.info("=" * 52)
    logger.info("  UrbanRoof DDR Generator — Pipeline Start")
    logger.info("=" * 52)
    logger.info("Model        : %s", config.AI_MODEL)
    logger.info("Inspection   : %s", config.INSPECTION_PDF)
    logger.info("Thermal      : %s", config.THERMAL_PDF)
    logger.info("Output dir   : %s", config.OUTPUT_DIR)

    # ------------------------------------------------------------------
    # Step 1: Parse documents
    # ------------------------------------------------------------------
    logger.info("STEP 1 — Document Parsing")

    try:
        inspection_doc = step_parse_document(config.INSPECTION_PDF)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(
            "Cannot parse inspection report '%s': %s",
            config.INSPECTION_PDF.name, e,
        )
        logger.error(
            "Expected location: %s",
            config.INSPECTION_PDF.resolve(),
        )
        sys.exit(1)

    thermal_doc: ParsedDocument | None = None
    try:
        thermal_doc = step_parse_document(config.THERMAL_PDF)
    except FileNotFoundError:
        logger.warning(
            "Thermal report '%s' not found. "
            "Conflict detection will use empty thermal data.",
            config.THERMAL_PDF.name,
        )
    except RuntimeError as e:
        logger.warning(
            "Thermal report could not be parsed: %s. "
            "Continuing without thermal data.",
            e,
        )

    # ------------------------------------------------------------------
    # Step 2: Extract images
    # ------------------------------------------------------------------
    logger.info("STEP 2 — Image Extraction")

    try:
        inspection_images = step_extract_images(
            config.INSPECTION_PDF,
            config.INSPECTION_IMAGES_DIR,
        )
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("Image extraction failed for inspection report: %s", e)
        sys.exit(1)

    thermal_images: ExtractionResult | None = None
    if thermal_doc is not None:
        try:
            thermal_images = step_extract_images(
                config.THERMAL_PDF,
                config.THERMAL_IMAGES_DIR,
            )
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(
                "Image extraction failed for thermal report: %s. "
                "Continuing without thermal images.",
                e,
            )

    # ------------------------------------------------------------------
    # Step 3: Extract observations using AI
    # ------------------------------------------------------------------
    logger.info("STEP 3 — AI Observation Extraction")

    try:
        inspection_result = step_extract_observations(inspection_doc)
    except EnvironmentError as e:
        logger.error("Environment error: %s", e)
        sys.exit(1)
    except (RuntimeError, ValueError) as e:
        logger.error("Observation extraction failed: %s", e)
        sys.exit(1)

    thermal_result: ObservationExtractionResult
    if thermal_doc is not None:
        try:
            thermal_result = step_extract_observations(thermal_doc)
        except (EnvironmentError, RuntimeError, ValueError) as e:
            logger.warning(
                "Thermal observation extraction failed: %s. "
                "Using empty thermal result.",
                e,
            )
            thermal_result = step_build_empty_extraction_result(
                config.THERMAL_PDF.name
            )
    else:
        thermal_result = step_build_empty_extraction_result(
            config.THERMAL_PDF.name
        )

    # ------------------------------------------------------------------
    # Step 4: Map observations to images
    # ------------------------------------------------------------------
    logger.info("STEP 4 — Image–Observation Mapping")

    mapping_result = ObservationImageMapper(
        observations=inspection_result.observations,
        image_result=inspection_images,
    ).build_map()

    logger.info(
        "Image mapping: %d/%d observations matched (%.1f%%).",
        mapping_result.observations_matched,
        inspection_result.count,
        mapping_result.match_rate,
    )

    # ------------------------------------------------------------------
    # Step 5: Detect conflicts between inspection and thermal
    # ------------------------------------------------------------------
    logger.info("STEP 5 — Conflict Detection")

    conflict_result: ConflictDetectionResult = ConflictDetector(
        inspection_result=inspection_result,
        thermal_result=thermal_result,
    ).detect()

    logger.info(
        "Conflicts: %d total (%d high severity).",
        conflict_result.count,
        len(conflict_result.high_severity),
    )

    # ------------------------------------------------------------------
    # Step 6: Assess severity
    # ------------------------------------------------------------------
    logger.info("STEP 6 — Severity Assessment")

    severity_result = SeverityAssessor(
        observations=inspection_result.observations,
        source_name=inspection_doc.source_path.name,
    ).assess()

    logger.info("Severity summary: %s", severity_result.summary())

    # ------------------------------------------------------------------
    # Step 7: Generate DDR report
    # ------------------------------------------------------------------
    logger.info("STEP 7 — DDR Report Generation")

    # Merge inspection + thermal images so thermal evidence photos
    # can also appear in the report under relevant observations.
    if thermal_images is not None:
        combined_images = ExtractionResult(
            images=inspection_images.images + thermal_images.images,
            total_found=(
                inspection_images.total_found + thermal_images.total_found
            ),
            duplicates_skipped=(
                inspection_images.duplicates_skipped
                + thermal_images.duplicates_skipped
            ),
            source_pdf=inspection_images.source_pdf,
        )
    else:
        combined_images = inspection_images

    try:
        report = DDRGenerator(
            observation_result=inspection_result,
            severity_result=severity_result,
            conflict_result=conflict_result,
            image_result=combined_images,
            property_address=config.PROPERTY_ADDRESS,
            inspected_by=config.INSPECTED_BY,
            inspection_date=config.INSPECTION_DATE,
            output_dir=config.OUTPUT_DIR,
        ).generate()
    except Exception as e:
        logger.error("DDR report generation failed: %s", e)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("=" * 52)
    logger.info("  Pipeline Complete")
    logger.info("=" * 52)
    logger.info("Report saved : %s", report.output_path.resolve())
    logger.info("Observations : %d", report.observation_count)
    logger.info("Conflicts    : %d", report.conflict_count)
    logger.info(
        "Generated at : %s",
        report.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
    )
    logger.info(
        "Open report  : file://%s",
        report.output_path.resolve(),
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()