"""
reasoning/observation_image_mapper.py

Maps extracted images to their corresponding observations using a
multi-signal confidence scoring system.

Why confidence scoring instead of binary match:
    The UrbanRoof inspection report embeds images throughout the document
    in ways that don't always align with a single page number. The same
    area (e.g. "Master Bedroom") may appear in headings, body text, captions,
    and summary tables across multiple pages. A binary page-match rule misses
    most images because:

        1. Observation.source_page is often None (the LLM rarely extracts
           reliable page numbers from dense inspection text).
        2. The image may be on a different page than where the observation
           text was extracted from.
        3. Multiple signals together (area name + issue keyword + heading)
           are far more reliable than any single signal alone.

Matching signals used (with weights):
    - Area name in image heading text      → strong signal (weight: 3)
    - Area name in image nearby body text  → moderate signal (weight: 2)
    - Issue keyword in nearby text         → supporting signal (weight: 1)
    - Page number match                    → supporting signal (weight: 1)

An image is assigned to an observation if its total score
meets or exceeds MATCH_THRESHOLD.
"""

import logging
from dataclasses import dataclass, field

from extraction.observation_extractor import IssueType, Observation
from parser.image_extractor import ExtractedImage, ExtractionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring configuration
# ---------------------------------------------------------------------------

# Minimum total score for an image to be assigned to an observation.
# Increase to be more selective, decrease to be more permissive.
MATCH_THRESHOLD: int = 2

# Signal weights — how much each matching signal contributes to the score.
WEIGHT_AREA_IN_HEADING: int = 3
WEIGHT_AREA_IN_BODY: int = 2
WEIGHT_ISSUE_KEYWORD: int = 1
WEIGHT_PAGE_MATCH: int = 1

# Maps issue types to keywords that might appear near relevant images.
# These are checked against image nearby_text to provide supporting evidence.
ISSUE_KEYWORDS: dict[IssueType, list[str]] = {
    IssueType.DAMPNESS:       ["dampness", "damp", "moisture", "wet"],
    IssueType.SEEPAGE:        ["seepage", "seeping", "leakage", "leak"],
    IssueType.CRACK:          ["crack", "cracks", "cracking", "fissure"],
    IssueType.TILE_GAP:       ["tile", "grout", "gap", "joint"],
    IssueType.HOLLOWNESS:     ["hollow", "hollowness", "debonding"],
    IssueType.VEGETATION:     ["vegetation", "moss", "algae", "growth", "plant"],
    IssueType.EFFLORESCENCE:  ["efflorescence", "salt", "white deposit"],
    IssueType.PAINT_SPALLING: ["spalling", "peeling", "flaking", "paint"],
    IssueType.STRUCTURAL:     ["structural", "crack", "spalling", "rcc", "beam", "column"],
    IssueType.PLUMBING:       ["plumbing", "pipe", "outlet", "drain", "trap"],
    IssueType.UNKNOWN:        [],
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ImageMatch:
    """
    Represents a scored match between an image and an observation.

    Attributes:
        image:       The matched ExtractedImage.
        score:       Total confidence score from all matching signals.
        signals:     List of signals that contributed to the score.
                     Useful for debugging and audit.
    """

    image: ExtractedImage
    score: int
    signals: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ImageMatch("
            f"image='{self.image.file_path.name}', "
            f"score={self.score}, "
            f"signals={self.signals})"
        )


@dataclass
class MappingResult:
    """
    Complete result of an image mapping run.

    Attributes:
        image_map:             Dict mapping observation fingerprint
                               → list of matched ExtractedImages.
                               Used directly by DDRGenerator for O(1) lookup.
        observations_matched:  Count of observations that got at least one image.
        observations_unmatched: Count of observations with no matched images.
        total_assignments:     Total number of image assignments across all
                               observations (one image can match multiple observations).
    """

    image_map: dict[str, list[ExtractedImage]] = field(default_factory=dict)
    observations_matched: int = 0
    observations_unmatched: int = 0
    total_assignments: int = 0

    @property
    def match_rate(self) -> float:
        """
        Percentage of observations that received at least one image.
        Returns 0.0 if there are no observations.
        """
        total = self.observations_matched + self.observations_unmatched
        if total == 0:
            return 0.0
        return round((self.observations_matched / total) * 100, 1)

    def __repr__(self) -> str:
        return (
            f"MappingResult("
            f"matched={self.observations_matched}, "
            f"unmatched={self.observations_unmatched}, "
            f"assignments={self.total_assignments}, "
            f"match_rate={self.match_rate}%)"
        )


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------

class ObservationImageMapper:
    """
    Matches extracted images to observations using multi-signal confidence
    scoring.

    For each (observation, image) pair, the matcher computes a score by
    checking several signals. Only images whose score meets MATCH_THRESHOLD
    are assigned to the observation.

    This approach handles the two key failure modes of simpler matchers:
        - source_page being None (common with LLM extraction)
        - Area name not appearing on the exact page of the image

    Usage:
        mapper = ObservationImageMapper(observations, image_result)
        result = mapper.build_map()

        # In DDRGenerator:
        images = result.image_map.get(observation.fingerprint, [])
    """

    def __init__(
        self,
        observations: list[Observation],
        image_result: ExtractionResult,
    ) -> None:
        self.observations = observations
        self.image_result = image_result

    def _score_match(
        self,
        observation: Observation,
        image: ExtractedImage,
    ) -> ImageMatch:
        """
        Computes a confidence score for an observation–image pair
        by checking multiple signals.

        Signal 1 — Area name in image headings (strongest signal):
            Headings are the most reliable indicator because they are
            typically section titles like "4.4.1 CEILING (HALL)" which
            directly name the area being photographed.

        Signal 2 — Area name in image body text (moderate signal):
            The surrounding text on the same page as the image often
            describes what is shown, e.g. "Dampness observed at Hall ceiling."

        Signal 3 — Issue keyword in nearby text (supporting signal):
            Issue-specific keywords near the image provide supporting
            evidence even when the area name is not exact.

        Signal 4 — Page number match (supporting signal):
            When source_page is available, an exact page match adds
            to the confidence. Not relied upon alone due to LLM
            extraction limitations.

        Args:
            observation: The observation to match against.
            image:       The extracted image to evaluate.

        Returns:
            ImageMatch with the total score and list of triggered signals.
        """
        score = 0
        signals: list[str] = []

        area_lower = observation.area.lower().strip()
        nearby_lower = image.nearby_text.lower()

        # Build area search terms — split compound areas like
        # "Master Bedroom Bathroom" into ["master bedroom bathroom",
        # "master bedroom", "bathroom"] for partial matching.
        area_terms = self._build_area_terms(area_lower)

        # --- Signal 1: Area name in headings ---
        # Extract heading text from nearby_text (headings are typically
        # uppercase or at the start of lines in inspection reports)
        heading_lines = [
            line.strip().lower()
            for line in image.nearby_text.split("\n")
            if line.strip().isupper() or line.strip().startswith("4.")
        ]
        heading_text = " ".join(heading_lines)

        if any(term in heading_text for term in area_terms):
            score += WEIGHT_AREA_IN_HEADING
            signals.append(f"area_in_heading (area='{area_lower}')")

        # --- Signal 2: Area name in body text ---
        elif any(term in nearby_lower for term in area_terms):
            score += WEIGHT_AREA_IN_BODY
            signals.append(f"area_in_body (area='{area_lower}')")

        # --- Signal 3: Issue keyword in nearby text ---
        issue_keywords = ISSUE_KEYWORDS.get(observation.issue_type, [])
        matched_keywords = [kw for kw in issue_keywords if kw in nearby_lower]
        if matched_keywords:
            score += WEIGHT_ISSUE_KEYWORD
            signals.append(f"issue_keyword (matched={matched_keywords[:2]})")

        # --- Signal 4: Page number match ---
        if (
            observation.source_page is not None
            and image.page_number == observation.source_page
        ):
            score += WEIGHT_PAGE_MATCH
            signals.append(f"page_match (page={image.page_number})")

        return ImageMatch(image=image, score=score, signals=signals)

    @staticmethod
    def _build_area_terms(area: str) -> list[str]:
        """
        Builds a list of search terms from an area name to support
        partial matching.

        For compound area names like "master bedroom bathroom", this returns
        all substrings so that an image captioned "master bedroom" still
        matches an observation for "master bedroom bathroom".

        Args:
            area: Lowercase area name string.

        Returns:
            List of search terms from most specific to least specific.
        """
        words = area.split()
        terms = [area]  # full name always first (most specific)

        # Add progressively shorter substrings (at least 2 words)
        for end in range(len(words) - 1, 0, -1):
            sub = " ".join(words[:end])
            if sub not in terms and len(sub) > 3:
                terms.append(sub)

        return terms

    def _find_matches_for_observation(
        self,
        observation: Observation,
    ) -> list[ExtractedImage]:
        """
        Scores all images against a single observation and returns
        those that meet the MATCH_THRESHOLD, sorted by score descending.

        Args:
            observation: The observation to find images for.

        Returns:
            List of matched ExtractedImage objects, best matches first.
        """
        scored: list[ImageMatch] = []

        for image in self.image_result.images:
            match = self._score_match(observation, image)
            if match.score >= MATCH_THRESHOLD:
                scored.append(match)
                logger.debug(
                    "Match: obs='%s/%s' ↔ img='%s' score=%d signals=%s",
                    observation.area,
                    observation.issue_type.value,
                    image.file_path.name,
                    match.score,
                    match.signals,
                )

        # Sort by score descending — best matches first in the DDR
        scored.sort(key=lambda m: m.score, reverse=True)

        return [m.image for m in scored]

    def build_map(self) -> MappingResult:
        """
        Builds the complete observation → images lookup map.

        Iterates all observations, scores all images for each, and
        assembles a MappingResult with the final lookup dict and
        summary statistics.

        The returned image_map is keyed by observation.fingerprint
        (an 8-character MD5 hash of area+issue_type) so DDRGenerator
        can do O(1) lookups during report rendering.

        Returns:
            MappingResult containing the image_map and match statistics.
        """
        logger.info(
            "Starting image mapping: %d observations, %d images.",
            len(self.observations),
            self.image_result.unique_count,
        )

        image_map: dict[str, list[ExtractedImage]] = {}
        matched = 0
        unmatched = 0
        total_assignments = 0

        for observation in self.observations:
            matched_images = self._find_matches_for_observation(observation)

            image_map[observation.fingerprint] = matched_images

            if matched_images:
                matched += 1
                total_assignments += len(matched_images)
                logger.debug(
                    "Observation '%s/%s' → %d image(s) assigned.",
                    observation.area,
                    observation.issue_type.value,
                    len(matched_images),
                )
            else:
                unmatched += 1
                logger.debug(
                    "Observation '%s/%s' → no images matched.",
                    observation.area,
                    observation.issue_type.value,
                )

        result = MappingResult(
            image_map=image_map,
            observations_matched=matched,
            observations_unmatched=unmatched,
            total_assignments=total_assignments,
        )

        logger.info(
            "Image mapping complete: %s",
            result,
        )

        return result

    def __repr__(self) -> str:
        return (
            f"ObservationImageMapper("
            f"observations={len(self.observations)}, "
            f"images={self.image_result.unique_count})"
        )