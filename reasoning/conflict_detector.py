"""
reasoning/conflict_detector.py

Detects conflicts between inspection report observations and thermal
report observations for the same area.

A conflict is a meaningful disagreement between two data sources that
the client needs to know about. There are three types:

    1. THERMAL_ONLY
       Thermal imaging detected moisture or an anomaly in an area
       where the visual inspection found nothing.
       This is the most important conflict type — thermal can detect
       hidden moisture that is invisible to the naked eye.

    2. INSPECTION_ONLY
       Visual inspection found a clear issue (crack, tile gap, dampness)
       in an area where thermal shows no anomaly.
       This may indicate a dry-phase issue or a thermal imaging limitation.

    3. TYPE_MISMATCH
       Both sources found issues in the same area, but the issue types
       are meaningfully different (not just different labels for the
       same symptom).
       Example: Inspection says structural crack, thermal shows no stress.

NOTE on false positives:
    Dampness and efflorescence in the same area are NOT a conflict —
    they are related moisture symptoms. This module uses a compatibility
    map to avoid flagging symptom variants as contradictions.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from extraction.observation_extractor import (
    IssueType,
    Observation,
    ObservationExtractionResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Issue compatibility — these pairs are NOT conflicts
# ---------------------------------------------------------------------------

# If inspection and thermal both map to any of the same compatibility group,
# they are considered consistent observations, not conflicts.
COMPATIBLE_ISSUE_GROUPS: list[frozenset[IssueType]] = [
    # Moisture symptom group — all related, not contradictory
    frozenset({
        IssueType.DAMPNESS,
        IssueType.SEEPAGE,
        IssueType.EFFLORESCENCE,
        IssueType.PAINT_SPALLING,
    }),
    # Surface deterioration group
    frozenset({
        IssueType.CRACK,
        IssueType.HOLLOWNESS,
        IssueType.STRUCTURAL,
    }),
    # Tile-related group
    frozenset({
        IssueType.TILE_GAP,
        IssueType.HOLLOWNESS,
    }),
]


def _are_compatible(issue_a: IssueType, issue_b: IssueType) -> bool:
    """
    Returns True if two issue types are considered compatible
    (related symptoms, not contradictory findings).

    Args:
        issue_a: Issue type from one source.
        issue_b: Issue type from the other source.

    Returns:
        True if they belong to the same compatibility group.
    """
    if issue_a == issue_b:
        return True
    for group in COMPATIBLE_ISSUE_GROUPS:
        if issue_a in group and issue_b in group:
            return True
    return False


# ---------------------------------------------------------------------------
# Enums and data models
# ---------------------------------------------------------------------------

class ConflictType(str, Enum):
    """
    The category of disagreement between inspection and thermal sources.

    THERMAL_ONLY:     Thermal found an issue, inspection found nothing.
    INSPECTION_ONLY:  Inspection found an issue, thermal found nothing.
    TYPE_MISMATCH:    Both found issues but they are incompatible types.
    """

    THERMAL_ONLY = "thermal_only"
    INSPECTION_ONLY = "inspection_only"
    TYPE_MISMATCH = "type_mismatch"


class ConflictSeverity(str, Enum):
    """
    How urgently this conflict needs attention in the DDR report.

    HIGH:   Thermal found hidden moisture the inspection missed.
            Requires immediate investigation.
    MEDIUM: Inspection found a visible issue thermal did not confirm.
            Worth noting but less urgent.
    LOW:    Different issue labels for what may be the same problem.
            Informational — mention in the report but not alarming.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Conflict:
    """
    Represents a single meaningful contradiction between two data sources
    for the same area of the property.

    Attributes:
        area:               The property area where the conflict was detected.
        conflict_type:      Category of disagreement.
        severity:           How urgently this needs attention.
        inspection_issues:  Issue types found by visual inspection (empty if none).
        thermal_issues:     Issue types found by thermal imaging (empty if none).
        explanation:        Client-friendly description of the specific disagreement.
    """

    area: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    inspection_issues: list[str]
    thermal_issues: list[str]
    explanation: str

    def __repr__(self) -> str:
        return (
            f"Conflict("
            f"area='{self.area}', "
            f"type='{self.conflict_type.value}', "
            f"severity='{self.severity.value}')"
        )


class ConflictDetectionResult:
    """
    Complete result of a conflict detection run.

    Provides the full conflict list and convenience methods
    for filtering by type or severity.
    """

    def __init__(
        self,
        conflicts: list[Conflict],
        areas_compared: int,
    ) -> None:
        self.conflicts = conflicts
        self.areas_compared = areas_compared

    @property
    def count(self) -> int:
        """Total number of conflicts detected."""
        return len(self.conflicts)

    @property
    def has_conflicts(self) -> bool:
        """True if any conflicts were detected."""
        return self.count > 0

    def get_by_type(self, conflict_type: ConflictType) -> list[Conflict]:
        """Returns all conflicts of a specific type."""
        return [c for c in self.conflicts if c.conflict_type == conflict_type]

    def get_by_severity(self, severity: ConflictSeverity) -> list[Conflict]:
        """Returns all conflicts of a specific severity level."""
        return [c for c in self.conflicts if c.severity == severity]

    @property
    def high_severity(self) -> list[Conflict]:
        """Shortcut: returns all HIGH severity conflicts."""
        return self.get_by_severity(ConflictSeverity.HIGH)

    def __repr__(self) -> str:
        return (
            f"ConflictDetectionResult("
            f"conflicts={self.count}, "
            f"areas_compared={self.areas_compared}, "
            f"high_severity={len(self.high_severity)})"
        )


# ---------------------------------------------------------------------------
# Conflict detector
# ---------------------------------------------------------------------------

class ConflictDetector:
    """
    Compares inspection observations with thermal observations to detect
    meaningful contradictions that should be reported to the client.

    Logic overview:
        For each area that appears in either source:
            - If thermal found issues but inspection found none → THERMAL_ONLY (HIGH)
            - If inspection found issues but thermal found none → INSPECTION_ONLY (MEDIUM)
            - If both found issues but types are incompatible  → TYPE_MISMATCH (LOW)
            - If both found compatible issue types             → no conflict

    Usage:
        detector = ConflictDetector(inspection_result, thermal_result)
        result = detector.detect()
        print(result)
        for conflict in result.high_severity:
            print(conflict)
    """

    def __init__(
        self,
        inspection_result: ObservationExtractionResult,
        thermal_result: ObservationExtractionResult,
    ) -> None:
        self.inspection_result = inspection_result
        self.thermal_result = thermal_result

    def _group_by_area(
        self, observations: list[Observation]
    ) -> dict[str, list[Observation]]:
        """
        Groups observations by normalised area name (lowercase).

        Args:
            observations: List of observations to group.

        Returns:
            Dict mapping lowercase area name → list of observations.
        """
        grouped: dict[str, list[Observation]] = {}
        for obs in observations:
            area_key = obs.area.lower().strip()
            grouped.setdefault(area_key, []).append(obs)
        return grouped

    def _build_explanation(
        self,
        conflict_type: ConflictType,
        area: str,
        inspection_issues: list[str],
        thermal_issues: list[str],
    ) -> str:
        """
        Builds a specific, client-friendly explanation for a conflict.

        Each conflict type gets a distinct explanation that accurately
        describes the disagreement rather than a generic fallback message.

        Args:
            conflict_type:      The type of conflict detected.
            area:               The area name (title-cased).
            inspection_issues:  List of issue type strings from inspection.
            thermal_issues:     List of issue type strings from thermal.

        Returns:
            A clear, specific explanation string for the DDR report.
        """
        inspection_str = ", ".join(inspection_issues) if inspection_issues else "no issue"
        thermal_str = ", ".join(thermal_issues) if thermal_issues else "no issue"

        match conflict_type:
            case ConflictType.THERMAL_ONLY:
                return (
                    f"Thermal imaging detected {thermal_str} in {area}, "
                    f"but the visual inspection found no issue in this area. "
                    f"Hidden moisture or subsurface damage may be present. "
                    f"Further investigation is recommended."
                )
            case ConflictType.INSPECTION_ONLY:
                return (
                    f"Visual inspection found {inspection_str} in {area}, "
                    f"but thermal imaging did not confirm this finding. "
                    f"The issue may be intermittent or not yet thermally detectable."
                )
            case ConflictType.TYPE_MISMATCH:
                return (
                    f"Inspection reported {inspection_str} in {area}, "
                    f"while thermal imaging indicated {thermal_str}. "
                    f"These findings are inconsistent and require professional review."
                )
            case _:
                return f"Conflicting findings detected in {area}."

    def _check_area(
        self,
        area: str,
        inspection_obs: list[Observation],
        thermal_obs: list[Observation],
    ) -> Conflict | None:
        """
        Evaluates a single area for conflicts between the two sources.

        Args:
            area:           Lowercase area key.
            inspection_obs: Inspection observations for this area (may be empty).
            thermal_obs:    Thermal observations for this area (may be empty).

        Returns:
            A Conflict object if a meaningful disagreement is found, None otherwise.
        """
        inspection_issues = {obs.issue_type for obs in inspection_obs}
        thermal_issues = {obs.issue_type for obs in thermal_obs}

        inspection_issue_strs = [i.value for i in inspection_issues]
        thermal_issue_strs = [i.value for i in thermal_issues]
        area_title = area.title()

        # --- Case 1: Thermal found issues, inspection found nothing ---
        # Most important conflict type. Thermal can detect hidden moisture
        # that a visual inspection completely misses.
        if thermal_issues and not inspection_issues:
            return Conflict(
                area=area_title,
                conflict_type=ConflictType.THERMAL_ONLY,
                severity=ConflictSeverity.HIGH,
                inspection_issues=[],
                thermal_issues=thermal_issue_strs,
                explanation=self._build_explanation(
                    ConflictType.THERMAL_ONLY,
                    area_title,
                    [],
                    thermal_issue_strs,
                ),
            )

        # --- Case 2: Inspection found issues, thermal found nothing ---
        # Less urgent — the issue is visible but thermal doesn't confirm it.
        if inspection_issues and not thermal_issues:
            return Conflict(
                area=area_title,
                conflict_type=ConflictType.INSPECTION_ONLY,
                severity=ConflictSeverity.MEDIUM,
                inspection_issues=inspection_issue_strs,
                thermal_issues=[],
                explanation=self._build_explanation(
                    ConflictType.INSPECTION_ONLY,
                    area_title,
                    inspection_issue_strs,
                    [],
                ),
            )

        # --- Case 3: Both found issues — check for incompatibility ---
        # If every inspection issue is compatible with every thermal issue,
        # there is no conflict (they are different descriptions of the same problem).
        # A conflict only exists if at least one pair is incompatible.
        if inspection_issues and thermal_issues:
            has_incompatible_pair = any(
                not _are_compatible(i_issue, t_issue)
                for i_issue in inspection_issues
                for t_issue in thermal_issues
            )

            if has_incompatible_pair:
                return Conflict(
                    area=area_title,
                    conflict_type=ConflictType.TYPE_MISMATCH,
                    severity=ConflictSeverity.LOW,
                    inspection_issues=inspection_issue_strs,
                    thermal_issues=thermal_issue_strs,
                    explanation=self._build_explanation(
                        ConflictType.TYPE_MISMATCH,
                        area_title,
                        inspection_issue_strs,
                        thermal_issue_strs,
                    ),
                )

        return None

    def detect(self) -> ConflictDetectionResult:
        """
        Runs the full conflict detection pass across all areas.

        Builds area maps from both sources, then evaluates every area
        that appears in either source for contradictions.

        Returns:
            ConflictDetectionResult with all detected conflicts and summary stats.
        """
        logger.info(
            "Starting conflict detection: '%s' vs '%s'",
            self.inspection_result.source_name,
            self.thermal_result.source_name,
        )

        inspection_map = self._group_by_area(self.inspection_result.observations)
        thermal_map = self._group_by_area(self.thermal_result.observations)

        # Union of all areas from both sources
        all_areas: set[str] = set(inspection_map.keys()) | set(thermal_map.keys())

        conflicts: list[Conflict] = []

        for area in sorted(all_areas):  # sorted for deterministic output
            inspection_obs = inspection_map.get(area, [])
            thermal_obs = thermal_map.get(area, [])

            conflict = self._check_area(area, inspection_obs, thermal_obs)

            if conflict:
                logger.debug("Conflict detected: %s", conflict)
                conflicts.append(conflict)

        result = ConflictDetectionResult(
            conflicts=conflicts,
            areas_compared=len(all_areas),
        )

        logger.info(
            "Conflict detection complete: %d conflicts across %d areas "
            "(%d high severity).",
            result.count,
            result.areas_compared,
            len(result.high_severity),
        )

        return result

    def __repr__(self) -> str:
        return (
            f"ConflictDetector("
            f"inspection='{self.inspection_result.source_name}', "
            f"thermal='{self.thermal_result.source_name}')"
        )