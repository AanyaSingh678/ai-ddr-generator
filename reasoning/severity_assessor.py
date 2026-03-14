"""
reasoning/severity_assessor.py

Assigns severity levels to extracted observations using a multi-factor
rule-based system.

Why rule-based instead of LLM for this step?
    Severity assessment must be consistent, auditable, and explainable.
    A client or reviewer must be able to trace exactly why an observation
    was rated High vs Medium. LLM-assigned severities vary between runs
    and cannot be justified with a clear rule. Rule-based logic is
    deterministic, testable, and transparent.

Severity factors considered (in priority order):
    1. Issue type          — base severity from known issue categories
    2. Leakage timing      — all-time leakage escalates severity
    3. Thermal confirmation — thermal-confirmed observations escalate
    4. Impact text         — keywords like "structural" escalate severity
    5. Probable cause text — known high-risk causes escalate severity
"""

import logging
from dataclasses import dataclass

from extraction.observation_extractor import (
    IssueType,
    LeakageTiming,
    Observation,
    SeverityLevel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity rules — base severity by issue type
# ---------------------------------------------------------------------------

# Base severity assigned to each issue type before contextual factors.
# These are starting points — the multi-factor logic below can escalate them.
BASE_SEVERITY: dict[IssueType, SeverityLevel] = {
    IssueType.PAINT_SPALLING:  SeverityLevel.LOW,
    IssueType.EFFLORESCENCE:   SeverityLevel.LOW,
    IssueType.TILE_GAP:        SeverityLevel.LOW,
    IssueType.VEGETATION:      SeverityLevel.MEDIUM,
    IssueType.HOLLOWNESS:      SeverityLevel.MEDIUM,
    IssueType.DAMPNESS:        SeverityLevel.MEDIUM,
    IssueType.SEEPAGE:         SeverityLevel.MEDIUM,
    IssueType.PLUMBING:        SeverityLevel.MEDIUM,
    IssueType.CRACK:           SeverityLevel.HIGH,
    IssueType.STRUCTURAL:      SeverityLevel.CRITICAL,
    IssueType.UNKNOWN:         SeverityLevel.LOW,
}

# Ordered severity levels for escalation logic.
# Index position represents severity rank (higher index = more severe).
SEVERITY_ORDER: list[SeverityLevel] = [
    SeverityLevel.LOW,
    SeverityLevel.MEDIUM,
    SeverityLevel.HIGH,
    SeverityLevel.CRITICAL,
]

# Keywords in impact or probable_cause text that trigger escalation.
# If any of these are found, severity is escalated by one level.
ESCALATION_KEYWORDS: list[str] = [
    "structural",
    "collapse",
    "reinforcement",
    "exposed steel",
    "spalling of concrete",
    "corroded",
    "foundation",
    "load bearing",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _escalate(severity: SeverityLevel) -> SeverityLevel:
    """
    Returns the next higher severity level.
    If already CRITICAL, stays at CRITICAL.

    Args:
        severity: Current severity level.

    Returns:
        Escalated severity level.
    """
    current_index = SEVERITY_ORDER.index(severity)
    escalated_index = min(current_index + 1, len(SEVERITY_ORDER) - 1)
    return SEVERITY_ORDER[escalated_index]


def _contains_escalation_keyword(text: str | None) -> bool:
    """
    Returns True if the text contains any keyword that indicates
    a more serious underlying condition.

    Args:
        text: Any string field from an observation (impact, probable_cause, etc.)
              Can be None — returns False safely.

    Returns:
        True if an escalation keyword is found.
    """
    if not text:
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in ESCALATION_KEYWORDS)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SeverityAssessment:
    """
    Severity analysis result for a single observation.

    Attributes:
        area:              Property area where the issue was found.
        issue_type:        Category of the issue.
        severity:          Final assigned severity level after all factors.
        reasoning:         Human-readable explanation of why this severity
                           was assigned. Used directly in the DDR report.
        base_severity:     Starting severity before contextual factors.
                           Useful for debugging and audit.
        escalation_flags:  List of factors that caused severity escalation.
                           Empty if no escalation occurred.
    """

    area: str
    issue_type: IssueType
    severity: SeverityLevel
    reasoning: str
    base_severity: SeverityLevel
    escalation_flags: list[str]

    @property
    def was_escalated(self) -> bool:
        """True if severity was escalated above the base level."""
        return len(self.escalation_flags) > 0

    def __repr__(self) -> str:
        escalated = f", escalated_by={self.escalation_flags}" if self.was_escalated else ""
        return (
            f"SeverityAssessment("
            f"area='{self.area}', "
            f"issue='{self.issue_type.value}', "
            f"severity='{self.severity.value}'"
            f"{escalated})"
        )


class SeverityAssessmentResult:
    """
    Complete result of a severity assessment run.

    Holds all assessments and provides convenience methods
    for filtering and summarising results — used by the
    DDR report generator to build the Severity Assessment section.
    """

    def __init__(
        self,
        assessments: list[SeverityAssessment],
        source_name: str,
    ) -> None:
        self.assessments = assessments
        self.source_name = source_name

    @property
    def count(self) -> int:
        """Total number of assessed observations."""
        return len(self.assessments)

    def get_by_severity(self, severity: SeverityLevel) -> list[SeverityAssessment]:
        """Returns all assessments at a specific severity level."""
        return [a for a in self.assessments if a.severity == severity]

    @property
    def critical(self) -> list[SeverityAssessment]:
        """Shortcut: all CRITICAL severity assessments."""
        return self.get_by_severity(SeverityLevel.CRITICAL)

    @property
    def high(self) -> list[SeverityAssessment]:
        """Shortcut: all HIGH severity assessments."""
        return self.get_by_severity(SeverityLevel.HIGH)

    @property
    def escalated(self) -> list[SeverityAssessment]:
        """Returns assessments where severity was escalated above base level."""
        return [a for a in self.assessments if a.was_escalated]

    def summary(self) -> dict[str, int]:
        """
        Returns a count of assessments per severity level.
        Useful for the Property Issue Summary section of the DDR.

        Returns:
            Dict mapping severity label → count.
            Example: {"critical": 1, "high": 3, "medium": 4, "low": 2}
        """
        return {
            level.value: len(self.get_by_severity(level))
            for level in SEVERITY_ORDER
        }

    def __repr__(self) -> str:
        return (
            f"SeverityAssessmentResult("
            f"source='{self.source_name}', "
            f"total={self.count}, "
            f"critical={len(self.critical)}, "
            f"high={len(self.high)})"
        )


# ---------------------------------------------------------------------------
# Severity assessor
# ---------------------------------------------------------------------------

class SeverityAssessor:
    """
    Assigns severity levels to observations using a multi-factor rule system.

    Assessment factors applied in order:
        1. Base severity    — determined by issue type from BASE_SEVERITY map.
        2. Leakage timing   — all-time leakage escalates by one level.
        3. Thermal flag     — thermal-confirmed issues escalate by one level.
        4. Keyword scan     — structural/damage keywords in impact or cause
                              escalate by one level.

    Each escalation is tracked in escalation_flags so the reasoning
    string in the final DDR is specific and auditable.

    Usage:
        assessor = SeverityAssessor(observations, source_name="inspection_report.pdf")
        result = assessor.assess()
        print(result)
        print(result.summary())
        for a in result.critical:
            print(a)
    """

    def __init__(
        self,
        observations: list[Observation],
        source_name: str = "unknown",
    ) -> None:
        self.observations = observations
        self.source_name = source_name

    def _assess_single(self, observation: Observation) -> SeverityAssessment:
        """
        Applies all severity factors to a single observation and
        builds the final SeverityAssessment with detailed reasoning.

        Factor 1 — Base severity from issue type.
        Factor 2 — Escalate if leakage occurs all the time (not just monsoon).
        Factor 3 — Escalate if thermal imaging confirmed the issue.
        Factor 4 — Escalate if impact or cause text contains high-risk keywords.

        Args:
            observation: A validated Observation from the extractor.

        Returns:
            SeverityAssessment with severity, reasoning, and escalation flags.
        """
        issue_type = observation.issue_type
        severity = BASE_SEVERITY.get(issue_type, SeverityLevel.LOW)
        base_severity = severity
        escalation_flags: list[str] = []
        reasoning_parts: list[str] = []

        # --- Factor 1: Base severity ---
        reasoning_parts.append(
            f"'{issue_type.value}' issues have a base severity of {severity.value}"
        )

        # --- Factor 2: Leakage timing ---
        if observation.leakage_timing == LeakageTiming.ALL_TIME:
            severity = _escalate(severity)
            escalation_flags.append("all-time leakage")
            reasoning_parts.append(
                "escalated because leakage occurs continuously, not only seasonally"
            )

        # --- Factor 3: Thermal confirmation ---
        # An observation is considered thermally confirmed if it has a source_page
        # from a thermal document. This flag is set externally by the pipeline
        # when merging inspection and thermal observations.
        # For now we check the description for thermal indicator keywords.
        thermal_keywords = ["thermal", "temperature", "°c", "ir ", "infrared"]
        description_lower = (observation.description or "").lower()
        if any(kw in description_lower for kw in thermal_keywords):
            severity = _escalate(severity)
            escalation_flags.append("thermal confirmation")
            reasoning_parts.append(
                "escalated because thermal imaging confirmed the presence of moisture"
            )

        # --- Factor 4: High-risk keyword scan ---
        if _contains_escalation_keyword(observation.impact):
            severity = _escalate(severity)
            escalation_flags.append("high-risk impact keywords")
            reasoning_parts.append(
                f"escalated due to high-risk language in impact description: "
                f"'{observation.impact[:80]}'"
            )
        elif _contains_escalation_keyword(observation.probable_cause):
            severity = _escalate(severity)
            escalation_flags.append("high-risk cause keywords")
            reasoning_parts.append(
                f"escalated due to high-risk language in probable cause: "
                f"'{observation.probable_cause[:80]}'"
            )

        # --- Build final reasoning string for DDR report ---
        reasoning = ". ".join(reasoning_parts).capitalize() + "."

        logger.debug(
            "Assessed '%s' / '%s': base=%s → final=%s (flags=%s)",
            observation.area,
            issue_type.value,
            base_severity.value,
            severity.value,
            escalation_flags or "none",
        )

        return SeverityAssessment(
            area=observation.area,
            issue_type=issue_type,
            severity=severity,
            reasoning=reasoning,
            base_severity=base_severity,
            escalation_flags=escalation_flags,
        )

    def assess(self) -> SeverityAssessmentResult:
        """
        Runs severity assessment across all observations.

        Returns:
            SeverityAssessmentResult containing all assessments
            and summary statistics.
        """
        logger.info(
            "Starting severity assessment: %d observations from '%s'.",
            len(self.observations),
            self.source_name,
        )

        assessments: list[SeverityAssessment] = []

        for observation in self.observations:
            assessment = self._assess_single(observation)
            assessments.append(assessment)

        result = SeverityAssessmentResult(
            assessments=assessments,
            source_name=self.source_name,
        )

        logger.info(
            "Assessment complete: %d total — %s",
            result.count,
            result.summary(),
        )

        return result

    def __repr__(self) -> str:
        return (
            f"SeverityAssessor("
            f"observations={len(self.observations)}, "
            f"source='{self.source_name}')"
        )