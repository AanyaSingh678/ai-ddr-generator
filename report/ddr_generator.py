"""
report/ddr_generator.py

Generates the final UrbanRoof-style Detailed Diagnostic Report (DDR)
from all processed pipeline data.

Output format: HTML
    HTML is chosen over plain Markdown because:
    - Renders correctly in any browser without extra tools
    - Supports inline images (base64 encoded)
    - Can be printed to PDF directly from the browser
    - Allows professional styling matching UrbanRoof's brand

DDR Structure (as required by assignment):
    1. Property Issue Summary
    2. Area-wise Observations  (with supporting images)
    3. Probable Root Cause
    4. Severity Assessment      (with reasoning)
    5. Recommended Actions
    6. Additional Notes
    7. Missing or Unclear Information
"""

import base64
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from extraction.observation_extractor import (
    IssueType,
    Observation,
    ObservationExtractionResult,
    SeverityLevel,
)
from parser.image_extractor import ExtractedImage, ExtractionResult
from reasoning.conflict_detector import (
    Conflict,
    ConflictDetectionResult,
    ConflictSeverity,
    ConflictType,
)
from reasoning.severity_assessor import (
    SeverityAssessment,
    SeverityAssessmentResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity display config — maps severity levels to colours and labels
# ---------------------------------------------------------------------------

SEVERITY_STYLE: dict[SeverityLevel, dict[str, str]] = {
    SeverityLevel.LOW: {
        "color": "#2e7d32",
        "background": "#e8f5e9",
        "border": "#a5d6a7",
        "label": "Low",
    },
    SeverityLevel.MEDIUM: {
        "color": "#e65100",
        "background": "#fff3e0",
        "border": "#ffcc80",
        "label": "Medium",
    },
    SeverityLevel.HIGH: {
        "color": "#b71c1c",
        "background": "#ffebee",
        "border": "#ef9a9a",
        "label": "High",
    },
    SeverityLevel.CRITICAL: {
        "color": "#ffffff",
        "background": "#b71c1c",
        "border": "#7f0000",
        "label": "Critical ⚠",
    },
}

CONFLICT_SEVERITY_STYLE: dict[ConflictSeverity, dict[str, str]] = {
    ConflictSeverity.HIGH: {
        "color": "#b71c1c",
        "background": "#ffebee",
        "border": "#ef9a9a",
        "label": "High Priority",
    },
    ConflictSeverity.MEDIUM: {
        "color": "#e65100",
        "background": "#fff3e0",
        "border": "#ffcc80",
        "label": "Medium Priority",
    },
    ConflictSeverity.LOW: {
        "color": "#1565c0",
        "background": "#e3f2fd",
        "border": "#90caf9",
        "label": "Low Priority",
    },
}


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _esc(text: str | None) -> str:
    """
    Escapes special HTML characters to prevent rendering issues.
    Returns 'Not Available' for None values.

    Args:
        text: Raw string that may contain HTML special characters.

    Returns:
        Safe HTML string.
    """
    if text is None:
        return "<em>Not Available</em>"
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _na(text: str | None) -> str:
    """
    Returns the text if present, or a styled 'Not Available' span.

    Args:
        text: Value that may be None.

    Returns:
        HTML string.
    """
    if not text or not str(text).strip():
        return '<span class="not-available">Not Available</span>'
    return _esc(text)


def _severity_badge(severity: SeverityLevel) -> str:
    """
    Returns an inline HTML badge for a severity level.

    Args:
        severity: The severity level to render.

    Returns:
        HTML span element with appropriate colour styling.
    """
    style = SEVERITY_STYLE.get(severity, SEVERITY_STYLE[SeverityLevel.LOW])
    return (
        f'<span class="severity-badge" '
        f'style="background:{style["background"]};'
        f'color:{style["color"]};'
        f'border:1px solid {style["border"]}">'
        f'{style["label"]}</span>'
    )


def _encode_image(image_path: Path) -> str | None:
    """
    Reads an image file and returns a base64-encoded data URI.
    This embeds the image directly into the HTML so the report
    is fully self-contained with no external file dependencies.

    Args:
        image_path: Path to the image file on disk.

    Returns:
        Base64 data URI string, or None if the file cannot be read.
    """
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        suffix = image_path.suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
        return f"data:image/{mime};base64,{image_data}"
    except OSError as e:
        logger.warning("Could not read image '%s': %s", image_path, e)
        return None


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DDRReport:
    """
    The generated DDR report with metadata.

    Attributes:
        html:           Complete HTML content of the report.
        output_path:    Path where the report was saved.
        generated_at:   Timestamp of generation.
        observation_count: Number of observations included.
        conflict_count:    Number of conflicts included.
    """

    html: str
    output_path: Path
    generated_at: datetime
    observation_count: int
    conflict_count: int

    def __repr__(self) -> str:
        return (
            f"DDRReport("
            f"output='{self.output_path.name}', "
            f"observations={self.observation_count}, "
            f"conflicts={self.conflict_count}, "
            f"generated_at='{self.generated_at.strftime('%Y-%m-%d %H:%M')}')"
        )


# ---------------------------------------------------------------------------
# DDR Generator
# ---------------------------------------------------------------------------

class DDRGenerator:
    """
    Assembles all pipeline outputs into a professional HTML DDR report.

    Inputs:
        observation_result:  Extracted, deduplicated observations.
        severity_result:     Severity assessments for each observation.
        conflict_result:     Detected conflicts between inspection and thermal.
        image_result:        Extracted images with page metadata.
        property_address:    Property address for the report header.
        inspected_by:        Inspector name for the report header.
        inspection_date:     Date of inspection for the report header.
        output_dir:          Directory where the HTML file will be saved.

    Usage:
        generator = DDRGenerator(
            observation_result=obs_result,
            severity_result=sev_result,
            conflict_result=conf_result,
            image_result=img_result,
            property_address="Flat No-8/63, Yamuna CHS, Mulund East",
            inspected_by="Tushar Rahane",
            inspection_date="July 24, 2023",
            output_dir=Path("output"),
        )
        report = generator.generate()
        print(report)
    """

    def __init__(
        self,
        observation_result: ObservationExtractionResult,
        severity_result: SeverityAssessmentResult,
        conflict_result: ConflictDetectionResult,
        image_result: ExtractionResult,
        property_address: str = "Not Available",
        inspected_by: str = "Not Available",
        inspection_date: str = "Not Available",
        output_dir: Path = Path("output"),
    ) -> None:
        self.observation_result = observation_result
        self.severity_result = severity_result
        self.conflict_result = conflict_result
        self.image_result = image_result
        self.property_address = property_address
        self.inspected_by = inspected_by
        self.inspection_date = inspection_date
        self.output_dir = output_dir

        # Build lookup maps for efficient access during rendering
        self._severity_map: dict[str, SeverityAssessment] = {
            f"{a.area.lower()}|{a.issue_type.value}": a
            for a in severity_result.assessments
        }
        self._images_by_page: dict[int, list[ExtractedImage]] = {}
        for img in image_result.images:
            self._images_by_page.setdefault(img.page_number, []).append(img)

    # ------------------------------------------------------------------
    # Section 1 — Property Issue Summary
    # ------------------------------------------------------------------

    def _section_issue_summary(self) -> str:
        """
        Builds Section 1: a high-level summary of all issue categories
        found, severity distribution, and total observation count.
        """
        observations = self.observation_result.observations

        if not observations:
            return "<p>No issues were detected in the provided documents.</p>"

        summary = self.severity_result.summary()
        issue_types = sorted({obs.issue_type.value for obs in observations})

        issue_list = "".join(
            f'<li class="issue-tag">{_esc(issue)}</li>'
            for issue in issue_types
        )

        severity_rows = "".join(
            f"""
            <tr>
                <td>{_esc(level.value.capitalize())}</td>
                <td>{_severity_badge(level)}</td>
                <td><strong>{count}</strong></td>
            </tr>"""
            for level, count in [
                (SeverityLevel.CRITICAL, summary.get("critical", 0)),
                (SeverityLevel.HIGH,     summary.get("high", 0)),
                (SeverityLevel.MEDIUM,   summary.get("medium", 0)),
                (SeverityLevel.LOW,      summary.get("low", 0)),
            ]
            if count > 0
        )

        return f"""
        <p>A total of <strong>{len(observations)}</strong> distinct issues were
        identified across the following categories:</p>
        <ul class="issue-tag-list">{issue_list}</ul>

        <h3>Severity Distribution</h3>
        <table class="summary-table">
            <thead>
                <tr><th>Severity</th><th>Level</th><th>Count</th></tr>
            </thead>
            <tbody>{severity_rows}</tbody>
        </table>
        """

    # ------------------------------------------------------------------
    # Section 2 — Area-wise Observations
    # ------------------------------------------------------------------

    def _get_images_for_area(self, observation: Observation) -> list[str]:
        """
        Finds images whose nearby_text contains the observation's area name.
        Used to place supporting images under the correct observation card.

        Args:
            observation: The observation to find images for.

        Returns:
            List of base64 data URIs for matching images.
        """
        area_lower = observation.area.lower()
        matched_uris: list[str] = []

        for img in self.image_result.images:
            if area_lower in img.nearby_text.lower():
                uri = _encode_image(img.file_path)
                if uri:
                    matched_uris.append(uri)

        return matched_uris

    def _observation_card(self, obs: Observation) -> str:
        """
        Renders a single observation as an HTML card with all available
        fields and any matching supporting images.

        Args:
            obs: A single validated observation.

        Returns:
            HTML string for the observation card.
        """
        key = f"{obs.area.lower()}|{obs.issue_type.value}"
        assessment = self._severity_map.get(key)
        severity_badge = _severity_badge(assessment.severity) if assessment else ""

        # Build field rows — only include fields that have data
        fields: list[tuple[str, str]] = [
            ("Floor",            _na(obs.floor)),
            ("Issue Type",       _esc(obs.issue_type.value.replace("_", " ").title())),
            ("Description",      _na(obs.description)),
            ("Probable Cause",   _na(obs.probable_cause)),
            ("Impact",           _na(obs.impact)),
            ("Leakage Timing",   _esc(obs.leakage_timing.value.replace("_", " ").title())),
            ("Recommended Action", _na(obs.recommended_action)),
        ]

        rows = "".join(
            f"""
            <tr>
                <td class="field-label">{label}</td>
                <td>{value}</td>
            </tr>"""
            for label, value in fields
        )

        # Supporting images
        image_uris = self._get_images_for_area(obs)
        images_html = ""
        if image_uris:
            img_tags = "".join(
                f'<img src="{uri}" alt="Supporting image for {_esc(obs.area)}" '
                f'class="observation-image">'
                for uri in image_uris
            )
            images_html = f'<div class="image-row">{img_tags}</div>'
        else:
            images_html = '<p class="no-image">Image Not Available</p>'

        return f"""
        <div class="observation-card">
            <div class="card-header">
                <span class="area-name">{_esc(obs.area)}</span>
                {severity_badge}
            </div>
            <table class="field-table">{rows}</table>
            <div class="image-section">
                <p class="image-label">Supporting Evidence</p>
                {images_html}
            </div>
        </div>
        """

    def _section_area_observations(self) -> str:
        """Builds Section 2: one card per observation."""
        observations = self.observation_result.observations

        if not observations:
            return "<p>No observations available.</p>"

        cards = "".join(
            self._observation_card(obs) for obs in observations
        )
        return f'<div class="observations-grid">{cards}</div>'

    # ------------------------------------------------------------------
    # Section 3 — Probable Root Cause
    # ------------------------------------------------------------------

    def _section_root_cause(self) -> str:
        """
        Builds Section 3: groups observations by their probable_cause
        and presents them as grouped findings.

        Observations without a stated cause are listed separately
        under 'Cause Not Stated in Report'.
        """
        observations = self.observation_result.observations

        if not observations:
            return "<p>No root cause data available.</p>"

        # Group areas by their probable cause
        cause_groups: dict[str, list[str]] = {}
        no_cause: list[str] = []

        for obs in observations:
            if obs.probable_cause and obs.probable_cause.strip():
                cause = obs.probable_cause.strip()
                cause_groups.setdefault(cause, []).append(obs.area)
            else:
                no_cause.append(obs.area)

        if not cause_groups and not no_cause:
            return "<p>No root cause information available.</p>"

        items = ""
        for cause, areas in cause_groups.items():
            area_tags = ", ".join(
                f'<span class="area-tag">{_esc(a)}</span>' for a in areas
            )
            items += f"""
            <div class="cause-block">
                <p class="cause-text">📌 {_esc(cause)}</p>
                <p class="cause-areas">Affected areas: {area_tags}</p>
            </div>
            """

        if no_cause:
            area_tags = ", ".join(
                f'<span class="area-tag">{_esc(a)}</span>' for a in no_cause
            )
            items += f"""
            <div class="cause-block cause-unknown">
                <p class="cause-text">📌 Cause not stated in source documents.</p>
                <p class="cause-areas">Affected areas: {area_tags}</p>
            </div>
            """

        return f'<div class="cause-list">{items}</div>'

    # ------------------------------------------------------------------
    # Section 4 — Severity Assessment
    # ------------------------------------------------------------------

    def _section_severity(self) -> str:
        """
        Builds Section 4: a table of all assessments ordered by severity
        (Critical first, Low last) with reasoning for each.
        """
        assessments = self.severity_result.assessments

        if not assessments:
            return "<p>No severity data available.</p>"

        # Sort: Critical → High → Medium → Low
        severity_rank = {
            SeverityLevel.CRITICAL: 0,
            SeverityLevel.HIGH: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.LOW: 3,
        }
        sorted_assessments = sorted(
            assessments,
            key=lambda a: severity_rank.get(a.severity, 99),
        )

        rows = "".join(
            f"""
            <tr>
                <td><strong>{_esc(a.area)}</strong></td>
                <td>{_esc(a.issue_type.value.replace("_", " ").title())}</td>
                <td>{_severity_badge(a.severity)}</td>
                <td class="reasoning-cell">{_esc(a.reasoning)}</td>
            </tr>"""
            for a in sorted_assessments
        )

        return f"""
        <table class="severity-table">
            <thead>
                <tr>
                    <th>Area</th>
                    <th>Issue</th>
                    <th>Severity</th>
                    <th>Reasoning</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """

    # ------------------------------------------------------------------
    # Section 5 — Recommended Actions
    # ------------------------------------------------------------------

    def _section_actions(self) -> str:
        """
        Builds Section 5: groups recommended actions by treatment type
        and lists the affected areas under each.

        Observations without a recommended action are listed separately.
        """
        observations = self.observation_result.observations

        # Group areas by recommended action
        action_groups: dict[str, list[str]] = {}
        no_action: list[str] = []

        for obs in observations:
            if obs.recommended_action and obs.recommended_action.strip():
                action = obs.recommended_action.strip()
                action_groups.setdefault(action, []).append(obs.area)
            else:
                no_action.append(obs.area)

        if not action_groups and not no_action:
            return "<p>No recommended actions available.</p>"

        items = ""
        for action, areas in action_groups.items():
            area_tags = ", ".join(
                f'<span class="area-tag">{_esc(a)}</span>' for a in areas
            )
            items += f"""
            <div class="action-block">
                <p class="action-text">🔧 {_esc(action)}</p>
                <p class="action-areas">Applies to: {area_tags}</p>
            </div>
            """

        if no_action:
            area_tags = ", ".join(
                f'<span class="area-tag">{_esc(a)}</span>' for a in no_action
            )
            items += f"""
            <div class="action-block action-unknown">
                <p class="action-text">🔧 No specific action recommended in source documents.</p>
                <p class="action-areas">Applies to: {area_tags}</p>
            </div>
            """

        return f'<div class="action-list">{items}</div>'

    # ------------------------------------------------------------------
    # Section 6 — Additional Notes (Conflicts)
    # ------------------------------------------------------------------

    def _section_additional_notes(self) -> str:
        """
        Builds Section 6: lists all detected conflicts between inspection
        and thermal data with their severity and specific explanation.
        """
        if not self.conflict_result.has_conflicts:
            return (
                "<p>No conflicting information was detected between the "
                "inspection report and thermal imaging data.</p>"
            )

        items = ""
        for conflict in self.conflict_result.conflicts:
            style = CONFLICT_SEVERITY_STYLE.get(
                conflict.severity,
                CONFLICT_SEVERITY_STYLE[ConflictSeverity.LOW],
            )
            conflict_label = conflict.conflict_type.value.replace("_", " ").title()

            inspection_str = (
                ", ".join(conflict.inspection_issues)
                if conflict.inspection_issues
                else "No issue found"
            )
            thermal_str = (
                ", ".join(conflict.thermal_issues)
                if conflict.thermal_issues
                else "No issue found"
            )

            items += f"""
            <div class="conflict-block" style="border-left:4px solid {style['border']};
                 background:{style['background']}">
                <div class="conflict-header">
                    <span class="conflict-area">{_esc(conflict.area)}</span>
                    <span class="conflict-type-badge"
                          style="color:{style['color']}">{conflict_label}</span>
                    <span class="conflict-severity"
                          style="color:{style['color']}">{style['label']}</span>
                </div>
                <table class="conflict-table">
                    <tr>
                        <td class="field-label">Inspection Finding</td>
                        <td>{_esc(inspection_str)}</td>
                    </tr>
                    <tr>
                        <td class="field-label">Thermal Finding</td>
                        <td>{_esc(thermal_str)}</td>
                    </tr>
                    <tr>
                        <td class="field-label">Explanation</td>
                        <td>{_esc(conflict.explanation)}</td>
                    </tr>
                </table>
            </div>
            """

        return f'<div class="conflict-list">{items}</div>'

    # ------------------------------------------------------------------
    # Section 7 — Missing or Unclear Information
    # ------------------------------------------------------------------

    def _section_missing_info(self) -> str:
        """
        Builds Section 7: scans all observations for None fields and
        reports exactly what information was absent from the source documents.

        This directly fulfils the assignment requirement to explicitly
        flag missing data rather than silently omitting it.
        """
        missing_entries: list[tuple[str, str]] = []

        for obs in self.observation_result.observations:
            if obs.floor is None:
                missing_entries.append((obs.area, "Floor / level not stated"))
            if obs.probable_cause is None:
                missing_entries.append((obs.area, "Probable cause not stated"))
            if obs.impact is None:
                missing_entries.append((obs.area, "Impact description not stated"))
            if obs.recommended_action is None:
                missing_entries.append((obs.area, "Recommended action not stated"))

        if not missing_entries:
            return (
                "<p>All expected fields were present in the source documents. "
                "No information is missing.</p>"
            )

        rows = "".join(
            f"""
            <tr>
                <td>{_esc(area)}</td>
                <td class="missing-cell">Not Available — {_esc(detail)}</td>
            </tr>"""
            for area, detail in missing_entries
        )

        return f"""
        <p>The following information was not available in the source documents:</p>
        <table class="missing-table">
            <thead>
                <tr><th>Area</th><th>Missing Information</th></tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------

    def _get_styles(self) -> str:
        """Returns the complete CSS for the report."""
        return """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 14px;
            color: #212121;
            background: #f5f5f5;
            padding: 0;
        }
        .report-wrapper { max-width: 960px; margin: 0 auto; background: #fff; }

        /* Header */
        .report-header {
            background: #1a1a1a;
            color: #fff;
            padding: 36px 40px 28px;
            border-bottom: 4px solid #f9a825;
        }
        .report-title {
            font-size: 28px;
            font-weight: 700;
            letter-spacing: 1px;
            color: #f9a825;
            text-decoration: underline;
        }
        .report-meta { margin-top: 16px; display: flex; gap: 40px; flex-wrap: wrap; }
        .meta-item { font-size: 13px; color: #bdbdbd; }
        .meta-item strong { color: #f9a825; display: block; font-size: 11px;
                            text-transform: uppercase; letter-spacing: 0.5px; }

        /* Sections */
        .section { padding: 32px 40px; border-bottom: 1px solid #e0e0e0; }
        .section:last-child { border-bottom: none; }
        .section-title {
            font-size: 18px;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 20px;
            padding-bottom: 8px;
            border-bottom: 2px solid #f9a825;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .section-number {
            background: #f9a825;
            color: #1a1a1a;
            font-size: 12px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 3px;
        }

        /* Issue tags */
        .issue-tag-list { list-style: none; display: flex; flex-wrap: wrap;
                          gap: 8px; margin: 12px 0; }
        .issue-tag {
            background: #e3f2fd;
            color: #1565c0;
            border: 1px solid #90caf9;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }

        /* Summary table */
        .summary-table { border-collapse: collapse; margin-top: 12px; }
        .summary-table th, .summary-table td {
            padding: 8px 16px;
            border: 1px solid #e0e0e0;
            text-align: left;
            font-size: 13px;
        }
        .summary-table th { background: #f5f5f5; font-weight: 600; }

        /* Severity badge */
        .severity-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.3px;
        }

        /* Observation cards */
        .observations-grid { display: flex; flex-direction: column; gap: 20px; }
        .observation-card {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            overflow: hidden;
        }
        .card-header {
            background: #1a1a1a;
            color: #fff;
            padding: 12px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .area-name { font-size: 15px; font-weight: 700; color: #f9a825; }
        .field-table { width: 100%; border-collapse: collapse; }
        .field-table tr { border-bottom: 1px solid #f0f0f0; }
        .field-table tr:last-child { border-bottom: none; }
        .field-label {
            width: 200px;
            padding: 8px 16px;
            background: #fafafa;
            font-weight: 600;
            font-size: 12px;
            color: #616161;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            vertical-align: top;
        }
        .field-table td:not(.field-label) {
            padding: 8px 16px;
            font-size: 13px;
            color: #424242;
        }

        /* Images */
        .image-section { background: #fafafa; padding: 12px 16px;
                          border-top: 1px solid #e0e0e0; }
        .image-label { font-size: 11px; font-weight: 700; color: #9e9e9e;
                       text-transform: uppercase; letter-spacing: 0.5px;
                       margin-bottom: 8px; }
        .image-row { display: flex; flex-wrap: wrap; gap: 12px; }
        .observation-image { max-width: 280px; max-height: 200px;
                              border-radius: 4px; border: 1px solid #e0e0e0;
                              object-fit: cover; }
        .no-image { font-size: 12px; color: #9e9e9e; font-style: italic; }

        /* Root cause & actions */
        .cause-list, .action-list { display: flex; flex-direction: column; gap: 12px; }
        .cause-block, .action-block {
            background: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-left: 4px solid #f9a825;
            border-radius: 4px;
            padding: 12px 16px;
        }
        .cause-unknown, .action-unknown { border-left-color: #9e9e9e; }
        .cause-text, .action-text { font-size: 13px; font-weight: 600;
                                     color: #212121; margin-bottom: 6px; }
        .cause-areas, .action-areas { font-size: 12px; color: #616161; }
        .area-tag {
            display: inline-block;
            background: #eeeeee;
            color: #424242;
            padding: 1px 8px;
            border-radius: 10px;
            font-size: 11px;
            margin: 2px;
        }

        /* Severity table */
        .severity-table { width: 100%; border-collapse: collapse; }
        .severity-table th, .severity-table td {
            padding: 10px 14px;
            border: 1px solid #e0e0e0;
            font-size: 13px;
            vertical-align: top;
        }
        .severity-table th { background: #1a1a1a; color: #f9a825;
                              font-weight: 600; text-align: left; }
        .severity-table tr:nth-child(even) { background: #fafafa; }
        .reasoning-cell { color: #616161; font-size: 12px; }

        /* Conflicts */
        .conflict-list { display: flex; flex-direction: column; gap: 16px; }
        .conflict-block { padding: 14px 18px; border-radius: 4px; }
        .conflict-header { display: flex; align-items: center; gap: 12px;
                           margin-bottom: 10px; flex-wrap: wrap; }
        .conflict-area { font-size: 15px; font-weight: 700; color: #212121; }
        .conflict-type-badge, .conflict-severity {
            font-size: 11px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.4px;
        }
        .conflict-table { width: 100%; border-collapse: collapse; }
        .conflict-table td { padding: 6px 12px; border: 1px solid rgba(0,0,0,0.08);
                              font-size: 13px; }

        /* Missing info */
        .missing-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        .missing-table th, .missing-table td {
            padding: 8px 14px;
            border: 1px solid #e0e0e0;
            font-size: 13px;
        }
        .missing-table th { background: #f5f5f5; font-weight: 600; }
        .missing-cell { color: #9e9e9e; font-style: italic; }

        /* Utilities */
        .not-available { color: #9e9e9e; font-style: italic; }
        h3 { font-size: 15px; font-weight: 600; color: #424242;
              margin: 16px 0 8px; }
        p { line-height: 1.6; color: #424242; }

        /* Footer */
        .report-footer {
            background: #1a1a1a;
            color: #9e9e9e;
            text-align: center;
            padding: 20px;
            font-size: 12px;
        }
        .report-footer strong { color: #f9a825; }

        @media print {
            body { background: #fff; }
            .report-wrapper { max-width: 100%; }
        }
        """

    # ------------------------------------------------------------------
    # HTML assembly
    # ------------------------------------------------------------------

    def _build_section(self, number: str, title: str, content: str) -> str:
        """
        Wraps section content in the standard DDR section HTML structure.

        Args:
            number:  Section number string (e.g. "1").
            title:   Section title text.
            content: Pre-rendered HTML content for the section body.

        Returns:
            Complete HTML for the section.
        """
        return f"""
        <section class="section">
            <h2 class="section-title">
                <span class="section-number">{number}</span>
                {_esc(title)}
            </h2>
            {content}
        </section>
        """

    def _build_header(self) -> str:
        """Builds the report header with property and inspection details."""
        return f"""
        <header class="report-header">
            <div class="report-title">Detailed Diagnostic Report</div>
            <div class="report-meta">
                <div class="meta-item">
                    <strong>Property Address</strong>
                    {_esc(self.property_address)}
                </div>
                <div class="meta-item">
                    <strong>Inspected By</strong>
                    {_esc(self.inspected_by)}
                </div>
                <div class="meta-item">
                    <strong>Inspection Date</strong>
                    {_esc(self.inspection_date)}
                </div>
                <div class="meta-item">
                    <strong>Report Generated</strong>
                    {datetime.now().strftime("%B %d, %Y at %H:%M")}
                </div>
            </div>
        </header>
        """

    def _build_html(self) -> str:
        """
        Assembles the complete HTML document from all sections.

        Returns:
            Complete, self-contained HTML string.
        """
        sections = "".join([
            self._build_section("1", "Property Issue Summary",
                                self._section_issue_summary()),
            self._build_section("2", "Area-wise Observations",
                                self._section_area_observations()),
            self._build_section("3", "Probable Root Cause",
                                self._section_root_cause()),
            self._build_section("4", "Severity Assessment",
                                self._section_severity()),
            self._build_section("5", "Recommended Actions",
                                self._section_actions()),
            self._build_section("6", "Additional Notes",
                                self._section_additional_notes()),
            self._build_section("7", "Missing or Unclear Information",
                                self._section_missing_info()),
        ])

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DDR — {_esc(self.property_address)}</title>
    <style>{self._get_styles()}</style>
</head>
<body>
    <div class="report-wrapper">
        {self._build_header()}
        <main>{sections}</main>
        <footer class="report-footer">
            <p>Generated by <strong>UrbanRoof DDR System</strong> &nbsp;|&nbsp;
               {datetime.now().strftime("%Y")} UrbanRoof Private Limited</p>
        </footer>
    </div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self) -> DDRReport:
        """
        Generates the complete DDR report and saves it to disk.

        Creates the output directory if it does not exist.
        The HTML file is named with a timestamp to avoid overwriting
        previous reports.

        Returns:
            DDRReport containing the HTML content, output path,
            and generation metadata.
        """
        logger.info(
            "Generating DDR report: %d observations, %d conflicts.",
            self.observation_result.count,
            self.conflict_result.count,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"DDR_Report_{timestamp}.html"

        html = self._build_html()

        output_path.write_text(html, encoding="utf-8")

        report = DDRReport(
            html=html,
            output_path=output_path,
            generated_at=datetime.now(),
            observation_count=self.observation_result.count,
            conflict_count=self.conflict_result.count,
        )

        logger.info("DDR report saved: '%s'", output_path)

        return report

    def __repr__(self) -> str:
        return (
            f"DDRGenerator("
            f"observations={self.observation_result.count}, "
            f"conflicts={self.conflict_result.count}, "
            f"images={self.image_result.unique_count})"
        )