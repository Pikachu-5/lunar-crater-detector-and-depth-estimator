"""Mission artifact export helpers: PDF, PNG, and CSV."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

import cv2
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode a numpy image as PNG bytes for download/export.

    Args:
        image: Grayscale or RGB image.

    Returns:
        PNG byte payload.
    """

    if image.ndim == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        bgr = image

    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("PNG encoding failed")
    return bytes(buf)


def crater_rows_to_csv(rows: list[dict[str, Any]]) -> bytes:
    """Convert crater analytics rows into CSV bytes.

    Args:
        rows: Table-like crater records.

    Returns:
        UTF-8 CSV bytes.
    """

    if len(rows) == 0:
        df = pd.DataFrame(columns=["crater_id", "depth_m", "safety_score", "zone"])
    else:
        df = pd.DataFrame(rows)

    return df.to_csv(index=False).encode("utf-8")


def _table_from_rows(rows: list[dict[str, Any]], max_rows: int = 20) -> Table:
    """Create compact ReportLab table from crater records.

    Args:
        rows: Crater rows with core fields.
        max_rows: Maximum rows included for readability.

    Returns:
        Styled ReportLab table.
    """

    header = [
        "Crater ID",
        "Depth (m)",
        "Score",
        "Zone",
        "Diameter (px)",
        "Shadow (px)",
    ]

    data = [header]
    for row in rows[:max_rows]:
        data.append(
            [
                row.get("crater_id", "-"),
                row.get("depth_m", "-"),
                row.get("safety_score", "-"),
                row.get("zone", "-"),
                row.get("diameter_px", "-"),
                row.get("shadow_length_px", "-"),
            ]
        )

    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f1a2d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#00ff9f")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#00ff9f")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Courier"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def build_mission_pdf(
    mission_id: str,
    crater_rows: list[dict[str, Any]],
    hazard_map: np.ndarray | None,
    recommended_coordinates: tuple[int, int] | None,
    overall_score: float,
) -> bytes:
    """Generate NASA-style PDF mission report in-memory.

    Reporting concept:
    Mission control workflows require reproducible artifacts that preserve both
    quantitative tables and visual hazard context for post-analysis or review.

    Args:
        mission_id: Mission identifier string.
        crater_rows: Final crater analytics rows.
        hazard_map: Optional RGB hazard map image.
        recommended_coordinates: Recommended landing coordinates.
        overall_score: Overall terrain safety score.

    Returns:
        PDF bytes for download.
    """

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=f"{mission_id} Mission Report",
        leftMargin=1.2 * cm,
        rightMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Courier-Bold",
        fontSize=18,
        textColor=colors.HexColor("#0f4f2f"),
    )
    mono = ParagraphStyle(
        "Mono",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=9,
        leading=12,
    )

    story = []
    story.append(Paragraph(f"NASA Mission Control Report - {mission_id}", title_style))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(f"Timestamp: {datetime.utcnow().isoformat()}Z", mono))
    story.append(Paragraph("Telemetry: NOMINAL", mono))
    story.append(Paragraph(f"Overall Terrain Safety Score: {overall_score:.2f}/100", mono))

    if recommended_coordinates is not None:
        story.append(
            Paragraph(
                f"Recommended Landing Coordinates: ({recommended_coordinates[0]}, {recommended_coordinates[1]})",
                mono,
            )
        )

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Crater Summary", styles["Heading3"]))
    story.append(_table_from_rows(crater_rows))
    story.append(Spacer(1, 0.35 * cm))

    if hazard_map is not None:
        png_bytes = image_to_png_bytes(hazard_map)
        img_buf = BytesIO(png_bytes)
        story.append(Paragraph("Final Hazard Map", styles["Heading3"]))
        story.append(Image(img_buf, width=15.6 * cm, height=15.6 * cm * (hazard_map.shape[0] / hazard_map.shape[1])))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
