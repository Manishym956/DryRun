from typing import Any, Dict
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


def build_pdf(report: Dict[str, Any], candidate_name: str, role: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 1 * inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, y, "AI Interview Report")
    y -= 0.3 * inch

    c.setFont("Helvetica", 11)
    c.drawString(1 * inch, y, f"Candidate: {candidate_name}")
    y -= 0.2 * inch
    c.drawString(1 * inch, y, f"Role: {role}")
    y -= 0.3 * inch

    def write_section(title: str, body: str, score: Any | None = None):
        nonlocal y
        if y < 1.2 * inch:
            c.showPage()
            y = height - 1 * inch
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1 * inch, y, title)
        y -= 0.22 * inch
        c.setFont("Helvetica", 10)
        if score is not None:
            c.drawString(1 * inch, y, f"Score: {score}")
            y -= 0.2 * inch
        for line in body.split("\n"):
            for chunk in [line[i:i+100] for i in range(0, len(line), 100)]:
                if y < 1.0 * inch:
                    c.showPage()
                    y = height - 1 * inch
                    c.setFont("Helvetica", 10)
                c.drawString(1 * inch, y, chunk)
                y -= 0.18 * inch
        y -= 0.12 * inch

    # New structure: scores is a list of {dimension, score, justification}
    def get_score(dim: str) -> Dict[str, Any]:
        for s in report.get("scores", []) or []:
            if s.get("dimension") == dim:
                return s
        return {"dimension": dim, "score": "-", "justification": ""}

    tech = get_score("Technical Knowledge")
    beh = get_score("Behavioral Fit")
    comm = get_score("Communication")

    write_section("Technical Knowledge", tech.get("justification", ""), tech.get("score"))
    write_section("Behavioral Fit", beh.get("justification", ""), beh.get("score"))
    write_section("Communication", comm.get("justification", ""), comm.get("score"))

    summary = report.get("overallFeedback", "")
    write_section("Summary", summary)

    c.setFont("Helvetica-Bold", 12)
    if y < 1.0 * inch:
        c.showPage()
        y = height - 1 * inch
    recommendation = report.get("hiringRecommendation", "Maybe")
    c.drawString(1 * inch, y, f"Recommendation: {recommendation}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()
