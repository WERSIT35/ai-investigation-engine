"""Generates data/synthetic/vendor-sla-paybridge.pdf (the PDF sample of the synthetic corpus). Needs reportlab."""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "vendor-sla-paybridge.pdf"
TEXT = [
    ("h1", "PayBridge card acquiring — service level agreement"),
    ("p", "This agreement covers the card acquiring connection between PayBridge and the payments platform. "
          "Effective from 2026-01-01."),
    ("h2", "Availability"),
    ("p", "PayBridge guarantees 99.95% monthly availability of its authorisation API. Planned maintenance is "
          "announced 5 business days in advance and is excluded."),
    ("h2", "Service credits"),
    ("p", "If availability falls below 99.95%, PayBridge credits 5% of the monthly fee; below 99.5%, 15%. "
          "Credits must be claimed within 30 days."),
    ("h2", "Incident communication"),
    ("p", "PayBridge notifies the platform's SRE team within 15 minutes of detecting an incident that affects "
          "authorisations. Delays caused inside the platform (for example in ledger-service) are not covered."),
]

styles = getSampleStyleSheet()
doc = SimpleDocTemplate(str(OUT), pagesize=A4, title="PayBridge card acquiring — service level agreement")
story = []
for kind, text in TEXT:
    story.append(Paragraph(text, styles["Heading1" if kind == "h1" else "Heading2" if kind == "h2" else "BodyText"]))
    story.append(Spacer(1, 8))
doc.build(story)
print("wrote", OUT)
