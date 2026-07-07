"""
One-off helper used to generate data/sample_inputs/hr_policy.pdf, a small
multi-page fictional HR policy document used to demonstrate PDF ingestion
and page-level citations. Not part of the runtime app -- run manually if
you need to regenerate the sample PDF:

    python data/sample_inputs/_generate_hr_policy_pdf.py
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

OUT_PATH = Path(__file__).parent / "hr_policy.pdf"

styles = getSampleStyleSheet()
doc = SimpleDocTemplate(str(OUT_PATH), pagesize=letter)
story = []

# --- Page 1: Title + Leave Policy intro ---
story.append(Paragraph("Northbridge Softworks — HR Policy Handbook", styles["Title"]))
story.append(Spacer(1, 12))
story.append(Paragraph("Section 1: Annual Leave", styles["Heading1"]))
story.append(Paragraph(
    "All full-time employees are entitled to paid annual leave, accrued monthly "
    "from their date of joining. Leave requests must be submitted through the HR "
    "portal at least 5 working days in advance for planned leave.", styles["Normal"]
))
story.append(PageBreak())

# --- Page 2 ---
story.append(Paragraph("Section 1: Annual Leave (continued)", styles["Heading1"]))
story.append(Paragraph(
    "Full-time confirmed employees are allowed 18 annual leaves per year. "
    "Unused annual leave may be carried forward up to a maximum of 6 days into "
    "the following calendar year; any balance beyond that is forfeited.",
    styles["Normal"]
))
story.append(Spacer(1, 12))
story.append(Paragraph("Section 2: Sick Leave", styles["Heading1"]))
story.append(Paragraph(
    "Employees are entitled to 10 paid sick leaves per year. A medical "
    "certificate is required for sick leave exceeding 2 consecutive days.",
    styles["Normal"]
))
story.append(PageBreak())

# --- Page 3 ---
story.append(Paragraph("Section 3: Parental Leave", styles["Heading1"]))
story.append(Paragraph(
    "Primary caregivers are entitled to 90 calendar days of paid parental leave. "
    "Secondary caregivers are entitled to 14 calendar days of paid parental leave, "
    "to be taken within 6 months of the child's birth or adoption.",
    styles["Normal"]
))
story.append(Spacer(1, 12))
story.append(Paragraph("Section 4: Probation Period", styles["Heading1"]))
story.append(Paragraph(
    "All new hires undergo a probation period of 3 months from their date of "
    "joining, during which either party may terminate employment with 2 weeks' "
    "written notice.", styles["Normal"]
))
story.append(PageBreak())

# --- Page 4 ---
story.append(Paragraph("Section 4: Probation Period (continued)", styles["Heading1"]))
story.append(Paragraph(
    "Confirmation of employment after probation requires a satisfactory "
    "performance review conducted jointly by the employee's manager and HR. "
    "Probation may be extended once, by a maximum of one additional month, if "
    "performance requires further evaluation.", styles["Normal"]
))
story.append(Spacer(1, 12))
story.append(Paragraph("Section 5: Code of Conduct", styles["Heading1"]))
story.append(Paragraph(
    "Employees are expected to maintain professional conduct at all times, "
    "avoid conflicts of interest, and report any violations of this handbook "
    "to People Operations at hr@northbridgesoftworks.example.", styles["Normal"]
))

doc.build(story)
print(f"Generated: {OUT_PATH}")
