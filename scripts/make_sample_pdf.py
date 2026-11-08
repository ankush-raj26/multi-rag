"""Generate a multi-page sample PDF for the RAG demo.

Run:  uv run python scripts/make_sample_pdf.py
Creates data/documents/acme_handbook.pdf — a fictional 10-page employee handbook,
one clear topic per page, so you can ask a question, see the cited page number,
and jump straight to that page in the UI.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "data" / "documents" / "acme_handbook.pdf"

# (page title, body) — each page holds distinct, checkable facts.
PAGES = [
    ("Acme Corp — Employee Handbook (2026)",
     "This handbook explains how we work at Acme Corp. Use the sidebar page selector "
     "to jump to any section. Each page covers one topic: leave, expenses, remote work, "
     "conduct, security, onboarding, benefits, IT support and key contacts."),
    ("1. Leave Policy",
     "Full-time employees receive 24 days of paid annual leave per year, accruing at 2 days "
     "per month. Sick leave is 12 days per year. Unused annual leave of up to 10 days may be "
     "carried over to the next year. Leave requests must be filed at least 5 working days in advance."),
    ("2. Expense Policy",
     "Business expenses up to 50 USD may be claimed without a receipt; above 50 USD a receipt is "
     "required. Travel must be booked through the Acme travel portal. Reimbursements are paid within "
     "14 days of approval. Client entertainment over 200 USD needs manager pre-approval."),
    ("3. Remote Work",
     "Employees may work remotely up to 3 days per week. Fully remote arrangements require director "
     "approval. A home-office stipend of 300 USD per year is available for equipment. Core collaboration "
     "hours are 11:00 to 16:00 in your local time zone."),
    ("4. Code of Conduct",
     "Treat colleagues, customers and partners with respect. Harassment and discrimination are not "
     "tolerated. Conflicts of interest must be disclosed to your manager. Report violations to the "
     "ethics line at ethics@acme.example within 48 hours of becoming aware."),
    ("5. Information Security",
     "Use the company password manager and enable multi-factor authentication on all accounts. "
     "Never share credentials. Company data must stay on approved, encrypted devices. Report any "
     "suspected phishing or breach to security@acme.example immediately; do not click suspicious links."),
    ("6. Onboarding",
     "New hires complete orientation in their first week: IT setup on day 1, policy review on day 2, "
     "and a team introduction on day 3. A 30-60-90 day plan is agreed with your manager. Your buddy "
     "answers day-to-day questions during the first month."),
    ("7. Benefits",
     "Acme provides private health insurance, a retirement plan with 5% employer match, and an annual "
     "learning budget of 1000 USD. Parental leave is 16 weeks for the primary caregiver and 4 weeks for "
     "the secondary caregiver. Gym membership is subsidised at 50%."),
    ("8. IT Support",
     "Raise IT tickets through the helpdesk portal or email helpdesk@acme.example. Standard requests are "
     "resolved within 2 business days; urgent incidents within 4 hours. Laptops are refreshed every 3 years. "
     "Lost or stolen devices must be reported within 1 hour so they can be remotely wiped."),
    ("9. Key Contacts",
     "HR: hr@acme.example. Payroll runs on the 25th of each month. Facilities: facilities@acme.example. "
     "The employee portal is at portal.acme.example. For anything not covered here, ask your manager or "
     "the people team."),
]


def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, title="Acme Corp Employee Handbook")
    flow = []
    for i, (title, body) in enumerate(PAGES):
        flow.append(Paragraph(title, styles["Title"] if i == 0 else styles["Heading1"]))
        flow.append(Spacer(1, 0.6 * cm))
        flow.append(Paragraph(body, styles["BodyText"]))
        flow.append(Spacer(1, 0.6 * cm))
        flow.append(Paragraph(f"(Acme Corp Employee Handbook — page {i + 1})", styles["Italic"]))
        if i != len(PAGES) - 1:
            flow.append(PageBreak())          # one topic per page
    doc.build(flow)
    print(f"wrote {OUT} ({len(PAGES)} pages)")


if __name__ == "__main__":
    build()
