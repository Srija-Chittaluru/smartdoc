"""Generate six sample company PDFs in documents/ so the project runs out of the box.

These are invented policies for a fictional company (Northwind Technologies).
Replace them with your own PDFs whenever you like - nothing in the pipeline is
specific to these files.

Run with:  python make_sample_pdfs.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from backend.config import DOCUMENTS_DIR

# Each document is a title plus a list of (section heading, body text) pairs.
DOCUMENTS = {
    "employee_handbook.pdf": (
        "Northwind Technologies - Employee Handbook",
        [
            (
                "1. Working Hours",
                "Standard working hours are 9:30 AM to 6:30 PM, Monday to Friday, with a "
                "one hour lunch break. Employees are expected to be reachable during the "
                "core collaboration window of 11:00 AM to 4:00 PM in their local time zone. "
                "Flexible start times between 8:00 AM and 10:30 AM may be arranged with a "
                "reporting manager, provided the core window is covered.",
            ),
            (
                "2. Annual Leave",
                "Full-time employees are entitled to 21 days of paid annual leave per "
                "calendar year. Leave accrues at 1.75 days per completed month of service. "
                "New joiners may take accrued leave during their probation period, but not "
                "more than 5 days. A maximum of 8 unused leave days may be carried forward "
                "into the following calendar year; anything beyond 8 days lapses on "
                "31 December. Leave requests must be submitted at least 7 calendar days in "
                "advance through the HR portal, except in emergencies.",
            ),
            (
                "3. Sick Leave",
                "Employees receive 12 days of paid sick leave per calendar year, separate "
                "from annual leave. A medical certificate is required for any absence of "
                "three consecutive days or longer. Sick leave does not carry forward.",
            ),
            (
                "4. Parental Leave",
                "Birthing parents are entitled to 26 weeks of paid maternity leave. "
                "Non-birthing parents are entitled to 6 weeks of paid paternity leave, "
                "which must be taken within 6 months of the child's birth or adoption. "
                "Both may be combined with annual leave.",
            ),
            (
                "5. Probation Period",
                "The standard probation period is 90 days from the date of joining. "
                "During probation, either party may terminate employment with 15 days "
                "notice. After confirmation, the notice period is 60 days.",
            ),
            (
                "6. Dress Code",
                "Northwind operates a smart casual dress code. Formal business attire is "
                "expected for client meetings and external presentations. There is no dress "
                "code on days when an employee is working remotely.",
            ),
            (
                "7. Code of Conduct",
                "All employees must treat colleagues, clients and vendors with respect. "
                "Harassment, discrimination and retaliation are grounds for immediate "
                "termination. Concerns may be raised anonymously through the ethics hotline "
                "at ethics@northwind.example, which is monitored by an external firm.",
            ),
        ],
    ),
    "it_security_policy.pdf": (
        "Northwind Technologies - IT Security Policy",
        [
            (
                "1. Purpose and Scope",
                "This policy applies to all employees, contractors and interns who access "
                "Northwind systems or data, on company-issued or personal devices.",
            ),
            (
                "2. Password Requirements",
                "Passwords must be at least 14 characters long and include upper case "
                "letters, lower case letters, numbers and at least one symbol. Passwords "
                "must be rotated every 90 days and the previous five passwords may not be "
                "reused. Sharing passwords with anyone, including IT staff, is prohibited. "
                "All employees must use the company password manager, 1Password, to store "
                "work credentials.",
            ),
            (
                "3. Multi-Factor Authentication",
                "Multi-factor authentication is mandatory for email, the VPN, the code "
                "repository and all cloud consoles. Hardware security keys are issued to "
                "engineering and finance staff. SMS-based codes are not permitted as a "
                "second factor because they are vulnerable to SIM swapping.",
            ),
            (
                "4. Remote Access and VPN",
                "Access to internal systems from outside the office requires the Northwind "
                "VPN. Public Wi-Fi may be used only with the VPN active. VPN sessions "
                "automatically disconnect after 12 hours and after 30 minutes of inactivity.",
            ),
            (
                "5. Device Security",
                "All laptops must have full-disk encryption enabled and the screen must "
                "lock after 5 minutes of inactivity. Company data may not be stored on "
                "personal USB drives. Lost or stolen devices must be reported to "
                "security@northwind.example within 2 hours of discovery.",
            ),
            (
                "6. Data Classification",
                "Data is classified as Public, Internal, Confidential or Restricted. "
                "Customer personal data and financial records are always Restricted. "
                "Restricted data may not be copied to personal devices, pasted into "
                "third-party AI tools, or emailed outside the company without written "
                "approval from the Data Protection Officer.",
            ),
            (
                "7. Incident Reporting",
                "Suspected phishing emails must be forwarded to phishing@northwind.example "
                "and then deleted. Any suspected breach must be reported immediately; the "
                "security team aims to acknowledge reports within 1 hour during business "
                "hours and within 4 hours outside them.",
            ),
        ],
    ),
    "expense_policy.pdf": (
        "Northwind Technologies - Travel and Expense Policy",
        [
            (
                "1. General Principles",
                "Employees should spend company money as carefully as they would spend "
                "their own. All expenses must be business related, reasonable, and "
                "supported by an itemised receipt.",
            ),
            (
                "2. Submission Deadlines",
                "Expense claims must be submitted within 30 days of the expense being "
                "incurred. Claims submitted after 60 days will not be reimbursed except "
                "with written approval from the Finance Director. Reimbursements are paid "
                "with the next payroll cycle following approval.",
            ),
            (
                "3. Approval Thresholds",
                "Expenses up to 5,000 INR are approved by the reporting manager. Expenses "
                "between 5,000 and 50,000 INR require department head approval. Anything "
                "above 50,000 INR requires Finance Director approval, obtained in advance "
                "rather than after the fact.",
            ),
            (
                "4. Air Travel",
                "Economy class is standard for all flights under 6 hours. Premium economy "
                "is permitted for flights over 6 hours. Business class requires prior "
                "approval from the Finance Director. Flights should be booked at least "
                "14 days in advance where the travel date is known.",
            ),
            (
                "5. Accommodation and Meals",
                "Hotel spending is capped at 6,000 INR per night in metro cities and "
                "4,000 INR per night elsewhere. The daily meal allowance is 1,500 INR per "
                "day of travel and does not require individual receipts. Alcohol is not "
                "reimbursable except at pre-approved client entertainment events.",
            ),
            (
                "6. Local Transport",
                "Taxis and ride-hailing services are reimbursable for business travel. "
                "Personal vehicle use is reimbursed at 12 INR per kilometre. Parking "
                "and tolls are reimbursable with receipts. Traffic fines are never "
                "reimbursable.",
            ),
            (
                "7. Non-Reimbursable Items",
                "The following are not reimbursed: personal entertainment, gym "
                "memberships, flight upgrades paid personally, laundry on trips shorter "
                "than four nights, and any expense already covered by a client.",
            ),
        ],
    ),
    "remote_work_policy.pdf": (
        "Northwind Technologies - Remote and Hybrid Work Policy",
        [
            (
                "1. Eligibility",
                "All confirmed employees are eligible for hybrid work after completing "
                "their 90 day probation period. Employees on a performance improvement plan "
                "are not eligible for fully remote work until the plan is closed.",
            ),
            (
                "2. Hybrid Schedule",
                "Employees are expected in the office three days per week, including "
                "Tuesday and Wednesday, which are company-wide collaboration days. The "
                "third day is chosen by each team. Fully remote arrangements require "
                "approval from both the reporting manager and the Head of People.",
            ),
            (
                "3. Home Office Stipend",
                "Northwind provides a one-time home office setup allowance of 25,000 INR, "
                "claimable within the first 90 days of eligibility, covering a desk, chair, "
                "monitor or lighting. In addition, remote employees receive a monthly "
                "internet reimbursement of 1,200 INR against a bill.",
            ),
            (
                "4. Availability Expectations",
                "Remote employees must be online and reachable during the core window of "
                "11:00 AM to 4:00 PM local time. Calendars must be kept current, and "
                "status must be updated in Slack when away for more than an hour.",
            ),
            (
                "5. Working from Another Country",
                "Working from outside the country of employment is permitted for a maximum "
                "of 30 days per calendar year and requires prior written approval from HR, "
                "because of tax and immigration implications. Requests must be submitted at "
                "least 21 days in advance.",
            ),
            (
                "6. Equipment and Support",
                "The IT helpdesk supports remote employees over chat and video between "
                "9:00 AM and 7:00 PM. Replacement hardware is couriered within 3 business "
                "days. Faulty equipment must be returned within 10 days of receiving a "
                "replacement.",
            ),
        ],
    ),
    "onboarding_guide.pdf": (
        "Northwind Technologies - New Hire Onboarding Guide",
        [
            (
                "1. Before Your First Day",
                "You will receive a welcome email with your company account details two "
                "business days before you join. Complete the digital paperwork in the HR "
                "portal and upload your identity and education documents before day one so "
                "that payroll can be set up in your first week.",
            ),
            (
                "2. Day One",
                "Your first day starts at 10:00 AM with an orientation session in the main "
                "meeting room or over video for remote joiners. You will collect your "
                "laptop and access badge from IT, set up multi-factor authentication, and "
                "meet your assigned onboarding buddy.",
            ),
            (
                "3. Your Onboarding Buddy",
                "Every new hire is paired with an onboarding buddy from a different team "
                "for the first 30 days. Your buddy is not your manager and is there for "
                "informal questions. Buddies are asked to check in at least twice a week "
                "during the first month.",
            ),
            (
                "4. First Week Checklist",
                "By the end of week one you should have: completed the security awareness "
                "training, joined your team's daily standup, been added to the relevant "
                "Slack channels and code repositories, submitted your bank details for "
                "payroll, and had a one-to-one with your reporting manager.",
            ),
            (
                "5. First 30, 60 and 90 Days",
                "Your manager will agree goals with you in your first week. The 30 day "
                "review focuses on whether you have the access and context you need. The "
                "60 day review focuses on your first independent piece of work. The 90 day "
                "review is the formal probation confirmation conversation.",
            ),
            (
                "6. Mandatory Training",
                "Three courses must be completed within the first 14 days: security "
                "awareness, data protection and workplace conduct. Each takes about "
                "45 minutes and is available in the learning portal. Completion is tracked "
                "and is a condition of probation confirmation.",
            ),
            (
                "7. Who to Ask",
                "For payroll and leave questions contact people@northwind.example. For "
                "laptop and access issues raise a ticket with the IT helpdesk. For anything "
                "you are not sure about, ask your onboarding buddy first.",
            ),
        ],
    ),
    # This one deliberately uses a different style - ALL CAPS, unnumbered
    # headings - to prove the chunker does not depend on one document format.
    "vendor_procurement_sop.pdf": (
        "VENDOR PROCUREMENT STANDARD OPERATING PROCEDURE",
        [
            (
                "PURPOSE",
                "This SOP governs how Northwind selects, onboards and contracts third-party "
                "vendors, and applies to every department that raises a purchase order.",
            ),
            (
                "VENDOR ONBOARDING",
                "Every new vendor must complete a security questionnaire and provide proof of "
                "insurance before the first purchase order is raised. Onboarding takes "
                "10 business days from receipt of a complete submission.",
            ),
            (
                "PURCHASE ORDER RULES",
                "No work may begin before a signed purchase order exists. Verbal commitments "
                "are not binding on Northwind. Purchase orders above 200,000 INR require two "
                "authorised signatures, one of which must be from Finance.",
            ),
            (
                "PAYMENT TERMS",
                "Standard payment terms are net 45 days from a valid invoice. Early payment "
                "discounts must be approved by Treasury. Northwind does not pay advances "
                "exceeding 20 percent of total contract value.",
            ),
            (
                "VENDOR REVIEW",
                "All active vendors are reviewed annually against delivery quality, security "
                "posture and price competitiveness. Vendors scoring below 3 out of 5 are "
                "placed on a 90 day improvement plan.",
            ),
            (
                "TERMINATION",
                "Contracts may be terminated for convenience with 60 days written notice, or "
                "with immediate effect in the event of a material security breach or a "
                "confirmed conflict of interest.",
            ),
        ],
    ),
    "performance_review_process.pdf": (
        "Northwind Technologies - Performance Review Process",
        [
            (
                "1. Review Cycle",
                "Northwind runs two formal review cycles per year: a mid-year review in "
                "July and an annual review in January. Compensation changes are decided "
                "only in the annual cycle and take effect from 1 April.",
            ),
            (
                "2. Eligibility",
                "Employees who joined on or before 30 September are eligible for the "
                "January annual review. Those who joined later receive their first formal "
                "review in the following July cycle, though probation reviews still apply.",
            ),
            (
                "3. Self Assessment",
                "The cycle opens with a self assessment, which must be submitted within "
                "10 business days. Employees are asked to list their three most "
                "significant contributions, evidence of impact, and two areas they want to "
                "develop.",
            ),
            (
                "4. Peer Feedback",
                "Each employee nominates three to five peers, and the manager may add up "
                "to two more. Peer feedback is shared with the employee in summarised form "
                "and is never attributed to a named individual.",
            ),
            (
                "5. Rating Scale",
                "Ratings are on a five point scale: 1 Needs Improvement, 2 Developing, "
                "3 Meets Expectations, 4 Exceeds Expectations, 5 Outstanding. Northwind "
                "does not operate a forced distribution or a quota for any rating.",
            ),
            (
                "6. Calibration",
                "Before ratings are released, department heads meet to calibrate them so "
                "that a rating of 4 means the same thing across teams. Calibration may "
                "change a proposed rating, and the manager must explain any change to the "
                "employee.",
            ),
            (
                "7. Promotions",
                "Promotion nominations are raised by the reporting manager during the "
                "annual cycle and require a rating of 4 or 5 in the two most recent cycles. "
                "Promotion decisions are made by a panel of at least three people, one of "
                "whom must be from outside the nominee's department.",
            ),
            (
                "8. Disagreeing With a Review",
                "An employee who disagrees with their rating may request a review by the "
                "Head of People within 10 business days of receiving it. The outcome of "
                "that review is final for the cycle.",
            ),
        ],
    ),
}


def build_pdf(filename: str, title: str, sections: list) -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Title"], fontSize=17, spaceAfter=18
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12.5,
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10.5, leading=15.5
    )

    story = [Paragraph(title, title_style)]
    for heading, body in sections:
        story.append(Paragraph(heading, heading_style))
        story.append(Paragraph(body, body_style))
    story.append(Spacer(1, 1 * cm))

    SimpleDocTemplate(
        str(DOCUMENTS_DIR / filename),
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        title=title,
    ).build(story)


def main():
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    for filename, (title, sections) in DOCUMENTS.items():
        build_pdf(filename, title, sections)
        print(f"  -  wrote documents/{filename}")
    print(f"\n{len(DOCUMENTS)} sample PDFs created. Next: python ingest.py\n")


if __name__ == "__main__":
    main()
