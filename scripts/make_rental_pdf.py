"""Generates the demo PDF data/docs/gear-rental-terms.pdf (2 pages)."""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

out = Path(__file__).resolve().parent.parent / "data" / "docs" / "gear-rental-terms.pdf"
s = getSampleStyleSheet()
page1 = [
    ("Gear Rental Program - Terms", "Title"),
    ("Pinecrest rents tents, backpacks and sleeping bags so you can try gear before you buy. Rentals are available for US addresses only.", "BodyText"),
    ("Prices", "Heading2"),
    ("A weekend rental (Friday to Monday) costs $25 per item. A full week costs $45 per item. A refundable $100 security deposit is held on your card for each rental order and released within 3 business days after the gear is returned and inspected.", "BodyText"),
    ("Delivery and return", "Heading2"),
    ("Rental gear arrives at least 1 business day before your start date. Return it with the prepaid label by the end date. Late returns cost $10 per item per day.", "BodyText"),
]
page2 = [
    ("Damage and cleaning", "Heading2"),
    ("Normal dirt is fine; we clean every item after each rental. Damage beyond normal use, such as burns, tears longer than 1 inch or broken poles, is charged at repair cost, up to the full retail price of the item.", "BodyText"),
    ("Buy what you rented", "Heading2"),
    ("If you buy the same new item within 30 days of returning a rental, the rental fee is taken off the purchase price.", "BodyText"),
]
story = []
for text, style in page1:
    story += [Paragraph(text, s[style]), Spacer(1, 8)]
story.append(PageBreak())
for text, style in page2:
    story += [Paragraph(text, s[style]), Spacer(1, 8)]
SimpleDocTemplate(str(out), pagesize=letter).build(story)
print("wrote", out)
