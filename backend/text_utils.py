"""Text cleaning shared by the notebook AND the API.

Why a separate file? The model must see text cleaned in exactly the same way during training (notebook) and during
prediction (Flask API). If the two used different cleaning code, predictions would silently get worse.
The notebook does `from text_utils import ticket_to_text`, and backend/app.py imports the same function.

Example:
    >>> ticket_to_text("Charged twice!!", "Hi, my card was billed $49 twice for INV-2025-0412.\\n\\nSent from my iPhone")
    'charged twice hi my card was billed _money_ twice for _invoice_'
"""
import re

# Lines that start "noise" blocks: everything from here to the end of the body is not written by the customer.
CUT_MARKERS = [
    r"^-+\s*original message\s*-+",      # -----Original Message-----
    r"^on .{0,80} wrote:",                # On Tue, Ana <...> wrote:
    r"^>",                                # quoted reply lines
]
# Single lines that are never useful for routing.
NOISE_LINES = [
    r"^sent from my \w+( phone)?$",
    r"^this email and any attachments are confidential",
    r"^please consider the environment before printing",
]

# Replace specific numbers by a placeholder word. The model should learn "an invoice number was mentioned",
# not memorise INV-2025-0412 (which will never appear again).
PLACEHOLDERS = [
    (r"\binv(?:oice)?[- ]?\d[\d-]*\b", " _invoice_ "),           # INV-2025-0412, invoice 88321
    (r"#?\bsf-?\d{4,}\b", " _order_ "),                          # #SF-48213, SF61009
    (r"[$€£₹]\s?\d[\d,]*(?:\.\d+)?", " _money_ "),               # $49, € 59, ₹2,999
    (r"\b(?:error|code)\s*(?:code\s*)?[a-z]?\d{3}\b", " _errorcode_ "),  # error 500, error code E104
    (r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", " _email_ "),               # email addresses
    (r"\+?\d[\d\s-]{7,}\d", " _phone_ "),                        # phone numbers
]


def strip_noise(body):
    """Remove quoted replies, 'Sent from my iPhone', legal disclaimers and HTML leftovers from an e-mail body."""
    if not isinstance(body, str):
        return ""
    body = re.sub(r"<[^>]+>|&nbsp;|&amp;", " ", body)           # HTML tags and entities from web forms
    kept = []
    for line in body.replace("\r", "").split("\n"):
        low = line.strip().lower()
        if any(re.match(p, low) for p in CUT_MARKERS):
            break                                                # the rest is an older message, not this ticket
        if any(re.match(p, low) for p in NOISE_LINES):
            continue
        kept.append(line)
    return "\n".join(kept)


def clean_text(text):
    """Lower case -> numbers to placeholders -> squash repeated letters -> keep letters, digits, _ -> collapse spaces.

    We KEEP stop words like "not", "can" and "no": "can not log in" is an important phrase for routing.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("’", "'")
    for pattern, repl in PLACEHOLDERS:
        text = re.sub(pattern, repl, text)
    text = re.sub(r"can't|cannot|cant\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)                   # "pleeease" -> "pleease", "!!!!" -> "!!"
    text = re.sub(r"[^a-z0-9_\s]", " ", text)                    # drop punctuation and other symbols
    return re.sub(r"\s+", " ", text).strip()


def ticket_to_text(subject, body):
    """The single string the models read: cleaned subject followed by the cleaned, de-noised body."""
    return (clean_text(subject) + " " + clean_text(strip_noise(body))).strip()
