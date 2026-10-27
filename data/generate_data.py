"""Generate the synthetic support-ticket dataset used in this project.

Run from the project root:  python data/generate_data.py
The output is data/support_tickets.csv. The seed is fixed, so everyone gets the same file.

The fictional company is "ShopFlow": a cloud point-of-sale (POS) and inventory app for small shops. It also SHIPS
hardware (card readers, receipt printers, barcode scanners, cash drawers). Customers write to the help desk by
email, chat or a web form, and every ticket must be routed to one of six teams and given a priority.

How a ticket is made:
  1. pick the TRUE team and a priority (priority depends on the team: bugs and lockouts are urgent more often),
  2. build the body from phrase pools: greeting, shop context, 1-3 issue sentences of that team, sometimes a sentence
     about ANOTHER team's topic (real tickets mix topics), urgency words that usually - but not always - match the
     priority, a closing line, a signature, "Sent from my iPhone", quoted e-mail replies or legal disclaimers,
  3. write a subject (specific, vague like "Help needed", or missing for chat),
  4. add typos, lower-case chat style or ALL CAPS anger,
  5. add messiness: some wrong labels (agents make mistakes), empty tickets and tickets submitted twice.

Nothing here is real customer data; it is invented for teaching.
"""
import os

import numpy as np
import pandas as pd

SEED = 42
N = 5000

TEAMS = ["Billing", "Technical Bug", "Account Access", "Feature Request", "Shipping/Hardware", "General Inquiry"]
TEAM_SHARE = [0.20, 0.26, 0.17, 0.13, 0.11, 0.13]
PRIORITIES = ["Low", "Medium", "High"]
# P(Low, Medium, High) for each team
PRIORITY_BY_TEAM = {
    "Billing": [0.30, 0.50, 0.20], "Technical Bug": [0.20, 0.45, 0.35], "Account Access": [0.15, 0.45, 0.40],
    "Feature Request": [0.70, 0.27, 0.03], "Shipping/Hardware": [0.30, 0.48, 0.22], "General Inquiry": [0.65, 0.30, 0.05],
}

# Values that fill the {slots} in the templates below
SLOTS = {
    "device": ["card reader", "receipt printer", "barcode scanner", "cash drawer", "card terminal", "label printer"],
    "plan": ["Starter", "Growth", "Pro", "Enterprise"],
    "amount": ["$49", "$129", "$29.99", "€59", "₹2,999", "$300", "$15", "€120", "$89"],
    "report": ["sales report", "end of day report", "inventory report", "tax summary", "staff report"],
    "time": ["since this morning", "for 3 days", "since yesterday", "since the last update", "all week",
             "since Monday", "for about an hour", "on and off for two weeks"],
    "stores": ["one store", "2 stores", "3 shops", "a small cafe", "a bakery", "a clothing boutique", "5 outlets",
               "a pharmacy", "a hardware store", "a food truck"],
    "city": ["Pune", "Leeds", "Austin", "Lisbon", "Nairobi", "Toronto", "Melbourne", "Dublin", "Chennai", "Denver"],
    "name": ["Priya", "Tom", "Ana", "Rahul", "Fatima", "James", "Mei", "Carlos", "Olu", "Sara", "Arjun", "Emma"],
    "order": ["#SF-48213", "#SF-50177", "#SF-39021", "#SF-61442", "SF61009", "#SF-45530"],
    "invoice": ["INV-2025-0412", "INV-2025-0977", "INV-2024-1203", "invoice 88321", "INV-2025-0150"],
    "browser": ["Chrome", "Safari", "the iPad app", "the Android app", "Edge", "the Windows app"],
    "error": ["error 500", "error code E104", "a blank white screen", "'Something went wrong'", "error 403",
              "a spinning wheel forever", "'sync failed'", "timeout error"],
    "integration": ["QuickBooks", "Xero", "our webshop", "Google Sheets", "Mailchimp", "our accounting software"],
}

# ---------------------------------------------------------------------------
# Phrase pools per team. "issues" are normal sentences, "severe" ones appear mainly in High-priority tickets.
# Some topics overlap ON PURPOSE (card readers appear in bugs and hardware; suspended accounts in billing and access).
# ---------------------------------------------------------------------------
POOLS = {
    "Billing": {
        "subjects": ["Charged twice", "Refund request", "Invoice question", "Wrong amount on invoice", "Billing issue",
                     "Cancel my subscription", "Upgrade plan", "Payment failed", "GST invoice needed",
                     "Why was I charged {amount}?", "Downgrade to {plan}", "Need a copy of {invoice}", "Overcharged",
                     "Subscription renewal", "Refund for unused months"],
        "issues": ["I was charged twice this month for the {plan} plan.",
                   "My card was billed {amount} but my plan is only {amount}.",
                   "Can you send me a copy of {invoice}? I need it for my accountant.",
                   "I want to cancel my subscription at the end of this billing cycle.",
                   "Please refund the extra charge of {amount}.",
                   "We downgraded to {plan} last month but we are still paying the old price.",
                   "Our invoice does not show our VAT number, can you add it?",
                   "The payment for our subscription failed and I updated the card, please retry the charge.",
                   "How is the proration calculated when we upgrade to {plan} in the middle of the month?",
                   "I need a GST invoice with our company name instead of my personal name.",
                   "We were charged for 5 user seats but we only have 3 staff accounts.",
                   "Is there a discount if we pay yearly instead of monthly?",
                   "The invoice amount does not match the price on your pricing page.",
                   "I closed one of my stores, please remove it from the bill.",
                   "Can I change the billing email to our accounts team?",
                   "We were charged after the free trial even though I cancelled it.",
                   "Please explain the extra line called 'add-on fee' on {invoice}.",
                   "Our bank says the refund you promised has not arrived yet."],
        "severe": ["You charged us {amount} three times and our bank account is now overdrawn.",
                   "Our account got suspended because of a failed payment and we cannot sell anything.",
                   "This unauthorised charge of {amount} must be reversed today."],
    },
    "Technical Bug": {
        "subjects": ["App keeps crashing", "Sync not working", "Bug in reports", "{error} when saving", "Inventory wrong",
                     "Checkout freezes", "Bug report", "Problem after update", "Totals are wrong",
                     "Barcode scan not adding item", "Export broken", "Offline sales missing", "Slow app",
                     "Integration with {integration} failing", "Receipts not printing from app"],
        "issues": ["The app crashes every time I open the {report}.",
                   "Sales made on the iPad are not syncing to the web dashboard {time}.",
                   "The {report} shows different totals than the actual sales.",
                   "I get {error} when I try to save a new product.",
                   "Inventory counts go negative after a refund is processed.",
                   "The checkout screen freezes when I add a discount.",
                   "When I scan a barcode the item is not added to the cart.",
                   "Exporting to CSV gives an empty file.",
                   "After the latest update {browser} is extremely slow.",
                   "Sales made in offline mode disappeared when the internet came back.",
                   "The sync with {integration} has been failing {time} with {error}.",
                   "Tax is calculated twice on some products.",
                   "The app does not detect the {device} even though it is paired.",
                   "Receipts print with strange symbols instead of the item names.",
                   "Product images are not loading in {browser}.",
                   "Discount codes are applied to the wrong items.",
                   "The page shows {error} when I click on customers.",
                   "Stock transfers between stores are not showing up.",
                   "The time on receipts is one hour behind after the clock change.",
                   "Reports page just shows {error}."],
        "severe": ["The whole POS is down and we cannot take any payments, all registers show {error}.",
                   "We lost all of today's sales data after the update, this is critical.",
                   "Checkout is completely broken in all {stores}, customers are waiting in line.",
                   "Production is down: every sale fails with {error}."],
    },
    "Account Access": {
        "subjects": ["Can't log in", "Password reset not working", "Locked out", "2FA problem", "Login issue",
                     "Reset email not received", "Account locked", "Add new user", "Staff PIN not working",
                     "Change account owner", "SSO login fails", "Lost phone for 2FA", "Access denied",
                     "Remove ex-employee access", "Login loop"],
        "issues": ["I cannot log in, it says my password is incorrect even after resetting it.",
                   "The password reset email never arrives, I checked spam too.",
                   "I lost my phone and cannot get the two-factor code.",
                   "My account is locked after too many login attempts.",
                   "Please add my new manager as an admin user.",
                   "A staff member's PIN stopped working on the register.",
                   "We need to change the account owner because the previous owner left.",
                   "Single sign-on with Google fails with {error}.",
                   "I keep getting logged out every few minutes in {browser}.",
                   "Please remove access for an employee who left yesterday.",
                   "The login page just reloads in a loop.",
                   "I changed my email address and now I cannot sign in.",
                   "My staff cannot see the reports, it says access denied.",
                   "The verification code says it has expired every time.",
                   "How do I give a cashier permission to issue refunds?",
                   "I forgot which email I used to sign up.",
                   "Our account says suspended, how do I get access back?"],
        "severe": ["I am completely locked out of the admin account and the store opens in an hour.",
                   "Someone changed our password and we think the account was hacked.",
                   "None of the staff can log in to the registers, we cannot open the shop."],
    },
    "Feature Request": {
        "subjects": ["Feature request", "Suggestion", "Idea for the app", "Would love dark mode", "Please add loyalty points",
                     "Request: multi-currency", "Can you add this?", "Wishlist", "Improvement idea",
                     "Integration with {integration}", "Custom receipt logo", "Bulk edit products",
                     "Feature idea: staff scheduling", "Export to Excel please", "Suggestion for reports"],
        "issues": ["It would be great if you could add a dark mode to the register screen.",
                   "Please add a loyalty points program for repeat customers.",
                   "We would love to export reports directly to Excel.",
                   "Could you support multiple currencies for tourists?",
                   "I wish I could put our logo on the receipts.",
                   "Please add a way to edit many products at once.",
                   "An integration with {integration} would save us hours every week.",
                   "It would help to schedule staff shifts inside the app.",
                   "Can you add a low stock alert by email?",
                   "Suggestion: let customers sign up for a newsletter at checkout.",
                   "We would like to split a bill between several cards.",
                   "Would be nice to have a weekly summary email of sales.",
                   "Please allow more than one discount on the same sale.",
                   "Could the {report} show profit and not only revenue?",
                   "It would be useful to have gift cards.",
                   "Can you make the product search smarter so typos still find items?",
                   "Any plans to support table ordering for restaurants?"],
        "severe": [],
    },
    "Shipping/Hardware": {
        "subjects": ["Where is my order?", "Tracking number", "{device} arrived damaged", "Wrong item shipped",
                     "Return a {device}", "Order {order}", "Replacement {device}", "Delivery delayed",
                     "Hardware not working", "Need more paper rolls", "Missing parts", "{device} won't turn on",
                     "Shipping address change", "Exchange device", "Hardware order status"],
        "issues": ["I ordered a {device} two weeks ago and it still has not arrived.",
                   "Can you send me the tracking number for order {order}?",
                   "The {device} arrived with a cracked screen.",
                   "You sent a {device} instead of the {device} I ordered.",
                   "I want to return the {device}, it is not what we need.",
                   "The {device} does not turn on even after charging it overnight.",
                   "The battery of the {device} only lasts two hours.",
                   "Please change the shipping address for order {order}.",
                   "The box was missing the power cable for the {device}.",
                   "How do I order more thermal paper rolls?",
                   "The cash drawer does not open when a sale is completed.",
                   "Our {device} is still under warranty, can we get a replacement?",
                   "Delivery says it was delivered but we never received it.",
                   "The courier lost our package with order {order}.",
                   "The receipt printer makes a grinding noise and the paper jams.",
                   "Do you ship hardware to {city}?"],
        "severe": ["Our only {device} is dead and we cannot take card payments today.",
                   "We open the new store tomorrow and the hardware for order {order} has not arrived."],
    },
    "General Inquiry": {
        "subjects": ["Question", "General question", "Information please", "Support hours?", "Does ShopFlow work in {city}?",
                     "Training session", "Talk to sales", "Data privacy question", "Partnership", "Just a question",
                     "Thank you!", "Documentation", "Demo request", "Feedback", "Hello"],
        "issues": ["What are your support hours on weekends?",
                   "Does ShopFlow work in {city}?",
                   "Do you offer training sessions for new staff?",
                   "I would like to talk to someone from sales about a bigger plan.",
                   "Where is our data stored and is it GDPR compliant?",
                   "We are a reseller and would like to discuss a partnership.",
                   "Just wanted to say thank you, the team was very helpful last week.",
                   "Where can I find the user guide?",
                   "Can I get a demo for my business partner?",
                   "Is there a phone number I can call?",
                   "How long do you keep our sales history?",
                   "Do you have a mobile app for the owner?",
                   "We are thinking of switching from another POS, how does the migration work?",
                   "Is ShopFlow suitable for a restaurant or only for shops?",
                   "Do you have an affiliate program?",
                   "What languages does the app support?"],
        "severe": [],
    },
}

GENERIC = ["It is not working.", "Please help.", "Can someone look into this?", "Not sure who to ask about this.",
           "This has happened before.", "Same problem as my colleague reported.", "Please check.",
           "I already tried restarting.", "Let me know what you need from me.", "Screenshots attached."]
GREETINGS = ["Hi,", "Hello team,", "Hi there,", "Dear support,", "Good morning,", "Hey,", "Hello,", "Hi ShopFlow team,"]
CONTEXT = ["We run {stores} in {city}.", "I am the owner of {stores}.", "We have used ShopFlow for 2 years.",
           "I manage {stores}.", "We are new customers.", "Our shop is in {city}."]
URGENT_HIGH = ["This is URGENT.", "Please help ASAP!!", "We are losing sales every minute.", "Need this fixed today.",
               "Urgent please, the whole team is blocked.", "Please call me immediately.", "This is critical for us."]
URGENT_MED = ["Would appreciate a reply soon.", "Please look into it this week.", "It is becoming annoying.",
              "Can you check today or tomorrow?"]
LOW_CUES = ["No rush.", "Whenever you get a chance.", "Not urgent, just curious.", "Low priority.",
            "Just for the future."]
CLOSINGS = ["Thanks.", "Thank you!", "Thanks in advance.", "Regards.", "Cheers.", "Best regards,", "Thx"]
SIGNATURES = ["{name}\nStore manager", "{name} | Owner\n{stores}, {city}", "-- \n{name}\nOperations",
              "Kind regards,\n{name}\nPhone: +1 555 0100", "{name}"]
NOISE = ["Sent from my iPhone",
         "Sent from my Android phone",
         "This email and any attachments are confidential and intended only for the addressee.",
         "On Tue, {name} <support@shopflow.example> wrote:\n> Thanks for contacting ShopFlow support.\n> How can we help?",
         "-----Original Message-----\nFrom: ShopFlow Support\nSubject: Ticket received",
         "Please consider the environment before printing this email.",
         "&nbsp;<br>"]
VAGUE_SUBJECTS = ["Help needed", "Issue", "Problem", "Question", "Urgent", "Re: my ticket", "Fwd: issue",
                  "Please help", "Support request", "(no subject)"]


def fill(template, rng):
    """Replace every {slot} with a random value. Each slot is filled independently."""
    out = template
    while "{" in out:
        start = out.index("{")
        end = out.index("}", start)
        key = out[start + 1:end]
        out = out[:start] + rng.choice(SLOTS[key]) + out[end + 1:]
    return out


def add_typos(text, rng, rate=0.06):
    """Swap or drop a letter in some longer words, like a person typing fast on a phone."""
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 4 and w.isalpha() and rng.random() < rate:
            j = int(rng.integers(1, len(w) - 2))
            if rng.random() < 0.5:
                words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]   # swap two letters
            else:
                words[i] = w[:j] + w[j + 1:]                      # drop a letter
    return " ".join(words)


def make_ticket(team, priority, channel, rng):
    pool = POOLS[team]
    parts = []
    if channel != "chat" and rng.random() < 0.6:
        parts.append(rng.choice(GREETINGS))
    if rng.random() < 0.3:
        parts.append(fill(rng.choice(CONTEXT), rng))

    # main issue sentences (sometimes a "severe" one for High priority)
    n_issue = int(rng.choice([1, 1, 2, 2, 3])) if channel != "chat" else int(rng.choice([1, 1, 2]))
    issues = list(rng.choice(pool["issues"], size=n_issue, replace=False))
    if pool["severe"] and ((priority == "High" and rng.random() < 0.5) or (priority == "Medium" and rng.random() < 0.06)):
        issues[0] = rng.choice(pool["severe"])
    parts += [fill(s, rng) for s in issues]

    # real tickets mix topics: a sentence about another team's topic
    if rng.random() < 0.25:
        other = rng.choice([t for t in TEAMS if t != team])
        extra = fill(rng.choice(POOLS[other]["issues"]), rng)
        parts.append("Also, " + extra[0].lower() + extra[1:] if rng.random() < 0.5 else extra)
    if rng.random() < 0.35:
        parts.append(rng.choice(GENERIC))

    # urgency words USUALLY match the priority, but people over- and under-state urgency
    r = rng.random()
    if priority == "High":
        cue = rng.choice(URGENT_HIGH) if r < 0.65 else (rng.choice(URGENT_MED) if r < 0.8 else None)
    elif priority == "Medium":
        cue = rng.choice(URGENT_MED) if r < 0.3 else (rng.choice(URGENT_HIGH) if r < 0.4 else None)
    else:
        cue = rng.choice(LOW_CUES) if r < 0.45 else (rng.choice(URGENT_HIGH) if r < 0.53 else None)
    if cue:
        parts.insert(int(rng.integers(len(parts) // 2, len(parts) + 1)), cue)

    if rng.random() < 0.5:
        parts.append(rng.choice(CLOSINGS))
    body = " ".join(parts)
    if channel == "email" and rng.random() < 0.55:
        body += "\n\n" + fill(rng.choice(SIGNATURES), rng)
    if channel == "email" and rng.random() < 0.3:
        body += "\n\n" + fill(rng.choice(NOISE), rng)

    # style: typos, lower-case chat, shouting
    if rng.random() < 0.3:
        body = add_typos(body, rng)
    if channel == "chat" and rng.random() < 0.5:
        body = body.lower()
    if priority == "High" and rng.random() < 0.06:
        body = body.upper()

    # subject: specific, vague, or missing (chat often has none)
    if channel == "chat" and rng.random() < 0.7:
        subject = ""
    elif rng.random() < 0.2:
        subject = rng.choice(VAGUE_SUBJECTS)
    else:
        subject = fill(rng.choice(pool["subjects"]), rng)
        if rng.random() < 0.1:
            subject = "Re: " + subject
    return subject, body


def make_dataset(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    teams = rng.choice(TEAMS, size=n, p=TEAM_SHARE)
    rows = []
    start = pd.Timestamp("2025-01-06")
    for i, team in enumerate(teams):
        priority = rng.choice(PRIORITIES, p=PRIORITY_BY_TEAM[team])
        channel = rng.choice(["email", "web", "chat"], p=[0.5, 0.3, 0.2])
        subject, body = make_ticket(team, priority, channel, rng)
        created = start + pd.Timedelta(days=int(rng.integers(0, 170)), hours=int(rng.integers(7, 22)),
                                       minutes=int(rng.integers(0, 60)))
        rows.append({"created_at": created, "channel": channel, "subject": subject, "body": body,
                     "team": team, "priority": priority})
    df = pd.DataFrame(rows)

    # very short tickets: only a few words, the rest of the details missing
    short = rng.choice(n, size=60, replace=False)
    df.loc[short, "body"] = rng.choice(["help", "pls call me", "see subject", "urgent", "?", "not working",
                                        "as discussed on the phone", ""], size=60)

    # agents sometimes route a ticket to the wrong team: 4% wrong labels
    wrong = rng.choice(n, size=int(0.04 * n), replace=False)
    df.loc[wrong, "team"] = [rng.choice([t for t in TEAMS if t != df.loc[i, "team"]]) for i in wrong]

    # completely empty tickets (form submitted by accident)
    empty = rng.choice(n, size=15, replace=False)
    df.loc[empty, ["subject", "body"]] = ""

    # customers who press "send" twice: same text, new ticket id, a few minutes later
    dupes = df.sample(80, random_state=seed).copy()
    dupes["created_at"] = dupes["created_at"] + pd.Timedelta(minutes=3)
    df = pd.concat([df, dupes], ignore_index=True).sort_values("created_at").reset_index(drop=True)
    df.insert(0, "ticket_id", [f"T{100000 + i}" for i in range(len(df))])
    df["created_at"] = df["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    return df[["ticket_id", "created_at", "channel", "subject", "body", "team", "priority"]]


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "support_tickets.csv")
    data = make_dataset()
    data.to_csv(out, index=False)
    print(f"Wrote {len(data)} rows to {out}")
    print(data["team"].value_counts().to_string())
