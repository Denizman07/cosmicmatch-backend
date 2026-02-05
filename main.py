import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

# PDF + chart
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = FastAPI(title="CosmicMatch API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MVP storage (Render disk is ephemeral)
REPORTS = {}  # report_id -> { "html": str, "pdf_path": str, "created_at": str }


class PreviewRequest(BaseModel):
    name: str
    age: int
    gender: str
    birth_date: str
    birth_time: str
    has_partner: bool
    partner_name: Optional[str] = None
    partner_gender: Optional[str] = None
    partner_birth_date: Optional[str] = None
    partner_birth_time: Optional[str] = None
    desired_text: Optional[str] = None


@app.get("/")
def home():
    return {"ok": True, "message": "CosmicMatch backend is running."}


def _prompt_preview(body: PreviewRequest) -> str:
    if body.has_partner:
        intent = "Write a short teaser preview (70–120 words) of romantic compatibility."
        partner_block = f"""
Partner:
- Name: {body.partner_name or "Unknown"}
- Gender: {body.partner_gender or "Unknown"}
- DOB: {body.partner_birth_date or "Unknown"}
- Time: {body.partner_birth_time or "Unknown"}
""".strip()
    else:
        intent = "Write a short teaser preview (70–120 words) about the user's love pattern and the kind of partner that suits them."
        partner_block = f"Desired partner (user words): {body.desired_text or 'Not provided.'}"

    return f"""
You are CosmicMatch, an elegant UK-English astrology-style relationship narrator.
Write ONLY in UK English. Romantic, mysterious, premium—never childish.
Subtle Shakespearean cadence is welcome, but do NOT quote Shakespeare directly.
Do NOT mention AI, APIs, prompts, or systems.

User:
- Name: {body.name}
- Age: {body.age}
- Gender: {body.gender}
- DOB: {body.birth_date}
- Birth time: {body.birth_time}

{partner_block}

Task:
{intent}

Rules:
- One short paragraph.
- 2–5 tasteful emojis total (✨🌙🪐💫).
- End with a line that invites them to unlock the full report.
""".strip()


def _prompt_full_report(body: PreviewRequest) -> str:
    # Full report text will be used for both HTML page and PDF.
    # "Hamlet vibe" = elegant gravity, no direct quotes.
    if body.has_partner:
        partner_block = f"""
Partner:
- Name: {body.partner_name or "Unknown"}
- Gender: {body.partner_gender or "Unknown"}
- DOB: {body.partner_birth_date or "Unknown"}
- Birth time: {body.partner_birth_time or "Unknown"}
""".strip()
        task = "Write a full 9-page premium compatibility report for this couple."
    else:
        partner_block = f"Desired partner (user words): {body.desired_text or 'Not provided.'}"
        task = "Write a full 9-page premium love + compatibility style report for the user (no partner provided)."

    return f"""
You are CosmicMatch, a premium UK-English astrologically-inspired relationship writer.
Write as if a seasoned human expert prepared it—confident, poetic, serious, romantic.
Subtle Shakespearean gravity (Hamlet-like) is welcome, but do NOT quote Shakespeare directly.
Do NOT mention AI, APIs, prompts, systems.

Audience: UK customers paying £19.99 for a premium report.
Style: mystical but grounded, elegant, never cringe, never childish. Use UK spellings.

STRUCTURE:
Return the report in clean Markdown with these sections (use these exact headings):
1) Cover
2) Compatibility Overview (scores)
3) Emotional & Communication
4) Attraction & Chemistry
5) Friction Points & Growth
6) Synastry Highlights (write as “Key Dynamics” without claiming exact degrees)
7) Timing & Near-Future Energy (next 3 months)
8) Action Steps (practical, emotionally intelligent)
9) Closing Vow (poetic ending)

Rules:
- Use 2–6 tasteful emojis per page-equivalent section (✨🌙🪐💫🕯️), not too many.
- Include a “Scoreboard” with 5 scores (0–10) and a short meaning line for each.
- Keep it premium and detailed.
- Never output Turkish.

User:
- Name: {body.name}
- Age: {body.age}
- Gender: {body.gender}
- DOB: {body.birth_date}
- Birth time: {body.birth_time}

{partner_block}

Task:
{task}
""".strip()


def _md_to_simple_html(md: str) -> str:
    # Very simple Markdown to HTML for MVP (headings + paragraphs + lists).
    # Good enough to show a premium-looking page; PDF is the "deliverable".
    html_lines = []
    for line in md.splitlines():
        line = line.rstrip()
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{line[2:]}</li>")
        elif line.strip() == "":
            html_lines.append("<br/>")
        else:
            html_lines.append(f"<p>{line}</p>")
    body = "\n".join(html_lines)
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>CosmicMatch Report</title>
  <style>
    body{{background:#06040b;color:#f5f2ff;font-family:system-ui;margin:0;padding:24px}}
    .wrap{{max-width:860px;margin:0 auto}}
    .card{{background:rgba(10,6,18,.75);border:1px solid rgba(180,120,255,.35);
      border-radius:18px;padding:22px;box-shadow:0 0 30px rgba(180,120,255,.12)}}
    h1,h2,h3{{color:#d9b7ff;letter-spacing:.2px}}
    p,li{{line-height:1.6;color:#f1ecff}}
    a.btn{{display:inline-block;margin-top:14px;padding:12px 16px;border-radius:999px;
      background:linear-gradient(90deg,#ffd700,#b58cff);color:#000;font-weight:800;text-decoration:none}}
    ul{{margin-top:0}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      {body}
    </div>
  </div>
</body>
</html>
""".strip()


def _make_placeholder_wheel_png(path: str) -> None:
    # Simple mystical "wheel" placeholder (not Swiss)
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = plt.subplot(111, projection="polar")
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # draw rings
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot([0, 2 * 3.14159], [r, r], linewidth=1)
    # spokes
    for i in range(12):
        theta = i * (2 * 3.14159 / 12)
        ax.plot([theta, theta], [0, 1.0], linewidth=1)
    plt.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def _make_pdf(report_md: str, pdf_path: str, wheel_path: str, title_line: str) -> None:
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    def header(page_title: str):
        c.setFont("Helvetica-Bold", 18)
        c.drawString(18 * mm, h - 22 * mm, page_title)
        c.setFont("Helvetica", 10)
        c.drawString(18 * mm, h - 28 * mm, "CosmicMatch • Premium Compatibility Report")

    # Split into "pages" roughly by headings (MVP)
    blocks = report_md.split("\n## ")
    pages = []
    for i, b in enumerate(blocks):
        if i == 0:
            pages.append(b.strip())
        else:
            pages.append(("## " + b).strip())

    # Cover page
    header("CosmicMatch ✨")
    c.setFont("Helvetica", 12)
    c.drawString(18 * mm, h - 40 * mm, title_line)
    c.drawString(18 * mm, h - 48 * mm, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    c.drawImage(wheel_path, 18 * mm, h - 150 * mm, width=70 * mm, height=70 * mm, mask="auto")
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(18 * mm, h - 160 * mm, "A mystic blueprint of love, written with care.")
    c.showPage()

    # Content pages
    for idx, content in enumerate(pages[:9], start=1):
        header(f"Chapter {idx}")
        y = h - 38 * mm
        c.setFont("Helvetica", 11)

        # simple line wrap
        text = c.beginText(18 * mm, y)
        text.setLeading(14)
        for raw in content.splitlines():
            line = raw.strip()
            if line == "":
                text.textLine("")
                continue
            # trim markdown symbols
            line = line.replace("### ", "").replace("## ", "").replace("# ", "")
            # hard wrap
            while len(line) > 95:
                text.textLine(line[:95])
                line = line[95:]
            text.textLine(line)
        c.drawText(text)

        # add wheel on a mid page
        if idx == 5:
            c.drawImage(wheel_path, w - 95 * mm, 30 * mm, width=70 * mm, height=70 * mm, mask="auto")

        c.showPage()

    c.save()


@app.post("/preview")
def preview(body: PreviewRequest):
    prompt = _prompt_preview(body)
    resp = client.responses.create(model="gpt-4o-mini", input=prompt)
    text = (resp.output_text or "").strip()
    return {"preview": text}


@app.post("/generate")
def generate(body: PreviewRequest):
    # Full report markdown from AI
    prompt = _prompt_full_report(body)
    resp = client.responses.create(model="gpt-4o-mini", input=prompt)
    report_md = (resp.output_text or "").strip()
    if not report_md:
        raise HTTPException(status_code=500, detail="Empty report from model.")

    report_id = uuid.uuid4().hex[:12]
    os.makedirs("/tmp/cosmicmatch", exist_ok=True)

    wheel_path = f"/tmp/cosmicmatch/wheel_{report_id}.png"
    pdf_path = f"/tmp/cosmicmatch/report_{report_id}.pdf"

    _make_placeholder_wheel_png(wheel_path)

    title_line = f"{body.name} & {body.partner_name}" if body.has_partner else f"{body.name}"
    _make_pdf(report_md, pdf_path, wheel_path, title_line)

    html = _md_to_simple_html(report_md)

    REPORTS[report_id] = {
        "html": html,
        "pdf_path": pdf_path,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    return {
        "report_id": report_id,
        "report_url": f"/report/{report_id}",
        "pdf_url": f"/download/{report_id}.pdf",
    }


@app.get("/report/{report_id}", response_class=HTMLResponse)
def report(report_id: str):
    item = REPORTS.get(report_id)
    if not item:
        raise HTTPException(status_code=404, detail="Report not found.")
    return item["html"]


@app.get("/download/{filename}")
def download(filename: str):
    # expects "xxxx.pdf"
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file.")
    report_id = filename.replace(".pdf", "")
    item = REPORTS.get(report_id)
    if not item:
        raise HTTPException(status_code=404, detail="Report not found.")
    path = item["pdf_path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File missing (server restarted).")
    return FileResponse(path, media_type="application/pdf", filename=f"cosmicmatch_{report_id}.pdf")
