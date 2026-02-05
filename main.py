import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

# WP'den çağırınca CORS takılmasın diye şimdilik açık bırakıyoruz.
# Canlıya geçince domain ekleyip kısıtlarız.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.get("/")
def home():
    return {"message": "🌙 CosmicMatch backend is alive ✨"}

class PreviewRequest(BaseModel):
    your_name: str | None = None
    your_gender: str | None = None

    has_partner: bool = False
    partner_name: str | None = None
    partner_gender: str | None = None

    desired_partner_text: str | None = None  # partner yoksa min 10 kelime gibi

@app.post("/preview")
def preview(body: PreviewRequest):
    your_name = (body.your_name or "You").strip()
    your_gender = (body.your_gender or "").strip()

    partner_name = (body.partner_name or "Partner").strip()
    partner_gender = (body.partner_gender or "").strip()
    desired_text = (body.desired_partner_text or "").strip()

    prompt = f"""
You are CosmicMatch, a mystical but serious astrology narrator.
Write a short teaser preview (2 paragraphs max) that makes the user want to unlock the full report.
Do NOT mention money, pricing, payment, Stripe, or "buy".
Use emojis lightly (max 3 emojis total).

User:
- Name: {your_name}
- Gender: {your_gender}
- Has partner: {body.has_partner}

If has partner:
- Partner name: {partner_name}
- Partner gender: {partner_gender}

If no partner:
- Desired partner description: {desired_text}
"""

    # Ucuz/hızlı model; sonra değiştirebiliriz.
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
    )

    text = (resp.output_text or "").strip()
    return {"preview": text}
