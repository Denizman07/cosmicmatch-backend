import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI


# --------------------
# APP
# --------------------
app = FastAPI(title="CosmicMatch API", version="0.1.0")

# CORS (şimdilik açık; prod’da domain kısıtları eklenir)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client (Render Environment'dan geliyor)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --------------------
# MODELS
# --------------------
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


# --------------------
# ROUTES
# --------------------
@app.get("/")
def home():
    return "CosmicMatch backend is running."


@app.post("/preview")
def preview(body: PreviewRequest):
    # Build prompt (UK English, romantic but serious, mystical)
    if body.has_partner:
        partner_block = f"""
Partner details:
- Name: {body.partner_name or "Unknown"}
- Gender: {body.partner_gender or "Unknown"}
- Date of birth: {body.partner_birth_date or "Unknown"}
- Time of birth: {body.partner_birth_time or "Unknown"}
"""
        intent_block = "Create a short, romantic-yet-serious cosmic compatibility preview for them as a couple."
    else:
        partner_block = ""
        desired = body.desired_text or "They want a meaningful, emotionally mature relationship."
        intent_block = f"Create a short, romantic-yet-serious cosmic preview about the kind of partner they are calling in. Their preferences: {desired}"

    prompt = f"""
You are CosmicMatch, an elegant UK English astrologically-inspired narrator.
Write in UK English only. Tone: mystical, warm, romantic but not childish. Serious, poetic.
Avoid Turkish completely.

User details:
- Name: {body.name}
- Age: {body.age}
- Gender: {body.gender}
- Date of birth: {body.birth_date}
- Time of birth: {body.birth_time}

{partner_block}

Task:
{intent_block}

Constraints:
- 1 short paragraph (70–120 words).
- Use a few tasteful emojis (2–5 total), e.g. ✨🌙🪐💫
- Do NOT mention “OpenAI”, “model”, “API”, “prompt”.
- Do NOT claim medical/legal certainty.
- End with a hopeful closing line.
""".strip()

    # Call model
    # (Render’da stabil olsun diye düşük maliyetli model kullanıyoruz; sonra yükseltiriz.)
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
    )

    text = (resp.output_text or "").strip()
    return {"preview": text}
