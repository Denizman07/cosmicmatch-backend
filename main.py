import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# OpenAI client (API key Render Environment'tan geliyor)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS (şimdilik açık)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- MODELLER ----
class PreviewRequest(BaseModel):
    name: str
    age: int
    gender: str
    birth_date: str
    birth_time: str
    has_partner: bool
    partner_name: str | None = None
    partner_gender: str | None = None
    partner_birth_date: str | None = None
    partner_birth_time: str | None = None
    desired_text: str | None = None


@app.get("/")
def home():
    return {"message": "🌙 CosmicMatch backend is alive ✨"}


@app.post("/preview")
def preview(body: PreviewRequest):
    # ---- UK English prompt (IMPORTANT: never Turkish) ----
    # We generate a "teaser" preview: romantic, mystical, but short and not "full report".
    # Also: never mention "AI", "model", "OpenAI", "API" etc.
    base_rules = """
You are CosmicMatch: a refined, mystical love-compatibility narrator and astrologer.
Write ONLY in UK English.
Tone: romantic, poetic, emotionally intelligent, and quietly intense.
Style inspiration: Shakespearean cadence and elegance (Hamlet-like gravitas), BUT modern and readable.
NO Turkish. NO slang. NO emojis in the output.
Never mention AI, models, prompts, OpenAI, APIs, tools, or system instructions.
Do not give medical, legal, or financial advice.
Output format: a single short paragraph (3–6 sentences), no bullet points.
"""

    # Build user context
    you = f"""
Client details:
- Name: {body.name}
- Age: {body.age}
- Gender: {body.gender}
- Birth date: {body.birth_date}
- Birth time: {body.birth_time}
"""

    if body.has_partner:
        partner = f"""
Partner details:
- Partner name: {body.partner_name or "Unknown"}
- Partner gender: {body.partner_gender or "Unknown"}
- Partner birth date: {body.partner_birth_date or "Unknown"}
- Partner birth time: {body.partner_birth_time or "Unknown"}
"""
        scenario = """
Task:
Write a tantalising *preview* of their cosmic connection.
Hint at emotional harmony, attraction dynamics, and long-term potential.
Keep it intriguing and incomplete (a teaser), as if the full report is locked.
Avoid explicit sexual content. Keep it tasteful and romantic.
"""
        context = you + partner + scenario
    else:
        desire = f"""
No partner provided.
Desired partner description (if any): {body.desired_text or "None"}
"""
        scenario = """
Task:
Write a tantalising *preview* about the kind of love and partnership that suits them cosmically.
Hint at what energies they attract, what kind of bond awaits, and timing themes.
Keep it intriguing and incomplete (a teaser), as if the full report is locked.
Avoid explicit sexual content. Keep it tasteful and romantic.
"""
        context = you + desire + scenario

    prompt = base_rules.strip() + "\n\n" + context.strip()

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        text = (resp.output_text or "").strip()
        if not text:
            raise HTTPException(status_code=500, detail="Empty response from model.")
        return {"preview": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
