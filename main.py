import os
from fastapi import FastAPI
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
    has_partner: bool
    partner_name: str | None = None
    partner_gender: str | None = None
    desired_text: str | None = None


# ---- ROUTES ----

@app.get("/")
def home():
    return {"message": "🌙 CosmicMatch backend is alive ✨"}


@app.post("/preview")
def preview(body: PreviewRequest):
    prompt = f"""
    Kullanıcı bilgileri:
    - İsim: {body.name}
    - Yaş: {body.age}
    - Cinsiyet: {body.gender}
    - Partneri var mı: {body.has_partner}

    """

    if body.has_partner:
        prompt += f"""
        Partner bilgileri:
        - İsim: {body.partner_name}
        - Cinsiyet: {body.partner_gender}
        """
    else:
        prompt += f"""
        İstenilen partner tanımı:
        - {body.desired_text}
        """

    prompt += """
    Buna göre kısa, romantik ve pozitif bir ilişki yorumu yap.
    """

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
    )

    text = (resp.output_text or "").strip()
    return {"preview": text}
