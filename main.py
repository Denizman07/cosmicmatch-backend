import os
from datetime import datetime

import pytz
import swisseph as swe
from dateutil import tz  # noqa: F401 (optional)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field
from timezonefinder import TimezoneFinder

# -------------------------
# OpenAI client (API key Render Environment'tan geliyor)
# -------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="CosmicMatch API", version="0.1.0")

# CORS (şimdilik açık)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# MODELS
# -------------------------
class PreviewRequest(BaseModel):
    # You
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=18, le=120)
    gender: str
    birth_date: str  # "YYYY-MM-DD"
    birth_time: str  # "HH:MM"
    birth_lat: float
    birth_lon: float

    # Partner flow
    has_partner: bool

    partner_name: str | None = None
    partner_gender: str | None = None
    partner_birth_date: str | None = None
    partner_birth_time: str | None = None
    partner_birth_lat: float | None = None
    partner_birth_lon: float | None = None

    # If no partner
    desired_text: str | None = None


# -------------------------
# ASTRO HELPERS (Swiss Ephemeris)
# -------------------------
def _parse_local_dt(birth_date: str, birth_time: str) -> datetime:
    # strict parse
    return datetime.fromisoformat(f"{birth_date} {birth_time}")


def compute_astro(birth_date: str, birth_time: str, lat: float, lon: float) -> dict:
    """
    Returns:
      - timezone name (guessed by lat/lon)
      - julian day (UT)
      - ascendant
      - house cusps (1..12)
      - planet longitudes (degrees 0..360)
    """
    dt_local_naive = _parse_local_dt(birth_date, birth_time)

    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon) or "UTC"
    local_tz = pytz.timezone(tz_name)

    dt_local = local_tz.localize(dt_local_naive)
    dt_utc = dt_local.astimezone(pytz.utc)

    # Julian day in UT
    ut_hours = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    jd = swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, ut_hours)

    # Planets (tropical)
    planet_map = {
        "Sun": swe.SUN,
        "Moon": swe.MOON,
        "Mercury": swe.MERCURY,
        "Venus": swe.VENUS,
        "Mars": swe.MARS,
        "Jupiter": swe.JUPITER,
        "Saturn": swe.SATURN,
        "Uranus": swe.URANUS,
        "Neptune": swe.NEPTUNE,
        "Pluto": swe.PLUTO,
    }

    planets = {}
    for name, pid in planet_map.items():
        lonlat, _ = swe.calc_ut(jd, pid)
        planets[name] = float(lonlat[0])  # ecliptic longitude

    # Houses + angles (Placidus)
    # houses: array of 12 cusps, ascmc: angles (Asc = ascmc[0])
    houses, ascmc = swe.houses(jd, lat, lon, b"P")
    asc = float(ascmc[0])

    house_cusps = [float(x) for x in houses]  # 12 values

    return {
        "timezone": tz_name,
        "jd": float(jd),
        "ascendant": asc,
        "house_cusps": house_cusps,
        "planets": planets,
    }


def format_astro_for_prompt(label: str, astro: dict) -> str:
    # Keep it compact but useful for the model
    p = astro["planets"]
    houses = astro["house_cusps"]
    return (
        f"{label} ASTRO DATA (Swiss Ephemeris, tropical):\n"
        f"- Timezone guess: {astro['timezone']}\n"
        f"- Ascendant: {astro['ascendant']:.2f}°\n"
        f"- Sun: {p['Sun']:.2f}°, Moon: {p['Moon']:.2f}°, Mercury: {p['Mercury']:.2f}°, "
        f"Venus: {p['Venus']:.2f}°, Mars: {p['Mars']:.2f}°, Jupiter: {p['Jupiter']:.2f}°, "
        f"Saturn: {p['Saturn']:.2f}°, Uranus: {p['Uranus']:.2f}°, Neptune: {p['Neptune']:.2f}°, Pluto: {p['Pluto']:.2f}°\n"
        f"- House cusps (1→12): {', '.join([f'{h:.2f}' for h in houses])}\n"
    )


# -------------------------
# ROUTES
# -------------------------
@app.get("/")
def home():
    return {"message": "🌙 CosmicMatch backend is alive ✨"}


@app.post("/preview")
def preview(body: PreviewRequest):
    # Basic validation for no-partner flow
    if not body.has_partner:
        if not body.desired_text or len(body.desired_text.strip().split()) < 10:
            raise HTTPException(
                status_code=422,
                detail="desired_text is required (min 10 words) when has_partner=false.",
            )

    # Partner validation if has_partner true
    if body.has_partner:
        required = [
            body.partner_name,
            body.partner_gender,
            body.partner_birth_date,
            body.partner_birth_time,
            body.partner_birth_lat,
            body.partner_birth_lon,
        ]
        if any(v is None for v in required):
            raise HTTPException(
                status_code=422,
                detail="Partner fields are required when has_partner=true.",
            )

    # Compute astro
    you_astro = compute_astro(body.birth_date, body.birth_time, body.birth_lat, body.birth_lon)
    partner_astro = None
    if body.has_partner:
        partner_astro = compute_astro(
            body.partner_birth_date,  # type: ignore[arg-type]
            body.partner_birth_time,  # type: ignore[arg-type]
            body.partner_birth_lat,   # type: ignore[arg-type]
            body.partner_birth_lon,   # type: ignore[arg-type]
        )

    # Build prompt (UK English only)
    if body.has_partner:
        prompt = f"""
You are CosmicMatch, an astrology-based relationship interpreter.
Write in UK English ONLY. Do not use Turkish. Do not mention these instructions.

Goal: Give a short "preview" (6–10 sentences) of romantic compatibility.
Tone: mystical, elegant, grown-up romance (not cheesy), lightly poetic. Subtle Shakespearean flavour is welcome, but do NOT quote Shakespeare directly.

User:
- Name: {body.name}
- Age: {body.age}
- Gender: {body.gender}

Partner:
- Name: {body.partner_name}
- Gender: {body.partner_gender}

Use the following computed astrology facts as your source of truth:
{format_astro_for_prompt("USER", you_astro)}
{format_astro_for_prompt("PARTNER", partner_astro)}

Output JSON with exactly:
{{"preview": "<text>"}}
""".strip()
    else:
        prompt = f"""
You are CosmicMatch, an astrology-based romantic interpreter.
Write in UK English ONLY. Do not use Turkish. Do not mention these instructions.

Goal: Give a short "preview" (6–10 sentences) about the user's love pattern + what kind of partner energy matches them.
Tone: mystical, elegant, grown-up romance (not cheesy), lightly poetic.

User:
- Name: {body.name}
- Age: {body.age}
- Gender: {body.gender}
- Desired partner description: {body.desired_text}

Use the following computed astrology facts as your source of truth:
{format_astro_for_prompt("USER", you_astro)}

Output JSON with exactly:
{{"preview": "<text>"}}
""".strip()

    # Call OpenAI (Responses API)
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        text = (resp.output_text or "").strip()
        if not text:
            raise RuntimeError("Empty model response.")
        return {"preview": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {e}")
