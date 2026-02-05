from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "🌙 CosmicMatch backend is alive ✨"}
