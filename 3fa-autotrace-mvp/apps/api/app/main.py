from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .tracer import trace_image

app = FastAPI(title="3FA AUTO TRACE API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
allow_origins=["https://threefa-autotrace-web.onrender.com"],
allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "3FA AUTO TRACE"}

@app.post("/api/trace")
async def trace(file: UploadFile = File(...)):
    allowed = {"image/png", "image/jpeg", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(400, "Upload PNG, JPG/JPEG atau WebP sahaja.")
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "Fail maksimum 15MB.")
    try:
        result = trace_image(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return result

@app.get("/api/subscription")
def subscription():
    return {
        "plan": "Pro",
        "price_myr": 49.90,
        "interval": "month",
        "status": "integration_pending",
        "message": "Sambungkan payment gateway untuk mengaktifkan recurring billing."
    }
