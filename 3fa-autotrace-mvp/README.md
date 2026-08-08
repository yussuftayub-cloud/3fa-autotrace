# 3FA AUTO TRACE — MVP

AI-powered raster-to-vector web app for printing workflows.

## Included
- Premium landing page
- PNG/JPG/WebP upload
- Basic automatic vector tracing using OpenCV contours
- SVG generation
- EPS generation
- SVG preview
- Subscription page prepared for RM49.90/month gateway integration
- Docker Compose

## Run locally

### Backend
```bash
cd apps/api
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd apps/web
npm install
npm run dev
```

Open http://localhost:3000

### Docker
```bash
docker compose up --build
```
