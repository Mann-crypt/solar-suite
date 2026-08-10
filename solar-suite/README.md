# Solar Suite — FastAPI backend

Your three tools (Loss Correction, RT Correction, Aeromal) as a FastAPI backend,
math ported 1:1 from the Streamlit app. Point your existing HTML/JS frontend at
these endpoints instead of the hand-ported JS math — see `frontend/index.html`
for the exact `fetch` calls to copy in.

## Run locally

```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 — the wiring reference frontend loads, and
http://localhost:8000/docs gives you interactive Swagger docs for every endpoint.

## Project layout

```
app/
  main.py              # FastAPI routes
  models/
    loss_correction.py # Fixed/Tracking x Cluster/non-Cluster branches
    rt_correction.py   # parabolic ramp fit
    aeromal.py          # curtailment + no-curtailment profile generation
frontend/
  index.html           # API wiring reference — replace with your real UI
Dockerfile
docker-compose.yml
Caddyfile              # reverse proxy + HTTPS + basic auth
```

## Before you deploy

1. **Replace `frontend/index.html`** with your actual black-blue app, wired to
   call the endpoints (see the reference file for the exact contract).
2. **Set the Aeromal password.** It's read from the `AEROMAL_PASSWORD`
   environment variable (not hardcoded) — copy `.env.example` to `.env`,
   fill in a real password, and docker-compose will pick it up automatically.
   If `.env` is missing, the app fails closed (no login succeeds) rather than
   silently using a default.
3. **Edit `Caddyfile`** — put your real internal domain (or the server's IP)
   and generate a password hash:
   ```bash
   docker run --rm caddy caddy hash-password --plaintext 'yourpassword'
   ```
4. **Test Loss Correction against a real workbook.** RT Correction and Aeromal
   were smoke-tested with synthetic data already; Loss Correction needs one of
   your actual Excel files (with `Area & Efficiency`, `Forecast Config`,
   `Config Tilt Angle`, `Tracking`, and the Backend Cal sheets) to confirm the
   sheet-parsing logic matches — the calculation code itself is copied
   line-for-line from your Streamlit branches.

## Deploy

```bash
docker compose up -d --build
```

Point your domain's DNS A record at the server, Caddy handles HTTPS
automatically via Let's Encrypt.

## API summary

| Endpoint | Purpose |
|---|---|
| `POST /api/upload` | Upload workbook, get `file_id` + detected cluster/non-cluster + input rows |
| `POST /api/loss-correction/fixed` | Fixed plant type — immediate result |
| `POST /api/loss-correction/tracking/optimize` | Tracking — runs DE (slow), returns `job_id` |
| `GET /api/status/{job_id}` | Poll job status for any async optimize call |
| `POST /api/loss-correction/tracking/recalculate` | Recompute chart from edited params, no DE rerun |
| `POST /api/rt-correction/optimize` | RT Correction — runs DE, returns `job_id` |
| `POST /api/rt-correction/recalculate` | Recompute from edited `w/n1/n2/b` |
| `POST /api/aeromal/login` | Check password |
| `POST /api/aeromal/curtailment` | Curtailment-mode profile |
| `POST /api/aeromal/no-curtailment` | No-curtailment 95th-percentile profile |

Uploaded files are cached in memory for 4 hours (`FILE_TTL_SECONDS` in
`main.py`) so a session doesn't have to re-upload on every request — this
replaces the caching fix from your Streamlit performance work.
