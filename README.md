# Dynamic ETA Forecast for Coaching Trains   
*Dynamic Forecast of Expected Time of Arrival (ETA) for Coaching Trains*

A working, end-to-end prototype: real historical delay data → PyTorch LSTM
model → FastAPI live-inference backend → browser dashboard that dynamically
re-forecasts ETA for every remaining station the instant a new live signal
(delay, weather, congestion) comes in.

---

## 1. What's inside

```
railway_eta_prototype/
├── raw_data/                  Real source dataset (see Data section)
├── data/
│   ├── build_dataset.py       Builds training data from real IR stats
│   ├── journeys.csv           Generated: ~89k station-events, 2,520 runs
│   └── stations.csv           Generated: station list + distances per train
├── models/
│   ├── model.py                PyTorch LSTM architecture
│   ├── train.py                Training script
│   ├── eta_lstm.pt             Trained model weights
│   ├── norm_stats.json         Feature normalization stats
│   └── eval_report.json        Model vs static-baseline comparison
├── api/
│   └── main.py                 FastAPI backend (live inference)
├── dashboard/
│   └── index.html               Control-room style live dashboard
├── requirements.txt
└── README.md
```

## 2. The data (real, not invented)

Indian Railways does not publish live GPS/NTES feeds publicly, so — as
scoped and agreed earlier — this prototype combines:

- **Real historical statistics**: 42 express trains on the
  Guwahati↔Delhi/Mumbai/Chennai/Kolkata corridor, with real per-station
  average delay and real delay-bucket probabilities (on-time / slight /
  significant / cancelled), sourced from a public Indian Railways delay
  dataset (Mar 2023–Mar 2024).
- **A simulated real-time layer on top of that real data**: for each train
  we Monte-Carlo sample 60 synthetic "runs", anchored to the real
  station-level delay distribution, and perturbed by weather and congestion
  — the exact real-world causal factors named in the problem statement.
  This is what lets the demo show *dynamic* re-forecasting rather than a
  static, one-shot number.

This is clearly flagged, not hidden: `data/build_dataset.py` documents
every assumption. Swapping in real live GPS/NTES/CRIS feeds later requires
no change to the model or API — only a different data source feeding the
same feature pipeline (see Section 5).

## 3. The model

A PyTorch **LSTM sequence regressor** (`models/model.py`):
- Input per station: distance covered/remaining, historical average delay
  at that station, live delay observed so far, time-of-day & day-of-week
  (cyclically encoded), weather, congestion.
- Output: predicted delay (→ ETA) at every station, computed fresh from
  whatever the live "as of now" conditions are — this is the "continuously
  refine its predictions" requirement from the problem statement.

**Result on held-out validation trains:**

| | MAE (minutes) |
|---|---|
| Static schedule baseline (today's practice — real historical average delay) | ~35 min |
| Our LSTM model | ~2.6 min |
| **Improvement** | **~92%** |

(Numbers regenerate each run in `models/eval_report.json` — small run-to-run
variance is expected since training uses random initialization.)

## 4. How to run it

Backend environment variables (keep these in `.env`; never put provider keys in
dashboard files):

- `RAILRADAR_API_KEY` is required for live train status and route data.
- `VISUAL_CROSSING_API_KEY` is optional; weather context is unavailable without it.
- `RAILRADAR_POLL_INTERVAL_SECONDS` is optional and defaults to `60`. Values are
  bounded to 30-3600 seconds and are served to the dashboard by `/api/health`.

```bash
# from the railway_eta_prototype/ folder
pip install -r requirements.txt

# 1) (already done, but to regenerate) rebuild the dataset
python3 data/build_dataset.py

# 2) (already done, but to retrain) train the model
python3 models/train.py

# 3) start the API
uvicorn api.main:app --reload --port 8001

# 4) open the dashboard
# just open dashboard/index.html directly in a browser
# (it calls http://localhost:8001 — keep the API running)
```

Try it live: pick a train, move the "current station" slider, then flip
weather to **Storm** + congestion to **High** and hit **Push Live Event** —
watch every downstream station's predicted ETA jump, while the static
baseline column stays exactly the same. That contrast *is* the pitch.

API docs (interactive): `http://localhost:8001/docs`

## Deploying the dashboard with Vercel

The GitHub Actions workflow in `.github/workflows/vercel-deploy.yml` deploys
the static dashboard to Vercel on every push to `main` and creates a Preview
deployment for pull requests targeting `main`.

1. Import this GitHub repository into Vercel and create the project. The
   included `vercel.json` serves `dashboard/` as the site output.
2. Add these GitHub Actions repository secrets from the Vercel project:
   `VERCEL_TOKEN`, `VERCEL_ORG_ID`, and `VERCEL_PROJECT_ID`.
3. Host the FastAPI service separately (for example, on a Python-capable
   service), then set `window.RAILCONNECT_API_URL` in `dashboard/config.js` to
   that service's public base URL. The current API uses PyTorch, which is not
   a practical fit for Vercel's serverless function package limits. The API
   permits cross-origin requests; keep provider keys in the backend's
   environment, never in dashboard files.
4. Push or merge to `main`. GitHub Actions builds and promotes the dashboard
   deployment to production automatically.

Pull request previews use the same configured API URL in `dashboard/config.js`.

## 5. Path from this prototype → full production system

This prototype is intentionally scoped to ~40 trains on one corridor so it
runs instantly and demos cleanly. Nothing in the architecture is
throwaway — here's what changes at each stage:

| Stage | What changes | What stays the same |
|---|---|---|
| **Pilot (1-2 real routes)** | Real GPS/NTES/CRIS live feed replaces the synthetic event generator; retrain on real live-tagged data | Model architecture, API contract, dashboard |
| **Zone rollout** | Add a message queue (Kafka) so live events stream in instead of being POSTed manually; move in-memory `LIVE_RUNS` state to Redis | Feature schema, LSTM core |
| **National scale** | Horizontally scale FastAPI behind a load balancer; TimescaleDB/PostgreSQL for historical + live data; per-zone model fine-tuning for regional variation | API endpoint design (`/simulate`, `/eta`) |
| **Full production** | Auth & role-based access, mobile app / station-display integration via the same REST API, monitoring/alerting, compliance & security audit | Core prediction pipeline |

The API is already designed the way the problem statement asks for:
`/trains/{id}/simulate` and `/trains/{id}/eta` are the exact integration
points a mobile app, station display, or control-room dashboard would call.

## 6. Known limitations (be upfront about these with judges)

- Inter-station distances are **approximated** (real spacing wasn't in the
  source dataset) — flagged in code, trivial to replace with real distance
  data from CRIS/GIS station master data.
- "Live" events are simulated, not pulled from an actual moving train —
  standard and expected for a hackathon prototype; the real-data-grounding
  strategy above is what makes this defensible rather than pure fabrication.
- Scoped to one corridor (42 trains) for demo speed; the model and API have
  no train-count ceiling — scaling is a data/infra question, not an
  architecture rewrite.
