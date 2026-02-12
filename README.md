# Edwin Analysis Power

Interactive Streamlit analytics dashboard for CSV exploration, cleaning, visualization, AI analysis, anomaly detection, and forecasting.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 127.0.0.1 --server.port 3001
```

## Features

- Multi-file CSV upload (stack/merge)
- Data cleaning tools (missing values, outliers, type conversion, text cleanup)
- KPI cards and comparison metrics
- Plotly visualizations
- AI chat/report (Gemini API key optional)
- Anomaly detection (Isolation Forest)
- Trend forecasting with confidence bands
- Export: filtered CSV, JSON summary, HTML report, session JSON
