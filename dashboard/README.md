# Executive Intelligence Dashboard (Streamlit)

Renders `results/dashboard_data.json` as the 7 PDF dashboard sections. The backend
shapes the data; this app is a thin renderer.

## Run
```bash
# 1. generate the data (needs the model server up)
python main.py ingest        # if you haven't collected/indexed yet
python main.py report        # writes results/dashboard_data.json

# 2. launch the dashboard
uv run streamlit run dashboard/app.py
```

## Sections
| Section | Source key in dashboard_data.json |
|---|---|
| 1 Company Overview | `company` |
| 2 Market Intelligence | `market_intelligence` |
| 3 Opportunity Monitor | `opportunities` |
| 4 Risk Monitor | `risks` |
| 5 Sentiment Analysis | `sentiment` (incl. `distribution`, `trend`) |
| 6 Strategic Recommendations | `recommendations` |
| 7 CEO Briefing | `briefing` |

The sidebar **Ask the agent** box calls the live tool-calling agent (`ask_ceo`),
so it needs the DataLab model server running.
