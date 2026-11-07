# Customer churn & subscriptions — clean Streamlit dashboard

This version removes the charge-band notes, exact-number tables and presentation/calc notes. It retains interactive charts, plan selector, KPI cards, insights and the updated header. No CSV or analytics.py is needed; it uses exact anonymised customer aggregates from the original dataset, unless an optional source CSV is supplied.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Upload **app.py**, **requirements.txt**, and **.streamlit/config.toml** to the repository root and select **app.py** as the main file. Commit changes to the deployed branch.

Note: the original CSV did not specify charge currency. Churn is descriptive rather than causal.
