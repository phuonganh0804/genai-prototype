# 🔍 GenAI Sentiment Analysis Dashboard

A small Streamlit app that classifies customer product reviews as **Positive /
Negative / Neutral** using the Anthropic API, then charts the sentiment
breakdown per product.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env          # then edit .env and add your real key
```

Get a key from [console.anthropic.com](https://console.anthropic.com):

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
.venv/bin/python -m streamlit run streamlit-app-customer-reviews.py
```

Click **Load Dataset**, then **Analyze Sentiment**, then filter by product.

## Data

`data/customer_reviews.csv` — 100 sample product reviews (`PRODUCT`, `DATE`,
`SUMMARY`, `SENTIMENT_SCORE`, `Order ID`). The app classifies the `SUMMARY`
text; `SENTIMENT_SCORE` is a pre-computed score it doesn't use.

## Notes

- Each analyzed review is one Claude API call (`claude-opus-5`); results are
  cached with `@st.cache_data` so re-running costs nothing.
- Model output is normalized to exactly `Positive` / `Negative` / `Neutral`
  before charting.
