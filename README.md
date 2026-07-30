# Causal Inference Dashboard

Interactive dashboard that applies propensity score matching and uplift modeling to the [Criteo Uplift dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) - a 14M-row advertising incrementality trial. Generates plain-English business recommendations via LLM.

A rigorous causal analysis goes beyond a simple "it converted higher, ship it" read and accounts for selection bias and heterogeneous effects. This project demonstrates what a proper causal pipeline looks like end-to-end.

## How it works

1. **Data** - Downloads the full Criteo Uplift Modeling Dataset (Diemert et al., 2018) - ~13.9M rows - and caches it as parquet. Treatment = ad bid placed, control = bid withheld, outcome = site visit.

2. **Propensity Score Matching** - Logistic regression estimates each user's probability of being treated, then 1:1 nearest-neighbor matching (caliper=0.05) creates balanced groups. SMD diagnostics confirm covariate balance before/after.

3. **ATE estimation** - Two-sample t-test on matched pairs gives the causal average treatment effect with 95% CI.

4. **T-Learner uplift** - Separate HistGBM models for treated/control predict P(visit|X). 80/20 train/test split, segmentation reported on held-out test set. The difference gives per-user CATE, segmented into quartiles for targeting decisions.

5. **Qini curve evaluation** - Out-of-sample evaluation of the uplift model's ranking ability. Qini coefficient measures area between model curve and random baseline, normalized by the perfect oracle.

6. **LLM summary** - Ships the stats to Claude/GPT/local LLM to generate a business-friendly interpretation.

## Results on Criteo data

- **PSM-adjusted ATE: +0.82pp** (95% CI: +0.80 to +0.83, p < 0.001)
- 11.9M matched pairs, zero dropped (excellent propensity overlap)
- Q4 users (best responders): +3.01pp uplift on test set
- Q1 users ("sleeping dogs"): -0.16pp - ads slightly *decrease* their visit rate
- **Qini coefficient: 0.088** (out-of-sample) - meaningfully better than random; typical for Criteo where ATE is small
- Naive ATE (+1.03pp) slightly overestimates the causal effect

The interesting finding is the heterogeneity. The overall effect is under 1pp, but Q4 responds at 3x the average while Q1 responds negatively. Ad spend optimization lives in this segmentation - stop wasting impressions on Q1 and concentrate on Q4.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add an API key if you want the LLM summary (the causal analysis works without any key):

```bash
cp .env.example .env
# Add ANTHROPIC_API_KEY or OPENAI_API_KEY
```

Run it:

```bash
cd app
uvicorn main:app --reload
```

First run downloads the dataset (~300MB), caches it as parquet, and runs the full analysis on all 14M rows. This takes ~8 minutes (logistic regression + PSM + GBM uplift on 14M rows). Results are cached in memory after that so the dashboard is instant on subsequent page loads. Restarting the server re-runs analysis but skips the download.

## LLM provider

Auto-detects based on what's in `.env`:
- **Anthropic** - if `ANTHROPIC_API_KEY` is set
- **OpenAI** - if `OPENAI_API_KEY` is set
- **Jan AI** - fallback, hits local server at `127.0.0.1:1337` (free, no key needed)

## Stack

Python (FastAPI, scikit-learn, scipy, pandas) + vanilla JS frontend with Chart.js. No React, no build step. The whole thing is ~500 lines of Python and a single HTML file.

## Project layout

```
app/
  main.py            FastAPI routes + static file serving
  data_loader.py     Criteo download + parquet caching
  causal_engine.py   PSM pipeline + T-Learner uplift model
  llm_engine.py      Multi-provider LLM integration
frontend/
  index.html         Dashboard UI (Chart.js)
```

## References

- Diemert, E. et al. (2018). "A Large Scale Benchmark for Uplift Modeling." *AdKDD Workshop*.
- Dataset: [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) (CC BY-NC-SA 4.0)
