# Bayesian HMM Pipeline Walkthrough

This guide explains how to use the newly implemented Bayesian HMM pipeline for macro regime detection.

## 1. Overview
The pipeline consists of two main components:
1.  **Offline Training (`scripts/train_hmm.py`)**: Periodically trains the Bayesian HMM using PyMC, identifies regimes, and saves parameters to `models/hmm_v1.npz`.
2.  **Daily Inference (`scripts/pipeline_daily.py`)**: Loads the saved model, runs forward-backward inference on daily data, and saves regime scores/probs to Supabase.

## 2. Prerequisites
You need `jax`, `jaxlib`, and `numpyro` installed for the training script.
```bash
uv pip install jax jaxlib numpyro
```

## 3. Training the Model
Run the training script to generate `models/hmm_v1.npz`. This script:
- Fetches data from FRED (via `pipeline_daily`).
- Constructs features (Growth, Inflation, Stress, etc.).
- Fits a 4-state HMM using NUTS sampling.
- automatically labels states (Goldilocks, Reflation, Stagflation, Recession) based on feature means.
- Saves model parameters, scaling factors, and score weights.

```bash
python -m scripts.train_hmm --start 2000-01-01 --draws 1000 --tune 500
```
*Note: This process may take several minutes depending on your CPU/GPU.*

## 4. Daily Inference
The daily pipeline `scripts/pipeline_daily.py` now automatically checks for `models/hmm_v1.npz`.
If the model file exists, it will:
- Load the model parameters.
- Transform the daily data using the *same* scaling factors used in training.
- Calculate regime probabilities (`prob_regime_0`...`3`) and `hmm_score` (0-100).
- Save these new columns to Supabase.

Run as usual:
```bash
python scripts/pipeline_daily.py --save-supabase
```

## 5. Feature Engineering
The features used for the HMM are defined in `make_hmm_features` in `scripts/pipeline_daily.py`:
- `cpi_gap`: Core CPI YoY - Median(36m)
- `orders_yoy`: Mfg New Orders YoY
- `orders_gap`: Orders MoM - Median(12m)
- `unrate_chg`: Unemployment Rate Change (3m)
- `yc_10y2y`: 10Y-2Y Spread
- `hy_oas`: High Yield OAS
- `vix`: VIX
- `stress_score`: Internal Stress Score
