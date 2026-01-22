from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az

# ---------------------------------------------------------------------
# Ensure we can import from project root
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.pipeline_daily import build_daily_state, FredClient, env, make_hmm_features


# -------------------------------------------------------------------------
# NumPyro HMM (states marginalized via forward algorithm)
# -------------------------------------------------------------------------
def hmm_forward_logp(emit: jnp.ndarray, logpi: jnp.ndarray, logA: jnp.ndarray) -> jnp.ndarray:
    alpha0 = logpi + emit[0]
    if emit.shape[0] == 1:
        return logsumexp(alpha0, axis=0)

    def step(alpha_prev, emit_t):
        alpha_t = emit_t + logsumexp(alpha_prev[:, None] + logA, axis=0)
        return alpha_t, alpha_t

    alpha_last, _ = lax.scan(step, alpha0, emit[1:])
    return logsumexp(alpha_last, axis=0)


def hmm_model(X: jnp.ndarray, K: int, sticky_strength: float = 25.0):
    """
    Gaussian HMM with correlated emissions per state.
    Hidden states are marginalized via forward algorithm.
    """
    T, D = X.shape
    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(K)))
    alpha_base = jnp.ones((K, K)) + sticky_strength * jnp.eye(K)
    A = numpyro.sample("A", dist.Dirichlet(alpha_base))

    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0).expand((K, D)))
    sd = numpyro.sample("sd", dist.Exponential(1.0).expand((K, D)))
    corr_chol = numpyro.sample(
        "corr_chol",
        dist.LKJCholesky(dimension=D, concentration=2.0),
        sample_shape=(K,),
    )
    chol = corr_chol * sd[..., None]

    mvn = dist.MultivariateNormal(loc=mu, scale_tril=chol)
    emit = jax.vmap(lambda x: mvn.log_prob(x))(X)  # (T,K)

    logA = jnp.log(A + 1e-30)
    logpi = jnp.log(pi + 1e-30)
    ll = hmm_forward_logp(emit, logpi, logA)
    numpyro.factor("hmm_ll", ll)


def fit_bayesian_hmm(
    X: np.ndarray,
    K: int = 4,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    sticky_strength: float = 25.0,
    seed: int = 42,
):
    X = jnp.asarray(X, dtype=jnp.float64)
    numpyro.set_host_device_count(chains)

    kernel = NUTS(hmm_model)
    mcmc = MCMC(
        kernel,
        num_warmup=tune,
        num_samples=draws,
        num_chains=chains,
        chain_method="sequential",
        progress_bar=True,
    )
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, X=X, K=K, sticky_strength=sticky_strength)
    return az.from_numpyro(mcmc)


# -------------------------------------------------------------------------
# Utilities: validation / resample / scoring target
# -------------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"[train_hmm] Required feature columns missing in {where}: {missing}\n"
            f"  - Available columns: {list(df.columns)}\n"
            f"  - Fix: update make_hmm_features(state) to output these, or adjust feature rules."
        )


def _resample_features(
    X_df: pd.DataFrame,
    freq: str,
    weekly_anchor: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    freq:
      - 'B' (business daily): no resample
      - 'D' (calendar daily): reindex to D + ffill
      - 'W' (weekly): resample to 'W-<anchor>' last + ffill
    """
    freq = freq.upper()
    meta: Dict[str, str] = {"freq": freq}

    if freq == "B":
        return X_df.copy(), meta

    if freq == "D":
        idx = pd.date_range(X_df.index.min(), X_df.index.max(), freq="D")
        out = X_df.reindex(idx).ffill()
        return out, meta

    if freq == "W":
        anchor = weekly_anchor.upper()
        rule = f"W-{anchor}"
        meta["weekly_anchor"] = anchor
        out = X_df.resample(rule).last().ffill()
        return out, meta

    raise ValueError(f"Unsupported freq={freq}. Use B, D, or W.")


def _min_obs_from_years(freq: str, min_years: float) -> int:
    freq = freq.upper()
    if freq == "B":
        per_year = 252
    elif freq == "D":
        per_year = 365
    elif freq == "W":
        per_year = 52
    else:
        raise ValueError(f"Unsupported freq={freq}. Use B, D, or W.")
    return int(np.ceil(float(min_years) * per_year))


def _fetch_spx_close(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    Use yfinance via pipeline_daily's YF fetch inside build_daily_state.
    We re-fetch here to keep train_hmm self-contained.
    """
    try:
        import yfinance as yf  # type: ignore
    except Exception as ex:
        raise RuntimeError(f"[train_hmm] yfinance required for scoring target: {ex}")

    df = yf.download(
        "^GSPC",
        start=str(start.date()),
        end=str((end + pd.Timedelta(days=1)).date()),
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError("[train_hmm] yfinance ^GSPC fetch failed (empty).")

    if isinstance(df.columns, pd.MultiIndex):
        s = df["Close"].iloc[:, 0]
    else:
        s = df["Close"]
    s = s.dropna()
    s.name = "spx_close"
    return s


def _align_spx_to_freq(spx_close: pd.Series, idx: pd.DatetimeIndex, freq: str, weekly_anchor: str) -> pd.Series:
    """
    Make SPX series on the SAME index as features after resample.
    - B: business-day index
    - D: calendar-day index
    - W: weekly index (W-ANCHOR)
    """
    freq = freq.upper()

    if freq == "B":
        s = spx_close.reindex(pd.date_range(spx_close.index.min(), idx.max(), freq="D")).ffill()
        s = s.reindex(idx, method="ffill")
        return s

    if freq == "D":
        s = spx_close.reindex(pd.date_range(spx_close.index.min(), idx.max(), freq="D")).ffill()
        s = s.reindex(idx, method="ffill")
        return s

    if freq == "W":
        rule = f"W-{weekly_anchor.upper()}"
        s = spx_close.reindex(pd.date_range(spx_close.index.min(), idx.max(), freq="D")).ffill()
        s = s.resample(rule).last().ffill()
        s = s.reindex(idx, method="ffill")
        return s

    raise ValueError(f"Unsupported freq={freq}.")


def _forward_log_return(spx: pd.Series, horizon: int) -> pd.Series:
    """
    y_t = log(SPX_{t+h} / SPX_t)
    """
    s = spx.astype(float)
    y = np.log(s.shift(-horizon) / s)
    y.name = f"fwd_logret_h{horizon}"
    return y


def _align_state_to_freq(
    state: pd.DataFrame, idx: pd.DatetimeIndex, freq: str, weekly_anchor: str
) -> pd.DataFrame:
    """
    Align state (daily business) series to the same index as features after resample.
    """
    freq = freq.upper()
    df = state.copy()

    if freq in ("B", "D"):
        daily_idx = pd.date_range(df.index.min(), idx.max(), freq="D")
        df = df.reindex(daily_idx).ffill()
        df = df.reindex(idx, method="ffill")
        return df

    if freq == "W":
        rule = f"W-{weekly_anchor.upper()}"
        daily_idx = pd.date_range(df.index.min(), idx.max(), freq="D")
        df = df.reindex(daily_idx).ffill()
        df = df.resample(rule).last().ffill()
        df = df.reindex(idx, method="ffill")
        return df

    raise ValueError(f"Unsupported freq={freq}.")


def _rolling_zscore(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=0)
    return (s - mu) / (sd + 1e-6)


def _risk_off_target(
    state: pd.DataFrame,
    idx: pd.DatetimeIndex,
    freq: str,
    weekly_anchor: str,
    window_years: float,
    min_periods_ratio: float,
) -> pd.Series:
    """
    Build a risk-off target so higher values indicate lower stress.
    Target = -mean(zscore(vix), zscore(hy_oas), zscore(stress_score))
    """
    cols = ["vix", "hy_oas", "stress_score"]
    missing = [c for c in cols if c not in state.columns]
    if missing:
        raise RuntimeError(f"[train_hmm] missing state columns for risk target: {missing}")

    aligned = _align_state_to_freq(state[cols], idx, freq=freq, weekly_anchor=weekly_anchor)
    aligned = aligned.apply(pd.to_numeric, errors="coerce")

    freq_u = freq.upper()
    if freq_u == "W":
        win = int(np.ceil(window_years * 52))
    elif freq_u == "D":
        win = int(np.ceil(window_years * 365))
    else:
        win = int(np.ceil(window_years * 252))
    minp = max(30, int(np.ceil(win * min_periods_ratio)))

    z_vix = _rolling_zscore(aligned["vix"], win, minp)
    z_hy = _rolling_zscore(aligned["hy_oas"], win, minp)
    z_stress = _rolling_zscore(aligned["stress_score"], win, minp)
    risk_off = -(z_vix + z_hy + z_stress) / 3.0
    risk_off.name = "risk_off_target"
    return risk_off


def _ridge_beta(Gamma: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    beta = (G'G + lam I)^-1 G'y
    Gamma: (T, K), y: (T,)
    """
    K = Gamma.shape[1]
    GTG = Gamma.T @ Gamma
    rhs = Gamma.T @ y
    beta = np.linalg.solve(GTG + lam * np.eye(K), rhs)
    return beta


# -------------------------------------------------------------------------
# Forward-Backward on numpy (posterior mean params) to get gamma
# -------------------------------------------------------------------------
def _logsumexp_np(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def _log_mvn_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """
    log N(x | mu, cov) for dense cov
    """
    D = x.shape[0]
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(D))
    diff = x - mu
    sol = np.linalg.solve(L, diff)
    quad = sol @ sol
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return float(-0.5 * (D * np.log(2.0 * np.pi) + logdet + quad))


def _emission_loglik(X: np.ndarray, mus: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """
    X: (T, D)
    mus: (K, D)
    covs: (K, D, D)
    return emit: (T, K)
    """
    T, D = X.shape
    K = mus.shape[0]
    emit = np.zeros((T, K), dtype=float)
    for k in range(K):
        for t in range(T):
            emit[t, k] = _log_mvn_pdf(X[t], mus[k], covs[k])
    return emit


def forward_backward_gamma(pi: np.ndarray, A: np.ndarray, emit: np.ndarray) -> np.ndarray:
    """
    Return smoothed gamma[t,k] = P(z_t=k | x_1:T) using log-domain FB.
    pi: (K,), A: (K,K), emit: (T,K) log p(x_t|k)
    """
    T, K = emit.shape
    logpi = np.log(pi + 1e-30)
    logA = np.log(A + 1e-30)

    logalpha = np.zeros((T, K), dtype=float)
    logbeta = np.zeros((T, K), dtype=float)

    logalpha[0] = logpi + emit[0]
    for t in range(1, T):
        tmp = logalpha[t - 1][:, None] + logA
        logalpha[t] = emit[t] + _logsumexp_np(tmp, axis=0)

    logbeta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        tmp = logA + emit[t + 1][None, :] + logbeta[t + 1][None, :]
        logbeta[t] = _logsumexp_np(tmp, axis=1)

    loggamma = logalpha + logbeta
    loggamma = loggamma - _logsumexp_np(loggamma, axis=1)[:, None]
    gamma = np.exp(loggamma)
    return gamma


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2000-01-01")
    p.add_argument("--end", type=str, default=None, help="default: today(UTC)")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=500)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sticky", type=float, default=25.0)

    p.add_argument("--freq", type=str, default="B", choices=["B", "D", "W"])
    p.add_argument("--weekly-anchor", type=str, default="FRI", choices=["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"])

    p.add_argument("--min-years", type=float, default=5.0)

    p.add_argument(
        "--score-target",
        type=str,
        default="risk_off",
        choices=["risk_off", "forward_return"],
        help="score target: risk_off (default) or forward_return",
    )
    p.add_argument("--score-z-window-years", type=float, default=5.0, help="rolling zscore window for risk_off target")
    p.add_argument("--score-z-min-periods-ratio", type=float, default=0.4, help="min periods ratio for rolling zscore")
    p.add_argument("--score-horizon", type=int, default=None, help="override horizon; default: W->13, else->63")
    p.add_argument("--score-lambda", type=float, default=1e-3, help="ridge lambda for beta_state")
    p.add_argument("--score-window-years", type=float, default=5.0, help="for diagnostics scaling window")

    p.add_argument("--model-path", type=str, default=None, help="default: models/hmm_v1.npz")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> int:
    try:
        import dotenv  # type: ignore
        dotenv.load_dotenv()
    except Exception:
        pass

    args = parse_args()

    fred_key = env("FRED_API_KEY")
    if not fred_key:
        print("FRED_API_KEY required", file=sys.stderr)
        return 1

    fred = FredClient(api_key=fred_key)

    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end) if args.end else pd.Timestamp.now(tz="UTC").normalize()
    start_date = start_date.tz_localize(None) if getattr(start_date, "tzinfo", None) else start_date
    end_date = end_date.tz_localize(None) if getattr(end_date, "tzinfo", None) else end_date

    print(f"[train_hmm] Fetching daily state from {start_date.date()} to {end_date.date()} ...")
    state = build_daily_state(fred, start_date, end_date)

    print("[train_hmm] Building features via make_hmm_features(state) ...")
    X_df = make_hmm_features(state)
    if not isinstance(X_df, pd.DataFrame) or X_df.empty:
        raise RuntimeError("[train_hmm] make_hmm_features returned empty dataframe.")

    if not isinstance(X_df.index, pd.DatetimeIndex):
        if "date" in X_df.columns:
            X_df = X_df.copy()
            X_df.index = pd.to_datetime(X_df["date"])
        else:
            raise RuntimeError("[train_hmm] make_hmm_features must return DataFrame with DatetimeIndex (or 'date' column).")

    X_df = X_df.sort_index()

    X_rs, meta_freq = _resample_features(X_df, freq=args.freq, weekly_anchor=args.weekly_anchor)

    min_obs = _min_obs_from_years(args.freq, args.min_years)
    if len(X_rs) < min_obs:
        raise RuntimeError(
            f"[train_hmm] Not enough observations for training.\n"
            f"  - freq={args.freq}, min_years={args.min_years}\n"
            f"  - need >= {min_obs} obs, got {len(X_rs)}\n"
            f"  - Fix: earlier --start or reduce --min-years"
        )

    X_clean = X_rs.ffill().dropna()
    if len(X_clean) < min_obs:
        raise RuntimeError(
            f"[train_hmm] Not enough non-NaN observations after cleaning.\n"
            f"  - need >= {min_obs} obs, got {len(X_clean)}"
        )

    feature_cols = list(map(str, X_clean.columns))
    _require_columns(X_clean, ["cpi_gap", "orders_yoy"], where="make_hmm_features output")

    X_val = X_clean.values.astype(float)

    mean = np.mean(X_val, axis=0)
    std = np.std(X_val, axis=0)
    X_scaled = (X_val - mean) / (std + 1e-6)

    print(f"[train_hmm] Dataset shape: {X_scaled.shape} (T, D)")
    print("[train_hmm] Starting HMM Sampling (NumPyro NUTS) ...")
    trace = fit_bayesian_hmm(
        X_scaled,
        K=args.k,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        sticky_strength=float(args.sticky),
        seed=int(args.seed),
    )
    print("[train_hmm] Sampling complete. Building continuous score ...")

    post = trace.posterior

    pi_mean = post["pi"].mean(dim=("chain", "draw")).values
    A_mean = post["A"].mean(dim=("chain", "draw")).values
    mus_mean = post["mu"].mean(dim=("chain", "draw")).values

    corr_chol = post["corr_chol"].values
    sd_post = post["sd"].values
    chol_mean = (corr_chol * sd_post[..., None]).mean(axis=(0, 1))
    cov_mean = np.zeros((args.k, X_scaled.shape[1], X_scaled.shape[1]))
    for k in range(args.k):
        cov_mean[k] = chol_mean[k] @ chol_mean[k].T

    emit = _emission_loglik(X_scaled, mus_mean, cov_mean)
    gamma = forward_backward_gamma(pi_mean, A_mean, emit)

    default_h = 13 if args.freq.upper() == "W" else 63
    horizon = int(args.score_horizon) if args.score_horizon is not None else default_h

    score_target = str(args.score_target).lower()
    if score_target == "forward_return":
        spx_close = _fetch_spx_close(start_date, end_date)
        spx_aligned = _align_spx_to_freq(spx_close, X_clean.index, args.freq, args.weekly_anchor)
        y = _forward_log_return(spx_aligned, horizon=horizon)
        score_mode = "B_continuous_expected_return"
    else:
        y = _risk_off_target(
            state,
            X_clean.index,
            args.freq,
            args.weekly_anchor,
            window_years=float(args.score_z_window_years),
            min_periods_ratio=float(args.score_z_min_periods_ratio),
        )
        score_mode = "B_continuous_risk_off"

    valid = y.notna().values
    Gamma = gamma[valid, :]
    yv = y.values[valid].astype(float)

    if len(yv) < max(200, args.k * 50):
        raise RuntimeError(
            f"[train_hmm] Not enough target observations for beta fit after target alignment.\n"
            f"  - target={score_target}, remaining={len(yv)}"
        )

    lam = float(args.score_lambda)
    beta_state = _ridge_beta(Gamma, yv, lam=lam)

    raw_score = (gamma @ beta_state).astype(float)
    raw_s = pd.Series(raw_score, index=X_clean.index, name="raw_score")

    freq = args.freq.upper()
    if freq == "W":
        win = int(np.ceil(args.score_window_years * 52))
    elif freq == "D":
        win = int(np.ceil(args.score_window_years * 365))
    else:
        win = int(np.ceil(args.score_window_years * 252))

    def _rolling_percentile(s: pd.Series, window: int) -> pd.Series:
        def pct_rank(a: np.ndarray) -> float:
            x = a[-1]
            return float(100.0 * (np.sum(a <= x) / len(a)))
        return s.rolling(window, min_periods=max(50, window // 5)).apply(pct_rank, raw=True)

    hmm_score_diag = _rolling_percentile(raw_s, win).rename("hmm_score_diag")

    save_dir = os.path.join(project_root, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = args.model_path or os.path.join(save_dir, "hmm_v1.npz")

    meta = {
        "version": "hmm_v1",
        "trained_at_utc": pd.Timestamp.utcnow().replace(tzinfo=None).isoformat(),
        "freq": meta_freq.get("freq", args.freq),
        "weekly_anchor": meta_freq.get("weekly_anchor", ""),
        "min_years": float(args.min_years),
        "min_obs": int(min_obs),
        "feature_cols": feature_cols,
        "score_mode": score_mode,
        "score_horizon": int(horizon),
        "score_lambda": float(lam),
        "score_window_years": float(args.score_window_years),
        "score_target": score_target,
        "score_components": ["vix", "hy_oas", "stress_score"] if score_target == "risk_off" else ["spx"],
        "score_z_window_years": float(args.score_z_window_years),
        "score_z_min_periods_ratio": float(args.score_z_min_periods_ratio),
    }
    meta_json = json.dumps(meta, ensure_ascii=True)

    np.savez(
        save_path,
        pi=pi_mean.astype(float),
        A=A_mean.astype(float),
        mus=mus_mean.astype(float),
        covs=cov_mean.astype(float),
        mean=mean.astype(float),
        std=std.astype(float),
        beta_state=beta_state.astype(float),
        meta_feature_cols=np.array(feature_cols, dtype="U64"),
        meta_json=np.array([meta_json], dtype="U2048"),
    )

    print(f"[train_hmm] beta_state: {np.round(beta_state, 6)}")
    print(f"[train_hmm] Model saved to: {save_path}")

    if args.debug:
        print("[train_hmm] meta_json:", meta_json)
        tail = pd.concat([raw_s, hmm_score_diag, y], axis=1).tail(10)
        print("\n[train_hmm] tail diagnostics:")
        print(tail.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
