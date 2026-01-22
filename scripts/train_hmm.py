import argparse
import os
import sys
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

# Ensure we can import from scripts
# If running from root as 'python -m scripts.train_hmm', this is fine.
# If running as 'python scripts/train_hmm.py', we need to add current dir to path.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.pipeline_daily import build_daily_state, FredClient, env, make_hmm_features

# -------------------------------------------------------------------------
# NumPyro HMM code (states marginalized via forward algorithm)
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
    emit = jax.vmap(lambda x: mvn.log_prob(x))(X)

    logA = jnp.log(A + 1e-30)
    logpi = jnp.log(pi + 1e-30)
    ll = hmm_forward_logp(emit, logpi, logA)
    numpyro.factor("hmm_ll", ll)


def fit_bayesian_hmm(
    X: np.ndarray,
    K: int = 4,
    draws: int = 1000,
    tune: int = 1000,
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
    )
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, X=X, K=K, sticky_strength=sticky_strength)
    return az.from_numpyro(mcmc)

# -------------------------------------------------------------------------
# Main Logic
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2000-01-01")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    try:
        import dotenv
        dotenv.load_dotenv()
    except:
        pass

    fred_key = env("FRED_API_KEY")
    if not fred_key:
        print("FRED_API_KEY required")
        sys.exit(1)

    fred = FredClient(api_key=fred_key)
    
    start_date = pd.to_datetime(args.start)
    end_date = pd.Timestamp.now(tz="UTC").normalize()
    
    print(f"Fetching data from {start_date.date()} to {end_date.date()}...")
    state = build_daily_state(fred, start_date, end_date)
    
    print(f"Building features...")
    X_df = make_hmm_features(state)
    
    # Drop initial NaNs
    X_clean = X_df.ffill().dropna()
    print(f"Dataset shape: {X_clean.shape}")
    
    if len(X_clean) < 252:
        print("Error: Dataset too small for training")
        sys.exit(1)
        
    X_val = X_clean.values
    
    # Standardize
    mean = np.mean(X_val, axis=0)
    std = np.std(X_val, axis=0)
    X_scaled = (X_val - mean) / (std + 1e-6)
    
    print("Starting HMM Sampling (NumPyro NUTS)... This may take a while.")
    trace = fit_bayesian_hmm(X_scaled, K=args.k, draws=args.draws, tune=args.tune, chains=args.chains)
    
    print("Sampling complete. Analyzing regimes...")
    
    # Extract Means
    post = trace.posterior
    
    # (K, D)
    mus_mean = post["mu"].mean(dim=("chain", "draw")).values
    
    # Auto-Labeling Heuristic
    # X columns: cpi_gap, orders_yoy, orders_gap, unrate_chg, yc_10y2y, hy_oas, vix, stress_score
    # Index:     0        1           2           3           4         5       6    7
    
    # We care about: orders_yoy (Growth), cpi_gap (Inflation)
    # orders_yoy is col 1
    # cpi_gap is col 0
    
    # However, data matches make_hmm_features output order.
    # Check make_hmm_features columns:
    # "cpi_gap", "orders_yoy", "orders_gap", "unrate_chg", "yc_10y2y", "hy_oas", "vix", "stress_score"
    
    col_cpi = 0
    col_growth = 1
    
    # Identify each state k
    # regime_map = {k: "Label"}
    # heuristic scores
        
    regime_labels = {}
    regime_scores = np.zeros(args.k)
    
    print("\nRegime Analysis (Standardized Means):")
    print(f"{'K':<3} {'Growth(OrdYoY)':<15} {'Infl(CPIGap)':<15} {'Label':<15} {'Score'}")
    
    for k in range(args.k):
        g = mus_mean[k, col_growth]
        i = mus_mean[k, col_cpi]
        
        # Determine Label
        # Goldilocks: G > 0, I < 0
        # Reflation: G > 0, I > 0
        # Stagflation: G < 0, I > 0
        # Recession: G < 0, I < 0
        # (Thresholds are 0 because standardized)
        
        if g >= 0 and i < 0:
            lab = "Goldilocks"
            score = 100
        elif g >= 0 and i >= 0:
            lab = "Reflation"
            score = 70
        elif g < 0 and i >= 0:
            lab = "Stagflation"
            score = 20
        else:
            lab = "Recession"
            score = 30
            
        regime_labels[k] = lab
        regime_scores[k] = score
        print(f"{k:<3} {g:<15.4f} {i:<15.4f} {lab:<15} {score}")
        
    
    # Save Model
    save_dir = os.path.join(parent_dir, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "hmm_v1.npz")
    
    # Reconstruct Covariances
    chol_mean = post["chol"].mean(dim=("chain", "draw")).values # (K, D, D)
    cov_mean = np.zeros((args.k, X_val.shape[1], X_val.shape[1]))
    for k in range(args.k):
        cov_mean[k] = chol_mean[k] @ chol_mean[k].T
        
    np.savez(
        save_path,
        pi=post["pi"].mean(dim=("chain", "draw")).values,
        A=post["A"].mean(dim=("chain", "draw")).values,
        mus=mus_mean,
        covs=cov_mean,
        mean=mean,
        std=std,
        score_weights=regime_scores,
        regime_labels=regime_labels # Note: npz cannot save dict easily unless pickled, 
                                    # but we can rely on score_weights for scoring.
    )
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    main()
