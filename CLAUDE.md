# CLAUDE.md — MBML 2026 project context

## What this is
DTU course 42186 (Model-Based Machine Learning), Spring 2026. Deadline **15 May 2026**.
Bayesian Gaussian Mixture Model over song audio features + per-user Dirichlet taste profiles.
Deliverables: `mood_model.ipynb` (all phases) + 6-page IEEE paper.

## Repo layout
```
MBML---2026/
├── phase0_data.ipynb       # ✓ DONE — data pipeline
├── PROJECT_PLAN.md         # authoritative spec, phases, PGM
├── course_content/         # lectures, solutions, exercise notebooks
└── data/                   # local only, not committed
    ├── songs_clean.csv     # produced by phase0; one row per song, 6 scaled features
    └── listens_clean.csv   # produced by phase0; (user, song, play_count, listened)
```
External raw data (sibling of repo clone):
- Last.fm 1K: `../lastfm-dataset-1K/`
- MSD summary: `../msd_summary_file.h5` (single HDF5, full 1M-song MSD)

## Data
| Dataset | Content | Size | Join key |
|---|---|---|---|
| Last.fm 1K | ~19M scrobbles, ~1000 users | 1.3 GB TSV | `(artist_norm, track_norm)` |
| MSD summary file | 1M songs, single HDF5, vectorised load | ~300 MB | same |
| songs_clean.csv | matched corpus, MinMax-scaled features | TBD rows | — |
| listens_clean.csv | pos+neg listen events | — | — |

Features: `loudness, tempo, key, mode, time_signature, duration` (all scaled to [0,1]).
Note: `energy` and `danceability` are all-zero in the summary file (only in per-track HDF5s) — replaced by `key` (chromatic 0–11) and `duration` (seconds).

## PGM (short form)
```
μ_k ~ Normal(μ₀, κ₀⁻¹·Σ_k)     Σ_k ~ InverseWishart(ν, Ψ)     [K moods]
θ_u ~ Dirichlet(α)                                                [U users]
z_s ~ Categorical(π)             x_s ~ MVN(μ_{z_s}, Σ_{z_s})    [S songs]
l_us ~ Bernoulli(σ(θ_u · e_{z_s}))                               [U×S listens]
```
Observed: `x_s` (audio features), `l_us` (listen events). Latent: `z_s, θ_u, μ_k, Σ_k`.

## Phases
| Phase | Status | Notebook | Inputs | Gate |
|---|---|---|---|---|
| 0 — data pipeline | ✓ DONE | `phase0_data.ipynb` | raw TSVs + HDF5 | `songs_clean.csv`, `listens_clean.csv` exist |
| 1 — baseline GMM | TODO | `mood_model.ipynb` | `songs_clean.csv` | ELBO converged, K moods not collapsed |
| 2 — NUTS comparison | TODO | same | same | R-hat < 1.05, SVI≈NUTS posteriors |
| 3 — user extension | TODO | same | both CSVs | θ_u posteriors interpretable |
| 4 — PPC | TODO | same | both CSVs | predictive histograms overlap real |
| 5 — IEEE paper | TODO | LaTeX | results | 6 pages, double-column |
| 6 — notebook polish | TODO | same | — | every cell has markdown explanation |

## Pyro patterns (canonical — match these exactly)

**SVI for GMM (Phase 1):**
```python
guide = AutoDiagonalNormal(pyro.poutine.block(model, hide=["z"]))
svi = SVI(model, guide, Adam({"lr": 1e-2}), TraceEnum_ELBO(max_plate_nesting=1))
```

**NUTS for GMM (Phase 2):** marginalise discrete z first:
```python
collapsed_model = infer_discrete(pyro.poutine.config_enumerate(model), first_available_dim=-2)
mcmc = MCMC(NUTS(collapsed_model), num_samples=500, warmup_steps=200, num_chains=2)
```

**Posterior predictive:**
```python
predictive = Predictive(model, posterior_samples=posterior_samples)
```

## Style conventions (must match course exercises)
```python
# Imports — always in this order
import numpy as np
import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
np.random.seed(42)
plt.style.use('ggplot')
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 8)
```
- Every code cell preceded by 2–5 sentence markdown explanation.
- LaTeX inline `$\theta_u$` and display `\begin{align}…\end{align}`.
- `##` major sections, `###` sub-sections.

## Curriculum scope — hard rules
**In scope (weeks 1–12):** Bayesian GMM, Dirichlet priors, plates, SVI (`TraceEnum_ELBO`, `AutoDiagonalNormal`), NUTS/MCMC, R-hat, ELBO, `Predictive`, hierarchical models, LDA-style plate structure.

**Never suggest:** normalising flows, diffusion, transformers, attention, GNNs, RL, Stan, PyMC, NumPyro. If tempted: write *"out of scope"* and stop.

## Key decisions (non-obvious — do not re-litigate)
- **Spotify deprecated** `/audio-features` on 27 Nov 2024 → switched to MSD + Last.fm.
- **6 features only** — `loudness, tempo, key, mode, time_signature, duration`. `energy`/`danceability` are all-zero in the summary file; `key` and `duration` are good substitutes. `hotttnesss` and segment pitches add noise.
- **5:1 negative sampling** — standard for implicit-feedback Bernoulli likelihoods.
- **Active-user filter ≥5** — below 5, θ_u posterior is dominated by prior.
- **K=6 moods** — sweep {4,6,8} in Phase 1, pick most interpretable.
- **MinMax not z-score** — `mode` and `time_signature` are categorical-ish; z-score distorts them.

## Reference notebooks (read before writing model code)
- `course_content/all_solutions_merged.ipynb` — all course solutions in one file.
  Priority: Week 8 (LDA/Pyro), Week 9 (MCMC), Week 10 (VI + Bayesian GMM).
