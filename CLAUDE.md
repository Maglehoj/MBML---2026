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
- tagtraum genre annotations: `../msd_tagtraum_cd2.cls` (~280k songs, 15 genres, tab-separated, uses TR... track IDs)
- MSD track→song ID mapping: `../unique_tracks.txt` (1M rows, `trackID<SEP>songID<SEP>artist<SEP>title`, bridges TR→SO IDs)

## Data
| Dataset | Content | Size | Join key |
|---|---|---|---|
| Last.fm 1K | ~19M scrobbles, ~1000 users | 1.3 GB TSV | `(artist_norm, track_norm)` |
| MSD summary file | 1M songs, single HDF5, vectorised load | ~300 MB | same |
| tagtraum cd2 | majority-vote genre labels, ~280k songs, 15 genres | ~280k rows TSV | `song_id` |
| songs_clean.csv | matched corpus, MinMax-scaled features + `genre` column | 292k rows | — |
| listens_clean.csv | pos+neg listen events | 16.8M rows | — |

Features: `loudness, tempo, key, mode, time_signature, duration` (all scaled to [0,1]).
Note: `energy` and `danceability` are all-zero in the summary file (only in per-track HDF5s) — replaced by `key` (chromatic 0–11) and `duration` (seconds).

## PGM (short form)
```
π ~ Dirichlet(5·1_K)
μ_k ~ N(0,I_3)   σ_k ~ LogNormal(0,0.5²I_3)   θ^key_k ~ Dir(1_12)   [K moods]
θ^ts_k ~ Dir(1_6)   p_k ~ Beta(2,2)                                   [K moods]
θ_u ~ Dirichlet(α)                                                     [U users]
z_s ~ Categorical(π)                                                   [S songs]
x^cont_s|z_s ~ N(μ_{z_s}, diag(σ²_{z_s}))   x^key_s|z_s ~ Cat(θ^key_{z_s})
x^ts_s|z_s ~ Cat(θ^ts_{z_s})                 x^mode_s|z_s ~ Bern(p_{z_s})
l_us ~ Bernoulli(σ(θ_u · e_{z_s}))                                    [U×S listens]
```
Observed: `x^cont, x^key, x^ts, x^mode` (audio features), `l_us` (listen events).
Latent: `z_s, θ_u, μ_k, σ_k, θ^key_k, θ^ts_k, p_k`.

## Phases
| Phase | Status | Notebook | Inputs | Gate |
|---|---|---|---|---|
| 0 — data pipeline | ✓ DONE | `phase0_data.ipynb` | raw TSVs + HDF5 + tagtraum cls | `songs_clean.csv` (with `genre`), `listens_clean.csv` exist |
| 1 — baseline GMM | ✓ DONE (Anton) | `phase1_mood_model_w-genres.ipynb` | `songs_clean.csv` | Mixed-likelihood, K auto-selected via ΔELBO, SVI converged, 4-seed ensemble |
| 2 — NUTS comparison | TODO | same | same | R-hat < 1.05, SVI≈NUTS posteriors |
| 3 — user extension | TODO | same | both CSVs | θ_u posteriors interpretable |
| 4 — PPC | TODO | same | both CSVs | predictive histograms overlap real |
| 5 — IEEE paper | TODO | LaTeX | results | 6 pages, double-column |
| 6 — notebook polish | TODO | same | — | every cell has markdown explanation |

## Pyro patterns (canonical — match these exactly)

**SVI for mixed-likelihood GMM (Phase 1):**
```python
# Model: Gaussian for continuous, Categorical for discrete, Bernoulli for binary
# Discrete latent z: use infer_discrete + config_enumerate + TraceEnum_ELBO
guide = AutoDiagonalNormal(pyro.poutine.block(model, hide=["z"]), init_scale=0.05)
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

# Setup — use seed 67 everywhere
np.random.seed(67)
torch.manual_seed(67)       # Phase 1+ only
pyro.set_rng_seed(67)       # Phase 1+ only
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
- **K=10 via ΔELBO elbow** — K is auto-selected by sweeping K=2..15, fitting 500 SVI steps each, and picking the elbow of the per-component ELBO gain. Yielded K=10. Genre-stratified init was considered but abandoned: genres and moods are different constructs, and the mixed-likelihood loss landscape is well-conditioned without it. Plain multi-restart K-means (`n_init=20`) on the 3 z-scored continuous features is used instead.
- **tagtraum ID mismatch** — tagtraum uses MSD track IDs (TR...), summary HDF5 uses Echo Nest song IDs (SO...). Bridge via `unique_tracks.txt` (`trackID<SEP>songID<SEP>artist<SEP>title`). Coverage ~54% after bridge. Genre saved in `songs_clean.csv` for post-hoc interpretation only — NOT a model feature and NOT used for K-means init.
- **MinMax not z-score** — `mode` and `time_signature` are categorical-ish; z-score distorts them.

## Reference notebooks (read before writing model code)
- `course_content/all_solutions_merged.ipynb` — all course solutions in one file.
  Priority: Week 8 (LDA/Pyro), Week 9 (MCMC), Week 10 (VI + Bayesian GMM).
