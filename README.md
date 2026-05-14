# Latent Mood & User-Taste Modelling with MSD + Last.fm

DTU course **42186 - Model-Based Machine Learning**, Spring 2026. Deadline: **15 May 2026**.

## What this project does

We build a Bayesian probabilistic model of music that jointly learns:

1. **K mood clusters** (Bayesian Gaussian Mixture) over six Echo Nest audio features from the Million Song Dataset summary file (1M songs).
2. **Per-user taste profiles** (Dirichlet distributions over moods) fitted to real listening histories from Last.fm 1K.

## Data (not committed - must be on disk)

| Dataset | Where | Size |
|---|---|---|
| Last.fm 1K scrobbles | `../lastfm-dataset-1K/` | 1.3 GB |
| MSD summary file | `../msd_summary_file.h5` | ~300 MB |
| tagtraum genre annotations | `../msd_tagtraum_cd2.cls` | ~280k rows |
| MSD track-to-song ID mapping | `../unique_tracks.txt` | ~1M rows |

Both audio and listening datasets are fully offline. No API keys required. The Spotify `/audio-features` endpoint was deprecated 27 Nov 2024; MSD provides equivalent Echo Nest features. The summary file covers the full 1M-song MSD and loads in seconds (single vectorised HDF5 read).

## Setup

```bash
pip install numpy pandas h5py scikit-learn matplotlib seaborn torch pyro-ppl jupyter
```

Place (or symlink) the data as siblings of this repo clone:
```
Desktop/Repos/
├── MBML---2026/              <- this repo
├── lastfm-dataset-1K/        <- Last.fm 1K folder
├── msd_summary_file.h5       <- MSD summary HDF5
├── msd_tagtraum_cd2.cls      <- tagtraum genre labels
└── unique_tracks.txt         <- MSD TR->SO ID bridge
```
No manual path editing needed - the notebooks resolve them via `Path.cwd().parent`.

## Running

**Phase 0 - data pipeline (run first, produces the clean CSVs):**
```bash
jupyter notebook phase0_data.ipynb  # Run All Cells - takes ~1-2 min on M-series Mac
```
Outputs: `data/songs_clean.csv` (matched corpus + genre column), `data/listens_clean.csv` (pos + neg listen events).

**Phase 1 - baseline GMM:**
```bash
jupyter notebook phase1_mood_model_w-genres.ipynb
```

## Phase overview

| Phase | Goal | Status |
|---|---|---|
| 0 | Data pipeline: load, join, normalise, save CSVs | ✓ Done |
| 1 | Baseline Bayesian GMM in Pyro (SVI) | ✓ Done |
| 2 | NUTS/MCMC comparison, R-hat diagnostics | ✓ Done |
| 3 | User-taste extension (Dirichlet theta_u + user/song biases + listen-event plate) | ✓ Done |
| 4 | Posterior predictive checks | Notebook drafted, not yet executed |
| 5 | IEEE 6-page paper (double-column LaTeX) | In progress (Results/Discussion/Conclusion empty) |
| 6 | Notebook polish to course style | TODO |

## Model sketch

Phase 1 (baseline - songs only):
```
pi           ~ Dirichlet(5 * 1_K)
mu_k         ~ N(0, I_3)                          [loudness, tempo, duration]
sigma_k      ~ LogNormal(0, 0.5^2 I_3)
theta_key_k  ~ Dirichlet(1_12)                    [chromatic key, 12 classes]
theta_ts_k   ~ Dirichlet(1_6)                     [time signature, 6 values]
p_k          ~ Beta(2, 2)                         [mode, binary]
z_s          ~ Categorical(pi)                    [latent mood of song s]
x_cont_s     ~ N(mu_{z_s}, diag(sigma^2_{z_s}))  [observed]
x_key_s      ~ Categorical(theta_key_{z_s})       [observed]
x_ts_s       ~ Categorical(theta_ts_{z_s})        [observed]
x_mode_s     ~ Bernoulli(p_{z_s})                 [observed]
```

Phase 3 extension adds (per-user taste, per-user activity, per-song popularity, learnable global scaling):
```
theta_u  ~ Dirichlet(0.5 * 1_K)                    [user taste over moods]
alpha_u  ~ N(0, 1)                                 [per-user activity bias]
gamma_s  ~ N(0, 1)                                 [per-song popularity bias]
b, beta  = pyro.param (learnable globals;          [global intercept + taste scaling]
           b init -1.6, beta init 5.0, beta>0)
l_us     ~ Bernoulli(sigma(b + beta * theta_{u, z_s} + alpha_u + gamma_s))
```
The MAP mood assignment $z_s$ is taken from the Phase 1 fit and treated as fixed in Phase 3, so the only discrete latent is gone and `Trace_ELBO` suffices. The two global params are registered with `pyro.param` and optimised by Adam alongside the variational parameters.

K is auto-selected via the ΔELBO elbow (K=10 from data). Continuous features are z-scored in Phase 1; all features MinMax-scaled to [0,1] in Phase 0. (`energy` and `danceability` are absent from the MSD summary file.)

## Repository layout

```
MBML---2026/
├── phase0_data.ipynb                  # data pipeline (done)
├── phase1_mood_model_w-genres.ipynb   # baseline GMM (done)
├── PROJECT_PLAN.md                    # full spec, PGM, per-phase instructions
├── CLAUDE.md                          # AI assistant context
├── RESPONSIBILITIES.txt               # team role assignments
└── course_content/                    # lectures, exercise solutions
```

## Constraints

This project is graded on curriculum alignment. Only techniques from DTU 42186 weeks 1-12 are used. No transformers, flows, diffusion, RL, Stan, PyMC, or NumPyro. See `PROJECT_PLAN.md §3` for the full in-scope / out-of-scope list.
