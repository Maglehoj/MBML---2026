# Latent Mood & User-Taste Modelling with MSD + Last.fm

DTU course **42186 — Model-Based Machine Learning**, Spring 2026. Deadline: **15 May 2026**.

## What this project does

We build a Bayesian probabilistic model of music that jointly learns:

1. **K mood clusters** (Bayesian Gaussian Mixture) over six Echo Nest audio features from the Million Song Dataset.
2. **Per-user taste profiles** (Dirichlet distributions over moods) fitted to real listening histories from Last.fm 1K.

The model answers: *given that user u listened to songs with a certain audio profile, what is their distribution over moods, and what new songs would they likely enjoy?*

## Data (not committed — must be on disk)

| Dataset | Where | Size |
|---|---|---|
| Last.fm 1K scrobbles | `/Users/magle/Desktop/Repos/lastfm-dataset-1K/` | 1.3 GB |
| MSD 10K subset (HDF5) | `/Users/magle/Desktop/Repos/MillionSongSubset/` | 2.6 GB |

Both datasets are fully offline. No API keys required. The Spotify `/audio-features` endpoint was deprecated 27 Nov 2024; MSD provides equivalent Echo Nest features.

## Setup

```bash
pip install numpy pandas h5py scikit-learn matplotlib seaborn torch pyro-ppl jupyter
```

Symlink or copy data into `data/`:
```bash
mkdir -p data/lastfm1k data/msd
ln -s /Users/magle/Desktop/Repos/lastfm-dataset-1K/* data/lastfm1k/
ln -s /Users/magle/Desktop/Repos/MillionSongSubset data/msd/MillionSongSubset
```

## Running

**Phase 0 — data pipeline (run first, produces the clean CSVs):**
```bash
jupyter notebook phase0_data.ipynb  # Run All Cells — takes ~5 min on M-series Mac
```
Outputs: `data/songs_clean.csv` (~1k–3k songs), `data/listens_clean.csv` (pos + neg listen events).

**Phases 1–4 — model notebook:**
```bash
jupyter notebook mood_model.ipynb   # not yet created
```

## Phase overview

| Phase | Goal | Status |
|---|---|---|
| 0 | Data pipeline: load, join, normalise, save CSVs | ✓ Done |
| 1 | Baseline Bayesian GMM in Pyro (SVI) | TODO |
| 2 | NUTS/MCMC comparison, R-hat diagnostics | TODO |
| 3 | User-taste extension (Dirichlet θ_u + listen-event plate) | TODO |
| 4 | Posterior predictive checks | TODO |
| 5 | IEEE 6-page paper (double-column LaTeX) | TODO |
| 6 | Notebook polish to course style | TODO |

## Model sketch

```
μ_k, Σ_k  ~ Normal-Wishart priors       [K mood components]
θ_u        ~ Dirichlet(α)               [user taste over moods]
z_s        ~ Categorical(π)             [latent mood of song s]
x_s        ~ MVN(μ_{z_s}, Σ_{z_s})     [observed audio features, D=6]
l_us       ~ Bernoulli(σ(θ_u · e_{z_s})) [did user u listen to song s?]
```

Audio features: `loudness, tempo, energy, danceability, mode, time_signature` — all MinMax-scaled to [0,1].

## Repository layout

```
MBML---2026/
├── phase0_data.ipynb    # data pipeline
├── mood_model.ipynb     # phases 1–4 (to be created)
├── PROJECT_PLAN.md      # full spec, PGM, per-phase instructions
├── CLAUDE.md            # AI assistant context (token-optimised)
└── course_content/      # lectures, exercise solutions
```

## Constraints

This project is graded on curriculum alignment. Only techniques from DTU 42186 weeks 1–12 are used. No transformers, flows, diffusion, RL, Stan, PyMC, or NumPyro. See `PROJECT_PLAN.md §3` for the full in-scope / out-of-scope list.
