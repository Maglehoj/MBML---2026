---
name: MBML 2026 project overview
description: DTU 42186 Bayesian GMM + user-taste model project — phases, data locations, deadline, constraints
type: project
---

DTU course 42186 (Model-Based Machine Learning), Spring 2026. Deadline **15 May 2026**.

**Goal:** Bayesian Gaussian Mixture over 6 Echo Nest audio features (MSD) + per-user Dirichlet taste profiles (Last.fm 1K). Deliverables: `mood_model.ipynb` + 6-page IEEE paper.

**Data paths (local, not committed):**
- Last.fm 1K: `/Users/magle/Desktop/Repos/lastfm-dataset-1K/`
- MSD 10K subset: `/Users/magle/Desktop/Repos/MillionSongSubset/`
- Clean outputs: `data/songs_clean.csv`, `data/listens_clean.csv`

**Phase status (as of 2026-05-06):**
- Phase 0 (data pipeline) — ✓ DONE (`phase0_data.ipynb`)
- Phases 1–6 — TODO

**Key constraints:** Pyro only (no Stan/PyMC/NumPyro), curriculum weeks 1–12 only, no transformers/flows/diffusion/RL.

**Key decisions:** Spotify deprecated → switched to MSD+Last.fm; MinMax not z-score (mode/time_signature are categorical-ish); 5:1 negative sampling; K=6 moods; active-user filter ≥5 listens.

**Why:** Spotify `/audio-features` endpoint deprecated 27 Nov 2024; MSD provides equivalent offline Echo Nest features.

**How to apply:** Always check PROJECT_PLAN.md for phase-level detail. Never suggest out-of-scope techniques. Match course exercise notebook style (ggplot, seed 42, figsize (12,8)).
