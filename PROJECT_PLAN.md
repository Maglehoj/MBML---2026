# Project plan — Latent Mood & User-Taste Modelling

A complete, self-contained roadmap for the MBML group project. Written so a fresh Claude session, a new teammate, or a future-you can pick up the project in a freshly cloned repo without re-reading any prior chat.

---

## 1  What this project is

We are building a **Bayesian probabilistic model of music** for the DTU course *Model-Based Machine Learning (42186), Spring 2026*. The model has two parts that are trained jointly:

1. A **Bayesian Gaussian mixture** over song audio features. Each component is a "mood" — a characteristic combination of `loudness, tempo, key, mode, time_signature, duration`. The model recovers $K$ mood clusters from songs alone.
2. A **per-user Dirichlet taste profile** over those moods, fitted to real listening histories. The user's preferences shape which songs they engage with via a Bernoulli listen-event likelihood.

The deliverables are a fully-explained Jupyter notebook plus a 6-page IEEE double-column paper, due **15 May 2026**.

### Why this dataset combination

We originally planned to use the Spotify API (`/audio-features`), but **Spotify deprecated this endpoint on 27 November 2024** — new apps now receive HTTP 403, with no replacement. The Million Song Dataset (MSD) was created in 2011 from the same Echo Nest features that Spotify later acquired and used internally, so MSD provides equivalent pre-computed features. Last.fm 1K provides far richer user listening histories than the Spotify user endpoint ever did. Both are fully offline, no API keys needed.

| Dataset | Provides | Size | Use in Phase 1 |
|---|---|---|---|
| **songs_clean.csv** | Matched corpus of songs with 6 audio features (loudness, tempo, key, mode, time_signature, duration), MinMax-scaled, + genre column | 292k rows | Primary input; features used for Gaussian, Categorical, Bernoulli likelihoods |
| **listens_clean.csv** | User listening events (user, song, play_count, listened binary) | 16.8M rows | Held for Phase 3 (user extension); not used in Phase 1 |
| Last.fm 1K source | ~1,000 users, ~19M scrobbles (history only) | 1.3 GB | Already processed → songs_clean.csv |
| MSD summary source | Pre-computed Echo Nest features for 1M songs | ~300 MB | Already processed → songs_clean.csv |

They join on a normalised `(artist_name, track_title)` key. Measured match rate: **45.6% of scrobbles (~8.7M / 19M rows)** after aggressive normalisation (strip live/remaster/version suffixes before lowercasing). Artist coverage is 81% — the gap is MSD missing specific hit tracks, not artists. Acceptable; noted as catalog-bias attrition in the paper. Note: `energy` and `danceability` are not stored in the summary file (all-zero); we use `key` and `duration` instead.

---

## 2  Repository layout

MBML---2026/
├── course_content/
│   ├── 00 - Every_Lecture.pdf
│   ├── all_solutions_merged.ipynb
│   ├── Course overview.pdf
│   ├── Minitests-with&without-answers.pdf
│   ├── Projects.pdf
│   └── phase0_data.ipynb
├── PROJECT_PLAN.md
└── README.md

data is found locally
---

## 3  Curriculum scope — the hard rule

The project is graded on how well it applies course concepts. Anything outside the curriculum **must not appear in the model** even if it would help.

### In scope (weeks 1–12)

| Week | Topic | What this project uses |
|---|---|---|
| 1 | Probability review | Bayes theorem |
| 2 | PGM foundations | Plates, conditional independence |
| 3 | PGM foundations II | Conjugate priors, generative processes |
| 4 | Mixture models | `pyro.sample`, `pyro.plate`, Categorical mixtures, Gaussian/Categorical/Bernoulli likelihoods, `TraceEnum_ELBO` |
| 5 | Regression | Bayesian linear/Poisson regression, PGM + neural net |
| 6 | Classification + hierarchical | Logistic regression, Beta-Bernoulli, hierarchical Dirichlet priors |
| 7 | Temporal | (not used) |
| 8 | Topic models | LDA-style Dirichlet-Categorical for discrete features (key, time_signature) |
| 9 | MCMC | NUTS in Pyro, R-hat diagnostics |
| 10 | Variational inference | ELBO, SVI, mean-field guide, `AutoDiagonalNormal`, K-selection via elbow |
| 11 | Generative models | (PPCA / VAE — relevant for context, not used directly) |
| 12 | Gaussian processes | (not used) |

### Explicitly out of scope — never suggest

- Normalising flows, diffusion models, transformers, attention, GNNs, RL.
- Any deep-learning technique not covered in week 5 (PGM + neural net) or week 11 (VAE).
- External probabilistic programming libraries (Stan, NumPyro, PyMC) — only Pyro.

If you are tempted to add something off-list, write *"out of scope"* and stop.

---

## 4  The PGM — full specification

### Plate notation

```
Hyperparameters (fixed, not learned):
  α        — Dirichlet concentration for user taste profiles
  μ₀, κ₀  — Normal-Wishart prior on mood means
  ν, Ψ    — Wishart prior on mood covariances

K-mood plate  (repeated K times):
  μ_k  ~ Normal(μ₀, κ₀⁻¹ · Σ_k)           — mean of mood k
  Σ_k  ~ InverseWishart(ν, Ψ)              — covariance of mood k

U-user plate  (repeated U times):
  θ_u  ~ Dirichlet(α)                      — taste profile of user u
                                              (distribution over K moods)

S-song plate  (repeated S times):
  z_s  ~ Categorical(π)                    — latent mood of song s
                                              (π = global mixture weights in baseline;
                                               θ_u in the user extension)
  x_s  ~ MultivariateNormal(μ_{z_s}, Σ_{z_s})
                                            — observed audio features
                                              (D-vector, D=6, from MSD)

U-user plate (activity bias, in addition to θ_u above):
  α_u  ~ N(0, 1)                           — per-user listening propensity

S-song plate (popularity bias):
  γ_s  ~ N(0, 1)                           — per-song popularity

Global learnable parameters (pyro.param, not random variables):
  b    — global intercept (init -1.6 ≈ σ⁻¹(0.166), matches base rate)
  β>0  — taste-scaling coefficient (init 5.0)

U×S listen-event plate  (one per observed (u,s) pair):
  l_us ~ Bernoulli(σ(b + β · θ_{u, z_s} + α_u + γ_s))
                                            — did user u listen to song s?
                                              σ = sigmoid; θ_{u, z_s} is the
                                              z_s-th coordinate of θ_u
                                              (l=1 from Last.fm, l=0 sampled at 5:1)
```

In Phase 3 the song-level latent $z_s$ is **not re-sampled**: we take the MAP assignment from the fitted Phase 1 guide and treat it as observed. This removes the only discrete latent in the listen-event submodel, so a plain `Trace_ELBO` is sufficient and no enumeration is needed.

### Variable key

| Symbol | Type | Observed? | Description |
|---|---|---|---|
| α | scalar/vector | fixed | Dirichlet prior concentration |
| μ₀, κ₀ | vector, scalar | fixed | Normal prior on mood means |
| ν, Ψ | scalar, matrix | fixed | Wishart prior on mood covariances |
| μ_k | D-vector (D=6) | latent | Mean audio feature vector of mood k |
| Σ_k | D×D matrix | latent | Covariance of mood k |
| θ_u | K-vector (simplex) | latent | User u's distribution over moods |
| α_u | scalar | latent (Phase 3) | Per-user activity bias, N(0,1) prior |
| γ_s | scalar | latent (Phase 3) | Per-song popularity bias, N(0,1) prior |
| b | scalar | learnable global param | Logit intercept, init −1.6 |
| β | positive scalar | learnable global param | Taste-scaling coefficient, init 5.0 |
| z_s | integer in {1..K} | latent (Phase 1) / fixed MAP (Phase 3) | Mood assignment of song s |
| x_s | D-vector | **observed** | Audio features of song s from MSD |
| l_us | binary | **observed** | Whether user u listened to song s |

### Baseline vs extension

**Baseline (Phase 1):** Ignore the user layer entirely.
- $z_s \sim \mathrm{Categorical}(\pi)$ where $\pi \sim \mathrm{Dirichlet}(\alpha_{\text{song}})$ is a global mixture weight.
- $x_s \sim \mathcal{N}(\mu_{z_s}, \Sigma_{z_s})$.
- Just a Bayesian Gaussian Mixture Model over audio features.
- Goal: recover $K$ interpretable mood clusters.

**Extension (Phase 3):**
- Add user plate, song-bias plate, and listen-event plate.
- $\theta_u \sim \mathrm{Dirichlet}(0.5 \cdot \mathbf{1}_K)$ per user.
- $\alpha_u \sim \mathcal{N}(0, 1)$ per user (activity bias) and $\gamma_s \sim \mathcal{N}(0, 1)$ per song (popularity bias).
- Learnable global parameters $b$ and $\beta > 0$ via `pyro.param`.
- $l_{us} \sim \mathrm{Bernoulli}(\sigma(b + \beta \cdot \theta_{u, z_s} + \alpha_u + \gamma_s))$ where $z_s$ is the fixed MAP from Phase 1.
- Goal: recover interpretable per-user taste profiles while separating mood preference from raw user activity and song popularity.

---

## 5  Style conventions (from `CONVENTIONS.md`)

Every project notebook must follow the same style as the course exercise notebooks. The examiners wrote those notebooks and will compare against them.

**Imports** (always in this order):
```python
import numpy as np
import torch                       # Phase 1+ only
import pyro                        # Phase 1+ only
import pyro.distributions as dist  # Phase 1+ only
import matplotlib.pyplot as plt
import seaborn as sns
```

**Setup**:
```python
np.random.seed(67)
torch.manual_seed(67)       # Phase 1+ only
pyro.set_rng_seed(67)       # Phase 1+ only
plt.style.use('ggplot')
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 8)
```

**Markdown style:** every code cell preceded by a markdown cell with 2–5 sentence explanation. LaTeX inline (`$\theta_u$`) and in displays (`\begin{align}…\end{align}`). `##` for major sections, `###` for sub-sections.

**Pyro model skeleton**:
```python
def model(data, K=...):
    # priors outside plates
    weights = pyro.sample("weights", dist.Dirichlet(torch.ones(K)))
    with pyro.plate("components", K):
        mu = pyro.sample("mu", dist.Normal(...).to_event(1))
        sigma = pyro.sample("sigma", dist.HalfCauchy(...).to_event(1))
    with pyro.plate("data", len(data)):
        z = pyro.sample("z", dist.Categorical(weights),
                        infer={"enumerate": "parallel"})
        pyro.sample("obs", dist.MultivariateNormal(mu[z], ...), obs=data)
```

**Plotting**: seaborn for high-level plots, matplotlib for tweaks; `figsize=(12,8)` from rcParams.

---

## 6  Phases — complete in order

### Phase 0 — Setup and data collection ✓ DONE

**Goal:** Both datasets downloaded, joined, cleaned, saved as two DataFrames.

**Status:** `phase0_data.ipynb` written (27 cells), data extracted, ready to run.

The notebook does, in order:

1. Imports + seed + plot config.
2. Load Last.fm scrobbles (~19M rows from a 1.3 GB TSV).
3. Load MSD summary HDF5 (`msd_summary_file.h5`) in a single vectorised read — all 1,000,000 songs in seconds.
4. Load `unique_tracks.txt` (TR→SO ID mapping) and tagtraum genre annotations (`msd_tagtraum_cd2.cls`). Tagtraum uses TR... track IDs; the summary HDF5 uses SO... song IDs — bridge via `unique_tracks.txt`. Result: ~157k songs (54%) get a genre label; remainder set to `"Unknown"`.
5. Normalise (artist, track) strings on both sides: lowercase, strip, replace `-`/`_` with space, **strip live/remaster/version suffixes** (everything after ` - ` or a `(` / `[`).
6. Inner-join MSD × Last.fm on `(artist_norm, track_norm)`.
7. Merge tagtraum genres into matched corpus on `song_id`; fill missing with `"Unknown"`.
8. Aggregate listen events; filter active users with ≥5 listens.
9. 5:1 negative sampling per user.
10. MinMax-scale the 6 features to $[0,1]$, drop NaN rows.
11. Save `data/songs_clean.csv` (includes `genre` column) and `data/listens_clean.csv`.
12. EDA: matched corpus size, genre coverage, listen-count distribution per user, feature correlation heatmap, feature histograms, summary printout.

**Gate** (must pass before Phase 1): `Matched corpus size`, `Active users (>=5 listens)`, `Positive listen events`, `Negative listen events`, genre coverage %, and `df_songs.describe()` of the 6 scaled features.

**Run:**
```bash
cd project/
jupyter notebook phase0_data.ipynb   # then Run All Cells
```

---

### Phase 1 — Baseline Bayesian GMM in Pyro ✓ DONE

**Goal:** A working Pyro model that clusters songs into K mood groups using audio features only. No users yet.

**Inputs:** `data/songs_clean.csv`.

**K selection:** K is auto-selected via the ΔELBO elbow (sweep K=2..15, 500 SVI steps each). Yielded K=10. Genre-stratified init was abandoned — genres and moods are different constructs, and the mixed-likelihood loss landscape is well-conditioned without it. Plain multi-restart K-means (`n_init=20`) on the 3 z-scored continuous features is used instead.

**Model (mixed likelihoods):**
```python
def model(X_cont, X_key, X_ts, X_mode, K, mu_prior_loc):
    D = X_cont.shape[1]  # 3 continuous features
    pi = pyro.sample("pi", dist.Dirichlet(5.0 * torch.ones(K)))

    with pyro.plate("moods", K):
        mu_cont   = pyro.sample("mu_cont",
            dist.Normal(torch.zeros(D), torch.ones(D)).to_event(1))
        sigma_cont = pyro.sample("sigma_cont",
            dist.LogNormal(torch.zeros(D), 0.5 * torch.ones(D)).to_event(1))
        theta_key = pyro.sample("theta_key", dist.Dirichlet(torch.ones(N_KEY)))
        theta_ts  = pyro.sample("theta_ts",  dist.Dirichlet(torch.ones(N_TS)))
        p_mode    = pyro.sample("p_mode",    dist.Beta(2.0, 2.0))

    with pyro.plate("songs", X_cont.shape[0]):
        z = pyro.sample("z", dist.Categorical(pi),
                        infer={"enumerate": "parallel"})
        pyro.sample("obs_cont", dist.Normal(mu_cont[z], sigma_cont[z]).to_event(1), obs=X_cont)
        pyro.sample("obs_key",  dist.Categorical(theta_key[z]), obs=X_key)
        pyro.sample("obs_ts",   dist.Categorical(theta_ts[z]),  obs=X_ts)
        pyro.sample("obs_mode", dist.Bernoulli(p_mode[z]),      obs=X_mode)
```

**Inference:** SVI with `AutoDiagonalNormal` over continuous latents and `TraceEnum_ELBO(max_plate_nesting=1)` to enumerate the discrete `z`.

```python
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

guide = AutoDiagonalNormal(pyro.poutine.block(model, hide=["z"]))
svi = SVI(model, guide, Adam({"lr": 1e-2}), TraceEnum_ELBO(max_plate_nesting=1))

losses = []
for step in range(2000):
    loss = svi.step(X)
    losses.append(loss)
```

**Deliverables:**
- ELBO convergence plot.
- Posterior mean / std of $\mu_k$ and $\sigma_k$ for each mood.
- Cluster-assignment plot over the data (a 2D PCA or pair-plot coloured by $z$).
- Brief markdown interpreting each mood ("loud + fast → high-energy", etc.).

**Gate:** ELBO has converged (flat for last 200 steps), $K$ moods are recoverable and not collapsed (no two near-identical $\mu_k$).

---

### Phase 2 — MCMC comparison via NUTS

**Goal:** Re-run inference on the baseline model using HMC/NUTS, compare posterior to SVI.

**Key step:** marginalise out the discrete `z` via `infer_discrete` before running NUTS — HMC has no gradients for discrete variables.

```python
from pyro.infer import MCMC, NUTS, infer_discrete

# Wrap the model so z is marginalised out by enumeration
collapsed_model = infer_discrete(
    pyro.poutine.config_enumerate(model),
    first_available_dim=-2,
)

nuts_kernel = NUTS(collapsed_model)
mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=200, num_chains=2)
mcmc.run(X)
posterior = mcmc.get_samples()
```

**Deliverables:**
- R-hat plot for each continuous parameter ($\hat R < 1.05$ everywhere).
- Trace plots ("hairy caterpillars").
- Side-by-side posterior of $\mu_k$ from SVI vs NUTS.
- Effective sample size summary.

**Gate:** R-hat < 1.05 for all parameters; SVI and NUTS posteriors qualitatively agree (means within ~1 posterior std of each other).

---

### Phase 3 — User-taste extension ✓ DONE

**Goal:** Add the user plate, song-bias plate, and listen-event plate. Recover interpretable per-user taste profiles while separating mood preference from overall user activity and song popularity.

**Inputs:** `data/songs_clean.csv` + `data/listens_clean.csv` + the MAP $z_s$ extracted from the fitted Phase 1 guide.

**Why this shape and not the simpler $\sigma(\theta_u \cdot e_{z_s})$:** Without bias terms, $\theta_u$ has to absorb every source of variation in the listen signal — that some users scrobble more than others, and that some songs are universally popular. The ELBO improved from ~7.5M to ~5.6M after adding $\alpha_u$ and $\gamma_s$, confirming the decomposition explains real variance. The learnable scalar $b$ removes the need to hard-code a magic-number offset to match the empirical listen base rate; $\beta > 0$ lets the optimiser pick the right strength for the taste-match term independent of the simplex constraint.

**Model:**
```python
def extended_model(user_idx, song_idx, mood_idx, listened, U, S, K):
    # Learnable global parameters
    base_logit  = pyro.param("base_logit",  torch.tensor(-1.6))       # σ⁻¹(0.166) ≈ base rate
    taste_scale = pyro.param("taste_scale", torch.tensor(5.0),
                             constraint=dist.constraints.positive)

    with pyro.plate("users", U):
        theta   = pyro.sample("theta",   dist.Dirichlet(0.5 * torch.ones(K)))
        alpha_u = pyro.sample("alpha_u", dist.Normal(0.0, 1.0))

    with pyro.plate("songs", S):
        gamma_s = pyro.sample("gamma_s", dist.Normal(0.0, 1.0))

    with pyro.plate("listens", len(listened)):
        taste_match = taste_scale * theta[user_idx, mood_idx]
        logit = base_logit + taste_match + alpha_u[user_idx] + gamma_s[song_idx]
        pyro.sample("obs", dist.Bernoulli(torch.sigmoid(logit)), obs=listened)
```

**Guide:** custom mean-field — Dirichlet `alpha_q` (positive-constrained) for $\theta_u$, Normal `(loc_u, scale_u)` for $\alpha_u$, Normal `(loc_s, scale_s)` for $\gamma_s$. `mood_idx` carries the Phase 1 MAP $z_s$ per listen event; no discrete latent remains, so `Trace_ELBO` is sufficient (no enumeration).

**Inference:** SVI with Adam (lr=$10^{-2}$), 1000 steps, seed 67.

**Deliverables (in notebook):**
- ELBO convergence plot.
- Per-user posterior taste profile bar charts (six representative users sorted by entropy: 2 concentrated, 2 middle, 2 dispersed).
- Population-level $U \times K$ posterior-mean heatmap.
- Dominant-mood distribution across users.

**Gate (passed):** ELBO converged (flat tail), at least some users have concentrated taste profiles below uniform entropy.

---

### Phase 4 — Posterior predictive checks

**Goal:** Verify the model can generate realistic song features and listen patterns.

**Procedure:**
- Draw $S$ posterior samples of $(\pi, \mu_{1..K}, \Sigma_{1..K}, \theta_{1..U})$.
- For each sample, sample fresh $z_s, x_s, l_{us}$ via ancestral sampling.
- Compare distributions of fake-vs-real for: per-feature histograms, listen-count distributions, mood-frequency histograms.

**Pyro pattern:**
```python
from pyro.infer import Predictive
predictive = Predictive(model, posterior_samples=posterior_samples)
samples = predictive(X_blank)  # samples["obs"] shape: (S, N, D)
```

**Deliverables:**
- 6 overlay histograms: real feature vs posterior-predictive feature.
- A density plot of observed-vs-predicted listen probabilities.
- Calibration plot (predicted probability vs empirical frequency, binned).

**Gate:** Posterior-predictive distributions visually overlap the real ones; calibration is reasonable.

---

### Phase 5 — IEEE 6-page paper

**Goal:** Write the paper. Sections:

1. **Introduction** — motivation, related work in implicit-feedback / mood modelling.
2. **Model** — formal PGM, generative process, mathematical specification.
3. **Inference** — SVI vs MCMC, why this combination, key Pyro idioms.
4. **Data** — Last.fm 1K + MSD subset, the join, EDA highlights.
5. **Results** — recovered moods, posterior summaries, posterior-predictive checks, SVI-vs-MCMC comparison.
6. **Discussion & limitations** — what the model captures and what it misses; honest about the corpus size and the matching attrition.

**Format:** IEEE double-column LaTeX template.

---

### Phase 6 — Notebook polish

**Goal:** make `mood_model.ipynb` fully self-explanatory.

- Every code cell preceded by a markdown explanation in course style (2–5 sentence paragraphs, LaTeX for math).
- All plots labelled and captioned.
- A final "Summary" markdown cell at the end recapping headline numbers.
- All hyperparameter choices documented inline (why $K=6$? why $\alpha=0.5$? etc.).

---

## 7  Status checklist

- [x] Master prompt read (`master_prompt_spotify_pgm.md`).
- [x] Cheat sheet and study notes built for mini-test 2 (in `Mini Tests/`).
- [x] Spotify-API approach abandoned (deprecated 27 Nov 2024).
- [x] Last.fm 1K downloaded and extracted to `data/lastfm1k/`.
- [x] MSD summary file at `../msd_summary_file.h5` (1,000,000 songs, single HDF5).
- [x] `CONVENTIONS.md` extracted from course exercises.
- [x] `README.md` written.
- [x] `phase0_data.ipynb` written and run end-to-end — produces `songs_clean.csv` (with `genre` column) and `listens_clean.csv`.
- [x] tagtraum genre annotations (`msd_tagtraum_cd2.cls`) integrated into Phase 0 via `unique_tracks.txt` TR→SO bridge (~54% coverage).
- [x] Phase 1 — baseline GMM (`phase1_mood_model_w-genres.ipynb`).
- [x] Phase 2 — NUTS comparison (`phase2_nuts_comparison.ipynb`): R̂=1.038, min ESS 69.9, median ESS 600, 93.3% SVI/NUTS agreement.
- [x] Phase 3 — user-taste extension (`phase3_user_taste.ipynb`): Dirichlet $\theta_u$ + $\alpha_u$, $\gamma_s$ biases + learnable global $b$, $\beta$.
- [ ] Phase 4 — posterior predictive checks (notebook drafted, not yet executed; `data/phase4_processed/` does not exist).
- [ ] Phase 5 — IEEE paper (Results, Discussion, Conclusion still empty; abstract/contribution table placeholder).
- [ ] Phase 6 — notebook polish.

---

## 8  Key decisions made (for future reference)

These are the non-obvious choices that future-you will want to be reminded of.

- **Dataset switch.** Original plan was Spotify API. The `/audio-features` endpoint was deprecated 27 November 2024. We switched to Last.fm 1K + MSD summary file (1M songs), joined on normalised `(artist, track)`. Matched corpus will be much larger than the old 10K subset.
- **Six features only.** `loudness, tempo, key, mode, time_signature, duration`. `energy` and `danceability` are all-zero in the summary file (only stored in per-track HDF5s). `key` (chromatic 0–11) and `duration` are musically meaningful substitutes. `hotttnesss` and segment pitches add noise.
- **5:1 negative sampling.** Standard for implicit-feedback Bernoulli likelihoods. Higher ratios (10:1) overweight the negatives; lower (1:1) underweight them.
- **Active-user filter ≥5 listens.** Below 5, the per-user $\theta_u$ posterior is dominated by the prior — we'd be reporting noise.
- **K=10 via ΔELBO elbow.** K is auto-selected by sweeping K=2..15, fitting 500 SVI steps each, and picking the elbow of the per-component ELBO gain. Genre-stratified init was considered but abandoned: genres and moods are different constructs, and the mixed-likelihood loss landscape is well-conditioned without it. Plain multi-restart K-means (`n_init=20`) on the 3 z-scored continuous features is used instead.
- **Phase 3 listen-likelihood shape.** The model uses $\sigma(b + \beta \cdot \theta_{u, z_s} + \alpha_u + \gamma_s)$ with learnable global $b$ (init $-1.6$, matches empirical base rate) and $\beta > 0$ (init $5.0$), plus per-user $\alpha_u \sim \mathcal{N}(0,1)$ and per-song $\gamma_s \sim \mathcal{N}(0,1)$. The simpler $\sigma(\theta_u \cdot e_{z_s})$ form forces $\theta_u$ to absorb activity and popularity variance, distorting taste estimates; the bias decomposition improved ELBO from ~7.5M to ~5.6M.
- **$z_s$ fixed at Phase 1 MAP in Phase 3.** The Phase 3 model does not re-sample $z_s$; it reads the MAP assignment from the fitted Phase 1 guide. This removes the only discrete latent in the listen submodel, so `Trace_ELBO` is sufficient (no enumeration / `infer_discrete` needed inside Phase 3).
- **tagtraum genre annotations.** `msd_tagtraum_cd2.cls` at `../msd_tagtraum_cd2.cls`. Uses MSD track IDs (TR...) — does NOT directly join to the summary HDF5 (which uses SO... Echo Nest song IDs). Bridge via `unique_tracks.txt` (`../unique_tracks.txt`). Genre coverage ~54% (157k of 292k matched songs). Used only for K-means initialisation — not a model feature. Falls back to n_init=20 multi-restart K-means if coverage drops below 30%.
- **MinMax scaling, not z-score.** `mode` and `time_signature` are categorical-ish; z-scoring would distort them. MinMax to $[0,1]$ keeps them on the same scale as the continuous features without changing their shape.
- **`TraceEnum_ELBO` for SVI, `infer_discrete` + NUTS for MCMC.** These are the canonical Pyro patterns the course uses for mixtures with discrete latents. Discrete latents have no gradients, so SVI must enumerate them and NUTS must marginalise them.

---

## 9  Constraints / hard rules (recap)

1. **Curriculum scope only.** Use only weeks 1–12 techniques. No transformers, no flows, no diffusion, no RL, no Stan/PyMC/NumPyro.
2. **Notation matches lectures.** $\pi$ = mixture weights, $\theta$ = Dirichlet variables, $z$ = discrete latents, $\alpha, \tau, \nu, \Psi$ = priors as in `master_prompt_spotify_pgm.md`.
3. **Style matches `Exercises/`.** numpy/matplotlib/seaborn imports, ggplot, `(12,8)` figsize, `np.random.seed(42)`. See `CONVENTIONS.md`.
4. **No code outside the curriculum.** If a technique would help but wasn't taught, write *"out of scope"* and stop.
5. **Read `Exercises/` notebooks before writing project code.** Especially `08 - LDA - Pyro - Solutions.ipynb`, `09 - Markov chain Monte Carlo methods - Solutions.ipynb`, and `10 - Variational inference - Bayesian GMM - solutions.ipynb` — they are the closest analogues to the project model.

---

## 10  Pointer to other docs

- `master_prompt_spotify_pgm.md` — full project specification with the original chat history, PGM diagrams, exact loader code, mathematical details. Authoritative.
- `CONVENTIONS.md` — extracted notebook style guide. The "what does a course-style notebook look like" reference.
- `README.md` — the human-friendly run guide.
- `Mini Tests/cheatsheet.pdf` and `Mini Tests/study_notes.pdf` — exam prep materials covering the same theoretical machinery the project uses (LDA, MCMC, VI, generative models).

---

**Deadline:** 15 May 2026.
**Today:** see `env` block at top of any prompt.
**Days remaining:** count backwards.
