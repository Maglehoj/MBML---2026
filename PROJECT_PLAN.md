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

| Dataset | Provides | Size | Source |
|---|---|---|---|
| **Last.fm 1K** | Real listening histories for ~1,000 users (user, timestamp, artist, track) | 1.3 GB extracted | http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html |
| **MSD summary file** | Pre-computed Echo Nest audio features for all 1,000,000 MSD songs, single HDF5 | ~300 MB | http://millionsongdataset.com/pages/getting-dataset/ |

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
| 4 | Mixture models | `pyro.sample`, `pyro.plate`, Categorical mixtures, `TraceEnum_ELBO` |
| 5 | Regression | Bayesian linear/Poisson regression, PGM + neural net |
| 6 | Classification + hierarchical | Logistic regression, hierarchical Dirichlet priors |
| 7 | Temporal | (not used) |
| 8 | Topic models | LDA-style plate structure mirrors users → moods → songs |
| 9 | MCMC | NUTS in Pyro, R-hat diagnostics |
| 10 | Variational inference | ELBO, SVI, mean-field guide, `AutoNormal` / `AutoDiagonalNormal` |
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

U×S listen-event plate  (one per observed (u,s) pair):
  l_us ~ Bernoulli(σ(θ_u · e_{z_s}))       — did user u listen to song s?
                                              σ = sigmoid, e_{z_s} = one-hot mood indicator
                                              (l=1 from Last.fm, l=0 sampled at 5:1)
```

### Variable key

| Symbol | Type | Observed? | Description |
|---|---|---|---|
| α | scalar/vector | fixed | Dirichlet prior concentration |
| μ₀, κ₀ | vector, scalar | fixed | Normal prior on mood means |
| ν, Ψ | scalar, matrix | fixed | Wishart prior on mood covariances |
| μ_k | D-vector (D=6) | latent | Mean audio feature vector of mood k |
| Σ_k | D×D matrix | latent | Covariance of mood k |
| θ_u | K-vector (simplex) | latent | User u's distribution over moods |
| z_s | integer in {1..K} | latent | Mood assignment of song s |
| x_s | D-vector | **observed** | Audio features of song s from MSD |
| l_us | binary | **observed** | Whether user u listened to song s |

### Baseline vs extension

**Baseline (Phase 1):** Ignore the user layer entirely.
- $z_s \sim \mathrm{Categorical}(\pi)$ where $\pi \sim \mathrm{Dirichlet}(\alpha_{\text{song}})$ is a global mixture weight.
- $x_s \sim \mathcal{N}(\mu_{z_s}, \Sigma_{z_s})$.
- Just a Bayesian Gaussian Mixture Model over audio features.
- Goal: recover $K$ interpretable mood clusters.

**Extension (Phase 3):**
- Add user plate and listen-event plate.
- $\theta_u \sim \mathrm{Dirichlet}(\alpha)$ per user.
- $l_{us} \sim \mathrm{Bernoulli}(\sigma(\theta_u^\top e_{z_s}))$.
- Goal: jointly learn mood clusters and user taste profiles.

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
np.random.seed(42)
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
4. Normalise (artist, track) strings on both sides: lowercase, strip, replace `-`/`_` with space, **strip live/remaster/version suffixes** (everything after ` - ` or a `(` / `[`).
5. Inner-join MSD × Last.fm on `(artist_norm, track_norm)`.
6. Aggregate listen events; filter active users with ≥5 listens.
7. 5:1 negative sampling per user.
8. MinMax-scale the 6 features to $[0,1]$, drop NaN rows.
9. Save `data/songs_clean.csv` and `data/listens_clean.csv`.
10. EDA: matched corpus size, listen-count distribution per user, feature correlation heatmap, feature histograms, summary printout.

**Gate** (must pass before Phase 1): `Matched corpus size`, `Active users (>=5 listens)`, `Positive listen events`, `Negative listen events`, and `df_songs.describe()` of the 6 scaled features.

**Run:**
```bash
cd project/
jupyter notebook phase0_data.ipynb   # then Run All Cells
```

---

### Phase 1 — Baseline Bayesian GMM in Pyro

**Goal:** A working Pyro model that clusters songs into $K$ mood groups using audio features only. No users yet.

**Inputs:** `data/songs_clean.csv`.

**Model:**
```python
def model(X, K=6):
    D = X.shape[1]  # 6 audio features
    pi = pyro.sample("pi", dist.Dirichlet(torch.ones(K) / K))

    with pyro.plate("moods", K):
        mu = pyro.sample("mu",
            dist.Normal(0.5 * torch.ones(D), 0.5 * torch.ones(D)).to_event(1))
        sigma = pyro.sample("sigma",
            dist.LogNormal(torch.zeros(D), torch.ones(D)).to_event(1))

    with pyro.plate("songs", X.shape[0]):
        z = pyro.sample("z", dist.Categorical(pi),
                        infer={"enumerate": "parallel"})
        pyro.sample("obs",
            dist.Normal(mu[z], sigma[z]).to_event(1),
            obs=X)
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

### Phase 3 — User-taste extension

**Goal:** Add the user plate and listen-event plate. Jointly learn mood clusters and user taste profiles.

**Inputs:** `data/songs_clean.csv` + `data/listens_clean.csv`.

**Model addition:**
```python
def extended_model(X, listens, K=6):
    # ... song-level priors as in Phase 1 ...

    U = listens["user_id"].nunique()
    with pyro.plate("users", U):
        theta = pyro.sample("theta", dist.Dirichlet(torch.ones(K) * 0.5))

    # ... song-level z and x_s sampling as in Phase 1 ...

    # listen events
    with pyro.plate("listens", len(listens)):
        # listen probability for (u_i, s_i) pair
        u_idx = listens["user_idx"].values
        s_idx = listens["song_idx"].values
        # one-hot mood indicator e_{z_s}
        z_per_listen = z[s_idx]
        score = theta[u_idx, z_per_listen]
        pyro.sample("l", dist.Bernoulli(score), obs=listens["listened"].values)
```

**Inference:** SVI again, with the larger latent space ($\theta_u$ for every user).

**Deliverables:**
- Posterior over $\theta_u$ for a few selected users — sanity-check that some users concentrate on one mood (say, electronic) while others spread across many.
- Per-user "taste profile" bar chart for 4–6 example users.
- Updated mood interpretation now that the user signal is in the loop.

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
- [x] `phase0_data.ipynb` written (27 cells).
- [ ] **Phase 0 run end-to-end** — produces `songs_clean.csv` and `listens_clean.csv`.
- [ ] Phase 1 — baseline GMM (`mood_model.ipynb`).
- [ ] Phase 2 — NUTS comparison.
- [ ] Phase 3 — user-taste extension.
- [ ] Phase 4 — posterior predictive checks.
- [ ] Phase 5 — IEEE paper.
- [ ] Phase 6 — notebook polish.

---

## 8  Key decisions made (for future reference)

These are the non-obvious choices that future-you will want to be reminded of.

- **Dataset switch.** Original plan was Spotify API. The `/audio-features` endpoint was deprecated 27 November 2024. We switched to Last.fm 1K + MSD summary file (1M songs), joined on normalised `(artist, track)`. Matched corpus will be much larger than the old 10K subset.
- **Six features only.** `loudness, tempo, key, mode, time_signature, duration`. `energy` and `danceability` are all-zero in the summary file (only stored in per-track HDF5s). `key` (chromatic 0–11) and `duration` are musically meaningful substitutes. `hotttnesss` and segment pitches add noise.
- **5:1 negative sampling.** Standard for implicit-feedback Bernoulli likelihoods. Higher ratios (10:1) overweight the negatives; lower (1:1) underweight them.
- **Active-user filter ≥5 listens.** Below 5, the per-user $\theta_u$ posterior is dominated by the prior — we'd be reporting noise.
- **K=6 moods.** Corresponds to the typical "mood wheel" granularity in music IR. We can sweep $K \in \{4, 6, 8\}$ in Phase 1 and pick the most interpretable.
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
