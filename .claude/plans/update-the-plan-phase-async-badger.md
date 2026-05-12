# Plan: Phase 3 Notebook Fix

## Context
Phase 3 (`phase3_user_taste.ipynb`) models per-user taste profiles as Dirichlet-distributed K-simplices and infers them via SVI from binary listen events. The gate currently **FAILS** because 99.7% of users collapse to Mood 8 (the smallest cluster, 0.8% of corpus), median posterior entropy is 0.178 (vs. uniform 2.303), 100% of users are "concentrated", and 0% are "diffuse".

The root cause is the scoring line in **cell 15**:
```python
score = torch.sigmoid(K * theta[user_idx, mood_idx])
```
`theta` is already a probability in `[0,1]` (Dirichlet sample). Multiplying by K=10 and applying sigmoid compresses the signal — uniform taste (0.1 per mood) maps to sigmoid(1) ≈ 0.73, leaving almost no room for the model to distinguish preference from indifference. The Dirichlet prior (α=0.5, sparse) then drives users to corners, and the gradient signal can't push back.

---

## Changes to make in `phase3_user_taste.ipynb`

### Change 1 — Fix the likelihood scoring (cell 15)

Remove the sigmoid + K-scaling. Use `theta[user_idx, mood_idx]` directly as the Bernoulli probability — it's already in `[0,1]`:

```python
# Before
score = torch.sigmoid(K * theta[user_idx, mood_idx])  # shape (L,)

# After
score = theta[user_idx, mood_idx]  # theta is a simplex → already in [0,1]
```

### Change 2 — Raise α from 0.5 to 1.0 (cell 15)

α=0.5 is sub-uniform (Dirichlet corners). α=1.0 is neutral — lets the data decide concentration vs. diffusion:

```python
# Before
ALPHA = 0.5  # sparse Dirichlet

# After
ALPHA = 1.0  # neutral Dirichlet prior
```

### Change 3 — Update the PGM markdown (cell 13)

Fix the listen likelihood equation to match the new model:

```
# Before
l_us ~ Bernoulli(σ(θ_u · e_{z_s}))

# After
l_us ~ Bernoulli(θ_u[z_s])
```

Update both the LaTeX display equation and the ASCII plate diagram in cell 13.

---

## Expected outcome after re-running

- Dominant mood distribution spread across multiple moods (not 99.7% Mood 8)
- Median posterior entropy noticeably above 0.178 (target > ~0.5)
- Mix of concentrated and diffuse users
- Gate cell 29: `Phase 3 gate: PASS`
