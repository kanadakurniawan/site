---
title: "Machine Learning for Precipitation Nowcasting"
description: "A technical overview of nowcasting methods — from radar extrapolation to deep generative models — and where the field is heading."
pubDatetime: 2026-08-30
tags: ["AI", "Meteorology", "nowcasting", "deep learning", "radar", "precipitation"]
---

Precipitation nowcasting is the problem of predicting rainfall up to ~2 hours ahead. It remains
hard because of the chaotic nature of convective systems and the steep decay of predictability.

## Formulation

Let $X_t \in \mathbb{R}^{H \times W}$ be a radar reflectivity field at time $t$. Nowcasting asks
for the conditional distribution

$$
p(X_{t+1}, \ldots, X_{t+\tau} \mid X_{t-k+1}, \ldots, X_{t})
$$

over the next $\tau$ frames. Most production systems predict the conditional mean and call it a
forecast.

## Approaches

1. **Optical flow extrapolation** — Lagrangian advection of the last observed field.
2. **ConvLSTM and successors** — recurrent networks with convolutional state transitions.
3. **Generative models** — GANs and diffusion models that sample realistic nowcasts calibrated
   on skill scores.

## A simple optical flow baseline

```python
import numpy as np

def extrapolate(sequence, velocity, lead_time):
    """Extrapolate radar frames along a constant velocity field."""
    h, w = velocity.shape[:2]
    out = np.empty_like(sequence[0])
    for y in range(h):
        for x in range(w):
            sy = int(np.clip(y - velocity[y, x, 0] * lead_time, 0, h - 1))
            sx = int(np.clip(x - velocity[y, x, 1] * lead_time, 0, w - 1))
            out[y, x] = sequence[-1][sy, sx]
    return out
```

The mean-squared error between the extrapolation and the observed frame drops sharply after
20–30 minutes, which motivates the current interest in learned, stochastic models.