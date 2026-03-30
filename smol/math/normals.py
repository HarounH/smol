#!/usr/bin/env python3
"""
sum_normals_viz.py

Visualize the distribution of a sum of m independent normal distributions, plus
a few individual component distributions, and save a PNG in the current directory.

Examples:
  python sum_normals_viz.py --mus 0 0 0 --sigmas 1 1 1 --samples 200000 --out sum_normals.png
  python sum_normals_viz.py --m 5 --mu-range -1 1 --sigma-range 0.5 2 --seed 42 --show-components 3

Notes:
- Each component is Normal(mu_i, sigma_i^2); you pass sigmas (std devs).
- If you provide --mus/--sigmas, their length defines m (unless you also pass --m).
- If you do NOT provide mus/sigmas, random parameters are generated from ranges.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot sum of independent normals and save figure.")
    p.add_argument("--m", type=int, default=None, help="Number of component normals.")
    p.add_argument("--mus", type=float, nargs="*", default=None,
                   help="List of means μ_i. Example: --mus 0 1 -0.5")
    p.add_argument("--sigmas", type=float, nargs="*", default=None,
                   help="List of std devs σ_i (>0). Example: --sigmas 1 0.5 2")

    p.add_argument("--mu-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=[-1.0, 1.0],
                   help="If mus not provided, sample μ_i uniformly from [LOW, HIGH].")
    p.add_argument("--sigma-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=[0.5, 2.0],
                   help="If sigmas not provided, sample σ_i uniformly from [LOW, HIGH]. Must be >0.")

    p.add_argument("--samples", type=int, default=200_000, help="Monte Carlo samples.")
    p.add_argument("--bins", type=int, default=120, help="Histogram bins for densities.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")

    p.add_argument("--show-components", type=int, default=3,
                   help="How many individual component distributions to visualize (PDF + sampled hist).")
    p.add_argument("--component-mode", choices=["first", "random", "largest_var"], default="largest_var",
                   help="Which components to show if show-components < m.")
    p.add_argument("--component-samples", type=int, default=60_000,
                   help="Samples per shown component for its histogram (kept smaller for speed/clarity).")

    p.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"),
                   help="Optional x-axis limits.")
    p.add_argument("--dpi", type=int, default=160, help="Output image DPI.")
    p.add_argument("--out", type=str, default=None,
                   help="Output filename (PNG). Default: sum_normals_m{m}.png")
    return p.parse_args()


def resolve_params(
    m_arg: Optional[int],
    mus: Optional[List[float]],
    sigmas: Optional[List[float]],
    mu_range: Tuple[float, float],
    sigma_range: Tuple[float, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if mus is not None and sigmas is not None:
        if len(mus) != len(sigmas):
            raise ValueError(f"--mus length ({len(mus)}) must match --sigmas length ({len(sigmas)}).")
        m = len(mus)
        if m_arg is not None and m_arg != m:
            raise ValueError(f"--m ({m_arg}) does not match len(--mus/--sigmas) ({m}).")
        mus_arr = np.array(mus, dtype=float)
        sigmas_arr = np.array(sigmas, dtype=float)
    else:
        if m_arg is None:
            raise ValueError("Provide --m, or provide both --mus and --sigmas.")
        m = m_arg
        mu_lo, mu_hi = mu_range
        s_lo, s_hi = sigma_range
        if s_lo <= 0 or s_hi <= 0:
            raise ValueError("--sigma-range bounds must be > 0.")
        mus_arr = rng.uniform(mu_lo, mu_hi, size=m)
        sigmas_arr = rng.uniform(s_lo, s_hi, size=m)

    if np.any(sigmas_arr <= 0):
        raise ValueError("All σ_i must be > 0.")

    return mus_arr, sigmas_arr


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def choose_component_indices(
    m: int,
    k: int,
    mode: str,
    sigmas: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    k = max(0, min(k, m))
    if k == 0:
        return np.array([], dtype=int)
    if mode == "first":
        return np.arange(k)
    if mode == "random":
        return rng.choice(m, size=k, replace=False)
    # largest_var
    return np.argsort(-(sigmas ** 2))[:k]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    mus, sigmas = resolve_params(
        m_arg=args.m,
        mus=args.mus,
        sigmas=args.sigmas,
        mu_range=(args.mu_range[0], args.mu_range[1]),
        sigma_range=(args.sigma_range[0], args.sigma_range[1]),
        rng=rng,
    )
    m = mus.shape[0]

    # Sample sum
    # shape: (samples, m) may be large; for m<=~50 this is fine. For huge m, chunking would be better.
    sum_samples = rng.normal(loc=mus, scale=sigmas, size=(args.samples, m)).sum(axis=1)

    # Theoretical sum distribution
    sum_mu = float(mus.sum())
    sum_sigma = float(np.sqrt((sigmas ** 2).sum()))

    # Choose components to show
    comp_idx = choose_component_indices(
        m=m,
        k=args.show_components,
        mode=args.component_mode,
        sigmas=sigmas,
        rng=rng,
    )

    # Determine plotting x-range
    if args.xlim is not None:
        x_min, x_max = args.xlim
    else:
        # Span enough to cover both sum and shown components
        # Sum approx within ±4σ; components within ±4σ.
        candidates = [(sum_mu - 4 * sum_sigma, sum_mu + 4 * sum_sigma)]
        for i in comp_idx:
            candidates.append((mus[i] - 4 * sigmas[i], mus[i] + 4 * sigmas[i]))
        x_min = min(c[0] for c in candidates)
        x_max = max(c[1] for c in candidates)

    xs = np.linspace(x_min, x_max, 900)

    # Plot
    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    # Sum histogram + theoretical PDF
    ax.hist(sum_samples, bins=args.bins, density=True, alpha=0.55, label="Sum (samples)")
    ax.plot(xs, normal_pdf(xs, sum_mu, sum_sigma), linewidth=2.2, label="Sum (theory PDF)")

    # Individual components: sampled hist (lighter) + PDF (line)
    for j, i in enumerate(comp_idx):
        # fewer samples per component for speed/clarity
        comp_samples = rng.normal(loc=mus[i], scale=sigmas[i], size=args.component_samples)
        ax.hist(comp_samples, bins=max(40, args.bins // 3), density=True, alpha=0.18,
                label=f"Comp {i+1} samples (μ={mus[i]:.2f}, σ={sigmas[i]:.2f})")
        ax.plot(xs, normal_pdf(xs, float(mus[i]), float(sigmas[i])), linewidth=1.6,
                label=f"Comp {i+1} PDF")

    ax.set_title(f"Sum of {m} independent Normals: μ_sum={sum_mu:.3f}, σ_sum={sum_sigma:.3f} (var={sum_sigma**2:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_xlim(x_min, x_max)
    ax.grid(True, alpha=0.25)

    # Keep legend manageable
    ax.legend(loc="best", fontsize=9, frameon=True)

    # Save
    out_name = args.out or f"sum_normals_m{m}.png"
    out_path = Path(out_name).resolve()
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print("Saved:", out_path)
    print("m =", m)
    print("mus   =", np.array2string(mus, precision=4, separator=", "))
    print("sigmas=", np.array2string(sigmas, precision=4, separator=", "))
    print(f"sum: mu={sum_mu:.6f}, sigma={sum_sigma:.6f}, var={sum_sigma**2:.6f}")
    if comp_idx.size:
        print("shown components (1-indexed):", (comp_idx + 1).tolist())


if __name__ == "__main__":
    main()
