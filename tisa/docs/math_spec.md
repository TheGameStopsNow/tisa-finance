# Mathematical Specification

This document restates the mathematical core of Transform-Invariant Segment
Alignment (TISA).

1. **Pre-normalization**: Input series are z-normalized unless `normalize=False`.
2. **Segmentization**: Segments are defined between consecutive extrema
   (including endpoints) with deterministic tie breaking and alternating
   directions. Each segment has duration `Δt`, amplitude `Δv`, and direction.
3. **Transforms**: The second series is evaluated under four variants: original,
   reversed, inverted, and reversed+inverted. Each transform is segmented
   independently.
4. **Costs**:
   - Normalization constants use medians across both segment sets: `T` for
     duration and `V` for amplitude (with `V ≥ 1e-8`).
   - Match cost (only equal directions):
     `α ((Δt_x - Δt_y)/T)^2 + β ((Δv_x - Δv_y)/V)^2`.
   - Gap cost: `γ (|Δv|/V)^2`.
5. **Dynamic Programming**: Standard alignment DP with match/x-gap/y-gap
   transitions, returning total cost `D_mode` for each transform. The global
   distance is `D* = min(D_mode)`.
6. **Mapping**: Backtracking provides segment mappings. For matched segments,
   optional per-point warping maps indices linearly within segments.

All computations are deterministic and support optional Numba acceleration.
