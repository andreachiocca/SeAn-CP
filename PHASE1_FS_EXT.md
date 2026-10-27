# Phase 1: corrected FS_ext SeAn-CP (Python)

This phase adds a Python implementation of the three-family SeAn-CP
candidate-plane search, with the Fatemi-Socie **extended** parameter as the
**only** implemented criterion. The historical MATLAB file `MAIN.m` is
preserved unchanged, allowing later reproduction audits.

## Parameter and conventions

For every unit plane normal `n`, the same complete fatigue parameter is
evaluated on that plane:

```text
FS_ext(n) = gamma_a(n) * (1 + k_FS * max(0, sigma_nn,max(n)) / Sy)
```

`gamma_a` is the engineering shear-strain **amplitude**, `Sy` is the
yield strength [MPa], and `sigma_nn,max` is the maximum plane-normal stress
over the whole loading cycle [MPa]. Stresses are in MPa; input small-strain
tensors contain *tensorial* shear strain, so engineering shear vectors are
formed by multiplying the in-plane projected strain by two.

The semi-analytical maximum is restricted to the three original plane
families. The full scan is an **independent numerical comparator**, not a
mathematically certified global maximum.

## Corrections against the original MATLAB code

- Compute shear strain and normal stress from the **same** plane normal.
  Several blocks of `MAIN.m` mixed first and third axes of the rotation
  matrix, which may misidentify the candidate-plane damage.
- Search all conjugate orientations of each extreme-eigenvector Mohr-circle
  family by using a full half-circle `0 <= alpha < pi`. This removes
  incomplete or inconsistent angle/sign selection in the original loops.
- Use `FS_ext` throughout. The maximum of the *full* FS damage parameter
  is used, never shear-only plane selection (conventional FS).
- Evaluate normal stress peaks and shear amplitudes from a **complete
  harmonic or sampled history**. Anchor states are used only to construct
  the three restricted candidate-plane families.

The search families follow `Delta E`, `S(t0)` and `S(t0+pi)`,
using the largest and smallest eigenvectors of each symmetric tensor.
For harmonic histories, `t0` defaults to the antipodal pair maximising
the Frobenius strain-tensor difference, with an explicit override available.
For sampled histories, the pair with the largest sampled Frobenius difference
is used. Choice of anchors can affect the restricted search under
non-proportional loading, and should be studied in Phase 3.

## Shear amplitude methods

| Cycle representation | `longest_chord` | `mcc` | `mce` |
| --- | --- | --- | --- |
| Complete single-frequency ellipse | Largest semiaxis | Minimum circumscribed circle radius = largest semiaxis | Square root of sum of squared semiaxes |
| Arbitrary sampled path | Half the maximum sampled chord length | Radius of minimum enclosing circle of samples | Unsupported; raises ValueError |

For any **complete single-frequency harmonic shear ellipse**, MCC and
longest-chord amplitudes coincide *by geometry*. MCE can differ. We
follow the amplitude convention already documented in the
[`multiaxial-fatigue` shear module](https://github.com/andreachiocca/multiaxial-fatigue/blob/main/src/multifat/models/utils/shear_amplitude.py).
The optional MCE choice is deliberately rejected for non-harmonic
sampled paths, because that convention does not specify a general
minimum circumscribed ellipse algorithm for arbitrary histories.

For a harmonic tensorial-strain history
`E(t)=Emean+Esin*sin(t)+Ecos*cos(t)`, define the engineering shear
Fourier coefficients as

```text
qsin = 2 * (I - n*n.T) * Esin * n
qcos = 2 * (I - n*n.T) * Ecos * n
```

The shear-path semiaxes are the singular values of the 3-by-2 matrix
`[qsin qcos]`. The full-cycle normal-stress maximum is

```text
n.T*Smean*n + hypot(n.T*Ssin*n, n.T*Scos*n)
```

The three methods are evaluated independently **on each candidate plane**;
the critical-plane ranking is recomputed for each amplitude definition.

## Installation and usage

```bash
python -m pip install -e '.[test]'
pytest -q
python examples/phase1_fs_ext.py
```

Import `HarmonicHistory`, `SampledHistory`, `seancp_fs_ext`,
`full_plane_scan`, and `evaluate_fs_ext` from `seancp`.
The `SearchResult` contains three `PlanePath` records with the normals,
angles, per-plane FS_ext values, and each path's best plane. This allows
the critical-plane surface and the three semi-analytical paths to be
plotted in a later phase without reconstructing their geometry.

## Scope and verification limits

This phase does **not** implement thick-wall specimen stresses, pressure,
actuator limits, multi-frequency MCE, nonlinear constitutive strain
reconstruction, or loading-condition optimisation. It accepts given
stress and strain histories. Verification tests cover analytical
amplitudes, pure shear, clipping, sampled non-proportional paths,
rotation and phase invariance, extrema checks, and the three restricted
search paths.

The Python implementation has been locally tested with the bundled
`pytest` suite. Independent reproduction against MATLAB and an
executable cross-check against the `multiaxial-fatigue` package remain
future verification activities. The three search families should not
be presented as guaranteeing the absolute critical plane.
