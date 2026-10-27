# Phase 3. Search for FS_ext limitations under harmonic triaxial loading

## Objective

Compare the **maximum Fatemi–Socie extended damage parameter** obtained by the
corrected SeAn-CP restricted search with an independent numerical full-plane
search. Loading histories are constrained to a thick-walled tubular specimen
subjected to independently controlled axial force, torque and internal pressure.
Only complete **single-frequency harmonic** histories are admitted. No other
critical-plane criterion or fatigue-life relation is considered.

The local stress/strain tensors come from the elastic Lamé and Saint-Venant
formulation implemented in Phase 2. The specimen geometry follows the supplied
`MAIN_v3.py`, with an internal diameter of 12 mm and a thickness of 1.5 mm.

## Running the study

Install from the repository root and run

```bash
python -m pip install -e '.[test,plot]'
python -m pytest -q
python examples/phase3_search.py --per-family 18 --out results/phase3
```

For a short functional check, use

```bash
python examples/phase3_search.py --per-family 2 --no-plots --out /tmp/phase3_quick
```

A completed screening can be reused only when its input signature matches the
current configuration and stored screening version.

```bash
python examples/phase3_search.py --per-family 18 --out results/phase3 --resume
```

The provided reproducible reference sweep uses random seed `23109`.
A new seed can be selected using `--seed` and changes all random cases.

## Loading-space definition

The tube is assumed closed-ended, with pressure-induced axial stress and the
user-specified axial force treated as an *additional* gauge-section force.
External pressure is set to zero. The three independent loading signals follow

\[
q(t)=q_m+q_a\sin(\omega t+\varphi_q)
\]

where the applied pressure is nonnegative throughout the cycle. The axial-force
harmonic provides the zero-phase reference. The sampling families are
`proportional`, `torque_90`, `pressure_90`, `mixed_90`, `general`, and
`mean_shifted`, in addition to ten fixed control histories. Each random family
contains 18 seeded histories for the reference run, giving 118 histories before
elastic screening. Their ranges are disclosed in `StudyConfig`.

The default material constants **illustrate a plausible elastic model**, but
are not measured specimen properties. They are `E=210000 MPa`, `nu=0.30`,
`Sy=900 MPa`, and `k_FS=0.4`. These choices must be replaced by material-specific
values before an experimental study. For this numerical investigation, histories
with *sampled* peak von Mises stress above `0.9 Sy` at the tube surfaces are
excluded. This is a preliminary screen, not a proof of elastic, fatigue, or
experimental admissibility. The sampled check does not establish continuous-time
extrema, and actuator limitations have not been applied.

## Damage parameter and amplitude definitions

On each candidate plane, the complete FS_ext parameter is

\[
FS_{\mathrm{ext}}(\mathbf n)=\gamma_a(\mathbf n)
\left(1+k_{FS}\frac{\max(0,\sigma_{n,\max}(\mathbf n))}{S_y}\right).
\]

The shear strain amplitude is based on *engineering* shear strain computed from
the tensorial strain history. The normal stress maximum is the exact harmonic
maximum. The normal-stress and shear contributions are evaluated on the same
plane. The wall-wide fatigue indicator is the maximum FS_ext evaluated at all
investigated radial positions.

The code records all three requested amplitude labels. For complete
single-frequency harmonic ellipses, the **longest chord** (half the path
diameter) and **MCC** (radius of the minimum circumscribed circle) are
mathematically identical. To avoid unnecessary duplicate calculations in the
large sweep, the MCC rows are copied from longest chord. Direct evaluation of
both algorithms is verified separately by automated tests. **MCE** is the
root-sum-square of the two shear-ellipse semiaxes, as defined in Phase 1.
The optimisation is repeated independently for MCE because the plane ranking
can change.

## Comparison and numerical checking

The study uses two stages.

1. The initial screen evaluates the inner, mean and outer wall radii, with a
   31 by 61 full-plane grid and multi-seed local refinement. The three SeAn-CP
   paths are sampled at 61 points each and refined locally. All error rankings
   use the wall-wide maximum, not the single most favourable local discrepancy.
2. The leading twelve distinct histories are refined at 11 radial locations
   using a 121 by 241 full-plane grid and multi-seed local refinement.
   SeAn-CP is independently refined using 361 samples per path. Four distinct
   histories are selected by the most negative wall-wide signed error. Their
   reference solutions are also checked on 41 by 81 and 181 by 361 grids, and
   with an independent seeded differential-evolution search on the hemisphere.

The signed error in percent is

\[
e=100\frac{\max_r FS^{\mathrm{SeAnCP}}_{\mathrm{ext}}(r)
-\max_r FS^{\mathrm{full}}_{\mathrm{ext}}(r)}
{\max_r FS^{\mathrm{full}}_{\mathrm{ext}}(r)}.
\]

A negative value denotes underprediction. For FS_ext near zero (below
`1e-7`), a relative percentage is not reported to avoid arbitrary rankings.
The full-plane comparator is a numerical optimisation, not a formal global
bound. The ranked results are the worst found **in the investigated sample**;
there is no claim of a global worst load case.

## Reference search results

The saved run generated **118** histories and retained **112** after the
sampled von Mises screen. The top four were all governed by MCE and the outer
surface (`r=7.5 mm`).

| Rank | Case ID | Full-plane FS_ext | SeAn-CP FS_ext | Signed wall-wide error |
| ---: | --- | ---: | ---: | ---: |
| 1 | `torque_90_0014` | 0.0024418485 | 0.0023498314 | -3.7683% |
| 2 | `general_0012` | 0.0014656566 | 0.0014166437 | -3.3441% |
| 3 | `mean_shifted_0008` | 0.0026022284 | 0.0025596790 | -1.6351% |
| 4 | `mixed_90_0004` | 0.0013581411 | 0.0013477234 | -0.7671% |

Among 112 accepted histories, three MCE results exceeded a 1% underprediction,
while no longest-chord or MCC result did. The largest absolute underprediction
with either longest chord or MCC was approximately 0.0141%. These findings
should not be extrapolated outside the tested material, geometry and load space.

For all four reported MCE cases, refining the grid from 41 by 81 to 181 by 361
changed the maximum negligibly at the displayed numerical precision. The
independent differential-evolution result agreed to better than `1e-9%` in the
reported comparisons. These observations support numerical convergence but
cannot certify global optimality.

## Outputs and figure interpretation

The script writes `screening_cases.csv`, `screening_radial.csv`,
`refined_candidates.csv`, `refined_radial.csv`, `worst_four.csv`,
`convergence_checks.csv`, `screening_signature.json`, and `manifest.json`.
It also generates `error_distribution.png` and, for each selected history,
**two FS_ext angle landscapes for each distinct amplitude definition**
(longest chord and MCE). The landscapes include a 2D colour map and a 3D
surface plot, each overlaid with the three SeAn-CP search paths. A through-wall
comparison figure is also generated.

The plane coordinates are the azimuth `phi` from 0 to 360 degrees and polar
angle `theta` from 0 to 90 degrees. Since `n` and `-n` define the same plane,
a hemisphere is sufficient. The periodic azimuth seam is handled explicitly;
`theta=0` has no unique azimuth. Coloured curves represent the three original
restricted search families, the white star represents the independently
refined full-plane maximum, and the open circle identifies the best restricted
plane. Some path segments may geometrically coincide.

The tabulated stress components and loading coefficients are in MPa, N and
N mm. Torques in the `worst_four.csv` file can be divided by 1000 to obtain
N m. The script preserves both sine and cosine coefficients, so every loading
history can be reproduced without inferring phase information from extrema.

## Limitations and next phase

These results do not verify a testing machine, a specimen's cyclic elastic
response, a pressure seal or a loading-control strategy. The mechanical model
ignores axial end effects, pressure connections, finite gauge length, plasticity
and instability. The sampling programme is broad but not a global load-space
optimisation. Four selected histories may share mechanisms or be practically
similar. Phase 4 should check machine capacities, force/pressure end thrust,
frequency and phase control, specimen strength, elastic shakedown, mean load
limits, and damage localisation before selecting experimental tests.
