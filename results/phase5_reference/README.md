# Frozen numerical outputs for Phase 5

These outputs were calculated from the four exact Phase 4 reference histories
in `../phase4_reference/selected_cases.json` using the implementation in
`seancp/feasibility.py` and `examples/phase5_feasibility.py`.

All comparisons are for `FS_ext` with single-frequency harmonics and
illustrative elastic constants. The projected histories are mathematical
fits, not experimentally validated load programs.

- `feasibility_cases.csv` gives the original-history peak stress and ideal
  face-force scale, plus an independent full-plane check.
- `projection_results.csv` gives tensor-space fitting errors and recalculated
  `FS_ext` errors for five stress models per case. The physically constrained
  tube fit includes an 11-radius through-wall error and sampled peak von Mises.
- `reduced_control_waveforms.csv` gives mean/sine/cosine coefficients, phases,
  ranges, and ideal 5 mm cube face forces for 2213 original, 1739 original,
  and 6920 fitted three-normal stress directions.
- `ideal_cube_face_tractions.csv` gives three traction components on every
  ideal face (six faces) for each of the four original histories. Forces on
  opposite faces are exactly balanced for each harmonic coefficient.
- `projected_histories.json` includes all numeric projected tensor histories,
  fitted fixed axes, and ideal positive internal pressure/force/torque controls.
- `manifest.json` records the numerical settings and verification status.

Run `python examples/phase5_feasibility.py --out results/phase5` from the
installed repository to regenerate all tables and the diagnostic PNG figures.
The PNGs are provided in a downloadable archive to avoid increasing the Git
repository size. See `PHASE5_REALIZABILITY_FS_EXT.md` for mechanical cautions.
