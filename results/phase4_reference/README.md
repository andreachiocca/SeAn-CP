# Phase 4 reference tensors and numerical results

The four selected cases are local harmonic stress-tensor histories, not
specimen or machine loading programs. Their full symmetric stress matrices,
material constants and Fourier coefficients are preserved in
`selected_cases.json`. Tensorial elastic strains are reconstructed by the
`elastic_strain` function in `seancp/unconstrained.py`.

`worst_four.csv` contains the full-plane and SeAn-CP MCE values, errors, plane
normals and sampled von Mises maxima. `other_amplitudes.csv` contains their
longest-chord and MCC counterparts. `independent_checks.csv` contains three
independent differential-evolution verifications. `manifest.json` records
the 8,000-history screening settings and the distinct-family selection rule.

The complete screening table, refined shortlist and 3D/2D damage figures are
provided in the Phase 4 downloadable archive. Regenerate them with the
example CLI script. No actual specimen load sequence is implied by these
results. See `PHASE4_UNCONSTRAINED_FS_EXT.md` for methods and limitations.
