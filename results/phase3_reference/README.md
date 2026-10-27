# Phase 3 reference sweep

This folder contains the selected numerical output from the harmonic study with
seed 23109 and 18 random loading cases per family. The full screening CSVs,
through-wall profiles, damage maps and 3D surface plots are included in the
separate downloadable results archive and can be regenerated with
`python examples/phase3_search.py --per-family 18`.

The ranked cases represent the four largest underpredictions in the investigated
finite sampling programme, not globally optimised loads. The reference is a
numerically refined full-plane search rather than a certified global maximum.
The applied yield strength is illustrative. Consult `PHASE3_FS_EXT.md` before
interpreting the numerical comparisons as candidate experiments.
