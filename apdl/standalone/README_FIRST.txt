ANSYS MAPDL 2025 R2 - RESULT SET SELECTION FIX V2

Your pressure-only FE export was excellent (711 elements; median full stress
relative error 0.142% against the exact Lamé solution). But the original
script then requested load step 2 via SET,2; ANSYS reported that this load
set did not exist in its .rst and stopped after writing basis 1.

This version selects SET,LAST after each static SOLVE, regardless of how
MAPDL indexed its load steps/substeps. It explicitly verifies that result
pseudo-time equals the requested unit-basis index 1/2/3. It will stop
instead of exporting pressure data falsely labelled as force or torsion.

1) Unzip; create TWO CLEAN working directories (smooth and notch).
2) Start with the smooth case. Run 01_RUN_smooth_tube.mac in its directory.
3) Inspect .err/.out for errors and fe_result_checks.csv.
4) ONLY IF the smooth case has three rows in fe_result_checks.csv, and all
   three basis indices in each FE CSV, run 02_RUN_one_sided_notch.mac in
   its own clean working directory.
5) Deliver fe_stress.csv, fe_strain.csv, fe_coordinates.csv,
   fe_result_checks.csv, and .err PLUS .out if an error occurs.

Expected UNIT load identifiers:
  1 = pressure: 1 MPa (closed-end pressure thrust included)
  2 = additional axial force: 1000 N
  3 = torsion: 1000 N mm

The three unit loads are INFLUENCE SOLUTIONS, not time samples.
The model uses E=210000 MPa, nu=0.3, ri=6 mm, ro=7.5 mm.

MAPDL was not available here for execution; code fix is statically checked.
