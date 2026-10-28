CORRECTED SELF-CONTAINED MAPDL files (2025 R2)

The original .err reported that *USE,export_notch could not locate a macro.
These two files inline the unit-load definitions and exporter to remove all
external .mac dependencies. Each input starts with BASIS_MODE=1.

RUN IN ORDER
1) Launch MAPDL with a CLEAN working directory (one per model).
2) File > Read Input from > 01_RUN_smooth_tube.mac
3) Verify the solver output log and three CSVs below.
4) Then use a SEPARATE CLEAN working directory and run
   02_RUN_one_sided_notch.mac

OUTPUT (in MAPDL working directory, not necessarily next to .mac)
fe_stress.csv
fe_strain.csv
fe_coordinates.csv
notched_tube.err (if any; job name is configured inside script)
MAPDL output (.out / .log / errors)

For each CSV there should be 3 load-basis indices (1 pressure, 2 force,
3 torque). The files are automatically reset at the start of each run.

Please send the three CSVs, the error/out file and a mesh screenshot.
These scripts have been statically inspected but cannot be run or
FE-validated in this environment without ANSYS MAPDL.
