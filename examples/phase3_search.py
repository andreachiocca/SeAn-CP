"""Reproduce the harmonic FS_ext triaxial loading survey."""
import argparse
from pathlib import Path
from seancp.study import StudyConfig,run_study

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("results/phase3"))
    parser.add_argument("--per-family",type=int,default=18)
    parser.add_argument("--seed",type=int,default=23109)
    parser.add_argument("--top",type=int,default=4)
    parser.add_argument("--no-plots",action="store_true")
    parser.add_argument("--resume",action="store_true",help="Reuse matching completed screening")
    args=parser.parse_args()
    result=run_study(args.out,n_per_family=args.per_family,top_n=args.top,
                     config=StudyConfig(seed=args.seed),plot=not args.no_plots,resume=args.resume)
    print(result["manifest"]["screening"])
    for row in result["worst"]:
        print(f"{row['case_id']:24s} {row['algorithm']:14s} "
              f"FS_ref={row['fs_ext_reference']:.8g} "
              f"FS_semi={row['fs_ext_seancp']:.8g} "
              f"error={row['error_wall_percent']:+.4f}% "
              f"r={row['governing_ref_radius_mm']:.2f} mm")
