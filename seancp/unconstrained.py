"""Unconstrained local-tensor harmonic FS_ext benchmark (Phase 4).

No specimen or actuator is assumed. Stresses and strains are symmetric,
three-dimensional, synchronous single-frequency harmonics. The default
elastic strain follows isotropic Hooke law from the stress coefficients.
The candidate-plane search algorithm is unchanged from Phases 1–3.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

import numpy as np
from scipy.optimize import differential_evolution

from .core import (HarmonicHistory, _anchors, evaluate_fs_ext,
                   full_plane_scan, seancp_fs_ext)

FAMILIES = (
    "proportional", "normal_offphase", "shear_rotating", "biaxial_mix",
    "principal_rotation", "noncoaxial_mean", "mixed_3d",
)
ALGORITHMS = ("longest_chord", "mcc", "mce")


@dataclass(frozen=True)
class TensorStudyConfig:
    """Units MPa, dimensionless strain. Values are illustrative, not measured."""
    young_mpa: float = 210000.0
    nu: float = 0.30
    sigma_y_mpa: float = 600.0
    k_fs: float = 0.40
    seed: int = 44713
    n_cases: int = 8000
    screen_normals: int = 600
    screen_path_angles: int = 49
    shortlist_per_family: int = 9
    # Normalise maximum sampled von Mises stress to 30%-92% of Sy.
    # This is NOT a specimen/actuator feasibility filter.
    von_mises_fraction: tuple[float, float] = (0.30, 0.92)

    def __post_init__(self):
        if (self.young_mpa <= 0 or not -1 < self.nu < 0.5 or
            self.sigma_y_mpa <= 0 or self.k_fs < 0 or self.seed < 0 or
            self.n_cases < 1 or self.screen_normals < 50 or
            self.screen_path_angles < 16 or self.shortlist_per_family < 1 or
            len(self.von_mises_fraction) != 2 or not
            (0 < self.von_mises_fraction[0] <= self.von_mises_fraction[1] <= 1)):
            raise ValueError("Invalid tensor study configuration")


@dataclass(frozen=True)
class TensorCase:
    id: int
    family: str
    stress_mean: np.ndarray
    stress_sin: np.ndarray
    stress_cos: np.ndarray


def elastic_strain(stress: np.ndarray, config: TensorStudyConfig) -> np.ndarray:
    """Isotropic linear elasticity, tensorial engineering-independent strain."""
    return ((1 + config.nu) * stress - config.nu * np.trace(stress) * np.eye(3)) / config.young_mpa


def history(case: TensorCase, config: TensorStudyConfig) -> HarmonicHistory:
    a, b, c = case.stress_mean, case.stress_sin, case.stress_cos
    return HarmonicHistory(a, b, c, elastic_strain(a, config),
                           elastic_strain(b, config), elastic_strain(c, config))


def _sym(rng) -> np.ndarray:
    q = rng.normal(size=(3, 3))
    return 0.5 * (q + q.T)


def _rotation(rng) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q


def generate_cases(config: TensorStudyConfig):
    """Deterministic, stratified generator of independent harmonic tensor loads.

    Family names describe stress-tensor structure, not specimen loading modes.
    Arrays are always symmetric. Scaling never modifies the sin/cos ratio.
    """
    rng = np.random.default_rng(config.seed)
    for i in range(config.n_cases):
        family = FAMILIES[i % len(FAMILIES)]
        if family == "proportional":
            sin, cos, mean = _sym(rng), np.zeros((3, 3)), _sym(rng)
        elif family == "normal_offphase":
            sin = np.diag(rng.normal(size=3))
            cos = np.diag(rng.normal(size=3))
            mean = np.diag(rng.normal(size=3))
        elif family == "shear_rotating":
            sin = np.zeros((3, 3)); cos = sin.copy()
            j, k = rng.choice(3, 2, replace=False)
            u, v = rng.choice(3, 2, replace=False)
            sin[j, k] = sin[k, j] = rng.normal()
            cos[u, v] = cos[v, u] = rng.normal()
            mean = np.diag(rng.normal(size=3))
        elif family == "biaxial_mix":
            sin, cos, mean = _sym(rng), _sym(rng), _sym(rng)
            for x in (sin, cos, mean):
                x[2, :] = 0; x[:, 2] = 0
        elif family == "principal_rotation":
            q, p = _rotation(rng), _rotation(rng)
            sin = q @ np.diag(rng.normal(size=3)) @ q.T
            cos = p @ np.diag(rng.normal(size=3)) @ p.T
            mean = np.diag(rng.normal(size=3))
        elif family == "noncoaxial_mean":
            sin, cos, mean = _sym(rng), _sym(rng), 2 * _sym(rng)
        else:
            sin, cos, mean = _sym(rng), _sym(rng), _sym(rng)
        sin *= rng.uniform(.4, 2.0)
        cos *= rng.uniform(.4, 2.0)
        mean *= rng.uniform(.25, 1.4)
        peak_vm = 0.
        for theta in np.arange(12) * (2 * np.pi / 12):
            s = mean + sin * np.sin(theta) + cos * np.cos(theta)
            d = s - np.trace(s) / 3 * np.eye(3)
            peak_vm = max(peak_vm, float(np.sqrt(1.5 * np.sum(d*d))))
        scale = (rng.uniform(*config.von_mises_fraction) * config.sigma_y_mpa / peak_vm)
        yield TensorCase(i, family, mean*scale, sin*scale, cos*scale)


def _normals(config: TensorStudyConfig):
    # Separate fixed seed so random case stream is unchanged by screening grid.
    q = np.random.default_rng(447131).normal(size=(config.screen_normals, 3))
    return q / np.linalg.norm(q, axis=1)[:, None]


def screen_case(case: TensorCase, config: TensorStudyConfig,
                normals: np.ndarray, angles: np.ndarray) -> list[dict]:
    h = history(case, config)
    diff, anchor0, anchor1, _ = _anchors(h, None)
    segments = []
    for tensor in (diff, anchor0, anchor1):
        _, eigenvectors = np.linalg.eigh(tensor)
        segments.append(np.cos(angles)[:, None] * eigenvectors[:, 2] +
                        np.sin(angles)[:, None] * eigenvectors[:, 0])
    paths = np.concatenate(segments)
    rows = []
    # MCC = longest_chord exactly for harmonic elliptical paths.
    for algo in ("longest_chord", "mce"):
        kw = dict(algorithm=algo, k_fs=config.k_fs, sigma_y=config.sigma_y_mpa)
        value_full = float(np.max(evaluate_fs_ext(h, normals, **kw)))
        value_sean = float(np.max(evaluate_fs_ext(h, paths, **kw)))
        row = dict(case_id=case.id, family=case.family, algorithm=algo,
                   preliminary_fs_full=value_full, preliminary_fs_seancp=value_sean,
                   preliminary_error_percent=100 * (value_sean / value_full - 1))
        rows.append(row)
        if algo == "longest_chord":
            rows.append({**row, "algorithm": "mcc"})
    return rows


def _normal_from_angles(x):
    theta, phi = x
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi), np.cos(theta)])


def differential_evolution_reference(h: HarmonicHistory, config: TensorStudyConfig,
                                     algorithm: str, seeds=(19, 101, 765)):
    """Numerical independent check on a hemisphere, not an analytical proof."""
    kw = dict(k_fs=config.k_fs, sigma_y=config.sigma_y_mpa, algorithm=algorithm)
    candidates = []
    for seed in seeds:
        def objective(x):
            return -float(evaluate_fs_ext(h, [_normal_from_angles(x)], **kw)[0])
        fit = differential_evolution(objective, [(0, np.pi/2), (0, 2*np.pi)],
                                     seed=seed, popsize=12, maxiter=120,
                                     tol=1e-11, polish=True)
        candidates.append((-float(fit.fun), _normal_from_angles(fit.x)))
    return max(candidates, key=lambda x: x[0]), np.array([v for v, n in candidates])


def refine_case(case: TensorCase, config: TensorStudyConfig, algorithm: str,
                *, confirm: bool = False) -> tuple[dict, dict]:
    h = history(case, config)
    kw = dict(k_fs=config.k_fs, sigma_y=config.sigma_y_mpa, algorithm=algorithm)
    semi = seancp_fs_ext(h, samples_per_path=181, **kw)
    ref, n_ref = full_plane_scan(h, n_theta=91, n_phi=181, **kw)
    record = {"case_id": case.id, "family": case.family, "algorithm": algorithm,
              "seancp_fs_ext": float(semi.value), "full_fs_ext": float(ref),
              "error_percent": float(100 * (semi.value/ref - 1))}
    convergence = {}
    if confirm:
        semi = seancp_fs_ext(h, samples_per_path=721, **kw)
        ref, n_ref = full_plane_scan(h, n_theta=181, n_phi=361, **kw)
        (global_value, global_normal), starts = differential_evolution_reference(h, config, algorithm)
        if semi.value > ref * (1 + 1e-7) or global_value > ref * (1 + 1e-6):
            raise RuntimeError("Full scan does not bound the restricted search/global check")
        record.update({"seancp_fs_ext": float(semi.value),
                       "full_fs_ext": float(ref),
                       "error_percent": float(100 * (semi.value/ref - 1)),
                       "semi_normal": semi.normal.tolist(),
                       "full_normal": n_ref.tolist(),
                       "anchor_phase_rad": float(semi.anchor_phase_rad),
                       "peak_von_mises_sampled_mpa": peak_vm(case)})
        convergence = {"case_id": case.id, "family": case.family,
                       "algorithm": algorithm, "full_scan": float(ref),
                       "independent_de": float(global_value),
                       "max_relative_difference_percent": float(100 * (global_value/ref - 1)),
                       "de_starts": starts.tolist(),
                       "de_normal": global_normal.tolist()}
    return record, convergence


def peak_vm(case: TensorCase):
    peak = 0.0
    for phase in np.arange(720)*2*np.pi/720:
        s = case.stress_mean + case.stress_sin*np.sin(phase) + case.stress_cos*np.cos(phase)
        d = s - np.trace(s)/3 * np.eye(3)
        peak = max(peak, float(np.sqrt(1.5*np.sum(d*d))))
    return peak


def write_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def case_data(case: TensorCase):
    return {"case_id":case.id,"family":case.family,
            "stress_mean_mpa":case.stress_mean.tolist(),
            "stress_sin_mpa":case.stress_sin.tolist(),
            "stress_cos_mpa":case.stress_cos.tolist()}


def case_from_data(data):
    return TensorCase(int(data["case_id"]), data["family"],
                      *(np.asarray(data[field], float) for field in
                        ("stress_mean_mpa", "stress_sin_mpa", "stress_cos_mpa")))


def run_study(out_dir: str | Path, config: TensorStudyConfig = TensorStudyConfig(),
              *, plot=True, retain_screen=True):
    """Screen many cases, refine candidates from each family and verify four families.

    Main outputs: worst_four.csv, selected_cases.json, independent_checks.csv,
    and 3D surface images including three SeAn-CP trajectories.
    """
    output = Path(out_dir); output.mkdir(parents=True, exist_ok=True)
    normals = _normals(config)
    angles = np.linspace(0, np.pi, config.screen_path_angles, endpoint=False)
    all_rows = []
    candidates = {}
    for case in generate_cases(config):
        candidates[case.id] = case
        all_rows.extend(screen_case(case, config, normals, angles))
    if retain_screen: write_csv(output / "screening.csv", all_rows)
    indexed = {name: [] for name in FAMILIES}
    for row in all_rows:
        if row["algorithm"] == "mce": indexed[row["family"]].append(row)
    shortlist = []
    for name, rows in indexed.items():
        picked = sorted(rows, key=lambda x: x["preliminary_error_percent"])
        shortlist.extend(picked[:config.shortlist_per_family])
    refined = []
    for row in shortlist:
        case = candidates[row["case_id"]]
        out, _ = refine_case(case, config, "mce")
        refined.append(out)
    refined.sort(key=lambda x: x["error_percent"])
    write_csv(output / "refined_candidates.csv", refined)
    # Distinct physical *tensor structures*. This is not the top four without
    # this condition, and the distinction is explicit in the output.
    selected = []
    seen = set()
    for row in refined:
        if row["family"] not in seen:
            selected.append(row); seen.add(row["family"])
        if len(selected) == 4: break
    checked = []; checks = []; comparison = []
    for row in selected:
        case = candidates[row["case_id"]]
        main, chk = refine_case(case, config, "mce", confirm=True)
        checked.append(main); checks.append(chk)
        for algo in ("longest_chord", "mcc"):
            rec, _ = refine_case(case, config, algo, confirm=True)
            comparison.append(rec)
    checked.sort(key=lambda x:x["error_percent"])
    write_csv(output / "worst_four.csv", checked)
    write_csv(output / "other_amplitudes.csv", comparison)
    write_csv(output / "independent_checks.csv", checks)
    payload = {"config": asdict(config),
               "selected_cases": [case_data(candidates[x["case_id"]]) for x in checked]}
    (output / "selected_cases.json").write_text(json.dumps(payload, indent=2)+"\n")
    manifest = {"description": "Unconstrained local tensor single-harmonic FS_ext screening",
                "total_cases": len(candidates), "candidate_evaluations": len(all_rows),
                "families": list(FAMILIES), "selected_families": [x["family"] for x in checked],
                "selected_ids": [x["case_id"] for x in checked],
                "config": asdict(config),
                "interpretation": "Selected worst candidate of each of four distinct tensor structures; not four unconstrained globally largest values",
                "assumptions": ["Linear elastic isotropic strain from stress",
                   "General symmetric three-dimensional tensors, no specimen or rig mapping",
                   "Von Mises normalisation is numerical and sampled, not rig feasibility",
                   "Only single-frequency harmonic histories; FS_ext only",
                   "MCC equals longest chord for a complete harmonic ellipse",
                   "Numerical refined scanning and differential evolution are not proofs of global optimality"]}
    (output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    if plot:
        from .unconstrained_plots import plot_comparison, plot_summary
        plot_summary(checked, output / "error_summary.png")
        for row in checked:
            h=history(candidates[row["case_id"]], config)
            filename=f"case_{row['case_id']:05d}_{row['family']}_mce"
            plot_comparison(h,config,output/(filename+"_3d.png"),
                            label=f"{row['family']} | case {row['case_id']}",three_d=True)
            plot_comparison(h,config,output/(filename+"_2d.png"),
                            label=f"{row['family']} | case {row['case_id']}",three_d=False)
    return {"worst":checked,"manifest":manifest}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/phase4_search")
    p.add_argument("--n-cases", type=int, default=8000)
    p.add_argument("--seed", type=int, default=44713)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-screening-csv",action="store_true")
    args=p.parse_args(argv)
    cfg=TensorStudyConfig(seed=args.seed,n_cases=args.n_cases)
    found=run_study(args.output,cfg,plot=not args.no_plots,
                    retain_screen=not args.no_screening_csv)
    for row in found["worst"]:
        print(f"{row['family']:22} case={row['case_id']:5d} MCE error={row['error_percent']:+.3f}%")


if __name__=="__main__":
    main()
