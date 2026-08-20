"""
validation.py
--------------
Task 3 / Section 3: Validate the BEMT tool in hover against the Knight &
Hefner (1937) rotor from the assignment handout.

Rotor geometry and airfoil model come directly from the handout table:
    R = 0.762 m, root cutout = 0.125 m, chord = 0.0508 m (constant, untapered)
    Cl = 5.75 * alpha,  Cd = 0.0113 + 1.25 * alpha^2   (alpha in radians)
    Blade number: choose 2/3/4/5 -- set NUM_BLADES below and justify choice
    in your report.

IMPORTANT: This script does NOT contain the actual experimental CT/CQ vs.
collective values -- those must be digitized from the Knight & Hefner
source (or your instructor-provided dataset) and placed in
`data/knight_hefner_template.csv` (rename it once filled in, e.g.
`data/knight_hefner_data.csv`). Do not submit results based on invented
experimental numbers.

Error metrics computed: RMSE and mean absolute percentage error (MAPE),
satisfying "quantify error using at least two metrics" (Task 3).
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

from environment import isa
from airfoil import LinearAirfoil
from rotor import Rotor, constant_chord, constant_twist
from bemt import run_bemt

# ---- Rotor & test-condition setup (from handout table) -------------------
NUM_BLADES = 2          # <-- choose 2/3/4/5 and justify in your report
RADIUS_M = 0.762
ROOT_CUTOUT_M = 0.125
CHORD_M = 0.0508
OMEGA_RAD_S = 2 * np.pi * 1250.0 / 60.0   # Knight & Hefner test speed: ~1250 RPM
                                            # (SET to the RPM used in your digitized figure)
ALTITUDE_M = 0.0
DISA_K = 0.0

airfoil = LinearAirfoil()  # uses handout's a0=5.75, Cd_min=0.0113, eps=1.25


def build_validation_rotor() -> Rotor:
    return Rotor(
        radius_m=RADIUS_M,
        root_cutout_m=ROOT_CUTOUT_M,
        num_blades=NUM_BLADES,
        chord_fn=constant_chord(CHORD_M),
        twist_fn=constant_twist(0.0),   # untwisted validation rotor
        name="Knight-Hefner validation rotor",
    )


def load_experimental_data(csv_path: str):
    coll_deg, CT_exp, CQ_exp = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            if row["collective_deg"].strip().startswith("#"):
                continue
            coll_deg.append(float(row["collective_deg"]))
            CT_exp.append(float(row["CT_exp"]))
            CQ_exp.append(float(row["CQ_exp"]))
    return np.array(coll_deg), np.array(CT_exp), np.array(CQ_exp)


def error_metrics(pred, exp):
    pred, exp = np.asarray(pred), np.asarray(exp)
    rmse = float(np.sqrt(np.mean((pred - exp) ** 2)))
    mape = float(np.mean(np.abs((pred - exp) / np.where(exp == 0, 1e-12, exp))) * 100)
    return rmse, mape


def run_validation(csv_path: str, collective_sweep_deg=None, plot=True, save_path=None):
    atmo = isa(ALTITUDE_M, DISA_K)
    rotor = build_validation_rotor()

    coll_exp_deg, CT_exp, CQ_exp = load_experimental_data(csv_path)
    sweep_deg = collective_sweep_deg if collective_sweep_deg is not None else coll_exp_deg

    CT_pred, CQ_pred, FM_pred = [], [], []
    for coll in sweep_deg:
        perf = run_bemt(rotor, lambda x: airfoil, OMEGA_RAD_S, np.radians(coll),
                         atmo.density_kg_m3, atmo.speed_of_sound_mps, v_axial=0.0)
        CT_pred.append(perf.CT)
        CQ_pred.append(perf.CQ)
        FM_pred.append(perf.figure_of_merit)

    CT_pred, CQ_pred = np.array(CT_pred), np.array(CQ_pred)

    # Only compare at collectives that actually have experimental data.
    CT_rmse, CT_mape = error_metrics(np.interp(coll_exp_deg, sweep_deg, CT_pred), CT_exp)
    CQ_rmse, CQ_mape = error_metrics(np.interp(coll_exp_deg, sweep_deg, CQ_pred), CQ_exp)

    print(f"CT: RMSE={CT_rmse:.5f}  MAPE={CT_mape:.2f}%")
    print(f"CQ: RMSE={CQ_rmse:.6f}  MAPE={CQ_mape:.2f}%")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        axes[0].plot(sweep_deg, CT_pred, "-", label="BEMT (predicted)")
        axes[0].plot(coll_exp_deg, CT_exp, "o", label="Knight & Hefner (experiment)")
        axes[0].set_xlabel("Collective pitch [deg]")
        axes[0].set_ylabel("Thrust coefficient, CT")
        axes[0].set_title("Thrust coefficient vs. collective")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(sweep_deg, CQ_pred, "-", label="BEMT (predicted)")
        axes[1].plot(coll_exp_deg, CQ_exp, "o", label="Knight & Hefner (experiment)")
        axes[1].set_xlabel("Collective pitch [deg]")
        axes[1].set_ylabel("Torque coefficient, CQ")
        axes[1].set_title("Torque coefficient vs. collective")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"BEMT validation, hover, sea level, B={NUM_BLADES} blades, "
                      f"Omega={OMEGA_RAD_S:.1f} rad/s")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.show()

    return dict(CT_rmse=CT_rmse, CT_mape=CT_mape, CQ_rmse=CQ_rmse, CQ_mape=CQ_mape)


if __name__ == "__main__":
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "data", "knight_hefner_data.csv")
    if not os.path.exists(csv_path):
        print("No digitized experimental data found. Copy "
              "data/knight_hefner_template.csv to data/knight_hefner_data.csv "
              "and fill in real Knight & Hefner values before running validation.")
    else:
        sweep = np.linspace(2, 14, 25)  # deg, dense sweep for smooth curves
        run_validation(csv_path, collective_sweep_deg=sweep,
                       save_path="outputs/validation_plot.png")
