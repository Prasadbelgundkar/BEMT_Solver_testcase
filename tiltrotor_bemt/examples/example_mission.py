"""
Task 9 & 10 / Demonstration Cases demo: one feasible mission and one
deliberately infeasible mission (e.g. insufficient fuel loaded), showing
that MissionInfeasibleError correctly identifies the failure point.

Replace the placeholder rotor / mass / power numbers with your team's
actual Task 5 tiltrotor design before using this for your report.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rotor import Rotor, constant_chord, constant_twist, linear_twist
from airfoil import LinearAirfoil
from mission import (MissionPlanner, MissionSegment, SegmentType,
                      PowerAvailableModel, FuelModel, DesignLimits,
                      MissionInfeasibleError)

# NOTE: PLACEHOLDER rotor/aircraft numbers, chosen only so this demo is
# internally self-consistent (hover collective/RPM matched to weight;
# cruise collective/RPM/speed matched to a non-stalled operating point per
# examples/example_axial_forward_flight.py). Replace every number in this
# file with your team's actual Task 5 design before using it in your report.
airfoil = LinearAirfoil()
rotor = Rotor(radius_m=0.762, root_cutout_m=0.125, num_blades=3,
              chord_fn=constant_chord(0.06),
              twist_fn=linear_twist(np.radians(30.0), np.radians(-20.0)))

power_model = PowerAvailableModel(sea_level_power_W=90_000.0)
fuel_model = FuelModel(sfc_kg_per_J=1.8e-8)  # placeholder SFC -- replace with your engine data
limits = DesignLimits(max_tip_mach=0.9, max_stall_fraction=0.10,
                       min_power_margin_frac=0.05, min_rpm=200, max_rpm=2600,
                       min_collective_deg=-20, max_collective_deg=30,
                       reserve_fuel_kg=5.0)

# NOTE: gross mass here is deliberately small (~110 kg) to match what this
# small placeholder rotor (R=0.762 m, same scale as the Knight & Hefner
# validation rotor) can actually lift -- it is NOT a real tiltrotor aircraft
# mass. Mission Planner v1 does not auto-trim collective to hold weight; the
# user supplies collective/RPM per segment (here chosen, via the sweeps in
# example_hover_and_design_study.py / example_axial_forward_flight.py, to
# be non-stalled and to roughly balance the toy aircraft's weight). Replace
# every number in this file with your Task 5 design; consider adding an
# automatic trim solve (root-find collective for T=W) as a Milestone 2
# improvement.


def build_planner(fuel_kg):
    return MissionPlanner(rotor=rotor, airfoil_provider=lambda x: airfoil,
                           num_rotors=2, empty_mass_kg=80.0, fuel_mass_kg=fuel_kg,
                           power_model=power_model, fuel_model=fuel_model, limits=limits)


def mission_profile():
    return [
        MissionSegment("Takeoff hover", SegmentType.HOVER, duration_s=60,
                        altitude_m=0, rpm=2200, collective_deg=-2, dt_s=5),
        MissionSegment("Vertical climb", SegmentType.VERTICAL_CLIMB, duration_s=60,
                        altitude_m=200, rpm=2200, collective_deg=0,
                        vertical_speed_mps=3.0, dt_s=5),
        MissionSegment("Cruise out", SegmentType.CRUISE, duration_s=900,
                        altitude_m=3000, rpm=2200, collective_deg=10,
                        cruise_speed_mps=40.0, wind_mps=-3.0, dt_s=30),
        MissionSegment("Payload drop", SegmentType.PAYLOAD_EVENT, duration_s=0,
                        altitude_m=3000, payload_delta_kg=-10.0),
        MissionSegment("Loiter", SegmentType.LOITER, duration_s=300,
                        altitude_m=3000, rpm=2200, collective_deg=-3, dt_s=15),
        MissionSegment("Cruise back", SegmentType.CRUISE, duration_s=900,
                        altitude_m=3000, rpm=2200, collective_deg=10,
                        cruise_speed_mps=40.0, wind_mps=3.0, dt_s=30),
        MissionSegment("Landing hover", SegmentType.HOVER, duration_s=60,
                        altitude_m=0, rpm=2200, collective_deg=-2, dt_s=5),
    ]


if __name__ == "__main__":
    print("=== Feasible mission (ample fuel) ===")
    planner = build_planner(fuel_kg=30.0)
    try:
        final_state = planner.run_mission(mission_profile())
        print(f"Mission complete. Final fuel: {final_state.fuel_mass_kg:.2f} kg, "
              f"final gross mass: {final_state.gross_mass_kg:.2f} kg, "
              f"total time: {final_state.time_s/60:.1f} min")
    except MissionInfeasibleError as e:
        print(f"UNEXPECTED FAILURE: {e}")

    print("\n=== Deliberately infeasible mission (insufficient fuel) ===")
    planner2 = build_planner(fuel_kg=6.0)  # too little fuel for the profile
    try:
        planner2.run_mission(mission_profile())
        print("Mission unexpectedly completed -- adjust the infeasible test case.")
    except MissionInfeasibleError as e:
        print(f"Mission correctly flagged as infeasible:")
        print(f"  Segment : {e.segment_name}")
        print(f"  Time    : {e.time_s:.1f} s")
        print(f"  Reason  : {e.reason}")
