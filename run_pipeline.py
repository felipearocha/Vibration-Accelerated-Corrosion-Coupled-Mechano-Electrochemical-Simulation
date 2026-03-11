"""
INTEGRITY CODE SERIES - Week 4
Master Pipeline Runner

Execution order:
    1. Run tests (abort if any fail)
    2. Run physics simulation (1-hour vibration, 100s at full resolution)
    3. Run parametric sweep (generates 50,000 physics-consistent data points)
    4. Train ML surrogate (GBR on parametric sweep output)
    5. Run validation suite
    6. Generate all visuals
    7. Print ROI summary
    8. Print cybersecurity threat model
    9. Print LinkedIn summary

Usage:
    cd integrity_code_series_week4_vibrocorrosion
    pip install -r requirements.txt
    python run_pipeline.py [--skip-tests] [--skip-sweep] [--skip-ml] [--skip-viz]
"""

import argparse
import subprocess
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def run_tests():
    section("STEP 1: UNIT TEST SUITE")
    result = subprocess.run([sys.executable, "tests/test_all.py"], capture_output=False)
    if result.returncode != 0:
        print("[ABORT] Tests failed. Fix before proceeding.")
        sys.exit(1)


def run_simulation():
    section("STEP 2: PHYSICS SIMULATION")
    from simulation.vibrocorrosion_engine import (
        MaterialParams, EnvironmentParams, VibrationParams, run_simulation
    )
    results = run_simulation(t_end_seconds=100.0, dt=1e-4)
    s = results["summary"]
    print(f"\n  Natural frequency:      {s['natural_freq_hz']:.2f} Hz")
    print(f"  Excitation frequency:   {s['excitation_freq_hz']:.2f} Hz")
    print(f"  RMS displacement:       {s['x_rms_um']:.2f} μm")
    print(f"  RMS surface stress:     {s['sigma_rms_MPa']:.2f} MPa")
    print(f"  Vibration-driven CR:    {s['CR_mean_mmyr']:.4f} mm/yr")
    print(f"  Static CR (no vib):     {s['CR_static_mmyr']:.4f} mm/yr")
    print(f"  Vibration Amplification Factor (VAF): {s['vibration_amplification']:.3f}x")
    print(f"  Fretting loss ({s['n_cycles_simulated']} cycles): {s['fretting_loss_nm']:.2f} nm")
    print(f"  Projected 1-yr depth:   {s['projected_1yr_depth_mm']:.3f} mm/yr")
    return results


def run_sweep():
    section("STEP 3: PARAMETRIC SWEEP (10,000+ data points)")
    from simulation.parametric_sweep import run_parametric_sweep
    os.makedirs("assets", exist_ok=True)
    df = run_parametric_sweep(output_path="assets/parametric_sweep.csv")
    print(f"\n  Dataset: {len(df):,} rows x {len(df.columns)} columns")
    print(f"  VAF range: {df['VAF'].min():.3f} to {df['VAF'].max():.3f}")
    return df


def run_ml(df):
    section("STEP 4: ML SURROGATE TRAINING (GBR)")
    from ml.gbr_surrogate import train_gbr
    result = train_gbr(df, model_dir="assets/models")
    m = result["metrics"]
    print(f"\n  Test R²:   {m['test_r2']:.4f}")
    print(f"  Test MAE:  {m['test_mae']:.5f} mm/yr")
    print(f"  Test RMSE: {m['test_rmse']:.5f} mm/yr")
    print(f"  Test MAPE: {m['test_mape']:.2f}%")
    print(f"  Monotonicity check: {m['monotone_force_check']}")
    return result


def run_validation():
    section("STEP 5: VALIDATION SUITE")
    from validation.validation_suite import run_all_validations
    return run_all_validations(df_path="assets/parametric_sweep.csv")


def run_visuals():
    section("STEP 6: VISUAL GENERATION")
    from visualization.viz_suite import run_all_visuals
    run_all_visuals()


def print_roi_summary():
    section("STEP 7: ROI ANALYSIS")
    print("""
  CONTEXT:
    Asset: subsea pipeline section with known vortex-induced vibration
    Inspection interval: currently 12 months (calendar-based)
    Asset consequence of failure: HIGH (product loss + HSE)

  WITHOUT vibration-corrosion coupling model:
    - Corrosion rate estimate: static model only (does not account for vibration)
    - Inspection interval: fixed 12 months regardless of vibration state
    - Missed inspections at resonance: possible (VAF can exceed 2-5x near resonance)
    - False alarm rate: not estimated

  WITH this system:
    - Real-time VAF computed from accelerometer + electrochemical probe
    - Inspection interval adjusted dynamically based on CR prediction
    - At VAF > 2.0: interval halved (6 months)
    - At VAF < 1.1: interval extended (18 months)

  ESTIMATED VALUE (order-of-magnitude, not guaranteed without site calibration):
    Inspection cost:                ~$50,000 per event (offshore)
    Avoided premature inspection:   1 per year saved  = ~$50,000/yr
    Avoided extended interval:      1 failure averted = depends on consequence
    Production downtime avoided:    1 day = $200,000-$500,000 (asset-dependent)

  ENGINEERING HOURS SAVED:
    Manual corrosion rate estimation:   4-8 hrs/event x 4 events/yr = 16-32 hrs
    With this system (automated):       0.5 hr/event for review = 2 hrs/yr
    Net saving:                         14-30 hrs/yr per asset

  SIMULATION TIME:
    Full ODE per query:         ~200ms
    GBR surrogate per query:    ~0.2ms
    Speedup:                    ~1000x  (justified ML deployment)

  UNCERTAINTY STATEMENT:
    All ROI figures are engineering estimates.
    Site-specific calibration of beta_stress is mandatory before deployment.
    Fretting synergy term not included: may underestimate total loss rate by 20-50%.
    External validation against coupon data required.
    """)


def print_cybersecurity():
    section("STEP 8: CYBERSECURITY SUMMARY")
    from cybersecurity.sensor_security import print_threat_model
    print_threat_model()


def print_linkedin():
    section("STEP 9: LINKEDIN TECHNICAL SUMMARY")
    with open("linkedin/linkedin_post.txt") as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests",  action="store_true")
    parser.add_argument("--skip-sweep",  action="store_true")
    parser.add_argument("--skip-ml",     action="store_true")
    parser.add_argument("--skip-viz",    action="store_true")
    args = parser.parse_args()

    t0 = time.time()

    if not args.skip_tests:
        run_tests()

    sim_results = run_simulation()

    df = None
    if not args.skip_sweep:
        df = run_sweep()
    else:
        import pandas as pd
        if os.path.exists("assets/parametric_sweep.csv"):
            df = pd.read_csv("assets/parametric_sweep.csv")
            print(f"[SKIP] Loaded existing sweep: {len(df):,} rows")

    if not args.skip_ml and df is not None:
        run_ml(df)

    run_validation()

    if not args.skip_viz:
        run_visuals()

    print_roi_summary()
    print_cybersecurity()
    print_linkedin()

    elapsed = time.time() - t0
    section("PIPELINE COMPLETE")
    print(f"  Total elapsed: {elapsed:.1f}s")
    print("\n  Outputs in: assets/")
    print("  This repository is intended to be zipped as:")
    print("  integrity_code_series_week4_vibrocorrosion.zip\n")


if __name__ == "__main__":
    main()
