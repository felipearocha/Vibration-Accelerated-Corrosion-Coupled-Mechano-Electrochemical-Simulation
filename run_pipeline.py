"""
INTEGRITY CODE SERIES - Bonus
Master Pipeline Runner

Execution order:
    1. Run tests (abort if any fail)
    2. Run physics simulation (1-hour vibration, 100s at full resolution)
    3. Run parametric sweep (generates 50,000 physics-consistent data points)
    4. Train ML surrogate (GBR on parametric sweep output)
    5. Run validation suite
    6. Generate all visuals
    7. Print cybersecurity threat model

Usage:
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
    print(f"  RMS displacement:       {s['x_rms_um']:.2f} um")
    print(f"  RMS surface stress:     {s['sigma_rms_MPa']:.2f} MPa")
    print(f"  Vibration-driven CR:    {s['CR_mean_mmyr']:.4f} mm/yr")
    print(f"  Static CR (no vib):     {s['CR_static_mmyr']:.4f} mm/yr")
    print(f"  Vibration Amplification Factor (VAF): {s['vibration_amplification']:.3f}x")
    print(f"  Fretting loss ({s['n_cycles_simulated']} cycles): {s['fretting_loss_nm']:.2f} nm")
    print(f"  Projected 1-yr depth:   {s['projected_1yr_depth_mm']:.3f} mm")
    return results


def run_sweep():
    section("STEP 3: PARAMETRIC SWEEP (50,000 data points)")
    from simulation.parametric_sweep import run_parametric_sweep
    os.makedirs("assets", exist_ok=True)
    df = run_parametric_sweep(output_path="assets/parametric_sweep.csv")
    print(f"\n  Dataset: {len(df):,} rows x {len(df.columns)} columns")
    dom = df[df["in_domain"] == 1]
    print(f"  In-domain rows: {len(dom):,}, VAF {dom['VAF'].min():.3f} to {dom['VAF'].max():.3f}")
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


def print_cybersecurity():
    section("STEP 7: CYBERSECURITY SUMMARY")
    from cybersecurity.sensor_security import print_threat_model
    print_threat_model()


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

    print_cybersecurity()

    elapsed = time.time() - t0
    section("PIPELINE COMPLETE")
    print(f"  Total elapsed: {elapsed:.1f}s")
    print("\n  Outputs in: assets/\n")


if __name__ == "__main__":
    main()
