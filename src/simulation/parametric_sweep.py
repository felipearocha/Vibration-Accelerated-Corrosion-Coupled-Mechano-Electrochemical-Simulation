"""
INTEGRITY CODE SERIES - Week 4
Parametric Sweep Engine

Generates >= 10,000 physics-consistent data points across the
vibration-corrosion design space for model training and validation.

Parameters swept:
    - Excitation frequency (Hz): 5 to 50
    - Excitation force amplitude (N): 100 to 2000
    - Damping ratio: 0.01 to 0.10
    - Overpotential (V): -0.05 to -0.30
    - pH: 3.5 to 7.0

Outputs:
    - CSV with all input parameters and computed outputs
    - At least 10,000 rows
"""

import numpy as np
import pandas as pd
from itertools import product
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from simulation.vibrocorrosion_engine import (
    MaterialParams, EnvironmentParams, VibrationParams,
    sdof_damped_response, compute_dynamic_stress,
    butler_volmer_stress_modified, faraday_mass_loss_rate
)

# Sweep grid definition
# 5 x 5 x 4 x 5 x 4 x 5 = 10,000 combinations
FREQ_GRID     = np.linspace(5.0,  50.0, 10)   # Hz
FORCE_GRID    = np.linspace(100.0, 2000.0, 10) # N
DAMP_GRID     = np.array([0.01, 0.03, 0.05, 0.08, 0.10])
OVERPOT_GRID  = np.linspace(-0.05, -0.30, 5)  # V
PH_GRID       = np.linspace(3.5, 7.0, 4)
STIFFNESS_GRID = np.array([1e6, 3e6, 5e6, 8e6, 12e6])  # N/m

# 10 x 10 x 5 x 5 x 4 x 5 = 50,000 -- too slow for full ODE per point
# Use analytical steady-state amplitude instead for sweep speed

# ============================================================
# ANALYTICAL STEADY-STATE SDOF AMPLITUDE (faster than ODE)
# ============================================================
def sdof_steady_state_amplitude(mass, stiffness, damping_ratio, F0, omega_exc):
    """
    Steady-state amplitude of damped SDOF under harmonic excitation.
    x_ss = (F0/k) / sqrt((1 - r²)² + (2*zeta*r)²)
    where r = omega_exc / omega_n
    """
    omega_n = np.sqrt(stiffness / mass)
    r = omega_exc / omega_n
    zeta = damping_ratio
    static_defl = F0 / stiffness
    denom = np.sqrt((1 - r**2)**2 + (2*zeta*r)**2)
    return static_defl / denom


def ph_to_overpotential_correction(pH: float, pH_ref: float = 4.5) -> float:
    """
    Simplified pH correction to corrosion driving force.
    Nernst shift: approximately -0.0592 * (pH - pH_ref) V at 25°C
    This is an approximation. Does not capture all pH-dependent effects.
    """
    return -0.0592 * (pH - pH_ref)


def run_parametric_sweep(output_path: str = "assets/parametric_sweep.csv") -> pd.DataFrame:
    """
    Generate parametric sweep dataset.

    For each parameter combination:
    1. Compute steady-state vibration amplitude (analytical)
    2. Compute peak and RMS stress
    3. Compute stress-modified corrosion rate
    4. Compute fretting wear rate

    Returns DataFrame with >= 10,000 rows.
    """
    mp = MaterialParams()
    mass = 50.0
    contact_area = 1e-4  # m²
    normal_force = 200.0
    slip_amplitude = 50e-6

    records = []

    print("[SWEEP] Starting parametric sweep...")
    count = 0

    for freq_hz, force_N, damp, overpot, pH, k in product(
        FREQ_GRID, FORCE_GRID, DAMP_GRID, OVERPOT_GRID, PH_GRID, STIFFNESS_GRID
    ):
        omega_exc = 2 * np.pi * freq_hz
        omega_n = np.sqrt(k / mass)
        fn_hz = omega_n / (2 * np.pi)

        # Steady-state displacement amplitude
        x_ss = sdof_steady_state_amplitude(mass, k, damp, force_N, omega_exc)
        x_rms = x_ss / np.sqrt(2)

        # Peak stress
        sigma_peak = (k * x_ss) / contact_area
        sigma_rms = sigma_peak / np.sqrt(2)

        # Clamp stress at yield (plasticity not modeled)
        sigma_peak_clamped = min(sigma_peak, mp.yield_strength)
        sigma_rms_clamped = min(sigma_rms, mp.yield_strength)

        # pH correction to overpotential
        overpot_corrected = overpot + ph_to_overpotential_correction(pH)

        ep = EnvironmentParams(pH=pH, overpotential=overpot_corrected)

        # Corrosion rate at RMS stress (time-averaged approximation)
        i_rms = butler_volmer_stress_modified(
            np.array([sigma_rms_clamped]), mp, ep
        )[0]
        CR_rms = faraday_mass_loss_rate(np.array([i_rms]), mp)[0]

        # Corrosion rate at zero stress (baseline)
        i_static = butler_volmer_stress_modified(
            np.array([0.0]), mp, ep
        )[0]
        CR_static = faraday_mass_loss_rate(np.array([i_static]), mp)[0]

        # Vibration amplification factor
        VAF = CR_rms / CR_static if CR_static > 0 else np.nan

        # Fretting wear rate (nm per 1000 cycles)
        V_wear_per_cycle = mp.K_wear * normal_force * slip_amplitude  # m³/cycle
        thickness_per_1000_cycles_nm = (V_wear_per_cycle * 1000 / contact_area) * 1e9

        # Resonance proximity flag
        r = freq_hz / fn_hz
        at_resonance = 1 if abs(r - 1.0) < 0.1 else 0

        records.append({
            "freq_hz": freq_hz,
            "force_N": force_N,
            "damping_ratio": damp,
            "overpotential_V": overpot,
            "pH": pH,
            "stiffness_Nm": k,
            "fn_hz": fn_hz,
            "freq_ratio_r": r,
            "x_ss_um": x_ss * 1e6,
            "x_rms_um": x_rms * 1e6,
            "sigma_peak_MPa": sigma_peak_clamped / 1e6,
            "sigma_rms_MPa": sigma_rms_clamped / 1e6,
            "sigma_clamped": 1 if sigma_peak > mp.yield_strength else 0,
            "i_corr_rms_Acm2": i_rms,
            "CR_rms_mmyr": CR_rms,
            "CR_static_mmyr": CR_static,
            "VAF": VAF,                              # Vibration Amplification Factor
            "fretting_nm_per_1000cycles": thickness_per_1000_cycles_nm,
            "at_resonance": at_resonance,
            "CR_total_mmyr": CR_rms + (thickness_per_1000_cycles_nm * 1e-6 * fn_hz * 3.156e7),
        })

        count += 1
        if count % 5000 == 0:
            print(f"  [{count:,} rows generated]")

    df = pd.DataFrame(records)
    print(f"[SWEEP] Complete. Total rows: {len(df):,}")
    print(f"[SWEEP] VAF range: {df['VAF'].min():.3f} to {df['VAF'].max():.3f}")
    print(f"[SWEEP] CR range: {df['CR_rms_mmyr'].min():.4f} to {df['CR_rms_mmyr'].max():.4f} mm/yr")
    print(f"[SWEEP] Rows with sigma clamped at yield: {df['sigma_clamped'].sum():,}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SWEEP] Saved to {output_path}")
    return df


if __name__ == "__main__":
    df = run_parametric_sweep()
    print(df.describe())
