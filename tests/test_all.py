"""
INTEGRITY CODE SERIES - Week 4
Test Suite

Runs unit tests for:
- Vibration ODE solver
- Faraday mass loss calculation
- Butler-Volmer current density
- Sensor security checks
- Input sanitization
- Audit log hash chain

Usage:
    python -m pytest tests/ -q
    python tests/test_all.py
"""

import sys
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.vibrocorrosion_engine import (
    run_simulation, MaterialParams, EnvironmentParams, VibrationParams,
    sdof_damped_response, compute_dynamic_stress,
    butler_volmer_stress_modified, faraday_mass_loss_rate,
    fretting_wear_rate
)
from simulation.parametric_sweep import (
    sdof_steady_state_amplitude, ph_to_overpotential_correction,
    fretting_rate_mmyr
)
from validation.validation_suite import (
    check_resonance_amplification, check_sdof_energy_balance
)
from cybersecurity.sensor_security import (
    SensorIntegrityChecker, AuditLog, sanitize_simulation_inputs
)

mp = MaterialParams()
ep = EnvironmentParams()
vp = VibrationParams()
t, x, v = sdof_damped_response(vp, t_end=2.0, dt=5e-4)
checker = SensorIntegrityChecker()


# ============================================================
# PHYSICS ENGINE TESTS
# ============================================================
def test_sdof_solver_returns_arrays():
    assert len(t) > 100 and len(x) == len(t)


def test_sdof_initial_displacement_zero():
    assert abs(x[0]) < 1e-12


def test_sdof_initial_velocity_zero():
    assert abs(v[0]) < 1e-12


def test_sdof_reaches_nonzero_amplitude():
    assert np.max(np.abs(x[-len(x)//4:])) > 1e-6


def test_stress_is_non_negative():
    assert np.all(compute_dynamic_stress(x, vp, mp) >= 0)


def test_faraday_cr_in_physical_range():
    CR = faraday_mass_loss_rate(np.array([1e-5]), mp)  # A/cm²
    assert 0.0001 < CR[0] < 50.0


def test_bv_at_zero_stress_and_overpotential_equals_i0():
    ep_zero = EnvironmentParams(overpotential=0.0)
    i_zero = butler_volmer_stress_modified(np.array([0.0]), mp, ep_zero)[0]
    assert abs(i_zero - mp.i0_ref) < 1e-12 * mp.i0_ref


def test_bv_tafel_slope_handcalc():
    # One Tafel slope of overpotential is one decade of current, by definition.
    ep_ba = EnvironmentParams(overpotential=mp.ba)
    i = butler_volmer_stress_modified(np.array([0.0]), mp, ep_ba)[0]
    assert i == pytest.approx(10.0 * mp.i0_ref)


def test_one_year_depth_equals_mean_rate_times_one_year():
    s = run_simulation(t_end_seconds=1.0, dt=1e-3)["summary"]
    assert s["projected_1yr_depth_mm"] == pytest.approx(s["CR_mean_mmyr"])


def test_bv_current_increases_with_stress():
    i_low = butler_volmer_stress_modified(np.array([0.0]), mp, ep)[0]
    i_high = butler_volmer_stress_modified(np.array([100e6]), mp, ep)[0]  # 100 MPa
    assert i_high > i_low


def test_fretting_wear_is_positive():
    assert fretting_wear_rate(vp, mp, n_cycles=1000) > 0


# ============================================================
# SECURITY MODULE TESTS
# ============================================================
def test_valid_frequency_passes_bounds_check():
    valid, reason = checker.check_bounds("S1", 15.0, "frequency_hz")
    assert valid, reason


def test_out_of_bounds_frequency_caught():
    valid, _ = checker.check_bounds("S1", 5000.0, "frequency_hz")
    assert not valid


def test_ph_above_14_caught():
    valid, _ = checker.check_bounds("S1", 15.0, "pH")
    assert not valid


def _sanitized():
    params = {"excitation_freq_hz": 5000.0, "pH": -5.0, "damping_ratio": 0.03}
    return sanitize_simulation_inputs(params)


def test_freq_clamped_to_max():
    assert _sanitized()["excitation_freq_hz"] == 200.0


def test_ph_clamped_to_min():
    assert _sanitized()["pH"] == 0.0


def test_valid_damping_unchanged():
    assert _sanitized()["damping_ratio"] == 0.03


def _audit_log(folder):
    log = AuditLog(log_path=os.path.join(folder, "test_audit.jsonl"))
    log.write("TEST_EVENT", {"key": "value1"})
    log.write("TEST_EVENT", {"key": "value2"})
    return log


def test_audit_log_chain_intact_after_two_writes():
    with tempfile.TemporaryDirectory() as folder:
        valid_chain, broken = _audit_log(folder).verify_chain()
    assert valid_chain, f"broken_at={broken}"


def test_tampered_audit_log_detected():
    with tempfile.TemporaryDirectory() as folder:
        log = _audit_log(folder)
        path = os.path.join(folder, "test_audit.jsonl")
        with open(path, "r") as f:
            content = f.read()
        with open(path, "w") as f:
            f.write(content.replace("value1", "TAMPERED"))
        valid_chain, _ = log.verify_chain()
    assert not valid_chain


# ============================================================
# PARAMETRIC CONSISTENCY TESTS
# ============================================================
def test_fretting_rate_handcalc():
    # 1 nm per 1000 cycles = 1e-9 mm per cycle; at 15 Hz over 3.156e7 s: 0.4734 mm/yr
    assert fretting_rate_mmyr(1.0, 15.0) == pytest.approx(0.4734)


def _slice_csv(folder, peak_r):
    rows = [{"force_N": 1000.0, "damping_ratio": 0.03, "overpotential_V": 0.15,
             "pH": 4.5, "stiffness_Nm": 5e6, "freq_ratio_r": r,
             "VAF": 2.0 if r == peak_r else 1.0} for r in (0.3, 0.6, 1.0)]
    path = os.path.join(folder, "slice.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_resonance_check_passes_on_peak_at_resonance():
    with tempfile.TemporaryDirectory() as folder:
        assert check_resonance_amplification(_slice_csv(folder, 1.0)) is True


def test_resonance_check_fails_on_peak_away_from_resonance():
    with tempfile.TemporaryDirectory() as folder:
        assert check_resonance_amplification(_slice_csv(folder, 0.3)) is False


def test_resonance_check_fails_without_data():
    assert check_resonance_amplification("no_such_file.csv") is False


def test_energy_balance_check_returns_plain_bool():
    assert check_sdof_energy_balance(sim_duration=3.0) is True


def test_cr_monotonically_increases_with_force():
    CRs = []
    for force in [100.0, 500.0, 1000.0, 2000.0]:
        omega = 2 * np.pi * 15.0
        x_ss = sdof_steady_state_amplitude(50.0, 5e6, 0.03, force, omega)
        sigma = min((5e6 * x_ss / np.sqrt(2)) / 1e-4, mp.yield_strength)
        i_c = butler_volmer_stress_modified(np.array([sigma]), mp, ep)[0]
        CRs.append(faraday_mass_loss_rate(np.array([i_c]), mp)[0])
    assert all(CRs[i] <= CRs[i+1] for i in range(len(CRs)-1))


def _cr_at_ph(pH):
    eta = 0.15 + ph_to_overpotential_correction(pH)
    ep_ph = EnvironmentParams(overpotential=eta, pH=pH)
    i_c = butler_volmer_stress_modified(np.array([50e6]), mp, ep_ph)[0]
    return faraday_mass_loss_rate(np.array([i_c]), mp)[0]


def test_cr_increases_as_ph_decreases():
    CRs_pH = [_cr_at_ph(pH) for pH in [7.0, 5.5, 4.5, 3.5]]
    assert all(CRs_pH[i] < CRs_pH[i+1] for i in range(len(CRs_pH)-1))


def test_cr_ratio_per_ph_unit_handcalc():
    # 59.2 mV per pH unit on a 120 mV/decade slope: 10**(0.0592/0.12) = 3.1141
    assert _cr_at_ph(3.5) / _cr_at_ph(4.5) == pytest.approx(3.1141, abs=1e-3)


def test_bv_cathodic_overpotential_lowers_anodic_current():
    ep_cath = EnvironmentParams(overpotential=-mp.ba)
    i = butler_volmer_stress_modified(np.array([0.0]), mp, ep_cath)[0]
    assert i == pytest.approx(0.1 * mp.i0_ref)



if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
