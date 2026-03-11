# INTEGRITY CODE SERIES - Week 4
## Vibration-Accelerated Corrosion: Coupled Mechano-Electrochemical Simulation

**ZIP name:** `integrity_code_series_week4_vibrocorrosion.zip`

---

## WHAT THIS IS

A physics-first engineering simulation of vibration-accelerated corrosion in X65 carbon steel pipe under CO2-saturated brine. The system couples:

- Damped single-degree-of-freedom structural vibration (SDOF)
- Stress-modified Butler-Volmer electrochemical kinetics
- Faraday mass loss rate (Corrosion)
- Archard fretting wear at pipe supports

An ML surrogate (GBR) is trained on 50,000+ physics-generated data points for real-time deployment. All governing equations, boundary conditions, and assumptions are explicit in source code.

---

## EXPLICIT LIMITATIONS

1. `beta_stress` (stress-activation coefficient) is phenomenological. Calibration against coupon tests required before production use.
2. Fretting-corrosion synergy term is NOT modeled. Total material loss may be underestimated by 20-50% in severe fretting contact.
3. No external experimental dataset was used for validation. Internal consistency only.
4. pH model uses Nernst approximation only.
5. Stress model assumes spring-force / contact-area approximation. FEA required for real geometry.

---

## GOVERNING PHYSICS

### 1. SDOF Vibration
```
m*x'' + c*x' + k*x = F0*sin(omega*t)
c = 2*zeta*sqrt(k*m)
BC: x(0) = 0, x'(0) = 0
```

### 2. Stress-Modified Butler-Volmer
```
i0_eff = i0_ref * exp(beta_stress * sigma / (R*T))
i_anodic = i0_eff * exp(alpha_a * F * |eta| / (R*T))
```

### 3. Faraday Mass Loss
```
dm/dt [g/(cm²·s)] = (i [A/cm²] * M [g/mol]) / (n * F [C/mol])
CR [mm/yr] = (dm/dt / rho) * 10 * 3.156e7
```

### 4. Archard Fretting Wear
```
V_wear = K_wear * W * s * N
thickness_loss = V_wear / contact_area
```

---

## REPOSITORY STRUCTURE

```
integrity_code_series_week4_vibrocorrosion/
├── run_pipeline.py              # Master orchestrator (start here)
├── requirements.txt
├── README.md
├── src/
│   ├── simulation/
│   │   ├── vibrocorrosion_engine.py    # Core physics engine
│   │   └── parametric_sweep.py        # 50,000-point dataset generator
│   ├── ml/
│   │   └── gbr_surrogate.py           # ML surrogate training and prediction
│   ├── validation/
│   │   └── validation_suite.py        # Physics consistency checks
│   ├── visualization/
│   │   └── viz_suite.py               # All visuals + GIF
│   └── cybersecurity/
│       └── sensor_security.py         # Threat model + sensor integrity
├── tests/
│   └── test_all.py                    # Unit test suite
├── assets/                            # Generated outputs (created at runtime)
│   ├── parametric_sweep.csv
│   ├── models/
│   │   ├── gbr_model.pkl
│   │   ├── scaler.pkl
│   │   └── model_manifest.json
│   ├── hero_3d_cr_surface.png
│   ├── secondary_time_domain.png
│   ├── secondary_residuals.png
│   ├── secondary_isorisk_map.png
│   ├── secondary_sensitivity.png
│   ├── vibrocorrosion_sweep.gif
│   ├── validation_report.json
│   └── audit_log.jsonl
├── notebooks/                         # Optional Jupyter exploration
└── linkedin/
    └── linkedin_post.txt
```

---

## EXECUTION ORDER

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (tests -> simulation -> sweep -> ML -> validation -> visuals)
python run_pipeline.py

# 3. OR run steps individually:
python tests/test_all.py
python src/simulation/vibrocorrosion_engine.py
python src/simulation/parametric_sweep.py
python src/ml/gbr_surrogate.py
python src/validation/validation_suite.py
python src/visualization/viz_suite.py
python src/cybersecurity/sensor_security.py

# 4. Skip slow steps if assets already exist:
python run_pipeline.py --skip-tests --skip-sweep --skip-ml
```

---

## OUTPUT FILE NAMING CONVENTIONS

| File | Description |
|------|-------------|
| `assets/parametric_sweep.csv` | 50,000+ row physics dataset |
| `assets/models/gbr_model.pkl` | Trained GBR model |
| `assets/models/scaler.pkl` | Feature scaler |
| `assets/hero_3d_cr_surface.png` | Primary hero visualization |
| `assets/secondary_*.png` | Secondary analysis charts |
| `assets/vibrocorrosion_sweep.gif` | Animated resonance sweep |
| `assets/validation_report.json` | Validation check results |
| `assets/audit_log.jsonl` | Hash-chained audit log |

---

## EXPECTED RUNTIMES (approximate, hardware-dependent)

| Step | Estimated Time |
|------|---------------|
| Unit tests | < 30s |
| Physics simulation (100s, dt=1e-4) | 30-90s |
| Parametric sweep (50k points) | 2-5 min |
| ML training (GBR, 400 trees) | 30-120s |
| Validation suite | 1-3 min |
| Visual generation | 2-5 min |
| GIF generation | 1-2 min |

---

## CYBERSECURITY ARCHITECTURE

Threats addressed: sensor spoofing, ML model tampering, data poisoning, audit log deletion, pipeline DoS.  
See `src/cybersecurity/sensor_security.py` for full threat model and mitigation code.  
Production deployment requires IEC 62443 zone/conduit model and HSM for model hash storage.

---

## STANDARDS REFERENCED (without specific clause citation)

- API 571: Damage Mechanisms Affecting Fixed Equipment (vibration-induced fatigue/corrosion listed)
- API 580/581: Risk-Based Inspection methodology (probability of failure framework)
- IEC 62443: Industrial Automation and Control Systems cybersecurity

**NOTE:** No specific clause numbers are cited as these require license access for verification.

---

## REPRODUCIBILITY

Fixed seeds: `random_state=42` in all sklearn calls.  
ODE solver: SciPy RK45 with `rtol=1e-8, atol=1e-10`.  
All parameters defined in `MaterialParams`, `EnvironmentParams`, `VibrationParams` dataclasses.  
Sweep grid explicitly defined in `parametric_sweep.py`.

---

*INTEGRITY CODE SERIES | Physics-First Engineering | Week 4*  
*Verification over visibility. Safety over novelty.*
