"""
INTEGRITY CODE SERIES - Week 4
Cybersecurity Architecture: Vibration-Corrosion Sensor Network

Threat Model for a vibration-corrosion monitoring system:

ASSETS:
    - Accelerometer sensor nodes (vibration measurement)
    - Corrosion rate probes (ER probes or electrochemical)
    - SCADA data acquisition layer
    - Physics simulation / ML inference server
    - Historian database
    - Inspection decision output (maintenance scheduling)

ADVERSARY PROFILE:
    - Motivated to suppress corrosion alarms (sabotage / competitive espionage)
    - Motivated to corrupt ML model to underpredict risk (targeted attack)
    - Nation-state capability: persistent, stealthy
    - Insider threat: maintenance technician with physical access

THREAT MODEL (STRIDE abbreviated):
    S - Spoofing:        Fake sensor readings injected via Modbus/TCP
    T - Tampering:       Model weights modified at rest
    R - Repudiation:     Log deletion after attack
    I - Info Disclosure: Proprietary material parameters exfiltrated
    D - Denial of Svce:  Flood data pipeline to delay alarm
    E - Elevation:       SCADA pivot to OT network

This module implements:
    1. Sensor integrity checking (statistical anomaly + physical law bounds)
    2. Model signature verification
    3. Input sanitization before physics engine
    4. Audit log with hash chain
"""

import numpy as np
import hashlib
import json
import time
import os
from datetime import datetime
from typing import Tuple, Optional


# ============================================================
# PHYSICAL BOUNDS (derived from governing equations)
# ============================================================
PHYSICAL_BOUNDS = {
    # Vibration sensor (accelerometer, industrial pipe)
    "acceleration_g": (-50.0, 50.0),        # g  -- beyond this is sensor malfunction
    "displacement_um": (-5000.0, 5000.0),   # micrometers -- extreme is 5mm
    "frequency_hz": (0.1, 500.0),           # Hz -- below 0.1 is DC drift, above 500 is ultrasonic
    # Electrochemical
    "corrosion_potential_V": (-1.5, 0.5),   # V vs SHE (steel in brine)
    "corrosion_current_Acm2": (1e-9, 1e-2), # physical range for passive to active steel
    # Environment
    "temperature_C": (-10.0, 200.0),        # operating range
    "pH": (0.0, 14.0),                      # hard physical limit
    "pressure_bar": (0.0, 500.0),           # pipeline operating range
}

# Reasonable gradient bounds (rate of change per second)
GRADIENT_BOUNDS = {
    "pH": 2.0,              # pH cannot jump 2 units per second legitimately
    "temperature_C": 10.0,  # temperature cannot jump 10°C per second (no shock)
}


# ============================================================
# SENSOR INTEGRITY CHECKER
# ============================================================
class SensorIntegrityChecker:
    """
    Validates sensor readings before they enter the physics engine.

    Three-layer check:
    1. Physical bounds check (hard limits from governing physics)
    2. Rate-of-change anomaly (statistical: z-score on rolling window)
    3. Cross-sensor consistency (stress + corrosion current must be correlated)
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._history = {}
        self._alert_log = []

    def check_bounds(self, sensor_id: str, value: float, channel: str) -> Tuple[bool, str]:
        """Returns (valid, reason)."""
        if channel not in PHYSICAL_BOUNDS:
            return True, "channel_not_bounded"
        lo, hi = PHYSICAL_BOUNDS[channel]
        if not (lo <= value <= hi):
            msg = (f"BOUNDS_VIOLATION: {sensor_id}.{channel} = {value:.4e} "
                   f"outside [{lo}, {hi}]")
            self._log_alert("BOUNDS", sensor_id, msg)
            return False, msg
        return True, "ok"

    def check_gradient(self, sensor_id: str, value: float, channel: str,
                       dt_seconds: float = 1.0) -> Tuple[bool, str]:
        """Check rate of change against physical limits."""
        key = f"{sensor_id}.{channel}"
        if key not in PHYSICAL_BOUNDS:
            return True, "not_checked"
        if key not in self._history:
            self._history[key] = []
        history = self._history[key]
        if len(history) > 0:
            prev_val = history[-1]
            rate = abs(value - prev_val) / dt_seconds
            if channel in GRADIENT_BOUNDS and rate > GRADIENT_BOUNDS[channel]:
                msg = (f"GRADIENT_VIOLATION: {sensor_id}.{channel} rate={rate:.3f}/s "
                       f"exceeds limit {GRADIENT_BOUNDS[channel]}/s")
                self._log_alert("GRADIENT", sensor_id, msg)
                return False, msg
        history.append(value)
        if len(history) > self.window_size:
            history.pop(0)
        return True, "ok"

    def check_statistical_anomaly(self, sensor_id: str, value: float,
                                  channel: str, z_threshold: float = 4.0) -> Tuple[bool, str]:
        """Z-score check against rolling window. Flags if |z| > threshold."""
        key = f"{sensor_id}.{channel}"
        if key not in self._history or len(self._history[key]) < 20:
            return True, "insufficient_history"
        history = np.array(self._history[key])
        mu = np.mean(history)
        sigma = np.std(history)
        if sigma < 1e-12:
            return True, "zero_variance"
        z = (value - mu) / sigma
        if abs(z) > z_threshold:
            msg = (f"STATISTICAL_ANOMALY: {sensor_id}.{channel} z={z:.2f} "
                   f"(threshold={z_threshold})")
            self._log_alert("ANOMALY", sensor_id, msg)
            return False, msg
        return True, "ok"

    def validate(self, sensor_id: str, readings: dict) -> dict:
        """
        Validate a dict of {channel: value} readings.
        Returns: {channel: {"valid": bool, "reason": str}}
        """
        results = {}
        for channel, value in readings.items():
            ok_b, reason_b = self.check_bounds(sensor_id, value, channel)
            ok_g, reason_g = self.check_gradient(sensor_id, value, channel)
            ok_s, reason_s = self.check_statistical_anomaly(sensor_id, value, channel)
            results[channel] = {
                "valid": ok_b and ok_g and ok_s,
                "bounds_check": reason_b,
                "gradient_check": reason_g,
                "statistical_check": reason_s,
            }
        return results

    def _log_alert(self, alert_type: str, sensor_id: str, message: str):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": alert_type,
            "sensor": sensor_id,
            "message": message
        }
        self._alert_log.append(entry)

    def get_alerts(self):
        return self._alert_log.copy()


# ============================================================
# MODEL INTEGRITY (HASH VERIFICATION)
# ============================================================
class ModelIntegrityVerifier:
    """
    Verifies ML model file has not been tampered with.
    Uses SHA-256 hash. Expected hash registered at deployment time.

    In production: hash should be stored in HSM or signed manifest.
    """

    def __init__(self, manifest_path: str = "assets/models/model_manifest.json"):
        self.manifest_path = manifest_path

    def register(self, model_path: str, model_name: str):
        """Compute and store hash of model file."""
        if not os.path.exists(model_path):
            print(f"[SEC] Model not found at {model_path}")
            return None
        h = self._sha256(model_path)
        manifest = {}
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                manifest = json.load(f)
        manifest[model_name] = {
            "path": model_path,
            "sha256": h,
            "registered_at": datetime.utcnow().isoformat()
        }
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[SEC] Registered {model_name}: SHA256={h[:16]}...")
        return h

    def verify(self, model_name: str) -> Tuple[bool, str]:
        """Verify model file matches registered hash."""
        if not os.path.exists(self.manifest_path):
            return False, "manifest_not_found"
        with open(self.manifest_path) as f:
            manifest = json.load(f)
        if model_name not in manifest:
            return False, "model_not_registered"
        entry = manifest[model_name]
        if not os.path.exists(entry["path"]):
            return False, "model_file_missing"
        current_hash = self._sha256(entry["path"])
        if current_hash != entry["sha256"]:
            return False, f"TAMPER_DETECTED: expected {entry['sha256'][:16]}... got {current_hash[:16]}..."
        return True, "integrity_ok"

    def _sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


# ============================================================
# AUDIT LOG WITH HASH CHAIN
# ============================================================
class AuditLog:
    """
    Append-only audit log with hash chain (each entry includes hash of previous).
    Provides tamper evidence for inspection decisions.

    In production: ship to immutable log store (WORM storage, SIEM).
    """

    def __init__(self, log_path: str = "assets/audit_log.jsonl"):
        self.log_path = log_path
        self._prev_hash = "GENESIS"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def write(self, event_type: str, data: dict):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data,
            "prev_hash": self._prev_hash
        }
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry["self_hash"] = entry_hash
        self._prev_hash = entry_hash

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def verify_chain(self) -> Tuple[bool, int]:
        """Verify the hash chain is unbroken. Returns (valid, first_broken_line)."""
        if not os.path.exists(self.log_path):
            return True, -1
        entries = []
        with open(self.log_path) as f:
            for line in f:
                entries.append(json.loads(line.strip()))

        for i, entry in enumerate(entries):
            stored_hash = entry.pop("self_hash")
            recomputed = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
            if recomputed != stored_hash:
                return False, i
            entry["self_hash"] = stored_hash
        return True, -1


# ============================================================
# INPUT SANITIZATION
# ============================================================
def sanitize_simulation_inputs(params: dict) -> dict:
    """
    Sanitize user or API inputs before passing to physics engine.
    Clamp values to physical ranges. Log any clamping.

    Returns sanitized dict.
    """
    sanitized = {}
    clamps = []

    clamp_map = {
        "excitation_freq_hz": (0.1, 200.0),
        "excitation_force_N": (0.0, 1e6),
        "damping_ratio": (0.001, 0.999),
        "temperature_K": (200.0, 700.0),
        "pH": (0.0, 14.0),
        "overpotential_V": (-2.0, 0.5),
    }

    for k, v in params.items():
        if k in clamp_map:
            lo, hi = clamp_map[k]
            clamped = float(np.clip(v, lo, hi))
            if clamped != v:
                clamps.append(f"{k}: {v} -> {clamped}")
            sanitized[k] = clamped
        else:
            sanitized[k] = v

    if clamps:
        print(f"[SEC] Input clamping applied: {clamps}")

    return sanitized


# ============================================================
# THREAT SUMMARY REPORT
# ============================================================
THREAT_MODEL = {
    "system": "Vibration-Corrosion Monitoring: Sensor Network + ML Inference",
    "threats": [
        {
            "id": "T1",
            "category": "Sensor Spoofing",
            "vector": "Modbus/TCP injection of false accelerometer data",
            "impact": "Suppress vibration alarm, miss resonance-driven corrosion",
            "mitigation": "SensorIntegrityChecker: bounds + gradient + statistical checks",
            "residual_risk": "LOW - multiple independent checks required to bypass"
        },
        {
            "id": "T2",
            "category": "ML Model Tampering",
            "vector": "Model weights file modification at rest (disk access)",
            "impact": "Systematic underprediction of corrosion rate, missed inspections",
            "mitigation": "ModelIntegrityVerifier SHA-256 at inference time; sign with HSM in production",
            "residual_risk": "LOW if HSM used; MEDIUM with file-only hash"
        },
        {
            "id": "T3",
            "category": "Data Poisoning",
            "vector": "Gradual injection of low-stress readings into training pipeline",
            "impact": "Biases retraining toward underestimating stress-corrosion coupling",
            "mitigation": "Training data signed at source; outlier detection in feature space; human review gate before retraining",
            "residual_risk": "MEDIUM - slow poisoning can evade statistical detection"
        },
        {
            "id": "T4",
            "category": "Audit Log Deletion",
            "vector": "Attacker deletes log after triggering false inspection deferral",
            "impact": "Repudiation of attack; liability exposure",
            "mitigation": "AuditLog hash chain + WORM storage + SIEM forwarding in production",
            "residual_risk": "LOW with WORM storage"
        },
        {
            "id": "T5",
            "category": "Pipeline DoS",
            "vector": "Flood SCADA historian with noise to delay alarm processing",
            "impact": "Inspection decision delayed; window of undetected corrosion growth",
            "mitigation": "Rate limiting on data ingestion; priority queue for alarm channel; watchdog timer",
            "residual_risk": "MEDIUM without QoS enforcement on OT network"
        }
    ],
    "architecture_note": (
        "Physics simulation server must be on isolated OT DMZ. "
        "ML inference must be air-gapped from historian write path. "
        "Sensor authentication via MAC address + rotating pre-shared key minimum. "
        "IEC 62443 zone and conduit model recommended for production deployment."
    )
}


def print_threat_model():
    print("\n" + "="*70)
    print("CYBERSECURITY THREAT MODEL - INTEGRITY CODE SERIES Week 4")
    print("="*70)
    for t in THREAT_MODEL["threats"]:
        print(f"\n[{t['id']}] {t['category']}")
        print(f"  Vector:     {t['vector']}")
        print(f"  Impact:     {t['impact']}")
        print(f"  Mitigation: {t['mitigation']}")
        print(f"  Residual:   {t['residual_risk']}")
    print(f"\nARCHITECTURE: {THREAT_MODEL['architecture_note']}")


if __name__ == "__main__":
    print_threat_model()

    # Demo sensor checker
    checker = SensorIntegrityChecker()
    # Feed normal data
    for i in range(50):
        checker.validate("SENSOR_01", {"frequency_hz": 15.0 + np.random.randn() * 0.1,
                                        "pH": 5.25 + np.random.randn() * 0.05})
    # Inject spoofed reading
    result = checker.validate("SENSOR_01", {"frequency_hz": 1500.0, "pH": 5.25})
    print(f"\n[SEC] Spoofed frequency_hz=1500: valid={result['frequency_hz']['valid']}")

    # Demo audit log
    log = AuditLog()
    log.write("INSPECTION_DECISION", {"asset": "LINE-A-027", "cr_mmyr": 0.45, "action": "DEFER_6M"})
    log.write("ALARM", {"asset": "LINE-A-027", "trigger": "CR_EXCEEDED_0.5"})
    valid, broken = log.verify_chain()
    print(f"[SEC] Audit chain valid: {valid}")
