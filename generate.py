#!/usr/bin/env python3
"""
generate.py - synthetic machine-condition data for predictive maintenance.

OUTPUT (in --out folder)
  predictive_maintenance_full.csv   extended dataset, 16 columns (below)
  cnc_mill_7col.csv, cnc_lathe_7col.csv, fdm_printer_7col.csv
                                    plain 7-column files: 6 sensor columns + condition

FULL DATASET COLUMNS
  machine_id        e.g. MILL-01, LATHE-03, FDM-02
  episode_id        one complete operating cycle of one machine (split train/test on this!)
  timestamp         reading time (one reading every 15 minutes of operating time)
  machine_type      CNC Mill / CNC Lathe / FDM Printer
  temperature       deg C   (spindle motor / extruder-stepper assembly)
  vibration         mm/s RMS
  motor_current     A
  rpm               rev/min
  operating_hours   total operating hours of the machine (keeps counting across episodes)
  workload          0-100 %
  health            hidden simulator truth, 1.0 (new) -> 0.0 (failed). NOT a model input
  failure           1 on the last row of an episode that ends in a failure, else 0
  rul_hours         remaining useful life in hours, capped at 200 (200 = no failure expected soon)
  fault_mode        active fault (e.g. Bearing wear, or two joined by " + ") or "No fault"
  fail_soon         1 if rul_hours <= 24, else 0
  condition         Normal / Warning / Critical (from health: >0.65 / 0.30-0.65 / <0.30)

USE IN TRAINING
  inputs  : temperature, vibration, motor_current, rpm, operating_hours, workload (+ machine_type)
  targets : condition, rul_hours, fail_soon, fault_mode
  never   : health, failure, rul_hours, fail_soon, fault_mode as inputs (they leak the answer)
  split   : by episode_id or machine_id, never by random rows

Usage:  python generate.py --rows 150000 --machines 5 --out data --seed 42
        (--rows is per machine type, so the default gives about 450,000 rows in total)
"""
import argparse
import os

import numpy as np
import pandas as pd

INTERVAL_H = 0.25          # one reading every 15 minutes of operating time
RUL_CAP_H = 200.0          # rul_hours is capped here
FAIL_SOON_H = 24.0         # fail_soon = rul_hours <= this
NORMAL_MAX, WARNING_MAX = 0.35, 0.70   # thresholds on degradation d = 1 - health

FEATURES = ["temperature", "vibration", "motor_current", "rpm", "operating_hours", "workload"]
FULL_COLUMNS = ["machine_id", "episode_id", "timestamp", "machine_type", *FEATURES,
                "health", "failure", "rul_hours", "fault_mode", "fail_soon", "condition"]

# fault mode = (temp_add_C, vib_mult, cur_mult, rpm_drop, rpm_jitter, temp_jitter_C), values at d = 1
AGEING = (6, 1.5, 0.10, 0.01, 0.01, 0.2)
CNC_MODES = {
    "Bearing wear":   (22, 8.0, 0.20, 0.02, 0.03, 0.5),
    "Motor overload": (35, 1.2, 0.90, 0.12, 0.02, 0.8),
    "Cooling fault":  (55, 0.4, 0.10, 0.00, 0.01, 1.5),
    "Belt/tool wear": (12, 4.5, 0.35, 0.08, 0.05, 0.5),
}
MACHINES = {
    "CNC Mill": dict(prefix="MILL", slug="cnc_mill", max_hours=15000,
                     temp_base=38, temp_load=20, vib_base=0.9, vib_load=0.8,
                     cur_idle=3.0, cur_load=9.0, rpm_min=3000, rpm_max=10000, modes=CNC_MODES),
    "CNC Lathe": dict(prefix="LATHE", slug="cnc_lathe", max_hours=15000,
                      temp_base=36, temp_load=18, vib_base=0.8, vib_load=0.7,
                      cur_idle=3.5, cur_load=10.0, rpm_min=500, rpm_max=4500, modes=CNC_MODES),
    "FDM Printer": dict(prefix="FDM", slug="fdm_printer", max_hours=8000,
                        temp_base=32, temp_load=14, vib_base=0.35, vib_load=0.35,
                        cur_idle=0.8, cur_load=1.6, rpm_min=600, rpm_max=1800,
                        modes={
                            "Axis wear":    (8, 6.0, 0.25, 0.03, 0.04, 0.3),
                            "Heater fault": (45, 0.5, 0.12, 0.00, 0.01, 3.0),
                            "Stepper stall": (18, 3.5, 1.00, 0.15, 0.06, 0.6),
                            "Extruder clog": (14, 1.8, 0.55, 0.06, 0.05, 0.8),
                        }),
}
SCENARIOS = ["healthy", "gradual", "near_failure", "sudden", "maintained"]
SCEN_P = [0.14, 0.38, 0.20, 0.14, 0.14]
PROFILES = {"idle": (5, 25), "light": (25, 45), "medium": (45, 70), "heavy": (70, 95)}
PROFILE_P = [0.15, 0.30, 0.35, 0.20]


def ar_series(rng, n, mean, phi, sigma, lo, hi):
    e = rng.normal(0, sigma, n)
    x = np.empty(n)
    x[0] = mean
    for i in range(1, n):
        x[i] = mean + phi * (x[i - 1] - mean) + e[i]
    return np.clip(x, lo, hi)


def ema(x, alpha):
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
    return y


def shape(rng, p):
    kind = rng.choice(["linear", "accel", "exp", "late"], p=[0.30, 0.30, 0.25, 0.15])
    if kind == "linear":
        return p
    if kind == "accel":
        return p ** 2
    if kind == "exp":
        k = rng.uniform(2.5, 5.0)
        return (np.exp(k * p) - 1) / (np.exp(k) - 1)
    s = 1 / (1 + np.exp(-12 * (p - rng.uniform(0.55, 0.85))))
    return (s - s[0]) / (s[-1] - s[0])


def fault_curve(rng, scen, n):
    """Hidden fault degradation (0..1) over an episode, and whether the episode ends in failure."""
    p = np.linspace(0, 1, n)
    d0 = rng.uniform(0.0, 0.08)
    if scen == "healthy":
        return rng.uniform(0, 0.04) + rng.uniform(0, 0.10) * p, False
    if scen == "gradual":
        fails = rng.random() >= 0.30
        dend = rng.uniform(0.85, 1.0) if fails else rng.uniform(0.40, 0.68)
        return d0 + (dend - d0) * shape(rng, p), fails
    if scen == "near_failure":
        d0 = rng.uniform(0.50, 0.72)
        return d0 + (rng.uniform(0.92, 1.0) - d0) * shape(rng, p), True
    if scen == "sudden":
        jump = max(2, int(n * rng.uniform(0.40, 0.85)))
        d = np.full(n, d0) + 0.05 * p
        dj = rng.uniform(0.45, 0.95)
        d[jump:] = dj + (1.0 - dj) * np.linspace(0, 1, n - jump)
        return d, True
    m_at = max(3, int(n * rng.uniform(0.55, 0.85)))     # maintained: degrade, serviced, healthy again
    peak = rng.uniform(0.50, 0.90)
    d = np.empty(n)
    d[:m_at] = d0 + (peak - d0) * shape(rng, np.linspace(0, 1, m_at))
    d[m_at:] = rng.uniform(0.0, 0.05) + 0.05 * np.linspace(0, 1, n - m_at)
    return d, False


def simulate_episode(rng, cfg, mtype, machine_id, episode_id, scen, n, hours_start, t0):
    d_fault, fails = fault_curve(rng, scen, n)
    d_fault = np.clip(d_fault + np.cumsum(rng.normal(0, 0.0015, n)), 0, 1)
    if fails:
        k = max(3, n // 20)
        d_fault[-k:] = np.linspace(d_fault[-k], 1.0, k)       # episode really ends in failure

    hours = hours_start + INTERVAL_H * np.arange(1, n + 1)
    age = np.clip(0.18 * (hours / cfg["max_hours"]) ** 1.5, 0, 0.3)
    d = 1 - (1 - d_fault) * (1 - age)
    dp = d ** 1.4

    mode, mode_name = AGEING, "No fault"
    if scen != "healthy":
        names = list(cfg["modes"])
        na = names[rng.integers(len(names))]
        mode, mode_name = cfg["modes"][na], na
        if rng.random() < 0.15:                                 # compound fault
            nb = names[rng.integers(len(names))]
            if nb != na:
                mode = tuple(0.65 * x + 0.65 * y for x, y in zip(cfg["modes"][na], cfg["modes"][nb]))
                mode_name = f"{na} + {nb}"
    t_add, v_mult, c_mult, r_drop, r_jit, t_jit = mode

    prof = list(PROFILES)[rng.choice(len(PROFILES), p=PROFILE_P)]
    w = ar_series(rng, n, rng.uniform(*PROFILES[prof]), 0.97, 5, 0, 100)
    w += rng.uniform(0, 15) * np.sin(2 * np.pi * np.arange(n) / rng.uniform(80, 300) + rng.uniform(0, 6.28))
    w = np.clip(w, 0, 100)
    load = w / 100
    sp = np.clip(0.7 * load + 0.3 * ar_series(rng, n, 0.5, 0.95, 0.05, 0, 1), 0, 1)
    load_t = ema(load, 0.08)
    amb = rng.uniform(-3, 8)

    temp = (cfg["temp_base"] + amb + cfg["temp_load"] * load_t + t_add * dp
            + rng.normal(0, 0.8 + t_jit * d))
    vib = (cfg["vib_base"] + cfg["vib_load"] * sp) * (1 + v_mult * dp) * (1 + rng.normal(0, 0.06, n))
    cur = (cfg["cur_idle"] + cfg["cur_load"] * load) * (1 + c_mult * dp) * (1 + rng.normal(0, 0.03, n))
    rpm = ((cfg["rpm_min"] + (cfg["rpm_max"] - cfg["rpm_min"]) * sp) * (1 - r_drop * dp)
           * (1 + rng.normal(0, 0.012 + r_jit * dp)))
    for i in np.flatnonzero(rng.random(n) < 0.003):             # random sensor spikes
        k = rng.integers(4)
        if k == 0:
            vib[i] *= rng.uniform(1.8, 3.2)
        elif k == 1:
            temp[i] += rng.uniform(5, 12)
        elif k == 2:
            cur[i] *= rng.uniform(1.3, 1.8)
        else:
            rpm[i] *= rng.uniform(0.7, 0.9)

    ttf = (n - 1 - np.arange(n)) * INTERVAL_H
    rul = np.minimum(ttf, RUL_CAP_H) if fails else np.full(n, RUL_CAP_H)
    failure = np.zeros(n, dtype=int)
    if fails:
        failure[-1] = 1
    fault = np.where((scen != "healthy") & (d_fault >= 0.12), mode_name, "No fault")
    cond = np.select([d < NORMAL_MAX, d < WARNING_MAX], ["Normal", "Warning"], "Critical")

    return pd.DataFrame({
        "machine_id": machine_id, "episode_id": episode_id,
        "timestamp": pd.date_range(t0, periods=n, freq="15min"),
        "machine_type": mtype,
        "temperature": np.round(np.maximum(temp, 15), 2),
        "vibration": np.round(np.maximum(vib, 0.05), 3),
        "motor_current": np.round(np.maximum(cur, 0.2), 3),
        "rpm": np.round(np.maximum(rpm, 0), 0).astype(int),
        "operating_hours": np.round(hours, 2),
        "workload": np.round(w + rng.normal(0, 1.5, n), 1).clip(0, 100),
        "health": np.round(1 - d, 4), "failure": failure, "rul_hours": np.round(rul, 2),
        "fault_mode": fault, "fail_soon": (rul <= FAIL_SOON_H).astype(int), "condition": cond,
    })[FULL_COLUMNS]


def generate_type(mtype, cfg, rows, n_machines, rng, ep_counter):
    machines = [dict(id=f"{cfg['prefix']}-{i + 1:02d}",
                     hours=rng.uniform(0, 0.3 * cfg["max_hours"]),
                     t=pd.Timestamp("2026-01-01") + pd.Timedelta(hours=float(rng.uniform(0, 500))))
                for i in range(n_machines)]
    parts, total, k = [], 0, 0
    while total < rows:
        m = machines[k % n_machines]
        k += 1
        scen = SCENARIOS[rng.choice(len(SCENARIOS), p=SCEN_P)]
        n = int(rng.integers(150, 600)) if scen == "near_failure" else \
            int(rng.integers(400, 1500)) if scen == "sudden" else int(rng.integers(600, 2400))
        ep_counter[0] += 1
        parts.append(simulate_episode(rng, cfg, mtype, m["id"], f"EP-{ep_counter[0]:05d}",
                                      scen, n, m["hours"], m["t"]))
        m["hours"] += n * INTERVAL_H
        m["t"] += pd.Timedelta(minutes=15 * n) + pd.Timedelta(hours=float(rng.uniform(12, 72)))
        total += n
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=150_000, help="rows per machine type (default 150000)")
    ap.add_argument("--machines", type=int, default=5, help="machines per type (default 5)")
    ap.add_argument("--out", default="data", help="output folder")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng, ep = np.random.default_rng(a.seed), [0]
    frames = []
    for mtype, cfg in MACHINES.items():
        df = generate_type(mtype, cfg, a.rows, a.machines, rng, ep)
        df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
        df[FEATURES + ["condition"]].to_csv(os.path.join(a.out, f"{cfg['slug']}_7col.csv"), index=False)
        frames.append(df)
        share = (df["condition"].value_counts(normalize=True) * 100).round(1)
        print(f"{mtype}: {len(df):,} rows, {df['episode_id'].nunique()} episodes, "
              f"{int(df['failure'].sum())} failures | "
              + " | ".join(f"{c} {share.get(c, 0)}%" for c in ["Normal", "Warning", "Critical"])
              + f" | fail_soon {df['fail_soon'].mean() * 100:.1f}%")
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(os.path.join(a.out, "predictive_maintenance_full.csv"), index=False)
    print(f"\nFull dataset: {len(full):,} rows x {full.shape[1]} columns -> {a.out}/predictive_maintenance_full.csv")


if __name__ == "__main__":
    main()