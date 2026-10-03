#!/usr/bin/env python3
"""
predict.py - loads the 3 saved models and predicts the state of a machine from its recent readings.

TEST THE MODELS (replays unseen machines):
    python predict.py --data data/predictive_maintenance_full.csv --models models

USE IN YOUR BACKEND:
    from predict import Predictor
    p = Predictor("models")
    result = p.predict(readings)      # readings = DataFrame of the machine's latest readings

`readings` needs these columns, oldest row first, ideally the last 48 readings (15-minute spacing):
    machine_type, temperature, vibration, motor_current, rpm, operating_hours, workload
Returns: condition, risk_percent, hours_left, anomaly, reasons, probabilities
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from train import TREND_SENSORS, add_features, risk_score

NEEDED = ["machine_type", "temperature", "vibration", "motor_current", "rpm", "operating_hours", "workload"]


class Predictor:
    def __init__(self, folder="models"):
        c = joblib.load(f"{folder}/condition_model.joblib")
        self.clf, self.features = c["model"], c["features"]
        self.rul = joblib.load(f"{folder}/rul_model.joblib")["model"]
        self.iso = joblib.load(f"{folder}/anomaly_model.joblib")["model"]

    def predict(self, readings):
        df = readings[NEEDED].copy().reset_index(drop=True)
        df["episode_id"] = "live"
        df = add_features(df)
        row = df.iloc[[-1]]
        X = row[self.features]
        proba = self.clf.predict_proba(X)
        condition = str(self.clf.predict(X)[0])
        risk = round(float(risk_score(proba, self.clf.classes_)[0]), 1)
        return {
            "condition": condition,
            "risk_percent": risk,
            "hours_left": round(float(np.clip(self.rul.predict(X)[0], 0, 200)), 0),   # 200 = no failure expected soon
            "anomaly": bool(self.iso.predict(X)[0] == -1),
            "reasons": self._reasons(row) if (condition != "Normal" or risk >= 25) else [],   # only explain Warning/Critical
            "probabilities": {c: round(float(p), 3) for c, p in zip(self.clf.classes_, proba[0])},
        }

    @staticmethod
    def _reasons(row):
        """Plain-language drivers: sensors whose 4-hour average moved most over the last ~8 hours."""
        out = []
        for c in TREND_SENSORS:
            avg, slope = float(row[f"{c}_avg"].iloc[0]), float(row[f"{c}_slope"].iloc[0])
            before = avg - slope
            if before > 0:
                out.append((abs(slope / before) * 100, f"{c.replace('_', ' ')} {'up' if slope > 0 else 'down'} "
                                                       f"{abs(slope / before) * 100:.0f}% over 8 h"))
        out.sort(reverse=True)
        return [t for pct, t in out[:2] if pct >= 5]


def replay(pred, ep, title, points=9, window=48):
    print(f"\n=== {title}: {ep['machine_id'].iloc[0]} ({ep['machine_type'].iloc[0]}) ===")
    print(f"{'true_left':>9} {'pred_left':>9} {'true':>9} {'pred':>9} {'risk%':>6} {'anomaly':>7}  reasons")
    for i in np.linspace(window, len(ep) - 1, points).astype(int):
        r = pred.predict(ep.iloc[i - window + 1: i + 1])
        t = ep.iloc[i]
        print(f"{t['rul_hours']:9.0f} {r['hours_left']:9.0f} {t['condition']:>9} {r['condition']:>9} "
              f"{r['risk_percent']:6.0f} {'yes' if r['anomaly'] else 'no':>7}  {'; '.join(r['reasons'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/predictive_maintenance_full.csv")
    ap.add_argument("--models", default="models")
    ap.add_argument("--seed", type=int, default=42, help="must match train.py so the episodes are unseen")
    a = ap.parse_args()
    pred = Predictor(a.models)

    df = pd.read_csv(a.data, parse_dates=["timestamp"]).sort_values(["episode_id", "timestamp"]).reset_index(drop=True)
    eps = df["episode_id"].unique()
    test = np.random.default_rng(a.seed).choice(eps, int(len(eps) * 0.25), replace=False)   # same split as train.py
    te = df[df["episode_id"].isin(test)]
    by = te.groupby("episode_id")

    fail = next(e for e, g in by if g["failure"].iloc[-1] == 1 and len(g) > 600)
    healthy = next(e for e, g in by if (g["condition"] == "Normal").all() and len(g) > 600)
    replay(pred, te[te["episode_id"] == fail], "FAILING machine (should climb Normal -> Warning -> Critical)")
    replay(pred, te[te["episode_id"] == healthy], "HEALTHY machine (should stay Normal, low risk)")


if __name__ == "__main__":
    main()
