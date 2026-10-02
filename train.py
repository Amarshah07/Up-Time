#!/usr/bin/env python3
"""
train.py - trains the three predictive-maintenance models on predictive_maintenance_full.csv

  1. condition model  : RandomForest   -> Normal / Warning / Critical (+ risk score 0-100)
  2. RUL model        : GradientBoost  -> remaining useful life in hours (alerts when <= 24 h)
  3. anomaly model    : IsolationForest trained on healthy rows -> "this reading looks abnormal"

Inputs are ONLY the 6 sensor columns + machine type + trend features computed per episode.
health / failure / rul_hours / fail_soon / fault_mode are never used as inputs (they leak the answer).
Train and test are split by episode, so the test machines' cycles are never seen in training.

Usage:  pip install numpy pandas scikit-learn joblib
        python train.py --data data/predictive_maintenance_full.csv --out models
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, mean_absolute_error,
                             precision_score, recall_score)

BASE = ["temperature", "vibration", "motor_current", "rpm", "operating_hours", "workload"]
TREND_SENSORS = ["temperature", "vibration", "motor_current", "rpm"]
WINDOW = 16                                   # 16 readings = 4 hours
TYPE_CODES = {"CNC Mill": 0, "CNC Lathe": 1, "FDM Printer": 2}
FEATURES = (BASE + [f"{c}_{k}" for c in TREND_SENSORS for k in ("avg", "slope", "std")] + ["type_code"])
CLASSES = ["Normal", "Warning", "Critical"]


def add_features(df):
    """Rolling features per episode. The backend must call this on the latest readings of a machine."""
    g = df.groupby("episode_id")
    for c in TREND_SENSORS:
        avg = g[c].transform(lambda s: s.rolling(WINDOW, min_periods=1).mean())
        df[f"{c}_avg"] = avg
        df[f"{c}_slope"] = avg.groupby(df["episode_id"]).diff(2 * WINDOW).fillna(0)   # change over ~8 h
        df[f"{c}_std"] = g[c].transform(lambda s: s.rolling(WINDOW, min_periods=2).std()).fillna(0)
    df["type_code"] = df["machine_type"].map(TYPE_CODES)
    return df


def risk_score(proba, classes):
    """0-100 risk from class probabilities: Warning counts half, Critical counts fully."""
    p = dict(zip(classes, proba.T))
    return 100 * (0.5 * p["Warning"] + 1.0 * p["Critical"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/predictive_maintenance_full.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--step", type=int, default=3, help="use every Nth training row (neighbours are near-duplicates)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.data, parse_dates=["timestamp"])
    df = df.sort_values(["episode_id", "timestamp"]).reset_index(drop=True)
    df = add_features(df)

    # ---- split by episode ----
    rng = np.random.default_rng(a.seed)
    eps = df["episode_id"].unique()
    test_eps = set(rng.choice(eps, int(len(eps) * 0.25), replace=False))
    is_test = df["episode_id"].isin(test_eps)
    tr = df[~is_test].iloc[::a.step]
    te = df[is_test]
    print(f"train rows {len(tr):,} ({(~is_test).sum():,} before subsampling) | test rows {len(te):,} "
          f"| test episodes {len(test_eps)}")

    # ---- 1. condition model ----
    clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced",
                                 n_jobs=-1, random_state=a.seed).fit(tr[FEATURES], tr["condition"])
    pred = clf.predict(te[FEATURES])
    print("\n=== 1. CONDITION MODEL (Normal / Warning / Critical) ===")
    print(classification_report(te["condition"], pred, labels=CLASSES, digits=3))
    print("confusion matrix (rows = true, cols = predicted; order Normal, Warning, Critical)")
    print(confusion_matrix(te["condition"], pred, labels=CLASSES))
    imp = pd.Series(clf.feature_importances_, FEATURES).sort_values(ascending=False)
    print("\ntop features driving the prediction:")
    print(imp.head(8).round(3).to_string())

    # ---- 2. remaining useful life ----
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=a.seed)
    reg.fit(tr[FEATURES], tr["rul_hours"])
    rul_pred = np.clip(reg.predict(te[FEATURES]), 0, 200)
    near = te["rul_hours"] < 200
    soon_pred = rul_pred <= 24
    print("\n=== 2. REMAINING USEFUL LIFE (hours, capped at 200) ===")
    print(f"MAE all rows            : {mean_absolute_error(te['rul_hours'], rul_pred):.1f} h")
    print(f"MAE when failure < 200 h: {mean_absolute_error(te['rul_hours'][near], rul_pred[near]):.1f} h")
    print(f"'failure within 24 h' alert -> precision {precision_score(te['fail_soon'], soon_pred):.2f}, "
          f"recall {recall_score(te['fail_soon'], soon_pred):.2f}")

    # ---- 3. anomaly detector (healthy data only) ----
    iso = IsolationForest(n_estimators=150, contamination=0.03, n_jobs=-1, random_state=a.seed)
    iso.fit(tr.loc[tr["condition"] == "Normal", FEATURES])
    flagged = iso.predict(te[FEATURES]) == -1
    print("\n=== 3. ANOMALY DETECTOR (share of rows flagged abnormal) ===")
    for c in CLASSES:
        print(f"{c:9s}: {100 * flagged[(te['condition'] == c).values].mean():5.1f} %")

    # ---- demo on one held-out machine that fails ----
    fail_eps = te.loc[te["failure"] == 1, "episode_id"].unique()
    if len(fail_eps):
        ep = te[te["episode_id"] == fail_eps[0]]
        idx = np.linspace(0, len(ep) - 1, 10).astype(int)
        rows = ep.iloc[idx]
        risk = risk_score(clf.predict_proba(rows[FEATURES]), clf.classes_)
        out = pd.DataFrame({
            "true_hours_left": rows["rul_hours"].values,
            "pred_hours_left": np.clip(reg.predict(rows[FEATURES]), 0, 200).round(0),
            "true_condition": rows["condition"].values,
            "pred_condition": clf.predict(rows[FEATURES]),
            "risk_%": risk.round(0),
            "anomaly": np.where(iso.predict(rows[FEATURES]) == -1, "yes", "no"),
        })
        print(f"\n=== DEMO: held-out {ep['machine_id'].iloc[0]} ({fail_eps[0]}), start -> failure ===")
        print(out.to_string(index=False))

    # ---- save ----
    joblib.dump({"model": clf, "features": FEATURES, "classes": list(clf.classes_)}, f"{a.out}/condition_model.joblib")
    joblib.dump({"model": reg, "features": FEATURES}, f"{a.out}/rul_model.joblib")
    joblib.dump({"model": iso, "features": FEATURES}, f"{a.out}/anomaly_model.joblib")
    print(f"\nSaved 3 models to {a.out}/")


if __name__ == "__main__":
    main()
