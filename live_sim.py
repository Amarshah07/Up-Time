#!/usr/bin/env python3
"""
live_sim.py - live machine fleet for the demo. Streams readings, runs the trained models on every
tick, raises alerts, and offers the demo controls.

Each machine replays a pre-generated episode (same physics as generate.py), one 15-minute reading
per tick, so the models always see data like they were trained on.

Controls
    set_scenario(id, "healthy" | "degrading" | "near_failure")   switch what the machine does
    fast_forward(id, "Critical")    age the machine quickly until it reaches that status
    service(id)                     "mark as serviced": logs maintenance and resets to healthy

Status shown to users = the worse of (model condition, risk band):  risk >= 25 -> Warning,  risk >= 60 -> Critical
Alerts fire when a worse status is held for 3 ticks in a row, and the machine alert level resets
after 5 ticks of Normal.

Quick console test:   python live_sim.py --models models
"""
import argparse
import threading
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

from generate import MACHINES, simulate_episode
from predict import Predictor

WINDOW = 48                 # readings the predictor needs (12 hours)
FF_STRIDE = 30              # readings advanced per tick while fast-forwarding (7.5 h)
HISTORY = 120               # points of history kept per machine for charts
ALERT_HOLD, CLEAR_HOLD = 3, 5
RISK_WARNING, RISK_CRITICAL = 25, 60
READING_COLS = ["temperature", "vibration", "motor_current", "rpm", "operating_hours", "workload"]
LEVEL = {"Normal": 0, "Warning": 1, "Critical": 2}
NAMES = {0: "Normal", 1: "Warning", 2: "Critical"}
ALIASES = {"normal": "healthy", "healthy": "healthy", "degrading": "degrading", "gradual": "degrading",
           "near_failure": "near_failure", "near-failure": "near_failure", "nearfailure": "near_failure"}
FLEET = [("MILL-01", "CNC Mill", "healthy"), ("MILL-02", "CNC Mill", "degrading"),
         ("LATHE-01", "CNC Lathe", "near_failure"), ("LATHE-02", "CNC Lathe", "healthy"),
         ("FDM-01", "FDM Printer", "degrading"), ("FDM-02", "FDM Printer", "healthy")]


class LiveSimulator:
    def __init__(self, models_dir="models", seed=7, fleet=FLEET):
        self.pred = Predictor(models_dir)
        self.rng = np.random.default_rng(seed)
        self.lock = threading.RLock()
        self.ep_count = self.alert_count = 0
        self.alerts, self.maintenance, self.m = [], [], {}
        for mid, mtype, scen in fleet:
            st = dict(id=mid, type=mtype, ff=None, failed=False, hist=deque(maxlen=HISTORY),
                      alert_level=0, streak=(0, 0), last=None)
            t0 = pd.Timestamp("2026-10-01") + pd.Timedelta(hours=float(self.rng.uniform(0, 48)))
            self._new_episode(st, ALIASES[scen], float(self.rng.uniform(500, 4000)), t0)
            self._observe(st)
            self.m[mid] = st

    # ---------- episode / prediction ----------
    def _new_episode(self, st, scenario, hours, t0):
        cfg = MACHINES[st["type"]]
        for _ in range(60):
            if scenario == "healthy":
                scen, n = "healthy", 1500
            elif scenario == "degrading":
                scen, n = "gradual", int(self.rng.integers(900, 1500))
            else:
                scen, n = "near_failure", int(self.rng.integers(300, 500))
            self.ep_count += 1
            ep = simulate_episode(self.rng, cfg, st["type"], st["id"], f"LIVE-{self.ep_count:04d}",
                                  scen, n, hours, t0).reset_index(drop=True)
            if scenario == "healthy" or ep["failure"].iloc[-1] == 1:
                break
        start = WINDOW
        if scenario == "degrading":                      # start about 60 h before the first non-Normal row
            bad = np.flatnonzero((ep["condition"] != "Normal").values)
            start = max(WINDOW, (int(bad[0]) if len(bad) else 300) - 240)
        st.update(ep=ep, idx=min(start, len(ep) - 2), scenario=scenario, ff=None, failed=False,
                  alert_level=0, streak=(0, 0))

    def _observe(self, st):
        ep, i = st["ep"], st["idx"]
        res = self.pred.predict(ep.iloc[max(0, i - WINDOW + 1): i + 1])
        row = ep.iloc[i]
        band = 2 if res["risk_percent"] >= RISK_CRITICAL else 1 if res["risk_percent"] >= RISK_WARNING else 0
        status = NAMES[max(LEVEL[res["condition"]], band)]
        readings = {c: round(float(row[c]), 3) for c in READING_COLS}
        readings["rpm"] = int(row["rpm"])
        snap = {"machine_id": st["id"], "machine_type": st["type"], "scenario": st["scenario"],
                "status": status, "condition": res["condition"], "risk_percent": res["risk_percent"],
                "hours_left": res["hours_left"], "anomaly": res["anomaly"], "reasons": res["reasons"],
                "probabilities": res["probabilities"], "readings": readings,
                "data_time": row["timestamp"].isoformat(timespec="seconds"), "updated_at": datetime.now().isoformat(timespec="seconds"),
                "fast_forwarding": st["ff"] is not None, "failed": st["failed"]}
        st["last"] = snap
        st["hist"].append({"time": snap["data_time"], "status": status, "risk_percent": res["risk_percent"],
                           **{c: readings[c] for c in ["temperature", "vibration", "motor_current", "rpm"]}})
        self._alert_rules(st, snap)

    def _alert_rules(self, st, snap):
        level = LEVEL[snap["status"]]
        lv, cnt = st["streak"]
        st["streak"] = (level, cnt + 1) if level == lv else (level, 1)
        cnt = st["streak"][1]
        if level > st["alert_level"] and cnt >= ALERT_HOLD:
            self.alert_count += 1
            why = ("; ".join(snap["reasons"]) + ". ") if snap["reasons"] else ""
            left = "" if snap["hours_left"] >= 200 else f" About {snap['hours_left']:.0f} h left."
            self.alerts.append({"id": self.alert_count, "machine_id": st["id"], "level": snap["status"],
                                "message": f"{st['id']} is {snap['status']} (risk {snap['risk_percent']:.0f}%). {why}{left}".strip(),
                                "data_time": snap["data_time"], "created_at": snap["updated_at"],
                                "acknowledged": False, "resolved": False})
            st["alert_level"] = level
        elif level == 0 and cnt >= CLEAR_HOLD:
            st["alert_level"] = 0

    # ---------- streaming ----------
    def _advance(self, st):
        if st["failed"]:
            return
        ep, last = st["ep"], len(st["ep"]) - 1
        st["idx"] = min(st["idx"] + (FF_STRIDE if st["ff"] else 1), last)
        if st["idx"] == last:
            if ep["failure"].iloc[-1] == 1:
                st["failed"] = True
            elif st["scenario"] == "healthy":          # healthy machines just keep running
                row = ep.iloc[-1]
                self._new_episode(st, "healthy", float(row["operating_hours"]),
                                  row["timestamp"] + pd.Timedelta(minutes=15))
        self._observe(st)
        if st["ff"] and (LEVEL[st["last"]["status"]] >= LEVEL[st["ff"]] or st["failed"]):
            st["ff"] = None
            st["last"]["fast_forwarding"] = False

    def tick(self):
        with self.lock:
            for st in self.m.values():
                self._advance(st)
            return self.snapshot_all()

    # ---------- read ----------
    def snapshot_all(self):
        with self.lock:
            return [st["last"] for st in self.m.values()]

    def get_machine(self, mid):
        with self.lock:
            st = self.m[mid]
            return {**st["last"], "history": list(st["hist"])}

    def list_alerts(self, only_open=False):
        with self.lock:
            return [a for a in self.alerts if not (only_open and a["resolved"])][::-1]

    def ack_alert(self, aid):
        with self.lock:
            for a in self.alerts:
                if a["id"] == aid:
                    a["acknowledged"] = True
                    return a
            raise KeyError(aid)

    def maintenance_log(self):
        with self.lock:
            return self.maintenance[::-1]

    # ---------- demo controls ----------
    def _restart(self, st, scenario):
        cur = st["ep"].iloc[st["idx"]]
        self._new_episode(st, scenario, float(cur["operating_hours"]), cur["timestamp"] + pd.Timedelta(minutes=15))
        self._observe(st)

    def set_scenario(self, mid, name):
        with self.lock:
            st = self.m[mid]
            if name.lower() not in ALIASES:
                raise ValueError(f"scenario must be one of healthy, degrading, near_failure (got '{name}')")
            self._restart(st, ALIASES[name.lower()])
            return st["last"]

    def fast_forward(self, mid, target="Critical"):
        with self.lock:
            st = self.m[mid]
            target = target.capitalize()
            if target not in ("Warning", "Critical"):
                raise ValueError("target must be Warning or Critical")
            if st["failed"]:
                raise ValueError(f"{mid} has failed - service it first")
            if st["scenario"] == "healthy":
                self._restart(st, "degrading")
            st["ff"] = target
            st["last"]["fast_forwarding"] = True
            return st["last"]

    def service(self, mid, note="Serviced - health reset"):
        with self.lock:
            st = self.m[mid]
            before = st["last"]
            cur = st["ep"].iloc[st["idx"]]
            self.maintenance.append({"machine_id": mid, "action": note, "status_before": before["status"],
                                     "risk_before": before["risk_percent"], "failed": before["failed"],
                                     "operating_hours": float(cur["operating_hours"]),
                                     "data_time": before["data_time"], "created_at": datetime.now().isoformat(timespec="seconds")})
            for a in self.alerts:
                if a["machine_id"] == mid:
                    a["resolved"] = True
            self._restart(st, "healthy")
            return st["last"]


def demo(models):
    sim = LiveSimulator(models)
    show = lambda s: print(f"{s['machine_id']:9s} {s['status']:8s} risk {s['risk_percent']:5.1f}%  "
                           f"left {s['hours_left']:4.0f} h  anomaly {'yes' if s['anomaly'] else 'no ':3s} "
                           f"{'FF ' if s['fast_forwarding'] else ''}{'FAILED' if s['failed'] else ''}  "
                           f"{'; '.join(s['reasons'])}")
    print("--- fleet at start ---")
    for _ in range(2):
        snaps = sim.tick()
    for s in snaps:
        show(s)
    print("\n--- fast-forwarding MILL-01 (healthy -> degrading -> Critical) ---")
    sim.fast_forward("MILL-01", "Critical")
    for t in range(80):
        sim.tick()
        s = sim.get_machine("MILL-01")
        if t % 3 == 0 or not s["fast_forwarding"]:
            show(s)
        if not s["fast_forwarding"]:
            break
    print("\nalerts:")
    for a in sim.list_alerts():
        print(" ", a["level"], "|", a["message"])
    print("\n--- servicing MILL-01 ---")
    show(sim.service("MILL-01"))
    print("maintenance log:", sim.maintenance_log())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="models")
    demo(ap.parse_args().models)
