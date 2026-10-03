#!/usr/bin/env python3
"""
live_api.py - FastAPI server for the live demo. Frontend (React) and mobile app call these endpoints.

Run:   pip install fastapi uvicorn
       uvicorn live_api:app --port 8000
Docs:  http://localhost:8000/docs   (try every endpoint in the browser)

GET  /machines                          all machines: status, risk, hours left, readings
GET  /machines/{id}                     one machine + recent history for charts
POST /machines/{id}/scenario/{name}     healthy | degrading | near_failure
POST /machines/{id}/fast-forward        ?target=Critical (or Warning)
POST /machines/{id}/service             "mark as serviced" (logs maintenance, resets to healthy)
GET  /alerts                            newest first (?only_open=true)
POST /alerts/{alert_id}/ack
GET  /maintenance                       maintenance history

Environment: MODELS_DIR (default "models"), TICK_SECONDS (default 1.5 seconds per new reading)
Each machine has a "status" field (Normal / Warning / Critical): show that one in the UI.
"""
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from live_sim import LiveSimulator

TICK_SECONDS = float(os.getenv("TICK_SECONDS", "1.5"))
sim = None


@asynccontextmanager
async def lifespan(app):
    global sim
    sim = LiveSimulator(os.getenv("MODELS_DIR", "models"))

    async def stream():
        while True:
            await asyncio.to_thread(sim.tick)
            await asyncio.sleep(TICK_SECONDS)

    task = asyncio.create_task(stream())
    yield
    task.cancel()


app = FastAPI(title="Predictive Maintenance Live API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def run(fn, *args):
    try:
        return fn(*args)
    except KeyError:
        raise HTTPException(404, "unknown machine or alert id")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/machines")
def machines():
    return sim.snapshot_all()


@app.get("/machines/{machine_id}")
def machine(machine_id: str):
    return run(sim.get_machine, machine_id)


@app.post("/machines/{machine_id}/scenario/{name}")
def scenario(machine_id: str, name: str):
    return run(sim.set_scenario, machine_id, name)


@app.post("/machines/{machine_id}/fast-forward")
def fast_forward(machine_id: str, target: str = "Critical"):
    return run(sim.fast_forward, machine_id, target)


@app.post("/machines/{machine_id}/service")
def service(machine_id: str):
    return run(sim.service, machine_id)


@app.get("/alerts")
def alerts(only_open: bool = False):
    return sim.list_alerts(only_open)


@app.post("/alerts/{alert_id}/ack")
def ack(alert_id: int):
    return run(sim.ack_alert, alert_id)


@app.get("/maintenance")
def maintenance():
    return sim.maintenance_log()
