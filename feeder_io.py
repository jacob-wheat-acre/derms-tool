"""feeder_io.py — OpenDSS solve, data extraction, and scenario helpers."""
from __future__ import annotations
import os
import warnings
import numpy as np
import opendssdirect as dss

warnings.filterwarnings("ignore", category=DeprecationWarning)

FEEDER_DIR = os.path.join(os.path.dirname(__file__), "feeders", "ieee123")
MASTER     = os.path.join(FEEDER_DIR, "Run_IEEE123Bus.DSS")

ANSI_LO, ANSI_HI = 0.95, 1.05


# ── DSS command helper ─────────────────────────────────────────────────────────

def _cmd(txt: str) -> str:
    dss.Text.Command(txt)
    return dss.Text.Result()


# ── Solve helpers ──────────────────────────────────────────────────────────────

def solve_base() -> None:
    _cmd(f'Redirect "{MASTER}"')
    _cmd("Solve")


def solve_with_pv(bus: str, kw: float, pf: float = 1.0) -> None:
    solve_base()
    dss.Circuit.SetActiveBus(bus)
    kv_base = dss.Bus.kVBase()          # line-to-neutral kV
    nodes   = dss.Bus.Nodes()           # e.g. [1,2,3] or [2] or [1,3]
    nph     = len(nodes)

    if nph == 3:
        kv = kv_base * 1.7321           # line-to-line
        bus1_str = bus
    else:
        kv = kv_base                    # line-to-neutral for 1- or 2-phase
        phase_str = ".".join(str(n) for n in nodes)
        bus1_str = f"{bus}.{phase_str}"

    kva = kw / max(pf, 0.01)
    _cmd(
        f"New PVSystem.DER_{bus} bus1={bus1_str} phases={nph} kVA={kva:.1f} "
        f"pf={pf:.3f} kV={kv:.4f} irradiance=1 Pmpp={kw:.1f}"
    )
    _cmd("Solve")


# ── Data extraction ────────────────────────────────────────────────────────────

def extract_buses() -> dict[str, dict]:
    """Return dict keyed by bus name with geometry and voltage data."""
    out: dict[str, dict] = {}
    for b in dss.Circuit.AllBusNames():
        if b.endswith("r"):          # skip regulator internal buses
            continue
        dss.Circuit.SetActiveBus(b)
        x, y = dss.Bus.X(), dss.Bus.Y()
        if x == 0.0 and y == 0.0:   # no coordinate data
            continue
        nodes   = dss.Bus.Nodes()   # e.g. [1,2,3] or [2]
        pv_ang  = dss.Bus.puVmagAngle()     # [V1,a1,V2,a2,...] pu
        vmags   = [pv_ang[i * 2] for i in range(len(nodes))]
        out[b] = dict(
            x=x, y=y, kv=dss.Bus.kVBase(),
            phases=nodes, vmag=vmags,
            vmin=min(vmags), vmax=max(vmags),
        )
    return out


def extract_lines(buses: dict) -> list[dict]:
    out = []
    n = dss.Lines.First()
    while n:
        name = dss.Lines.Name()
        b1   = dss.Lines.Bus1().split(".")[0].lower()
        b2   = dss.Lines.Bus2().split(".")[0].lower()
        nph  = dss.Lines.Phases()
        norm = dss.Lines.NormAmps()
        dss.Circuit.SetActiveElement(f"Line.{name}")
        cur  = dss.CktElement.CurrentsMagAng()
        imags = [cur[i * 2] for i in range(nph)] if cur else []
        imax  = max(imags) if imags else 0.0
        pct   = imax / norm * 100 if norm > 0 else 0.0
        if b1 in buses and b2 in buses:
            out.append(dict(
                name=name, b1=b1, b2=b2,
                x1=buses[b1]["x"], y1=buses[b1]["y"],
                x2=buses[b2]["x"], y2=buses[b2]["y"],
                nph=nph, norm_amps=norm, pct=pct,
            ))
        n = dss.Lines.Next()
    return out


def extract_loads() -> list[dict]:
    out = []
    n = dss.Loads.First()
    while n:
        name = dss.Loads.Name()
        dss.Circuit.SetActiveElement(f"Load.{name}")
        bus  = dss.CktElement.BusNames()[0].split(".")[0].lower()
        out.append(dict(name=name, bus=bus,
                        kw=dss.Loads.kW(), kvar=dss.Loads.kvar(),
                        phases=dss.Loads.Phases()))
        n = dss.Loads.Next()
    return out


def extract_caps() -> list[dict]:
    out = []
    n = dss.Capacitors.First()
    while n:
        name = dss.Capacitors.Name()
        dss.Circuit.SetActiveElement(f"Capacitor.{name}")
        bus = dss.CktElement.BusNames()[0].split(".")[0].lower()
        out.append(dict(name=name, bus=bus, kvar=dss.Capacitors.kvar()))
        n = dss.Capacitors.Next()
    return out


def extract_summary() -> dict:
    tp  = dss.Circuit.TotalPower()   # [P_kW, Q_kvar] negative = load convention
    lss = dss.Circuit.Losses()       # [P_watts, Q_vars]
    buses = extract_buses()
    vmins = [d["vmin"] for d in buses.values()]
    n_lo  = sum(1 for v in vmins if v < ANSI_LO)
    n_hi  = sum(1 for v in vmins if v > ANSI_HI)
    return dict(
        load_kw=-tp[0], load_kvar=-tp[1],
        loss_kw=lss[0] / 1000,
        n_lo_violations=n_lo,
        n_hi_violations=n_hi,
        converged=dss.Solution.Converged(),
    )


# ── Hosting capacity sweep ─────────────────────────────────────────────────────

def hosting_capacity_sweep(
    load_buses: list[str],
    step_kw: float = 100,
    max_kw: float  = 2000,
    progress_cb    = None,
) -> dict[str, float]:
    """
    For each bus, find the maximum PV (kW) that keeps all bus voltages
    within ANSI A limits. Returns {bus: max_kw}.
    """
    results: dict[str, float] = {}
    total = len(load_buses)

    for idx, bus in enumerate(load_buses):
        hc = 0.0
        for kw in np.arange(step_kw, max_kw + step_kw, step_kw):
            solve_with_pv(bus, float(kw))
            if not dss.Solution.Converged():
                break
            vmags = dss.Circuit.AllBusMagPu()
            if any(v > ANSI_HI for v in vmags) or any(v < ANSI_LO for v in vmags):
                break
            hc = float(kw)
        results[bus] = hc
        if progress_cb:
            progress_cb((idx + 1) / total)

    # Restore base case
    solve_base()
    return results
