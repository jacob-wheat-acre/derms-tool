"""_dss_worker.py — DSS solves in a fresh subprocess to avoid libdss_capi re-init crash.

Modes:
  base  <master> <out.pkl> [kv_min]
  der   <master> <bus> <kw> <pf> <out.pkl>
  hc    <master> <buses_json> <step_kw> <max_kw> <out.pkl>  (prints PROGRESS:x.xxxx to stdout)
"""
from __future__ import annotations
import sys, os, pickle, json
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import opendssdirect as dss
from feeder_io import (
    solve_base, extract_buses, extract_lines, extract_loads,
    extract_caps, extract_switches, extract_fuses, extract_regulators,
    extract_substation_transformers, extract_feeder_zones,
    extract_topology_lines, extract_load_xfmr_lines,
    extract_cap_states, extract_reg_taps,
    extract_summary, ANSI_LO, ANSI_HI, _cmd,
)


def _write(path: str, data) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def mode_base(master: str, out_pkl: str, kv_min: float | None = None) -> None:
    solve_base(master)
    buses   = extract_buses()
    lines   = extract_lines(buses)
    loads   = extract_loads()
    caps    = extract_caps()
    summary = extract_summary()

    if kv_min is not None:
        candidates = []
        for b in buses:
            dss.Circuit.SetActiveBus(b)
            if dss.Bus.kVBase() >= kv_min:
                candidates.append(b)
        load_buses = sorted(candidates)
    else:
        load_buses = sorted({ld["bus"] for ld in loads if ld["bus"] in buses})

    switches         = extract_switches(buses)
    fuses            = extract_fuses(buses)
    regulators       = extract_regulators(buses)
    sub_transformers = extract_substation_transformers(buses)
    feeder_zones     = extract_feeder_zones(buses)
    topology_lines   = extract_topology_lines()
    load_xfmr_lines  = extract_load_xfmr_lines(buses)

    _write(out_pkl, dict(
        buses=buses, lines=lines, loads=loads,
        caps=caps, switches=switches, fuses=fuses, regulators=regulators,
        sub_transformers=sub_transformers, feeder_zones=feeder_zones,
        topology_lines=topology_lines, load_xfmr_lines=load_xfmr_lines,
        summary=summary, load_buses=load_buses,
    ))


def mode_der(master: str, bus: str, kw: float, pf: float, out_pkl: str) -> None:
    solve_base(master)

    dss.Circuit.SetActiveBus(bus)
    kv_base = dss.Bus.kVBase()
    nodes   = dss.Bus.Nodes()
    nph     = len(nodes)

    if nph == 3:
        kv = kv_base * 1.7321
        bus1_str = bus
    else:
        kv = kv_base
        phase_str = ".".join(str(n) for n in nodes)
        bus1_str = f"{bus}.{phase_str}"

    kva = kw / max(pf, 0.01)
    _cmd(
        f"New PVSystem.DER_{bus} bus1={bus1_str} phases={nph} kVA={kva:.1f} "
        f"pf={pf:.3f} kV={kv:.4f} irradiance=1 Pmpp={kw:.1f}"
    )
    _cmd("Solve")

    _write(out_pkl, dict(buses=extract_buses(), summary=extract_summary()))


def mode_hc(master: str, buses_json: str, step_kw: float, max_kw: float, out_pkl: str) -> None:
    buses_list: list[str] = json.loads(buses_json)
    solve_base(master)

    results: dict[str, float] = {}
    total = len(buses_list)

    for idx, bus in enumerate(buses_list):
        hc = 0.0
        pv_name = f"DER_{bus}_hc"
        pv_added = False

        dss.Circuit.SetActiveBus(bus)
        kv_base = dss.Bus.kVBase()
        nodes   = dss.Bus.Nodes()
        nph     = len(nodes)
        if nph == 3:
            kv = kv_base * 1.7321
            bus1_str = bus
        else:
            kv = kv_base
            phase_str = ".".join(str(n) for n in nodes)
            bus1_str = f"{bus}.{phase_str}"

        for kw in np.arange(step_kw, max_kw + step_kw, step_kw):
            kva = float(kw)
            if not pv_added:
                _cmd(
                    f"New PVSystem.{pv_name} bus1={bus1_str} phases={nph} "
                    f"kVA={kva:.1f} pf=1.000 kV={kv:.4f} irradiance=1 Pmpp={kw:.1f}"
                )
                pv_added = True
            else:
                _cmd(f"Edit PVSystem.{pv_name} kVA={kva:.1f} Pmpp={kw:.1f}")

            _cmd("Solve")
            if not dss.Solution.Converged():
                break
            vmags = dss.Circuit.AllBusMagPu()
            if any(v > ANSI_HI for v in vmags) or any(v < ANSI_LO for v in vmags):
                break
            hc = float(kw)

        if pv_added:
            _cmd(f"Disable PVSystem.{pv_name}")

        results[bus] = hc
        print(f"PROGRESS:{(idx + 1) / total:.4f}", flush=True)

    _write(out_pkl, results)


def mode_vvo(
    master: str,
    objective: str,
    sub_cap_bus: str,
    sub_cap_kv: float,
    sub_cap_kvar: float,
    ems_voltage_pu: float,
    base_source_pu: float,
    out_pkl: str,
) -> None:
    from modules.vvo.optimizer import run_scenarios
    result = run_scenarios(
        master       = master,
        objective    = objective,
        sub_cap_bus  = sub_cap_bus,
        sub_cap_kv   = sub_cap_kv,
        sub_cap_kvar = sub_cap_kvar,
        ems_voltage_pu  = ems_voltage_pu,
        base_source_pu  = base_source_pu,
    )
    _write(out_pkl, result)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "base":
        kv_min = float(sys.argv[4]) if len(sys.argv) > 4 else None
        mode_base(sys.argv[2], sys.argv[3], kv_min)
    elif mode == "der":
        mode_der(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6])
    elif mode == "hc":
        mode_hc(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6])
    elif mode == "vvo":
        mode_vvo(
            master         = sys.argv[2],
            objective      = sys.argv[3],
            sub_cap_bus    = sys.argv[4],
            sub_cap_kv     = float(sys.argv[5]),
            sub_cap_kvar   = float(sys.argv[6]),
            ems_voltage_pu = float(sys.argv[7]),
            base_source_pu = float(sys.argv[8]),
            out_pkl        = sys.argv[9],
        )
    else:
        sys.exit(f"Unknown mode: {mode}")
