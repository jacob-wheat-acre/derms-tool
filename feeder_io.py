"""feeder_io.py — OpenDSS solve, data extraction, and scenario helpers."""
from __future__ import annotations
import os
import warnings
import numpy as np
import opendssdirect as dss

warnings.filterwarnings("ignore", category=DeprecationWarning)

ANSI_LO, ANSI_HI = 0.95, 1.05

_ROOT = os.path.dirname(__file__)

FEEDERS: dict[str, dict] = {
    "IEEE 123-Bus": {
        "master": os.path.join(_ROOT, "feeders", "ieee123", "Run_IEEE123Bus.DSS"),
        "source_bus": "150",
        "source_pu": 1.00,
        "default_der_bus": "57",
        "caption": "IEEE 123-Bus Test Case (Kersting 1991) · 4.16 kV · 91 loads",
        "vvo_sub_cap_bus": "149",
        "vvo_sub_cap_kv": 4.16,
        "vvo_sub_cap_kvar": 900,
        "profile_intro": (
            "The feeder starts at **Bus 150** (the substation), set at 1.00 pu. Voltage "
            "generally drops as you move away from the substation because current flowing "
            "through line resistance causes a voltage drop. The regulators (transformer tap "
            "changers) boost voltage at key points to keep everything within **ANSI A limits: "
            "0.95–1.05 pu**.\n\n"
            "Because this is an unbalanced distribution feeder, phases A, B, and C are shown "
            "separately — a single-phase lateral on phase B only appears in the blue trace. "
            "Hover over any point to see the bus and exact voltage."
        ),
        "der_intro": (
            "When a customer or developer installs PV on a feeder, voltage **rises** at that "
            "bus and upstream — potentially pushing buses above the ANSI limit. The DERMS needs "
            "to detect this and either curtail the PV or dispatch reactive absorption from smart "
            "inverters."
        ),
    },
    "IEEE 13-Bus": {
        "master": os.path.join(_ROOT, "feeders", "ieee13", "Run_IEEE13Bus.DSS"),
        "source_bus": "650",
        "source_pu": 1.05,
        "default_der_bus": "671",
        "caption": "IEEE 13-Node Test Case (Kersting 2001) · 4.16 kV · 8 load buses",
        "vvo_sub_cap_bus": "632",
        "vvo_sub_cap_kv": 4.16,
        "vvo_sub_cap_kvar": 600,
        "profile_intro": (
            "The feeder starts at **Bus 650** (the substation) with a source voltage of 1.06 pu. "
            "A voltage regulator between 650 and 632 maintains voltage near 1.02 pu at the head "
            "of the feeder. This feeder is **heavily unbalanced** — single-phase and two-phase "
            "laterals create significant voltage differences between phases A, B, and C.\n\n"
            "Bus 634 is on a 480 V wye secondary; its per-unit voltage is relative to the 480 V "
            "base. Hover over any point to see the bus and exact voltage."
        ),
        "der_intro": (
            "The 13-bus feeder is heavily loaded and unbalanced, making it an excellent testbed "
            "for DER impact analysis. Adding PV on a single-phase lateral (e.g. bus 652 or 611) "
            "creates strong voltage imbalance. The DERMS must detect and correct per-phase "
            "overvoltage — not just aggregate feeder voltage."
        ),
    },
    "IEEE 9500-Node": {
        "master": os.path.join(_ROOT, "feeders", "ieee9500", "Run_IEEE9500.DSS"),
        "source_bus": "sourcebus",
        "source_pu": 1.05,
        "default_der_bus": "190-8593",
        "vvo_sub_cap_bus": "190-8593",
        "vvo_sub_cap_kv": 12.47,
        "vvo_sub_cap_kvar": 1800,
        "caption": "IEEE 9500-Node Test Feeder (PNNL) · 12.47 kV · ~9,500 nodes · geographic coordinates",
        "large_feeder": True,
        "has_latlon": True,
        "der_kv_min": 1.0,
        "hc_sample_n": 60,
        "profile_intro": (
            "This is the **PNNL IEEE 9500-node test feeder** — a large-scale realistic distribution "
            "system located near Kennewick, Washington. It feeds from a 115 kV transmission bus "
            "through 69 kV and 12.47 kV distribution feeders down to 120/240 V service buses.\n\n"
            "The base case already has **110 undervoltage buses** on the 120 V secondary — a "
            "realistic condition in heavily loaded residential feeders. Bus coordinates are "
            "geographic (latitude/longitude), so the network map is a true geographic view."
        ),
        "der_intro": (
            "With ~9,500 nodes and 3-phase primary feeders at 12.47 kV, the 9500-node feeder "
            "demonstrates DERMS at utility scale. PV is added at the 12.47 kV primary level — "
            "typical for community solar or commercial rooftop systems that interconnect above "
            "the service transformer. Watch how a single large installation affects voltages "
            "across the entire geographic footprint of the feeder."
        ),
    },
}

_current_master: str = ""


# ── DSS command helper ─────────────────────────────────────────────────────────

def _cmd(txt: str) -> str:
    dss.Text.Command(txt)
    return dss.Text.Result()


# ── Solve helpers ──────────────────────────────────────────────────────────────

def solve_base(master: str = "") -> None:
    global _current_master
    if master:
        _current_master = master
    _cmd("Clear")
    _cmd("Set DefaultBaseFrequency=60")
    import tempfile as _tf
    _cmd(f'Redirect "{_current_master}"')
    _cmd("Set ShowExport=No")
    _cmd(f"Set DataPath={_tf.gettempdir()}")
    _cmd("Solve")


def solve_with_pv(bus: str, kw: float, pf: float = 1.0, reload_base: bool = True) -> None:
    if reload_base:
        solve_base()
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


# ── Data extraction ────────────────────────────────────────────────────────────

def extract_buses() -> dict[str, dict]:
    """Return dict keyed by bus name with geometry and voltage data."""
    out: dict[str, dict] = {}
    for b in dss.Circuit.AllBusNames():
        if b.endswith("r"):
            continue
        dss.Circuit.SetActiveBus(b)
        x, y = dss.Bus.X(), dss.Bus.Y()
        if x == 0.0 and y == 0.0:
            continue
        nodes   = dss.Bus.Nodes()
        pv_ang  = dss.Bus.puVmagAngle()
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


def extract_load_xfmr_lines(buses: dict) -> list[dict]:
    """Line-like dicts for distribution service transformers (< 5 MVA).
    Connects the primary MV bus to the LV secondary bus so maps can show the
    visual link between feeder laterals and customer service points.
    """
    reg_xfmr_names: set[str] = set()
    n = dss.RegControls.First()
    while n:
        reg_xfmr_names.add(dss.RegControls.Transformer().lower())
        n = dss.RegControls.Next()

    out = []
    n = dss.Transformers.First()
    while n:
        name = dss.Transformers.Name()
        if name.lower() not in reg_xfmr_names:
            kva = dss.Transformers.kVA()
            dss.Transformers.Wdg(1)
            kv1 = dss.Transformers.kV()
            if not (kva >= 5000 and kv1 >= 10.0):
                dss.Circuit.SetActiveElement(f"Transformer.{name}")
                bnames = dss.CktElement.BusNames()
                if len(bnames) >= 2:
                    b1 = bnames[0].split(".")[0].lower()
                    b2 = bnames[1].split(".")[0].lower()
                    if b1 in buses and b2 in buses and b1 != b2:
                        out.append(dict(
                            name=f"xfmr:{name}", b1=b1, b2=b2,
                            x1=buses[b1]["x"], y1=buses[b1]["y"],
                            x2=buses[b2]["x"], y2=buses[b2]["y"],
                            nph=dss.CktElement.NumPhases(),
                            norm_amps=0, pct=0,
                        ))
        n = dss.Transformers.Next()
    return out


def extract_substation_transformers(buses: dict) -> list[dict]:
    """Return large power transformers (≥5 MVA, primary ≥10 kV) that have mapped coordinates.
    Excludes regulator-controlled transformers (those are returned by extract_regulators).
    """
    reg_xfmr_names: set[str] = set()
    n = dss.RegControls.First()
    while n:
        reg_xfmr_names.add(dss.RegControls.Transformer().lower())
        n = dss.RegControls.Next()

    out = []
    n = dss.Transformers.First()
    while n:
        name = dss.Transformers.Name()
        if name.lower() in reg_xfmr_names:
            n = dss.Transformers.Next()
            continue
        kva = dss.Transformers.kVA()
        dss.Transformers.Wdg(1)
        kv1 = dss.Transformers.kV()
        dss.Transformers.Wdg(2)
        kv2 = dss.Transformers.kV()
        if kva >= 5000 and kv1 >= 10.0:
            dss.Circuit.SetActiveElement(f"Transformer.{name}")
            b = dss.CktElement.BusNames()[0].split(".")[0].lower()
            if b in buses:
                out.append(dict(
                    name=name, bus=b, kva=kva, kv1=kv1, kv2=kv2,
                    x=buses[b]["x"], y=buses[b]["y"],
                ))
        n = dss.Transformers.Next()
    return out


def extract_feeder_zones(buses: dict) -> dict[str, str]:
    """Return {bus_name: meter_name} for circuits with >1 EnergyMeter, else {}."""
    if dss.Meters.Count() <= 1:
        return {}
    zones: dict[str, str] = {}
    n = dss.Meters.First()
    while n:
        name = dss.Meters.Name()
        for branch in dss.Meters.AllBranchesInZone():
            dss.Circuit.SetActiveElement(branch)
            for bname in dss.CktElement.BusNames():
                b = bname.split(".")[0].lower()
                if b in buses:
                    zones[b] = name
        n = dss.Meters.Next()
    return zones


def extract_switches(buses: dict) -> list[dict]:
    """Lines that are switches (breakers, sectionalizers, tie switches)."""
    out = []
    n = dss.Lines.First()
    while n:
        is_sw = dss.Lines.IsSwitch()
        if is_sw:
            name = dss.Lines.Name()
            b1   = dss.Lines.Bus1().split(".")[0].lower()
            b2   = dss.Lines.Bus2().split(".")[0].lower()
            nph  = dss.Lines.Phases()
            dss.Circuit.SetActiveElement(f"Line.{name}")
            is_open = bool(dss.CktElement.IsOpen(1, 0))
            # Use bus with known coordinates
            x = y = None
            for b in (b1, b2):
                if b in buses:
                    x, y = buses[b]["x"], buses[b]["y"]
                    break
            if x is not None:
                out.append(dict(name=name, x=x, y=y, nph=nph, is_open=is_open))
        n = dss.Lines.Next()
    return out


def extract_fuses(buses: dict) -> list[dict]:
    out = []
    n = dss.Fuses.First()
    while n:
        name = dss.Fuses.Name()
        dss.Circuit.SetActiveElement(f"Fuse.{name}")
        bnames = dss.CktElement.BusNames()
        if bnames:
            b = bnames[0].split(".")[0].lower()
            if b in buses:
                out.append(dict(name=name, x=buses[b]["x"], y=buses[b]["y"]))
        n = dss.Fuses.Next()
    return out


def extract_regulators(buses: dict) -> list[dict]:
    out = []
    n = dss.RegControls.First()
    while n:
        name  = dss.RegControls.Name()
        xfmr  = dss.RegControls.Transformer()
        vreg  = dss.RegControls.ForwardVreg()
        band  = dss.RegControls.ForwardBand()
        dss.Circuit.SetActiveElement(f"Transformer.{xfmr}")
        bnames = dss.CktElement.BusNames()
        if bnames:
            b = bnames[0].split(".")[0].lower()
            if b in buses:
                out.append(dict(name=name, xfmr=xfmr,
                                x=buses[b]["x"], y=buses[b]["y"],
                                vreg=vreg, band=band))
        n = dss.RegControls.Next()
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


def extract_cap_states(buses: dict | None = None) -> list[dict]:
    """Capacitors with current on/off state and optional bus coordinates."""
    out = []
    n = dss.Capacitors.First()
    while n:
        name = dss.Capacitors.Name()
        dss.Circuit.SetActiveElement(f"Capacitor.{name}")
        bus    = dss.CktElement.BusNames()[0].split(".")[0].lower()
        states = list(dss.Capacitors.States())
        entry  = dict(
            name=name, bus=bus,
            kvar=dss.Capacitors.kvar(),
            phases=dss.CktElement.NumPhases(),
            states=states,
            on=all(s == 1 for s in states),
        )
        if buses and bus in buses:
            entry["x"] = buses[bus]["x"]
            entry["y"] = buses[bus]["y"]
        out.append(entry)
        n = dss.Capacitors.Next()
    return out


def extract_reg_taps() -> list[dict]:
    """Regulator tap number, vreg setpoint, and deadband for each RegControl."""
    out = []
    n = dss.RegControls.First()
    while n:
        name = dss.RegControls.Name()
        out.append(dict(
            name=name,
            tap=dss.RegControls.TapNumber(),
            vreg=dss.RegControls.ForwardVreg(),
            band=dss.RegControls.ForwardBand(),
            ptratio=dss.RegControls.PTratio(),
        ))
        n = dss.RegControls.Next()
    return out


def extract_td_seam_metrics() -> dict:
    """Reactive power and voltage metrics at the T-D boundary (vsource)."""
    import math
    tp  = dss.Circuit.TotalPower()
    lss = dss.Circuit.Losses()
    src_kw            = -tp[0]
    reactive_import   = -tp[1]   # positive = importing Q from transmission
    loss_kw           = lss[0] / 1000.0
    denom             = math.sqrt(src_kw ** 2 + reactive_import ** 2)
    pf                = abs(src_kw) / denom if denom > 0 else 1.0
    vmags             = [v for v in dss.Circuit.AllBusMagPu() if v > 0.3]
    n_lo              = sum(1 for v in vmags if v < ANSI_LO)
    n_hi              = sum(1 for v in vmags if v > ANSI_HI)
    v_dev             = math.sqrt(
        sum((v - 1.0) ** 2 for v in vmags) / max(len(vmags), 1)
    ) if vmags else 0.0
    return dict(
        reactive_import_kvar = round(reactive_import, 1),
        active_power_kw      = round(src_kw, 1),
        loss_kw              = round(loss_kw, 2),
        power_factor         = round(pf, 4),
        n_lo_violations      = n_lo,
        n_hi_violations      = n_hi,
        v_deviation_pu       = round(v_dev, 5),
    )


def extract_summary() -> dict:
    tp  = dss.Circuit.TotalPower()
    lss = dss.Circuit.Losses()
    buses = extract_buses()
    n_lo  = sum(1 for d in buses.values() if d["vmin"] < ANSI_LO)
    n_hi  = sum(1 for d in buses.values() if d["vmax"] > ANSI_HI)
    return dict(
        load_kw=-tp[0], load_kvar=-tp[1],
        loss_kw=lss[0] / 1000,
        n_lo_violations=n_lo,
        n_hi_violations=n_hi,
        converged=dss.Solution.Converged(),
    )


# ── Complete topology extraction (no coordinate filter) ───────────────────────

def extract_topology_lines() -> list[dict]:
    """All lines, switches, and transformers without coordinate filtering.
    Used by the PSPS optimizer to build a complete network graph.
    Includes transformer elements so source buses behind regulators stay connected.
    """
    out = []

    n = dss.Lines.First()
    while n:
        name   = dss.Lines.Name()
        b1     = dss.Lines.Bus1().split(".")[0].lower()
        b2     = dss.Lines.Bus2().split(".")[0].lower()
        nph    = dss.Lines.Phases()
        length = dss.Lines.Length()
        is_sw  = bool(dss.Lines.IsSwitch())
        dss.Circuit.SetActiveElement(f"Line.{name}")
        is_open = bool(dss.CktElement.IsOpen(1, 0)) if is_sw else False
        out.append(dict(
            name=name, b1=b1, b2=b2, nph=nph,
            length=length, is_switch=is_sw, is_open=is_open,
        ))
        n = dss.Lines.Next()

    n = dss.Transformers.First()
    while n:
        name = dss.Transformers.Name()
        dss.Circuit.SetActiveElement(f"Transformer.{name}")
        bnames = dss.CktElement.BusNames()
        if len(bnames) >= 2:
            b1 = bnames[0].split(".")[0].lower()
            b2 = bnames[1].split(".")[0].lower()
            out.append(dict(
                name=f"xfmr:{name}", b1=b1, b2=b2,
                nph=dss.CktElement.NumPhases(),
                length=0, is_switch=False, is_open=False,
            ))
        n = dss.Transformers.Next()

    n = dss.Reactors.First()
    while n:
        name = dss.Reactors.Name()
        dss.Circuit.SetActiveElement(f"Reactor.{name}")
        bnames = dss.CktElement.BusNames()
        if len(bnames) >= 2:
            b1 = bnames[0].split(".")[0].lower()
            b2 = bnames[1].split(".")[0].lower()
            if b1 != b2:
                out.append(dict(
                    name=f"reactor:{name}", b1=b1, b2=b2,
                    nph=dss.CktElement.NumPhases(),
                    length=0, is_switch=False, is_open=False,
                ))
        n = dss.Reactors.Next()

    return out


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

    solve_base()
    return results
