"""modules/vvo/optimizer.py — VVO optimization engine for T-D reactive coordination.

Models the gap described in the working notes: substation bus capacitor banks at
the T-D interface are effectively uncontrolled by either EMS or ADMS.  Four
operational scenarios are compared in a 2 × 2 matrix:

              │ Bus cap: relay (current) │ Bus cap: VVO scope (Phase 1) │
  ────────────┼──────────────────────────┼──────────────────────────────┤
  Fixed source│       "current"          │          "phase1"            │
  (ADMS today)│                          │                              │
  ────────────┼──────────────────────────┼──────────────────────────────┤
  EMS voltage │     "phase2_only"        │          "phase1_2"          │
  target      │                          │    (full coordination)       │

The optimizer sweeps all cap-bank switching combinations (exhaustive; both
feeders have ≤ 4 bank groups → 16 combos) and reports loss / reactive import /
voltage deviation under each scenario.
"""
from __future__ import annotations

import math
import opendssdirect as dss

from feeder_io import solve_base, extract_buses, extract_td_seam_metrics, _cmd, ANSI_LO, ANSI_HI

_SUB_CAP = "VVO_SUB_CAP"


# ── Cap grouping ───────────────────────────────────────────────────────────────

def _all_cap_names() -> list[str]:
    names, n = [], dss.Capacitors.First()
    while n:
        names.append(dss.Capacitors.Name())
        n = dss.Capacitors.Next()
    return names


def build_cap_groups(extra_names: list[str] | None = None) -> list[list[str]]:
    """Group A/B/C per-phase caps into bank groups (switched together).
    The virtual sub cap is excluded from the main loop and added via extra_names only.
    """
    names = [n for n in _all_cap_names() if n.lower() != _SUB_CAP.lower()]
    grouped: dict[str, list[str]] = {}

    for name in names:
        upper = name.upper()
        if upper and upper[-1] in ("A", "B", "C"):
            prefix = upper[:-1]
            peers  = [c for c in names if c.upper()[:-1] == prefix
                      and c != name and c.upper()[-1] in ("A", "B", "C")]
            if peers:
                grouped.setdefault(prefix, []).append(name)
                continue
        grouped[name.upper()] = [name]

    groups = list(grouped.values())

    for extra in (extra_names or []):
        groups.append([extra])

    return groups


# ── State helpers ──────────────────────────────────────────────────────────────

def _snapshot_cap_states() -> dict[str, list[int]]:
    snap, n = {}, dss.Capacitors.First()
    while n:
        snap[dss.Capacitors.Name()] = list(dss.Capacitors.States())
        n = dss.Capacitors.Next()
    return snap


def _restore_cap_states(snap: dict[str, list[int]]) -> None:
    for name, st in snap.items():
        _cmd(f"Capacitor.{name}.states=[{' '.join(str(s) for s in st)}]")


def _set_group(group: list[str], on: bool) -> None:
    val = 1 if on else 0
    for name in group:
        _cmd(f"Capacitor.{name}.states=[{val}]")


def _set_source_pu(pu: float) -> None:
    _cmd(f"Edit vsource.source pu={pu:.5f}")


# ── Virtual substation cap ─────────────────────────────────────────────────────

def _inject_sub_cap(bus: str, kv: float, kvar: float) -> None:
    """Inject the virtual substation cap once (disabled). Safe to call again."""
    existing = {n.lower() for n in _all_cap_names()}
    if _SUB_CAP.lower() in existing:
        _cmd(f"Edit Capacitor.{_SUB_CAP} kvar={kvar:.1f}")
    else:
        _cmd(
            f"New Capacitor.{_SUB_CAP} Bus1={bus} kV={kv:.4f} "
            f"kvar={kvar:.1f} phases=3 conn=wye enabled=no"
        )
    _cmd(f"Disable Capacitor.{_SUB_CAP}")


def _enable_sub_cap(on: bool) -> None:
    _cmd(f"{'Enable' if on else 'Disable'} Capacitor.{_SUB_CAP}")


# ── Metrics & objective ────────────────────────────────────────────────────────

def _metrics() -> dict:
    return extract_td_seam_metrics()


def _obj(m: dict, objective: str) -> float:
    if objective == "losses":
        return m["loss_kw"]
    if objective == "reactive":
        return abs(m["reactive_import_kvar"])
    if objective == "voltage":
        return (m["v_deviation_pu"] * 100
                + m["n_lo_violations"] * 10
                + m["n_hi_violations"] * 10)
    # composite
    return (m["loss_kw"] / 200
            + abs(m["reactive_import_kvar"]) / 2000
            + m["v_deviation_pu"] * 5)


# ── Exhaustive sweep ───────────────────────────────────────────────────────────

def _sweep(groups: list[list[str]], objective: str) -> tuple[int, dict]:
    best_mask, best_m, best_v = 0, None, float("inf")
    for mask in range(2 ** len(groups)):
        for k, grp in enumerate(groups):
            _set_group(grp, bool((mask >> k) & 1))
        _cmd("Solve")
        if not dss.Solution.Converged():
            continue
        m = _metrics()
        v = _obj(m, objective)
        if v < best_v:
            best_v, best_mask, best_m = v, mask, m
    return best_mask, best_m


# ── Action list ────────────────────────────────────────────────────────────────

def _actions(
    groups: list[list[str]],
    best_mask: int,
    base_snap: dict[str, list[int]],
    include_sub: bool,
) -> list[dict]:
    actions = []
    for k, grp in enumerate(groups):
        is_sub = any(g.lower() == _SUB_CAP.lower() for g in grp)
        on_now = bool((best_mask >> k) & 1)
        if is_sub:
            if include_sub:
                actions.append(dict(
                    asset=_SUB_CAP + " (virtual substation)",
                    was="Relay — unmanaged",
                    now="VVO ON" if on_now else "VVO OFF",
                    note="Phase 1: asset brought under VVO dispatch",
                ))
        else:
            rep    = grp[0]
            base_on = all(s == 1 for s in base_snap.get(rep, [0]))
            if on_now != base_on:
                label = " + ".join(grp) if len(grp) > 1 else grp[0]
                actions.append(dict(
                    asset=label,
                    was="ON" if base_on else "OFF",
                    now="ON" if on_now else "OFF",
                    note="",
                ))
    return actions


# ── Public: run all four scenarios ─────────────────────────────────────────────

def run_scenarios(
    master: str,
    objective: str,
    sub_cap_bus: str,
    sub_cap_kv: float,
    sub_cap_kvar: float,
    ems_voltage_pu: float,
    base_source_pu: float,
) -> dict:
    """Load the feeder once, run all four T-D coordination scenarios, return results."""

    # ── Load base case ────────────────────────────────────────────────────────
    solve_base(master)
    _cmd("Solve")
    base_snap    = _snapshot_cap_states()
    base_metrics = _metrics()
    base_groups  = build_cap_groups()   # original caps only

    # Inject virtual sub cap (disabled)
    _inject_sub_cap(sub_cap_bus, sub_cap_kv, sub_cap_kvar)

    _CONFIGS = [
        ("current",     False, None),
        ("phase1",      True,  None),
        ("phase2_only", False, ems_voltage_pu),
        ("phase1_2",    True,  ems_voltage_pu),
    ]

    scenarios: dict[str, dict] = {}

    for key, include_sub, ems_pu in _CONFIGS:

        # ── Reset to base state ───────────────────────────────────────────────
        _restore_cap_states(base_snap)
        _enable_sub_cap(False)
        _set_source_pu(ems_pu if ems_pu is not None else base_source_pu)
        if include_sub:
            _enable_sub_cap(True)
        _cmd("Solve")

        pre_metrics = _metrics()

        # ── Build groups for this scenario ────────────────────────────────────
        groups = build_cap_groups(extra_names=[_SUB_CAP] if include_sub else None)

        # ── Sweep ─────────────────────────────────────────────────────────────
        if groups:
            best_mask, best_m = _sweep(groups, objective)
        else:
            best_mask, best_m = 0, pre_metrics

        # ── Apply optimal, capture voltage map ────────────────────────────────
        for k, grp in enumerate(groups):
            _set_group(grp, bool((best_mask >> k) & 1))
        _cmd("Solve")
        opt_buses = extract_buses()

        scenarios[key] = dict(
            pre_opt_metrics = pre_metrics,
            optimal         = best_m,
            cap_actions     = _actions(groups, best_mask, base_snap, include_sub),
            opt_buses       = opt_buses,
        )

    # ── Restore everything ────────────────────────────────────────────────────
    _restore_cap_states(base_snap)
    _enable_sub_cap(False)
    _set_source_pu(base_source_pu)
    _cmd("Solve")

    return dict(
        baseline        = base_metrics,
        scenarios       = scenarios,
        sub_cap_kvar    = sub_cap_kvar,
        ems_voltage_pu  = ems_voltage_pu,
        base_source_pu  = base_source_pu,
        objective       = objective,
        base_cap_groups = [[g for g in grp] for grp in base_groups],
    )
