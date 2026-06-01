"""modules/vvo/ui.py — Volt/VAR Optimization: T-D Reactive Coordination module.

Models the EMS/ADMS reactive coordination gap described in the working notes.
Three tabs:
  1. Baseline State    — current feeder reactive profile and equipment inventory
  2. Scenario Compare  — 2×2 T-D coordination scenario matrix
  3. Handshake Report  — structured Phase-2 data exchange between EMS and ADMS
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Scenario labels ────────────────────────────────────────────────────────────

_SCENARIO_META = {
    "current": {
        "col": "Current state",
        "desc": "Feeder VVO only. Substation bus cap on local relay. Fixed source voltage.",
    },
    "phase1": {
        "col": "Phase 1 — bus cap in VVO",
        "desc": "Substation bus cap brought under ADMS VVO dispatch. Source still fixed.",
    },
    "phase2_only": {
        "col": "Phase 2 — EMS boundary",
        "desc": "Feeder VVO only, but source voltage = EMS target (not assumed fixed).",
    },
    "phase1_2": {
        "col": "Phase 1 + 2 — full coordination",
        "desc": "Bus cap in VVO scope AND EMS voltage target as boundary condition.",
    },
}

_OBJ_LABELS = {
    "losses":    "Minimize losses (kW)",
    "reactive":  "Minimize reactive import (kVAR)",
    "voltage":   "Minimize voltage deviation",
    "composite": "Composite (losses + reactive + voltage)",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _delta_color(val: float, better: str = "lower") -> str:
    if val == 0:
        return "off"
    return "inverse" if better == "lower" else "normal"


def _metric_delta(new: float, base: float, better: str = "lower") -> tuple[float, str]:
    d = new - base
    return round(d, 2), _delta_color(d, better)


def _voltage_map(buses: dict, lines: list[dict], feeder_cfg: dict) -> go.Figure:
    fig = go.Figure()
    lx, ly = [], []
    for ln in lines:
        lx += [ln["x1"], ln["x2"], None]
        ly += [ln["y1"], ln["y2"], None]
    if lx:
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines",
                                  line=dict(color="#CFD8DC", width=0.8),
                                  hoverinfo="skip", showlegend=False))

    _large   = feeder_cfg.get("large_feeder", False)
    BusTr    = go.Scattergl if _large else go.Scatter
    bus_list = list(buses.keys())
    vmins    = [buses[b]["vmin"] for b in bus_list]

    fig.add_trace(BusTr(
        x=[buses[b]["x"] for b in bus_list],
        y=[buses[b]["y"] for b in bus_list],
        mode="markers",
        marker=dict(
            size=3 if _large else 6,
            color=vmins,
            colorscale="RdYlGn",
            cmin=0.93, cmax=1.05,
            colorbar=dict(title="V (pu)", thickness=12, len=0.6),
            line=dict(width=0),
        ),
        hovertemplate=[
            f"<b>{b}</b><br>Vmin {buses[b]['vmin']:.3f} pu<extra></extra>"
            for b in bus_list
        ],
        showlegend=False,
    ))

    src = feeder_cfg["source_bus"]
    if src in buses:
        fig.add_trace(go.Scatter(
            x=[buses[src]["x"]], y=[buses[src]["y"]], mode="markers",
            marker=dict(symbol="star", size=14, color="#FFD600",
                        line=dict(color="#333", width=1.5)),
            name="T-D boundary",
            hovertemplate=f"<b>Bus {src}</b> (T-D seam)<extra></extra>",
        ))

    xs = [d["x"] for d in buses.values()]
    ys = [d["y"] for d in buses.values()]
    px = (max(xs) - min(xs)) * 0.05
    py = (max(ys) - min(ys)) * 0.05
    fig.update_layout(
        xaxis=dict(visible=False, range=[min(xs) - px, max(xs) + px]),
        yaxis=dict(visible=False, range=[min(ys) - py, max(ys) + py],
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=0, t=30, b=10),
        height=450,
    )
    return fig


# ── Tab 1: Baseline State ──────────────────────────────────────────────────────

def _tab_baseline(base: dict, feeder_cfg: dict) -> None:
    buses     = base["buses"]
    lines     = base["lines"]
    caps      = base.get("caps", [])
    regulators = base.get("regulators", [])
    summary   = base.get("summary", {})

    with st.sidebar:
        st.subheader("About this module")
        st.markdown("""
**VVO** (Volt/VAR Optimization) minimizes distribution system losses and reactive
power import from the transmission system by dispatching capacitor banks and
voltage regulators.

**The T-D coordination gap**: substation bus capacitor banks at the interface
between EMS (transmission) and ADMS (distribution) are often on local relay
control — managed by neither control centre. This module models what is recovered
when they are brought under coordinated VVO dispatch.
        """)

    st.subheader("Feeder Reactive Profile — Baseline")

    m1, m2, m3, m4 = st.columns(4)
    load_kw   = summary.get("load_kw",   0)
    load_kvar = summary.get("load_kvar", 0)
    loss_kw   = summary.get("loss_kw",   0)
    n_lo      = summary.get("n_lo_violations", 0)
    n_hi      = summary.get("n_hi_violations", 0)

    import math
    denom = math.sqrt(load_kw ** 2 + load_kvar ** 2)
    pf    = abs(load_kw) / denom if denom > 0 else 1.0

    m1.metric("Reactive import", f"{load_kvar:,.0f} kVAR",
              help="Q flowing from transmission into distribution (at source bus)")
    m2.metric("Active losses",   f"{loss_kw:.1f} kW",
              delta=f"{loss_kw / max(load_kw, 1) * 100:.2f}% of load",
              delta_color="inverse")
    m3.metric("Power factor",    f"{pf:.3f}")
    m4.metric("Voltage violations",
              f"{n_lo + n_hi}",
              delta=f"{n_lo} low / {n_hi} high",
              delta_color="inverse" if (n_lo + n_hi) > 0 else "off")

    st.divider()

    map_col, tbl_col = st.columns([2, 1])

    with map_col:
        fig = _voltage_map(buses, lines, feeder_cfg)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    with tbl_col:
        st.markdown("**Capacitor banks**")
        if caps:
            st.dataframe(
                pd.DataFrame([{
                    "Name": c["name"],
                    "Bus":  c["bus"],
                    "kVAR": c["kvar"],
                } for c in caps]),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No capacitors in this feeder.")

        if regulators:
            st.markdown("**Voltage regulators**")
            st.dataframe(
                pd.DataFrame([{
                    "Name": r["name"],
                    "Vreg (V)": r["vreg"],
                    "Band (V)": r["band"],
                } for r in regulators]),
                hide_index=True,
                use_container_width=True,
            )

    st.caption(
        f"Virtual substation bus cap: bus **{feeder_cfg['vvo_sub_cap_bus']}**, "
        f"{feeder_cfg['vvo_sub_cap_kvar']:,} kVAR — represents the T-D interface "
        "asset currently on local relay, outside ADMS VVO scope."
    )


# ── Tab 2: Scenario Comparison ─────────────────────────────────────────────────

def _tab_scenarios(feeder_cfg: dict, feeder_name: str,
                   worker_run, load_pkl, tmp_pkl) -> None:

    st.subheader("T-D Reactive Coordination — Scenario Analysis")
    st.markdown(
        "Compare four operational modes across two axes: "
        "**whether the substation bus cap is in VVO scope** (Phase 1) and "
        "**whether VVO receives the actual transmission bus voltage** instead "
        "of assuming a fixed source (Phase 2)."
    )

    with st.sidebar:
        st.subheader("Parameters")
        objective = st.radio(
            "Optimization objective",
            list(_OBJ_LABELS.keys()),
            index=list(_OBJ_LABELS.keys()).index("reactive"),
            format_func=lambda k: _OBJ_LABELS[k],
            key=f"vvo_obj_{feeder_name}",
        )
        sub_cap_kvar = st.slider(
            "Virtual substation bus cap (kVAR)",
            min_value=300, max_value=5000,
            value=feeder_cfg["vvo_sub_cap_kvar"],
            step=300,
            key=f"vvo_kvar_{feeder_name}",
        )
        st.caption(
            f"Injected at bus **{feeder_cfg['vvo_sub_cap_bus']}** "
            f"({feeder_cfg['vvo_sub_cap_kv']} kV) — represents the "
            "stranded substation bus asset."
        )
        _ems_default = round(max(0.95, float(feeder_cfg["source_pu"]) - 0.03), 2)
        ems_pu = st.slider(
            "EMS voltage target (pu)",
            min_value=0.95, max_value=1.05,
            value=_ems_default,
            step=0.01,
            format="%.2f",
            key=f"vvo_ems_{feeder_name}",
        )
        st.caption(
            f"Phase 2: EMS-communicated transmission bus voltage "
            f"(base = {feeder_cfg['source_pu']:.2f} pu). "
            "Default shows a stressed day — move right to return to nominal."
        )
        run_btn = st.button("⚡ Run Scenario Analysis", type="primary",
                             key=f"vvo_run_{feeder_name}")

    cache_key = f"vvo_result_{feeder_name}_{objective}_{sub_cap_kvar}_{ems_pu:.3f}"

    if run_btn:
        pkl = tmp_pkl()
        args = [
            "vvo",
            feeder_cfg["master"],
            objective,
            feeder_cfg["vvo_sub_cap_bus"],
            str(feeder_cfg["vvo_sub_cap_kv"]),
            str(sub_cap_kvar),
            f"{ems_pu:.5f}",
            f"{feeder_cfg['source_pu']:.5f}",
            pkl,
        ]
        with st.spinner(
            "Running T-D coordination analysis — "
            "4 scenarios × 16 cap combinations…"
        ):
            ok, out, err = worker_run(args)
        if not ok:
            st.error(f"VVO worker failed:\n{err}")
            import os; os.unlink(pkl)
            return
        result = load_pkl(pkl)
        import os; os.unlink(pkl)
        st.session_state[cache_key] = result

    result = st.session_state.get(cache_key)
    if result is None:
        st.info("Set parameters and press **Run Scenario Analysis**.")
        return

    baseline = result["baseline"]
    scenarios = result["scenarios"]

    # ── 2 × 2 metric grid ─────────────────────────────────────────────────────
    st.markdown("### Results")

    _ROWS = [
        ("reactive_import_kvar", "Reactive import (kVAR)", "lower"),
        ("loss_kw",              "Losses (kW)",            "lower"),
        ("power_factor",         "Power factor",           "higher"),
        ("n_lo_violations",      "Low-V violations",       "lower"),
        ("n_hi_violations",      "High-V violations",      "lower"),
    ]

    _ORDER = ["current", "phase1", "phase2_only", "phase1_2"]
    headers = ["Metric"] + [_SCENARIO_META[k]["col"] for k in _ORDER]
    tbl_rows = []
    for field, label, better in _ROWS:
        row = [label]
        for key in _ORDER:
            opt = scenarios[key]["optimal"]
            if opt is None:
                row.append("—")
                continue
            val  = opt.get(field, 0)
            base = baseline.get(field, val)
            d    = val - base
            sign = "+" if d >= 0 else ""
            if field == "power_factor":
                row.append(f"{val:.3f} ({sign}{d:+.3f})")
            elif abs(d) < 0.005 and field not in ("n_lo_violations", "n_hi_violations"):
                row.append(f"{val:,.2f}")
            else:
                row.append(f"{val:,.2f} ({sign}{d:,.2f})")
        tbl_rows.append(row)

    df = pd.DataFrame(tbl_rows, columns=headers)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # ── Phase benefit summary ─────────────────────────────────────────────────
    st.markdown("### Benefit decomposition")
    bc1, bc2, bc3 = st.columns(3)
    base_ri  = baseline.get("reactive_import_kvar", 0)
    cur_ri   = (scenarios["current"]["optimal"]   or {}).get("reactive_import_kvar", base_ri)
    p1_ri    = (scenarios["phase1"]["optimal"]    or {}).get("reactive_import_kvar", base_ri)
    p2_ri    = (scenarios["phase2_only"]["optimal"] or {}).get("reactive_import_kvar", base_ri)
    p12_ri   = (scenarios["phase1_2"]["optimal"]  or {}).get("reactive_import_kvar", base_ri)

    bc1.metric(
        "Phase 1 benefit",
        f"{abs(cur_ri - p1_ri):,.0f} kVAR",
        help="Reactive import reduction from adding substation bus cap to VVO scope",
    )
    bc2.metric(
        "Phase 2 benefit",
        f"{abs(cur_ri - p2_ri):,.0f} kVAR",
        help="Reactive import reduction from supplying EMS voltage boundary condition",
    )
    bc3.metric(
        "Full coordination",
        f"{abs(cur_ri - p12_ri):,.0f} kVAR",
        help="Total reactive import reduction with Phase 1 + 2 together",
    )

    # ── Action lists ──────────────────────────────────────────────────────────
    st.markdown("### VVO dispatch actions")
    tabs = st.tabs([_SCENARIO_META[k]["col"] for k in _ORDER])
    for tab, key in zip(tabs, _ORDER):
        with tab:
            st.caption(_SCENARIO_META[key]["desc"])
            actions = scenarios[key].get("cap_actions", [])
            if actions:
                st.dataframe(
                    pd.DataFrame(actions).rename(columns={
                        "asset": "Asset", "was": "Was", "now": "Now", "note": "Note",
                    }),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.success("No switching actions — base case is already optimal.")

    # ── Voltage maps ──────────────────────────────────────────────────────────
    st.markdown("### Voltage maps — optimal state per scenario")
    map_tabs = st.tabs([_SCENARIO_META[k]["col"] for k in _ORDER])
    lines = st.session_state.get(f"base_{feeder_name}", {}).get("lines", [])
    for tab, key in zip(map_tabs, _ORDER):
        with tab:
            opt_buses = scenarios[key].get("opt_buses")
            if opt_buses:
                fig = _voltage_map(opt_buses, lines, feeder_cfg)
                st.plotly_chart(fig, use_container_width=True,
                                config={"scrollZoom": True})
            else:
                st.warning("Voltage map not available for this scenario.")


# ── Tab 3: Reactive Capability Handshake ──────────────────────────────────────

def _tab_handshake(feeder_cfg: dict, feeder_name: str) -> None:
    st.subheader("Phase 2 — Reactive Capability Exchange (EMS ↔ ADMS)")
    st.markdown(
        "In Phase 2, real-time data flows across the T-D seam in both directions. "
        "This tab shows what that exchange looks like in structured form — "
        "the information that would flow over ICCP or middleware between the "
        "Transmission Control Center (EMS) and Distribution Control Center (ADMS)."
    )

    result = None
    for key in st.session_state:
        if key.startswith(f"vvo_result_{feeder_name}_"):
            result = st.session_state[key]
            break

    if result is None:
        st.info("Run the **Scenario Analysis** tab first to populate the handshake report.")
        return

    baseline  = result["baseline"]
    p12       = result["scenarios"]["phase1_2"]
    opt       = p12.get("optimal") or baseline
    sub_kvar  = result["sub_cap_kvar"]
    ems_pu    = result["ems_voltage_pu"]
    base_pu   = result["base_source_pu"]

    ems_col, adms_col = st.columns(2)

    with ems_col:
        st.markdown("#### EMS → ADMS")
        st.caption(
            "What the Transmission Control Center communicates to the "
            "Distribution Control Center as VVO boundary conditions."
        )
        st.markdown(f"""
| Field | Value |
|---|---|
| Transmission bus voltage target | **{ems_pu:.3f} pu** |
| Delta from base case | **{(ems_pu - base_pu):+.3f} pu** |
| Reactive flow through substation transformer (baseline) | **{baseline['reactive_import_kvar']:,.0f} kVAR** |
| Voltage schedule valid for | Rolling 15-min horizon |
| Source | EMS real-time state estimator |
        """)
        if ems_pu < base_pu - 0.005:
            st.warning(
                f"⚠ Transmission bus is depressed ({ems_pu:.3f} pu vs. "
                f"assumed {base_pu:.3f} pu). Distribution VVO must compensate "
                "with local reactive resources — this is the Phase 2 coordination scenario."
            )
        else:
            st.success("Transmission bus voltage is near nominal — no stress condition.")

    with adms_col:
        st.markdown("#### ADMS → EMS")
        st.caption(
            "What the Distribution Control Center reports back to the "
            "Transmission Control Center from the VVO dispatch plan."
        )
        import math
        # Reactive capability: all caps rated kVAR
        cap_total   = sum(c["kvar"] for c in st.session_state.get(
            f"base_{feeder_name}", {}).get("caps", []))
        cap_total  += sub_kvar   # include the virtual substation cap
        opt_ri      = opt.get("reactive_import_kvar", baseline["reactive_import_kvar"])
        base_ri     = baseline["reactive_import_kvar"]
        reduction   = max(0.0, base_ri - opt_ri)
        burden      = opt_ri
        st.markdown(f"""
| Field | Value |
|---|---|
| Total reactive capability (all caps) | **{cap_total:,.0f} kVAR** |
| Reactive capability dispatched | **{reduction:,.0f} kVAR** |
| Remaining reactive burden at transformer | **{burden:,.0f} kVAR** |
| Active losses (optimised) | **{opt.get('loss_kw', 0):,.1f} kW** |
| Voltage violations (optimised) | **{opt.get('n_lo_violations', 0) + opt.get('n_hi_violations', 0)}** |
| Substation bus cap status | **{"In VVO dispatch" if sub_kvar > 0 else "On relay"}** |
        """)

    st.divider()
    st.markdown("#### Recommended data exchange fields for ICCP/middleware configuration")
    st.dataframe(
        pd.DataFrame([
            {"Direction": "EMS → ADMS", "Signal": "Trans. bus voltage (pu)", "Update rate": "1 min", "Phase": "2"},
            {"Direction": "EMS → ADMS", "Signal": "Reactive flow at xfmr (kVAR)", "Update rate": "1 min", "Phase": "2"},
            {"Direction": "EMS → ADMS", "Signal": "Voltage schedule (15 min)", "Update rate": "15 min", "Phase": "2–3"},
            {"Direction": "EMS → ADMS", "Signal": "System stress flag", "Update rate": "Event", "Phase": "2–3"},
            {"Direction": "ADMS → EMS", "Signal": "Available reactive cap (kVAR)", "Update rate": "5 min", "Phase": "2"},
            {"Direction": "ADMS → EMS", "Signal": "VVO dispatch plan", "Update rate": "On change", "Phase": "2–3"},
            {"Direction": "ADMS → EMS", "Signal": "Reactive burden at xfmr (kVAR)", "Update rate": "1 min", "Phase": "2"},
            {"Direction": "ADMS → EMS", "Signal": "Sub bus cap state (on/off/relay)", "Update rate": "On change", "Phase": "1–2"},
        ]),
        hide_index=True,
        use_container_width=True,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def render(
    feeder_cfg: dict,
    base: dict,
    feeder_name: str,
    worker_run,
    load_pkl,
    tmp_pkl,
) -> None:
    tab_baseline, tab_scenarios, tab_handshake = st.tabs([
        "📊 Baseline State",
        "⚡ Scenario Analysis",
        "🤝 Handshake Report",
    ])

    with tab_baseline:
        _tab_baseline(base, feeder_cfg)

    with tab_scenarios:
        _tab_scenarios(feeder_cfg, feeder_name, worker_run, load_pkl, tmp_pkl)

    with tab_handshake:
        _tab_handshake(feeder_cfg, feeder_name)
