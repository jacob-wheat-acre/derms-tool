"""app.py — DERMS Explorer: IEEE 123-bus distribution feeder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from feeder_io import (
    ANSI_LO, ANSI_HI,
    solve_base, solve_with_pv,
    extract_buses, extract_lines, extract_loads, extract_caps, extract_summary,
    hosting_capacity_sweep,
)

st.set_page_config(page_title="DERMS Explorer", layout="wide",
                   page_icon="⚡")

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("DERMS Explorer")
    st.caption("IEEE 123-Bus Distribution Feeder · OpenDSS")
    st.divider()

    st.subheader("What is DERMS?")
    st.markdown("""
A **Distributed Energy Resource Management System** gives the utility
visibility and control over all DER on a feeder — solar PV, storage,
EVs, and smart inverters.

The three core functions:

**1. Visibility** — real-time voltage, current, and power at every node.
Without it, operators are flying blind as DER changes the power flow
direction on feeders designed for one-way flow.

**2. Volt-VAR Optimization** — dispatch reactive power (Q) from smart
inverters (IEEE 1547-2018) to keep all bus voltages inside ANSI A
(0.95–1.05 pu). This is the most immediate operational use.

**3. DER Dispatch** — coordinate charge/discharge of storage and curtailment
of PV to manage peak load, congestion, or frequency events.
    """)
    st.divider()
    st.caption("Feeder: IEEE 123-Bus Test Case (Kersting 1991) · 4.16 kV · 91 loads")


# ── Load base case (once per session) ─────────────────────────────────────────

if "base" not in st.session_state:
    with st.spinner("Loading IEEE 123-bus feeder via OpenDSS…"):
        solve_base()
        buses   = extract_buses()
        lines   = extract_lines(buses)
        loads   = extract_loads()
        caps    = extract_caps()
        summary = extract_summary()
        load_buses = sorted({ld["bus"] for ld in loads if ld["bus"] in buses})
        st.session_state["base"] = dict(
            buses=buses, lines=lines, loads=loads,
            caps=caps, summary=summary, load_buses=load_buses,
        )

base       = st.session_state["base"]
buses      = base["buses"]
lines      = base["lines"]
loads      = base["loads"]
caps       = base["caps"]
summary    = base["summary"]
load_buses = base["load_buses"]

# Bus X/Y arrays for plotting
bus_list = list(buses.keys())
xp = {b: buses[b]["x"] for b in bus_list}
yp = {b: buses[b]["y"] for b in bus_list}


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _v_color(v: float) -> str:
    if v < ANSI_LO:
        return "#B71C1C"
    if v > ANSI_HI:
        return "#E65100"
    if v < 0.97:
        return "#F57F17"
    if v > 1.03:
        return "#F9A825"
    return "#2E7D32"


def _line_color(pct: float) -> str:
    if pct >= 100: return "#B71C1C"
    if pct >= 90:  return "#E65100"
    if pct >= 75:  return "#F57F17"
    return "#1B5E20"


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_net, tab_vp, tab_der, tab_hc = st.tabs([
    "🗺️ Network", "📊 Voltage Profile", "☀️ DER Impact", "🏠 Hosting Capacity",
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Network map
# ══════════════════════════════════════════════════════════════════════════════

with tab_net:
    s = summary
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total load", f"{s['load_kw']:.0f} kW")
    m2.metric("Feeder losses", f"{s['loss_kw']:.1f} kW",
              delta=f"{s['loss_kw']/s['load_kw']*100:.1f}% of load", delta_color="inverse")
    m3.metric("Buses", str(len(buses)))
    m4.metric("ANSI low violations", str(s["n_lo_violations"]),
              delta="under 0.95 pu" if s["n_lo_violations"] else "none", delta_color="inverse")
    m5.metric("ANSI high violations", str(s["n_hi_violations"]),
              delta="over 1.05 pu" if s["n_hi_violations"] else "none", delta_color="inverse")

    fig = go.Figure()

    # Lines
    for ln in lines:
        clr = _line_color(ln["pct"])
        fig.add_trace(go.Scatter(
            x=[ln["x1"], ln["x2"], None],
            y=[ln["y1"], ln["y2"], None],
            mode="lines",
            line=dict(color=clr, width=2 if ln["nph"] == 3 else 1.2,
                      dash="dot" if ln["nph"] == 1 else "solid"),
            hoverinfo="skip", showlegend=False,
        ))

    # Capacitor banks — blue diamonds
    cap_bus_set = {c["bus"] for c in caps}
    cap_x = [xp[b] for b in cap_bus_set if b in xp]
    cap_y = [yp[b] for b in cap_bus_set if b in yp]
    if cap_x:
        fig.add_trace(go.Scatter(
            x=cap_x, y=cap_y, mode="markers",
            marker=dict(symbol="diamond", size=14, color="#1565C0",
                        line=dict(color="white", width=1.5)),
            name="Capacitor bank",
            hovertemplate="Cap bank<extra></extra>",
        ))

    # Buses — colored by min phase voltage
    node_colors = [_v_color(buses[b]["vmin"]) for b in bus_list]
    hover_text  = [
        f"<b>Bus {b}</b><br>"
        f"Phases: {buses[b]['phases']}<br>"
        f"V = {', '.join(f'{v:.4f}' for v in buses[b]['vmag'])} pu<br>"
        f"V_min = {buses[b]['vmin']:.4f} pu"
        "<extra></extra>"
        for b in bus_list
    ]
    fig.add_trace(go.Scatter(
        x=[xp[b] for b in bus_list],
        y=[yp[b] for b in bus_list],
        mode="markers+text",
        marker=dict(size=10, color=node_colors,
                    line=dict(color="white", width=1)),
        text=bus_list,
        textposition="top center",
        textfont=dict(size=8, color="#333333"),
        hovertemplate=hover_text,
        showlegend=False,
    ))

    # Legend patches (manual)
    for label, clr in [("< 0.95 pu (low)", "#B71C1C"),
                        ("0.95–1.05 pu (OK)", "#2E7D32"),
                        ("> 1.05 pu (high)", "#E65100")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=clr),
            name=label,
        ))

    xs = [v for b in bus_list for v in [xp[b]]]
    ys = [v for b in bus_list for v in [yp[b]]]
    pad_x = (max(xs) - min(xs)) * 0.05
    pad_y = (max(ys) - min(ys)) * 0.05

    fig.update_layout(
        xaxis=dict(visible=False, range=[min(xs)-pad_x, max(xs)+pad_x]),
        yaxis=dict(visible=False, range=[min(ys)-pad_y, max(ys)+pad_y],
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="white", paper_bgcolor="white",
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.85)"),
        title=dict(text="Node color = min phase voltage  |  "
                        "Line width = phases  |  ◆ = capacitor bank",
                   font=dict(size=11), x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, width="stretch",
                    config={"scrollZoom": True, "displayModeBar": True})

    with st.expander("Capacitor banks and regulators"):
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Capacitor banks**")
            st.dataframe(pd.DataFrame(caps), hide_index=True, width="stretch")
        with cc2:
            st.markdown("**Load summary by phase count**")
            df_ld = pd.DataFrame(loads)
            ph_sum = df_ld.groupby("phases").agg(
                count=("name","count"), kW=("kw","sum"), kvar=("kvar","sum")
            ).reset_index()
            ph_sum.columns = ["Phases", "Loads", "Total kW", "Total kvar"]
            st.dataframe(ph_sum, hide_index=True, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Voltage profile
# ══════════════════════════════════════════════════════════════════════════════

with tab_vp:
    st.subheader("Voltage Profile — Base Case")

    st.markdown("""
The feeder starts at **Bus 150** (the substation), set at 1.00 pu. Voltage generally
drops as you move away from the substation because current flowing through line
resistance causes a voltage drop. The regulators (transformer tap changers) boost
voltage at key points to keep everything within **ANSI A limits: 0.95–1.05 pu**.

Because this is an unbalanced distribution feeder, phases A, B, and C are shown
separately — a single-phase lateral on phase B only appears in the blue trace.
Hover over any point to see the bus and exact voltage.
    """)

    # Build per-phase data, sorted numerically by bus number
    def _bus_sort_key(b):
        try: return int(b)
        except: return 9999

    sorted_buses = sorted(bus_list, key=_bus_sort_key)

    ph_data = {1: [], 2: [], 3: []}   # {phase: [(bus_num, v_pu), ...]}
    for b in sorted_buses:
        d = buses[b]
        for i, ph in enumerate(d["phases"]):
            if ph in ph_data:
                ph_data[ph].append((b, d["vmag"][i]))

    vfig = go.Figure()

    vfig.add_hrect(y0=ANSI_LO, y1=ANSI_HI, fillcolor="rgba(144,238,144,0.15)",
                   line_width=0)
    vfig.add_hline(y=ANSI_LO, line_dash="dash", line_color="#B71C1C",
                   annotation_text="0.95 pu — ANSI low limit",
                   annotation_position="bottom right", annotation_font_size=10)
    vfig.add_hline(y=ANSI_HI, line_dash="dash", line_color="#E65100",
                   annotation_text="1.05 pu — ANSI high limit",
                   annotation_position="top right", annotation_font_size=10)

    ph_colors = {1: "#1565C0", 2: "#C62828", 3: "#2E7D32"}
    ph_names  = {1: "Phase A", 2: "Phase B", 3: "Phase C"}

    for ph in [1, 2, 3]:
        pts = ph_data[ph]
        if not pts:
            continue
        bus_labels = [p[0] for p in pts]
        voltages   = [p[1] for p in pts]
        dot_colors = [_v_color(v) for v in voltages]
        vfig.add_trace(go.Scatter(
            x=bus_labels, y=voltages,
            mode="markers",
            name=ph_names[ph],
            marker=dict(size=7, color=dot_colors,
                        line=dict(color=ph_colors[ph], width=1.5)),
            hovertemplate=(
                f"<b>%{{x}}</b> — {ph_names[ph]}<br>"
                "V = %{y:.4f} pu<extra></extra>"
            ),
        ))

    vfig.update_layout(
        xaxis=dict(title="Bus number (sorted numerically)",
                   type="category", tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(title="Voltage (pu)", range=[0.90, 1.10]),
        plot_bgcolor="white", paper_bgcolor="white",
        height=430, margin=dict(l=10, r=10, t=20, b=60),
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.85)"),
    )
    st.plotly_chart(vfig, width="stretch")

    # Summary callout
    all_v = [v for pts in ph_data.values() for _, v in pts]
    n_ok  = sum(1 for v in all_v if ANSI_LO <= v <= ANSI_HI)
    n_low = sum(1 for v in all_v if v < ANSI_LO)
    n_hi  = sum(1 for v in all_v if v > ANSI_HI)
    vmin, vmax = min(all_v), max(all_v)

    if n_low == 0 and n_hi == 0:
        st.success(
            f"**All {n_ok} phase-bus readings are within ANSI A.** "
            f"Voltage ranges from {vmin:.4f} to {vmax:.4f} pu. "
            f"The regulators are keeping the profile flat. Add DER in the next tab "
            f"to see how PV pushes voltages up and potentially causes overvoltage violations."
        )
    else:
        st.warning(
            f"**{n_low} readings below 0.95 pu, {n_hi} above 1.05 pu** — "
            f"range {vmin:.4f}–{vmax:.4f} pu. Check the network map for which buses are affected."
        )

    # Worst buses table
    with st.expander("All buses — sorted by voltage"):
        flat_rows = []
        for b in sorted_buses:
            d = buses[b]
            for i, ph in enumerate(d["phases"]):
                flat_rows.append({
                    "Bus": b,
                    "Phase": {1:"A",2:"B",3:"C"}.get(ph, str(ph)),
                    "Voltage (pu)": round(d["vmag"][i], 4),
                    "Status": ("LOW" if d["vmag"][i] < ANSI_LO else
                               "HIGH" if d["vmag"][i] > ANSI_HI else "OK"),
                })
        flat_rows.sort(key=lambda r: r["Voltage (pu)"])
        st.dataframe(pd.DataFrame(flat_rows), hide_index=True, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — DER impact
# ══════════════════════════════════════════════════════════════════════════════

with tab_der:
    st.subheader("DER Impact — Add PV and Re-solve")

    st.markdown("""
This is the core DERMS visibility problem. When a customer or developer installs
PV on a feeder, voltage **rises** at that bus and upstream — potentially pushing
buses above the ANSI limit. The DERMS needs to detect this and either curtail
the PV or dispatch reactive absorption from smart inverters.
    """)

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        def _bus_label(b):
            ph = buses[b]["phases"]
            ph_str = "".join({1:"A",2:"B",3:"C"}.get(p,"?") for p in ph)
            return f"Bus {b} ({ph_str})"
        der_bus = st.selectbox(
            "PV interconnection bus", load_buses,
            index=load_buses.index("57") if "57" in load_buses else 0,
            format_func=_bus_label,
        )
    with dc2:
        der_kw = st.slider("PV size (kW)", 50, 2000, 500, 50)
    with dc3:
        der_pf = st.slider("Power factor", 0.80, 1.00, 1.00, 0.01,
                           help="Unity = no reactive output. <1.0 = absorbing Q (lead).")

    run_der = st.button("▶ Solve with PV", type="primary")

    if run_der:
        with st.spinner("Re-solving with PV added…"):
            try:
                solve_with_pv(der_bus, float(der_kw), float(der_pf))
                der_buses = extract_buses()
                der_sum   = extract_summary()
                st.session_state["der_result"] = dict(
                    buses=der_buses, summary=der_sum,
                    bus=der_bus, kw=der_kw, pf=der_pf,
                )
                # Restore base case so other tabs aren't affected
                solve_base()
            except Exception as e:
                st.error(f"Solve error: {e}")
                st.session_state.pop("der_result", None)

    dr = st.session_state.get("der_result")
    if dr is not None:
        db = dr["buses"]
        ds = dr["summary"]

        # Metrics comparison
        dm1, dm2, dm3, dm4 = st.columns(4)
        dm1.metric("Net load after PV", f"{ds['load_kw']:.0f} kW",
                   delta=f"{ds['load_kw']-summary['load_kw']:+.0f} kW vs base",
                   delta_color="inverse")
        dm2.metric("Feeder losses", f"{ds['loss_kw']:.1f} kW",
                   delta=f"{ds['loss_kw']-summary['loss_kw']:+.1f} kW",
                   delta_color="inverse")
        dm3.metric("ANSI low violations", str(ds["n_lo_violations"]),
                   delta=f"{ds['n_lo_violations']-summary['n_lo_violations']:+d} vs base",
                   delta_color="inverse")
        dm4.metric("ANSI high violations", str(ds["n_hi_violations"]),
                   delta=f"{ds['n_hi_violations']-summary['n_hi_violations']:+d} vs base",
                   delta_color="inverse")

        # Interpretation
        dv_at_bus = (db[der_bus]["vmin"] - buses[der_bus]["vmin"]
                     if der_bus in db and der_bus in buses else 0)
        n_new_hi  = ds["n_hi_violations"] - summary["n_hi_violations"]
        if n_new_hi > 0:
            st.warning(
                f"**{n_new_hi} new overvoltage violation(s)** — {der_kw} kW of PV at Bus "
                f"{der_bus} raised voltages above 1.05 pu. A DERMS would detect this and "
                f"either curtail the PV output or dispatch reactive absorption (Q<0) from "
                f"the IEEE 1547-2018 inverter."
            )
        elif dv_at_bus > 0.005:
            st.success(
                f"**No violations** — PV at Bus {der_bus} raised the local voltage "
                f"by {dv_at_bus:+.4f} pu but all buses remain within ANSI A. "
                f"The feeder can host this DER without Volt-VAR support."
            )
        else:
            st.info("PV added and solved — check the voltage comparison below.")

        # Before/after voltage comparison
        st.markdown("**Before vs. after voltage — all buses**")
        common = [b for b in bus_list if b in db]
        delta_v = {b: db[b]["vmin"] - buses[b]["vmin"] for b in common}

        comp_rows = sorted(
            [dict(bus=b, v_base=buses[b]["vmin"], v_pv=db[b]["vmin"],
                  delta=delta_v[b]) for b in common],
            key=lambda r: -r["delta"],
        )

        cfig = go.Figure()
        cfig.add_hline(y=ANSI_LO, line_dash="dash", line_color="#B71C1C")
        cfig.add_hline(y=ANSI_HI, line_dash="dash", line_color="#E65100")
        cfig.add_trace(go.Bar(
            x=[r["bus"] for r in comp_rows],
            y=[r["v_base"] for r in comp_rows],
            name="Base", marker_color="#607D8B",
            hovertemplate="Bus %{x}<br>Base V = %{y:.4f} pu<extra></extra>",
        ))
        cfig.add_trace(go.Bar(
            x=[r["bus"] for r in comp_rows],
            y=[r["v_pv"] for r in comp_rows],
            name=f"+{der_kw} kW PV @ Bus {der_bus}",
            marker_color="#F9A825",
            hovertemplate="Bus %{x}<br>With PV V = %{y:.4f} pu<extra></extra>",
        ))
        cfig.update_layout(
            barmode="overlay", xaxis_title="Bus (sorted by ΔV, largest first)",
            yaxis_title="Voltage (pu)", yaxis=dict(range=[0.90, 1.10]),
            plot_bgcolor="white", paper_bgcolor="white",
            height=380, margin=dict(l=10, r=10, t=20, b=40),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(cfig, width="stretch")

        # ΔV network map
        st.markdown("**ΔV network map** — how much each bus voltage changed")
        dv_vals = np.array([delta_v.get(b, 0.0) for b in bus_list])
        dv_max  = float(max(abs(dv_vals.min()), abs(dv_vals.max()), 0.001))

        nfig = go.Figure()
        for ln in lines:
            nfig.add_trace(go.Scatter(
                x=[ln["x1"], ln["x2"], None], y=[ln["y1"], ln["y2"], None],
                mode="lines", line=dict(color="#CCCCCC", width=1),
                hoverinfo="skip", showlegend=False,
            ))

        nfig.add_trace(go.Scatter(
            x=[xp[b] for b in bus_list],
            y=[yp[b] for b in bus_list],
            mode="markers+text",
            marker=dict(
                size=12, color=dv_vals.tolist(),
                colorscale="RdYlGn_r",
                cmin=-dv_max, cmax=dv_max,
                showscale=True,
                colorbar=dict(title="ΔV (pu)", thickness=12, len=0.75, x=1.01),
                line=dict(color="white", width=1),
            ),
            text=bus_list,
            textposition="top center",
            textfont=dict(size=7, color="#333"),
            hovertemplate=[
                f"<b>Bus {b}</b><br>ΔV = {delta_v.get(b,0):+.4f} pu<extra></extra>"
                for b in bus_list
            ],
            showlegend=False,
        ))

        # Mark PV bus
        if der_bus in xp:
            nfig.add_trace(go.Scatter(
                x=[xp[der_bus]], y=[yp[der_bus]], mode="markers",
                marker=dict(symbol="star", size=18, color="#F9A825",
                            line=dict(color="black", width=1.5)),
                name=f"PV @ Bus {der_bus}", hoverinfo="skip",
            ))

        nfig.update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=500, margin=dict(l=10, r=60, t=20, b=10),
            legend=dict(x=0.01, y=0.01),
            title=dict(text="Green = voltage rose  |  Red = voltage dropped  |  ★ = PV bus",
                       font=dict(size=11), x=0.5, xanchor="center"),
        )
        st.plotly_chart(nfig, width="stretch",
                        config={"scrollZoom": True, "displayModeBar": True})


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Hosting capacity
# ══════════════════════════════════════════════════════════════════════════════

with tab_hc:
    st.subheader("Hosting Capacity — Maximum PV per Bus")

    st.markdown("""
**Hosting capacity** is the maximum amount of PV a bus can absorb before
any bus on the feeder exceeds ANSI A voltage limits (0.95–1.05 pu).
It's the key output DERMS uses to answer: *"Can this interconnection be
approved?"* and *"How much can the feeder absorb without infrastructure upgrades?"*

The sweep injects PV at one bus at a time and increases the size until a
violation appears. Run time is roughly 1–3 minutes for the full 123-bus feeder.
    """)

    hc1, hc2 = st.columns(2)
    with hc1:
        hc_step = st.select_slider(
            "Sweep step size (kW)", options=[50, 100, 150, 200], value=100,
            help="Smaller = more precise but slower. 100 kW is a good starting point.",
        )
    with hc2:
        hc_max = st.select_slider(
            "Maximum PV to test (kW)", options=[500, 1000, 1500, 2000], value=1000,
        )

    sweep_btn = st.button("📈 Run Hosting Capacity Sweep", type="primary",
                          key="hc_sweep_btn")

    if sweep_btn:
        prog = st.progress(0.0, text="Running hosting capacity sweep…")
        try:
            hc_results = hosting_capacity_sweep(
                load_buses,
                step_kw=float(hc_step),
                max_kw=float(hc_max),
                progress_cb=lambda v: prog.progress(v, text=f"Sweep {v*100:.0f}%…"),
            )
            prog.empty()
            st.session_state["hc_results"] = hc_results
        except Exception as e:
            prog.empty()
            st.error(f"Sweep error: {e}")

    hcr = st.session_state.get("hc_results")
    if hcr is not None:
        total_hc = sum(hcr.values())
        n_zero   = sum(1 for v in hcr.values() if v == 0)
        n_max    = sum(1 for v in hcr.values() if v >= hc_max)

        hm1, hm2, hm3 = st.columns(3)
        hm1.metric("Total hosting capacity", f"{total_hc:.0f} kW",
                   delta=f"across {len(hcr)} buses")
        hm2.metric("Buses at zero capacity", str(n_zero),
                   delta="already voltage-constrained" if n_zero else "none",
                   delta_color="inverse" if n_zero else "off")
        hm3.metric("Buses at max tested", str(n_max),
                   delta=f"can host ≥ {hc_max:.0f} kW",
                   delta_color="normal" if n_max else "off")

        if n_zero:
            st.info(
                f"**{n_zero} buses show zero hosting capacity** — these buses already "
                f"sit near the ANSI limit in the base case. Even a small PV injection "
                f"causes a violation elsewhere on the feeder, usually upstream near the "
                f"substation or at buses that are already high."
            )

        # Network map — colored by hosting capacity
        hc_vals = np.array([hcr.get(b, 0.0) for b in bus_list])

        hfig = go.Figure()
        for ln in lines:
            hfig.add_trace(go.Scatter(
                x=[ln["x1"], ln["x2"], None], y=[ln["y1"], ln["y2"], None],
                mode="lines", line=dict(color="#CCCCCC", width=1.2),
                hoverinfo="skip", showlegend=False,
            ))

        hfig.add_trace(go.Scatter(
            x=[xp[b] for b in bus_list],
            y=[yp[b] for b in bus_list],
            mode="markers+text",
            marker=dict(
                size=13, color=hc_vals.tolist(),
                colorscale="RdYlGn",
                cmin=0, cmax=float(hc_max),
                showscale=True,
                colorbar=dict(title="Hosting<br>cap (kW)", thickness=12,
                              len=0.75, x=1.01),
                line=dict(color="white", width=1),
            ),
            text=bus_list,
            textposition="top center",
            textfont=dict(size=7, color="#333"),
            hovertemplate=[
                f"<b>Bus {b}</b><br>Hosting capacity = {hcr.get(b,0):.0f} kW<extra></extra>"
                for b in bus_list
            ],
            showlegend=False,
        ))

        hfig.update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=550, margin=dict(l=10, r=70, t=30, b=10),
            title=dict(
                text="Green = high hosting capacity  |  Red = voltage-constrained",
                font=dict(size=11), x=0.5, xanchor="center",
            ),
        )
        st.plotly_chart(hfig, width="stretch",
                        config={"scrollZoom": True, "displayModeBar": True})

        # Sortable table
        st.markdown("**Hosting capacity by bus**")
        hc_df = pd.DataFrame([
            dict(Bus=b, **{"Hosting Capacity (kW)": hcr[b],
                           "Status": ("Constrained" if hcr[b] == 0 else
                                      f"≥ {hc_max:.0f} kW" if hcr[b] >= hc_max else
                                      f"{hcr[b]:.0f} kW")})
            for b in sorted(hcr, key=lambda x: -hcr[x])
        ])
        st.dataframe(hc_df, hide_index=True, width="stretch")
