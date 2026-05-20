"""app.py — DERMS Explorer: IEEE distribution feeder analysis via OpenDSS."""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from feeder_io import ANSI_LO, ANSI_HI, FEEDERS

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dss_worker.py")


def _worker_run(args: list[str]) -> tuple[bool, str, str]:
    r = subprocess.run([sys.executable, _WORKER] + args, capture_output=True, text=True)
    return r.returncode == 0, r.stdout, r.stderr


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _tmp_pkl() -> str:
    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    return path

st.set_page_config(page_title="DERMS Explorer", layout="wide",
                   page_icon="⚡")

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("DERMS Explorer")

    feeder_name = st.selectbox("IEEE Test Feeder", list(FEEDERS.keys()), key="feeder")
    feeder_cfg  = FEEDERS[feeder_name]

    st.caption(feeder_cfg["caption"])
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


# ── Load base case (cached per feeder) ────────────────────────────────────────

cache_key = f"base_{feeder_name}"

if cache_key not in st.session_state:
    with st.spinner(f"Loading {feeder_name} via OpenDSS…"):
        _pkl = _tmp_pkl()
        _args = ["base", feeder_cfg["master"], _pkl]
        _kv_min = feeder_cfg.get("der_kv_min")
        if _kv_min is not None:
            _args.append(str(_kv_min))
        _ok, _out, _err = _worker_run(_args)
        if not _ok:
            os.unlink(_pkl)
            st.error(f"DSS worker failed:\n{_err}")
            st.stop()
        st.session_state[cache_key] = _load_pkl(_pkl)
        os.unlink(_pkl)

base       = st.session_state[cache_key]
buses      = base["buses"]
lines      = base["lines"]
loads      = base["loads"]
caps       = base["caps"]
switches         = base.get("switches", [])
fuses            = base.get("fuses", [])
regulators       = base.get("regulators", [])
sub_transformers = base.get("sub_transformers", [])
feeder_zones     = base.get("feeder_zones", {})
summary      = base["summary"]
load_buses   = base["load_buses"]

# Zone → color mapping (sorted meter names get stable colors across reruns)
_FEEDER_PALETTE = ["#0288D1", "#E64A19", "#7B1FA2", "#00897B"]
_zone_names  = sorted(set(feeder_zones.values()))
_zone_color  = {z: _FEEDER_PALETTE[i % len(_FEEDER_PALETTE)] for i, z in enumerate(_zone_names)}
_zone_color[None] = "#78909C"   # backbone / unassigned

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

    # Lines — grouped by style to avoid thousands of individual traces
    # When feeder zones exist, color by zone rather than loading
    _segs: dict = {}
    for ln in lines:
        if feeder_zones:
            zone = feeder_zones.get(ln["b1"]) or feeder_zones.get(ln["b2"])
            clr = _zone_color.get(zone, "#78909C")
            w   = 1.5 if ln["nph"] == 3 else 0.8
            d   = "solid"
        else:
            clr = _line_color(ln["pct"])
            w   = 2 if ln["nph"] == 3 else 1.2
            d   = "dot" if ln["nph"] == 1 else "solid"
        key = (clr, w, d)
        if key not in _segs:
            _segs[key] = {"x": [], "y": []}
        _segs[key]["x"] += [ln["x1"], ln["x2"], None]
        _segs[key]["y"] += [ln["y1"], ln["y2"], None]
    for (clr, w, d), seg in _segs.items():
        fig.add_trace(go.Scatter(
            x=seg["x"], y=seg["y"],
            mode="lines",
            line=dict(color=clr, width=w, dash=d),
            hoverinfo="skip", showlegend=False,
        ))

    _large = feeder_cfg.get("large_feeder", False)

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

    # Voltage regulators — orange squares
    if regulators:
        fig.add_trace(go.Scatter(
            x=[r["x"] for r in regulators],
            y=[r["y"] for r in regulators],
            mode="markers",
            marker=dict(symbol="square", size=13, color="#E65100",
                        line=dict(color="white", width=1.5)),
            name="Voltage regulator",
            hovertemplate=[
                f"<b>Reg {r['name']}</b><br>Transformer: {r['xfmr']}<br>"
                f"V_set = {r['vreg']:.0f} V  Band = ±{r['band']:.0f} V<extra></extra>"
                for r in regulators
            ],
        ))

    # Substation transformers — large gold squares
    if sub_transformers:
        fig.add_trace(go.Scatter(
            x=[t["x"] for t in sub_transformers],
            y=[t["y"] for t in sub_transformers],
            mode="markers",
            marker=dict(symbol="square", size=18, color="#F9A825",
                        line=dict(color="#333", width=1.5)),
            name="Substation transformer",
            hovertemplate=[
                f"<b>{t['name']}</b><br>"
                f"{t['kv1']:.0f} / {t['kv2']:.3f} kV<br>"
                f"{t['kva']/1000:.0f} MVA<extra></extra>"
                for t in sub_transformers
            ],
        ))

    # Fuses — magenta triangles
    if fuses:
        fig.add_trace(go.Scatter(
            x=[f["x"] for f in fuses],
            y=[f["y"] for f in fuses],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#AD1457",
                        line=dict(color="white", width=1.5)),
            name="Fuse",
            hovertemplate=[
                f"<b>Fuse {f['name']}</b><extra></extra>" for f in fuses
            ],
        ))

    # Switches — green squares (closed) or red squares (open)
    sw_closed = [s for s in switches if not s["is_open"]]
    sw_open   = [s for s in switches if s["is_open"]]
    _sw_size  = 8 if _large else 11
    if sw_closed:
        fig.add_trace(go.Scatter(
            x=[s["x"] for s in sw_closed],
            y=[s["y"] for s in sw_closed],
            mode="markers",
            marker=dict(symbol="square", size=_sw_size, color="#1B5E20",
                        line=dict(color="white", width=1.5)),
            name="Switch (closed)",
            hovertemplate=[
                f"<b>Switch {s['name']}</b><br>{s['nph']}-phase · closed<extra></extra>"
                for s in sw_closed
            ],
        ))
    if sw_open:
        fig.add_trace(go.Scatter(
            x=[s["x"] for s in sw_open],
            y=[s["y"] for s in sw_open],
            mode="markers",
            marker=dict(symbol="square-open", size=_sw_size, color="#B71C1C",
                        line=dict(color="#B71C1C", width=2)),
            name="Switch (open)",
            hovertemplate=[
                f"<b>Switch {s['name']}</b><br>{s['nph']}-phase · open<extra></extra>"
                for s in sw_open
            ],
        ))

    # Source bus — gold star
    _src = feeder_cfg["source_bus"]
    if _src in xp:
        fig.add_trace(go.Scatter(
            x=[xp[_src]], y=[yp[_src]], mode="markers",
            marker=dict(symbol="star", size=18, color="#FFD600",
                        line=dict(color="#333", width=1.5)),
            name=f"Feeder head (Bus {_src})",
            hovertemplate=f"<b>Feeder head — Bus {_src}</b><extra></extra>",
        ))

    # Buses — colored by feeder zone (if multi-feeder) or min phase voltage
    BusTrace = go.Scattergl if _large else go.Scatter
    if feeder_zones:
        node_colors = [_zone_color.get(feeder_zones.get(b), "#78909C") for b in bus_list]
    else:
        node_colors = [_v_color(buses[b]["vmin"]) for b in bus_list]
    hover_text = [
        f"<b>Bus {b}</b><br>"
        f"V_min = {buses[b]['vmin']:.4f} pu  V_max = {buses[b]['vmax']:.4f} pu"
        + (f"<br>Feeder: {feeder_zones.get(b, 'backbone')}" if feeder_zones else "")
        + "<extra></extra>"
        for b in bus_list
    ]
    fig.add_trace(BusTrace(
        x=[xp[b] for b in bus_list],
        y=[yp[b] for b in bus_list],
        mode="markers" if _large else "markers+text",
        marker=dict(size=4 if _large else 10, color=node_colors,
                    line=dict(color="white", width=0.5 if _large else 1),
                    opacity=0.8 if _large else 1.0),
        text=None if _large else bus_list,
        textposition="top center",
        textfont=dict(size=8, color="#333333"),
        hovertemplate=hover_text,
        showlegend=False,
    ))

    # Voltage violation overlay (open rings) when buses are colored by feeder zone
    if feeder_zones:
        for vio_buses_list, label, clr in [
            ([b for b in bus_list if buses[b]["vmin"] < ANSI_LO], "Undervoltage (< 0.95 pu)", "#B71C1C"),
            ([b for b in bus_list if buses[b]["vmax"] > ANSI_HI], "Overvoltage (> 1.05 pu)", "#E65100"),
        ]:
            if vio_buses_list:
                fig.add_trace(go.Scatter(
                    x=[xp[b] for b in vio_buses_list],
                    y=[yp[b] for b in vio_buses_list],
                    mode="markers",
                    marker=dict(symbol="circle-open", size=10, color=clr,
                                line=dict(color=clr, width=2.5)),
                    name=label, hoverinfo="skip",
                ))

    # Legend
    if feeder_zones:
        for z in _zone_names:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=_zone_color[z], symbol="circle"),
                name=f"Feeder {z}",
            ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color="#78909C"), name="Backbone",
        ))
    else:
        for label, clr in [("< 0.95 pu (low)", "#B71C1C"),
                            ("0.95–1.05 pu (OK)", "#2E7D32"),
                            ("> 1.05 pu (high)", "#E65100")]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=clr),
                name=label,
            ))

    xs = [xp[b] for b in bus_list]
    ys = [yp[b] for b in bus_list]
    pad_x = (max(xs) - min(xs)) * 0.05
    pad_y = (max(ys) - min(ys)) * 0.05

    fig.update_layout(
        uirevision=cache_key,
        xaxis=dict(visible=False, range=[min(xs)-pad_x, max(xs)+pad_x]),
        yaxis=dict(visible=False, range=[min(ys)-pad_y, max(ys)+pad_y],
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="white", paper_bgcolor="white",
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.85)",
                    itemsizing="constant"),
        title=dict(
            text=("Node/line color = feeder zone  |  ○ = ANSI violation  |  ★ = feeder head"
                  if feeder_zones else
                  "Node color = min phase voltage  |  Line width = phases  |  ★ = feeder head"),
            font=dict(size=11), x=0.5, xanchor="center",
        ),
    )
    st.plotly_chart(fig, width="stretch",
                    config={"scrollZoom": True, "displayModeBar": True})

    with st.expander("Capacitor banks and load summary"):
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
    st.markdown(feeder_cfg["profile_intro"])

    def _bus_sort_key(b):
        try: return int(b)
        except: return 9999

    sorted_buses = sorted(bus_list, key=_bus_sort_key)

    ph_data = {1: [], 2: [], 3: []}
    for b in sorted_buses:
        d = buses[b]
        for i, ph in enumerate(d["phases"]):
            if ph in ph_data:
                ph_data[ph].append((b, d["vmag"][i]))

    _large_vp = feeder_cfg.get("large_feeder", False)
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

    if _large_vp:
        # For large feeders: histogram of voltage distribution per phase
        for ph in [1, 2, 3]:
            pts = ph_data[ph]
            if not pts:
                continue
            voltages = [p[1] for p in pts]
            vfig.add_trace(go.Histogram(
                x=voltages, name=ph_names[ph],
                marker_color=ph_colors[ph], opacity=0.65,
                xbins=dict(start=0.80, end=1.15, size=0.005),
                hovertemplate=f"{ph_names[ph]}: %{{x:.3f}}–%{{x:.3f}} pu<br>%{{y}} buses<extra></extra>",
            ))
        vfig.update_layout(
            barmode="overlay",
            xaxis=dict(title="Voltage (pu)", range=[0.80, 1.15]),
            yaxis=dict(title="Number of buses"),
        )
        st.caption(
            "Large feeder: showing voltage **distribution** (histogram) rather than per-bus scatter. "
            "Each bar = number of phase-bus readings in that 0.005 pu voltage bin."
        )
    else:
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
        )

    vfig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=430, margin=dict(l=10, r=10, t=20, b=60),
        legend=dict(x=0.01, y=0.99 if _large_vp else 0.01,
                    bgcolor="rgba(255,255,255,0.85)"),
    )

    all_v = [v for pts in ph_data.values() for _, v in pts]
    n_ok  = sum(1 for v in all_v if ANSI_LO <= v <= ANSI_HI)
    n_low = sum(1 for v in all_v if v < ANSI_LO)
    n_hi  = sum(1 for v in all_v if v > ANSI_HI)
    vmin, vmax = min(all_v), max(all_v)

    if n_low == 0 and n_hi == 0:
        st.success(
            f"**All {n_ok} phase-bus readings are within ANSI A.** "
            f"Voltage ranges from {vmin:.4f} to {vmax:.4f} pu. "
            f"Add DER in the next tab to see how PV pushes voltages up."
        )
    else:
        st.warning(
            f"**{n_low} readings below 0.95 pu, {n_hi} above 1.05 pu** — "
            f"range {vmin:.4f}–{vmax:.4f} pu. Check the network map for which buses are affected."
        )

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
    st.markdown(feeder_cfg["der_intro"])

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        def _bus_label(b):
            ph = buses[b]["phases"]
            ph_str = "".join({1:"A",2:"B",3:"C"}.get(p,"?") for p in ph)
            return f"Bus {b} ({ph_str})"
        _def_bus = feeder_cfg["default_der_bus"]
        _def_idx = load_buses.index(_def_bus) if _def_bus in load_buses else 0
        der_bus = st.selectbox(
            "PV interconnection bus", load_buses,
            index=_def_idx,
            format_func=_bus_label,
            key=f"der_bus_{feeder_name}",
        )
    with dc2:
        der_kw = st.slider("PV size (kW)", 50, 2000, 500, 50,
                           key=f"der_kw_{feeder_name}")
    with dc3:
        der_pf = st.slider("Power factor", 0.80, 1.00, 1.00, 0.01,
                           help="Unity = no reactive output. <1.0 = absorbing Q (lead).",
                           key=f"der_pf_{feeder_name}")

    run_der = st.button("▶ Solve with PV", type="primary",
                        key=f"run_der_{feeder_name}")

    der_key = f"der_result_{feeder_name}"

    if run_der:
        with st.spinner("Re-solving with PV added…"):
            _pkl = _tmp_pkl()
            _ok, _out, _err = _worker_run([
                "der", feeder_cfg["master"], der_bus,
                str(float(der_kw)), str(float(der_pf)), _pkl,
            ])
            if not _ok:
                os.unlink(_pkl)
                st.error(f"Solve error:\n{_err}")
                st.session_state.pop(der_key, None)
            else:
                _dr = _load_pkl(_pkl)
                os.unlink(_pkl)
                st.session_state[der_key] = dict(
                    buses=_dr["buses"], summary=_dr["summary"],
                    bus=der_bus, kw=der_kw, pf=der_pf,
                )

    dr = st.session_state.get(der_key)
    if dr is not None:
        db = dr["buses"]
        ds = dr["summary"]

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

        st.markdown("**ΔV network map** — how much each bus voltage changed")
        dv_vals = np.array([delta_v.get(b, 0.0) for b in bus_list])
        dv_max  = float(max(abs(dv_vals.min()), abs(dv_vals.max()), 0.001))

        nfig = go.Figure()
        _lx, _ly = [], []
        for ln in lines:
            _lx += [ln["x1"], ln["x2"], None]
            _ly += [ln["y1"], ln["y2"], None]
        if _lx:
            nfig.add_trace(go.Scatter(
                x=_lx, y=_ly,
                mode="lines", line=dict(color="#CCCCCC", width=1),
                hoverinfo="skip", showlegend=False,
            ))

        _large = feeder_cfg.get("large_feeder", False)
        _BT = go.Scattergl if _large else go.Scatter
        nfig.add_trace(_BT(
            x=[xp[b] for b in bus_list],
            y=[yp[b] for b in bus_list],
            mode="markers" if _large else "markers+text",
            marker=dict(
                size=4 if _large else 12, color=dv_vals.tolist(),
                colorscale="RdYlGn_r",
                cmin=-dv_max, cmax=dv_max,
                showscale=True,
                colorbar=dict(title="ΔV (pu)", thickness=12, len=0.75, x=1.01),
                line=dict(color="white", width=0.5 if _large else 1),
                opacity=0.8 if _large else 1.0,
            ),
            text=None if _large else bus_list,
            textposition="top center",
            textfont=dict(size=7, color="#333"),
            hovertemplate=[
                f"<b>Bus {b}</b><br>ΔV = {delta_v.get(b,0):+.4f} pu<extra></extra>"
                for b in bus_list
            ],
            showlegend=False,
        ))

        if der_bus in xp:
            nfig.add_trace(go.Scatter(
                x=[xp[der_bus]], y=[yp[der_bus]], mode="markers",
                marker=dict(symbol="star", size=18, color="#F9A825",
                            line=dict(color="black", width=1.5)),
                name=f"PV @ Bus {der_bus}", hoverinfo="skip",
            ))

        nfig.update_layout(
            uirevision=der_key,
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
violation appears.
    """)

    hc1, hc2 = st.columns(2)
    with hc1:
        hc_step = st.select_slider(
            "Sweep step size (kW)", options=[50, 100, 150, 200], value=100,
            key=f"hc_step_{feeder_name}",
        )
    with hc2:
        hc_max = st.select_slider(
            "Maximum PV to test (kW)", options=[500, 1000, 1500, 2000], value=1000,
            key=f"hc_max_{feeder_name}",
        )

    sweep_btn = st.button("📈 Run Hosting Capacity Sweep", type="primary",
                          key=f"hc_sweep_btn_{feeder_name}")

    hc_key = f"hc_results_{feeder_name}"

    if sweep_btn:
        _sample_n = feeder_cfg.get("hc_sample_n")
        _sweep_buses = load_buses
        if _sample_n and len(load_buses) > _sample_n:
            _step = max(1, len(load_buses) // _sample_n)
            _sweep_buses = load_buses[::_step][:_sample_n]
            st.info(
                f"Sampling {len(_sweep_buses)} of {len(load_buses)} primary buses "
                f"to keep the sweep time manageable for this large feeder."
            )
        _pkl = _tmp_pkl()
        _hc_cmd = [
            sys.executable, _WORKER, "hc",
            feeder_cfg["master"], json.dumps(_sweep_buses),
            str(float(hc_step)), str(float(hc_max)), _pkl,
        ]
        prog = st.progress(0.0, text="Running hosting capacity sweep…")
        _proc = subprocess.Popen(_hc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _line in _proc.stdout:
            _line = _line.strip()
            if _line.startswith("PROGRESS:"):
                _v = float(_line.split(":")[1])
                prog.progress(_v, text=f"Sweep {_v*100:.0f}%…")
        _proc.wait()
        prog.empty()
        if _proc.returncode != 0:
            os.unlink(_pkl)
            st.error(f"Sweep error:\n{_proc.stderr.read()}")
        else:
            st.session_state[hc_key] = _load_pkl(_pkl)
            os.unlink(_pkl)

    hcr = st.session_state.get(hc_key)
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
                f"causes a violation elsewhere on the feeder."
            )

        hc_vals = np.array([hcr.get(b, 0.0) for b in bus_list])

        hfig = go.Figure()
        _lx, _ly = [], []
        for ln in lines:
            _lx += [ln["x1"], ln["x2"], None]
            _ly += [ln["y1"], ln["y2"], None]
        if _lx:
            hfig.add_trace(go.Scatter(
                x=_lx, y=_ly,
                mode="lines", line=dict(color="#CCCCCC", width=1.2),
                hoverinfo="skip", showlegend=False,
            ))

        _large = feeder_cfg.get("large_feeder", False)
        _BT = go.Scattergl if _large else go.Scatter
        hfig.add_trace(_BT(
            x=[xp[b] for b in bus_list],
            y=[yp[b] for b in bus_list],
            mode="markers" if _large else "markers+text",
            marker=dict(
                size=4 if _large else 13, color=hc_vals.tolist(),
                colorscale="RdYlGn",
                cmin=0, cmax=float(hc_max),
                showscale=True,
                colorbar=dict(title="Hosting<br>cap (kW)", thickness=12,
                              len=0.75, x=1.01),
                line=dict(color="white", width=0.5 if _large else 1),
                opacity=0.8 if _large else 1.0,
            ),
            text=None if _large else bus_list,
            textposition="top center",
            textfont=dict(size=7, color="#333"),
            hovertemplate=[
                f"<b>Bus {b}</b><br>Hosting capacity = {hcr.get(b,0):.0f} kW<extra></extra>"
                for b in bus_list
            ],
            showlegend=False,
        ))

        hfig.update_layout(
            uirevision=hc_key,
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

        st.markdown("**Hosting capacity by bus**")
        hc_df = pd.DataFrame([
            dict(Bus=b, **{"Hosting Capacity (kW)": hcr[b],
                           "Status": ("Constrained" if hcr[b] == 0 else
                                      f"≥ {hc_max:.0f} kW" if hcr[b] >= hc_max else
                                      f"{hcr[b]:.0f} kW")})
            for b in sorted(hcr, key=lambda x: -hcr[x])
        ])
        st.dataframe(hc_df, hide_index=True, width="stretch")
