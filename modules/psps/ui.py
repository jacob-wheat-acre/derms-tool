"""modules/psps/ui.py — PSPS module UI.

Geographic mode (9500-node, has_latlon=True):
  Tab 1  🗺️ Risk Layers  — tier + fuel overlays on the network map
  Tab 2  ⚡ Optimize     — new switch placement optimizer
  Tab 3  💰 Cost Analysis — sensitivity curve (investment vs. PSPS impact)

Manual mode (123-bus, 13-bus):
  Tab 1  🔥 Risk Zone    — select at-risk buses on the map
  Tab 2  ⚡ Optimize     — open existing switches to isolate the zone
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .optimizer import (
    annotate_graph_with_risk,
    add_virtual_switches,
    build_graph,
    count_service_pts_affected,
    find_new_switch_locations,
    get_at_risk_edges,
    get_candidate_switches,
    get_energized_buses,
    optimize_isolation,
    run_sensitivity_analysis,
)
from .risk_data import (
    TIER_META,
    FUEL_META,
    annotate_buses,
    get_risk_zones_for_plotly,
    get_fuel_zones_for_plotly,
    get_fuel_type_descriptions,
)


def render(feeder_cfg: dict, base: dict, feeder_name: str) -> None:
    has_latlon = feeder_cfg.get("has_latlon", False)
    if has_latlon:
        _render_geo(feeder_cfg, base, feeder_name)
    else:
        _render_manual(feeder_cfg, base, feeder_name)


# ══════════════════════════════════════════════════════════════════════════════
# Geographic mode  (IEEE 9500-Node)
# ══════════════════════════════════════════════════════════════════════════════

def _render_geo(feeder_cfg: dict, base: dict, feeder_name: str) -> None:
    buses            = base["buses"]
    lines            = base["lines"]
    loads            = base["loads"]
    switches         = base.get("switches", [])
    topology_lines   = base.get("topology_lines", [])
    load_xfmr_lines  = base.get("load_xfmr_lines", [])
    sub_transformers = base.get("sub_transformers", [])
    source_bus       = feeder_cfg["source_bus"].lower()

    service_buses = set(base.get("load_buses", []))
    total_svc_pts = len(service_buses)

    with st.sidebar:
        st.subheader("What is PSPS?")
        st.markdown("""
**Public Safety Power Shutoff** — de-energizes distribution lines during extreme
fire weather to prevent wildfire ignition.

**This module** uses synthetic wildfire risk data for the Kennewick, WA area
(the geographic location of the IEEE 9500-Node feeder) to demonstrate
optimal isolation point planning.
        """)

    if not topology_lines:
        st.warning("Topology data unavailable — reload the feeder.")
        return

    # Build annotated graph
    G = build_graph(topology_lines, loads, buses)
    bus_risk = annotate_buses(buses)
    annotate_graph_with_risk(G, bus_risk)
    candidates = get_candidate_switches(G)

    # Precompute per-tier bus lists (for display)
    buses_by_tier: dict[int, list[str]] = {0: [], 1: [], 2: [], 3: []}
    for b, info in bus_risk.items():
        buses_by_tier[info["tier"]].append(b)

    tab_layers, tab_opt, tab_cost = st.tabs([
        "🗺️ Risk Layers", "⚡ Optimize", "💰 Cost Analysis",
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 1 — Risk Layers
    # ──────────────────────────────────────────────────────────────────────────
    with tab_layers:
        ctrl_col, map_col = st.columns([1, 3])

        with ctrl_col:
            st.subheader("Layer Controls")

            show_tiers = st.multiselect(
                "Wildfire Risk Zones",
                options=[1, 2, 3],
                default=[1, 2, 3],
                format_func=lambda t: TIER_META[t]["label"],
                key=f"show_tiers_{feeder_name}",
            )
            show_fuels = st.checkbox("Wildfire Fuels layer", value=False,
                                     key=f"show_fuels_{feeder_name}")
            show_switches = st.checkbox("Existing switches", value=True,
                                        key=f"show_sw_{feeder_name}")

            st.divider()
            st.markdown("**Service points by zone**")

            svc_by_tier = {t: 0 for t in range(4)}
            for b in service_buses:
                svc_by_tier[bus_risk.get(b, {}).get("tier", 0)] += 1

            for t in [3, 2, 1, 0]:
                meta  = TIER_META[t]
                count = svc_by_tier[t]
                pct   = count / max(total_svc_pts, 1) * 100
                st.markdown(
                    f"<span style='color:{meta['color']};font-weight:600'>"
                    f"{meta['label']}</span>: {count} ({pct:.0f}%)",
                    unsafe_allow_html=True,
                )

        with map_col:
            fig = _build_base_map(buses, lines, feeder_cfg, load_xfmr_lines, sub_transformers)

            risk_zones  = get_risk_zones_for_plotly()
            fuel_zones  = get_fuel_zones_for_plotly()

            # Fuel zones (drawn first, underneath risk zones)
            if show_fuels:
                seen_fuel_labels: set[str] = set()
                for fz in fuel_zones:
                    meta = FUEL_META[fz["type"]]
                    show_legend = fz["type"] not in seen_fuel_labels
                    seen_fuel_labels.add(fz["type"])
                    fig.add_trace(go.Scatter(
                        x=fz["lons"], y=fz["lats"],
                        fill="toself",
                        fillcolor=meta["fill"],
                        line=dict(color=meta["color"], width=1, dash="dot"),
                        name=fz["type"],
                        showlegend=show_legend,
                        legendgroup=f"fuel_{fz['type']}",
                        hovertemplate=f"<b>{fz['type']}</b><extra></extra>",
                        mode="lines",
                    ))

            # Risk zone polygons
            seen_tier_labels: set[int] = set()
            for rz in risk_zones:
                if rz["tier"] not in show_tiers:
                    continue
                show_legend = rz["tier"] not in seen_tier_labels
                seen_tier_labels.add(rz["tier"])
                fig.add_trace(go.Scatter(
                    x=rz["lons"], y=rz["lats"],
                    fill="toself",
                    fillcolor=rz["fill_color"],
                    line=dict(color=rz["line_color"], width=1.5),
                    name=rz["label"],
                    showlegend=show_legend,
                    legendgroup=f"tier_{rz['tier']}",
                    hovertemplate=f"<b>{rz['name']}</b><br>{rz['desc'][:80]}…<extra></extra>",
                    mode="lines",
                ))

            # Switches
            if show_switches:
                sw_closed = [s for s in switches if not s["is_open"]]
                if sw_closed:
                    fig.add_trace(go.Scatter(
                        x=[s["x"] for s in sw_closed],
                        y=[s["y"] for s in sw_closed],
                        mode="markers",
                        marker=dict(symbol="square", size=5, color="#1B5E20",
                                    line=dict(color="white", width=0.5)),
                        name="Switch (closed)",
                        hovertemplate=[
                            f"<b>Switch {s['name']}</b><extra></extra>" for s in sw_closed
                        ],
                    ))

            fig.update_layout(height=620)
            st.plotly_chart(fig, use_container_width=True,
                            config={"scrollZoom": True, "displayModeBar": True})

        # Zone descriptions
        with st.expander("Risk zone descriptions"):
            for rz in risk_zones:
                meta = TIER_META[rz["tier"]]
                st.markdown(
                    f"**{rz['name']}** "
                    f"<span style='color:{meta['color']}'>{meta['label']}</span> — "
                    f"{rz['desc']}",
                    unsafe_allow_html=True,
                )
        with st.expander("Fuel type descriptions"):
            for ftype, desc in get_fuel_type_descriptions().items():
                st.markdown(f"**{ftype}** — {desc}")

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 2 — Optimize
    # ──────────────────────────────────────────────────────────────────────────
    with tab_opt:
        oc1, oc2 = st.columns([1, 2])

        with oc1:
            st.subheader("Parameters")
            psps_tier = st.radio(
                "PSPS trigger threshold",
                options=[3, 2],
                format_func=lambda t: (
                    "Tier 3 only (High risk)"  if t == 3 else
                    "Tier 2 + 3 (Mod. + High)"
                ),
                key=f"psps_tier_{feeder_name}",
            )
            budget_n = st.slider(
                "New switches to add", 0, 20, 3,
                key=f"budget_n_{feeder_name}",
            )

            at_risk_buses = {
                b for b in G.nodes()
                if G.nodes[b].get("risk_tier", 0) >= psps_tier
            }
            risk_edges = get_at_risk_edges(G, min_tier=psps_tier)

            n_at_risk_svc = sum(1 for b in service_buses if b in at_risk_buses)
            n_risk_edges  = len(risk_edges)

            st.divider()
            st.metric("At-risk service points", n_at_risk_svc,
                      delta=f"{n_at_risk_svc/max(total_svc_pts,1)*100:.1f}% of total")
            st.metric("At-risk line segments",  n_risk_edges)
            st.metric("Existing candidate switches", len(candidates))

            run_btn = st.button("🔥 Run Optimizer", type="primary",
                                key=f"run_opt_{feeder_name}")

        opt_key = f"psps_geo_result_{feeder_name}_{psps_tier}_{budget_n}"

        if run_btn:
            with st.spinner("Finding new switch locations and optimal isolation plan…"):
                new_locs   = find_new_switch_locations(
                    G, source_bus, at_risk_buses, service_buses,
                    budget_n=budget_n, risk_edges=risk_edges,
                )
                G_aug      = add_virtual_switches(G, new_locs)
                sw_aug     = get_candidate_switches(G_aug)
                result     = optimize_isolation(G_aug, source_bus, risk_edges, sw_aug)
                baseline   = optimize_isolation(G,     source_bus, risk_edges, candidates)
                svc_base   = count_service_pts_affected(baseline, service_buses, at_risk_buses)
                svc_opt    = count_service_pts_affected(result,   service_buses, at_risk_buses)
                st.session_state[opt_key] = dict(
                    result=result, baseline=baseline,
                    svc_base=svc_base, svc_opt=svc_opt,
                    new_locs=new_locs, psps_tier=psps_tier, budget_n=budget_n,
                )

        saved = st.session_state.get(opt_key)
        if saved is None:
            with oc2:
                st.info("Set parameters and press **Run Optimizer**.")
            return

        result   = saved["result"]
        baseline = saved["baseline"]
        svc_base = saved["svc_base"]
        svc_opt  = saved["svc_opt"]
        new_locs = saved["new_locs"]

        with oc2:
            st.subheader("Results")

            kept_on_base = total_svc_pts - svc_base["total_affected"]
            kept_on_opt  = total_svc_pts - svc_opt["total_affected"]
            gained       = kept_on_opt - kept_on_base

            if gained > 0:
                st.success(
                    f"**{gained:,} additional customers can stay on** with "
                    f"{len(new_locs)} new switch(es) installed — "
                    f"{kept_on_opt:,} of {total_svc_pts:,} customers remain energized."
                )
            else:
                st.info(
                    f"{kept_on_opt:,} of {total_svc_pts:,} customers remain energized "
                    f"during the PSPS event."
                )

            # Impact comparison
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Customers kept on — baseline",  kept_on_base,
                       help="Customers remaining energized with existing switches only")
            mc2.metric("Customers kept on — optimized", kept_on_opt,
                       delta=f"{gained:+,}",
                       delta_color="normal",
                       help="Customers remaining energized after adding recommended switches")
            mc3.metric("Collateral baseline",  svc_base["collateral"])
            mc4.metric("Collateral optimized", svc_opt["collateral"],
                       delta=f"{svc_opt['collateral']-svc_base['collateral']:+d}",
                       delta_color="inverse")

            if new_locs:
                st.markdown(f"**{len(new_locs)} recommended new switch location(s):**")
                sw_df = pd.DataFrame([
                    {
                        "Location (edge)":       c["edge_name"],
                        "At-risk pts isolated":  c["at_risk_isolated"],
                        "Safe collateral":        c["safe_collateral"],
                    }
                    for c in new_locs
                ])
                st.dataframe(sw_df, hide_index=True)

            ops = result.get("switches_to_open", [])
            if ops:
                st.markdown(f"**{len(ops)} switch operation(s) to execute:**")
                ops_df = pd.DataFrame([{
                    "Switch": s,
                    "Type": "New (recommended)" if s.startswith("vsw:") else "Existing",
                    "Action": "OPEN",
                } for s in ops])
                st.dataframe(ops_df, hide_index=True)

        # Result map
        rfig = _build_result_map(
            buses, lines, feeder_cfg, bus_risk,
            result, new_locs, switches, psps_tier,
            load_xfmr_lines=load_xfmr_lines,
            sub_transformers=sub_transformers,
        )
        st.plotly_chart(rfig, use_container_width=True,
                        config={"scrollZoom": True, "displayModeBar": True})

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 3 — Cost Analysis
    # ──────────────────────────────────────────────────────────────────────────
    with tab_cost:
        cc1, cc2 = st.columns([1, 2])

        with cc1:
            st.subheader("Parameters")
            cost_per_sw = st.number_input(
                "Cost per new switch ($)",
                min_value=10_000, max_value=500_000,
                value=75_000, step=5_000,
                key=f"cost_sw_{feeder_name}",
            )
            max_sw = st.slider("Max new switches to evaluate", 5, 20, 10,
                               key=f"max_sw_{feeder_name}")
            sens_tier = st.radio(
                "PSPS tier threshold",
                options=[3, 2],
                format_func=lambda t: "Tier 3 only" if t == 3 else "Tier 2 + 3",
                key=f"sens_tier_{feeder_name}",
            )
            run_sens = st.button("📈 Run Sensitivity Analysis", type="primary",
                                 key=f"run_sens_{feeder_name}")

        sens_key = f"psps_sens_{feeder_name}_{sens_tier}_{max_sw}"

        if run_sens:
            sens_at_risk = {
                b for b in G.nodes()
                if G.nodes[b].get("risk_tier", 0) >= sens_tier
            }
            sens_edges = get_at_risk_edges(G, min_tier=sens_tier)
            with st.spinner("Running sensitivity analysis (may take ~30 s)…"):
                rows = run_sensitivity_analysis(
                    G, source_bus, sens_at_risk, service_buses, sens_edges,
                    max_new_switches=max_sw,
                    cost_per_switch=cost_per_sw,
                )
                st.session_state[sens_key] = rows

        rows = st.session_state.get(sens_key)
        if rows is None:
            with cc2:
                st.info("Press **Run Sensitivity Analysis** to compute the investment curve.")
            return

        with cc2:
            st.subheader("Collateral PSPS Impact vs. Investment")

            # Plotly line chart
            sfig = go.Figure()
            sfig.add_trace(go.Scatter(
                x=[r["cumulative_cost"] / 1_000 for r in rows],
                y=[r["collateral"] for r in rows],
                mode="lines+markers",
                name="Collateral service pts",
                line=dict(color="#B71C1C", width=2),
                marker=dict(size=8),
                hovertemplate=(
                    "$%{x:.0f}k investment<br>"
                    "%{y} collateral service points<extra></extra>"
                ),
            ))
            sfig.add_trace(go.Scatter(
                x=[r["cumulative_cost"] / 1_000 for r in rows],
                y=[r["collateral_reduction"] for r in rows],
                mode="lines+markers",
                name="Collateral pts saved vs. baseline",
                line=dict(color="#2E7D32", width=2, dash="dash"),
                marker=dict(size=8),
                hovertemplate=(
                    "$%{x:.0f}k investment<br>"
                    "%{y} pts saved vs. no-switch baseline<extra></extra>"
                ),
            ))
            sfig.update_layout(
                xaxis_title="Cumulative investment ($k)",
                yaxis_title="Service points",
                plot_bgcolor="white", paper_bgcolor="white",
                height=380, margin=dict(l=10, r=10, t=30, b=50),
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(sfig, use_container_width=True)

            # Table
            tbl = pd.DataFrame([{
                "New switches": r["n_new"],
                "Investment":   f"${r['cumulative_cost']/1000:.0f}k",
                "Total affected": r["total_affected"],
                "Collateral":   r["collateral"],
                "Saved vs. baseline": r["collateral_reduction"],
                "% reduction":  f"{r['pct_reduction']:.0f}%",
            } for r in rows])
            st.dataframe(tbl, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Manual mode  (IEEE 123-Bus, IEEE 13-Bus)
# ══════════════════════════════════════════════════════════════════════════════

def _render_manual(feeder_cfg: dict, base: dict, feeder_name: str) -> None:
    buses          = base["buses"]
    lines          = base["lines"]
    loads          = base["loads"]
    switches       = base.get("switches", [])
    topology_lines = base.get("topology_lines", [])
    source_bus     = feeder_cfg["source_bus"].lower()

    total_load_kw = sum(ld["kw"] for ld in loads)

    with st.sidebar:
        st.subheader("What is PSPS?")
        st.markdown("""
**Public Safety Power Shutoff** de-energizes distribution lines during extreme fire weather
to prevent wildfires ignited by electrical equipment.

Select at-risk buses on the map, then run the optimizer to find the minimum-impact
switching plan using existing switches.
        """)

    if not topology_lines:
        st.warning("Topology data unavailable — reload the feeder.")
        return

    G          = build_graph(topology_lines, loads, buses)
    candidates = get_candidate_switches(G)

    tab_zone, tab_opt = st.tabs(["🔥 Risk Zone", "⚡ Optimize"])

    # ── Tab 1 — Risk Zone ─────────────────────────────────────────────────────
    with tab_zone:
        col_map, col_ctrl = st.columns([3, 1])

        with col_ctrl:
            st.subheader("Define Risk Zone")

            bus_xs = [d["x"] for d in buses.values()]
            bus_ys = [d["y"] for d in buses.values()]
            x_mid  = (min(bus_xs) + max(bus_xs)) / 2
            y_mid  = (min(bus_ys) + max(bus_ys)) / 2

            predefined_zones: dict[str, list[str]] = {
                "None — no zone selected":  [],
                "Northern half":            [b for b, d in buses.items() if d["y"] > y_mid],
                "Southern half":            [b for b, d in buses.items() if d["y"] <= y_mid],
                "Eastern half":             [b for b, d in buses.items() if d["x"] > x_mid],
                "Western half":             [b for b, d in buses.items() if d["x"] <= x_mid],
            }

            scenario = st.selectbox(
                "Predefined fire zone",
                list(predefined_zones.keys()),
                key=f"psps_zone_{feeder_name}",
            )
            preset_buses = set(predefined_zones[scenario])
            extra_buses  = st.multiselect(
                "Add individual buses",
                options=sorted(buses.keys()),
                default=[],
                key=f"psps_extra_{feeder_name}",
            )
            risk_buses = preset_buses | set(extra_buses)

            if risk_buses:
                st.success(f"{len(risk_buses)} buses in risk zone")
            else:
                st.info("Select a zone or add buses above.")

        risk_edge_set: set[tuple[str, str]] = set()
        for ln in topology_lines:
            b1, b2 = ln["b1"], ln["b2"]
            if b1 in risk_buses and b2 in risk_buses and not ln.get("is_switch"):
                risk_edge_set.add((min(b1, b2), max(b1, b2)))

        st.session_state[f"psps_risk_buses_{feeder_name}"]  = risk_buses
        st.session_state[f"psps_risk_edges_{feeder_name}"]  = list(risk_edge_set)

        with col_map:
            fig = _build_base_map(buses, lines, feeder_cfg)

            risk_lx: list = []
            risk_ly: list = []
            for ln in lines:
                if ln["b1"] in risk_buses and ln["b2"] in risk_buses:
                    risk_lx += [ln["x1"], ln["x2"], None]
                    risk_ly += [ln["y1"], ln["y2"], None]
            if risk_lx:
                fig.add_trace(go.Scatter(
                    x=risk_lx, y=risk_ly, mode="lines",
                    line=dict(color="#B71C1C", width=3),
                    name=f"At-risk segments ({len(risk_edge_set)})",
                    hoverinfo="skip",
                ))

            risk_list = [b for b in buses if b in risk_buses]
            if risk_list:
                fig.add_trace(go.Scatter(
                    x=[buses[b]["x"] for b in risk_list],
                    y=[buses[b]["y"] for b in risk_list],
                    mode="markers",
                    marker=dict(size=11, color="#B71C1C",
                                line=dict(color="white", width=1.5)),
                    name="At-risk buses",
                    hovertemplate=[
                        f"<b>Bus {b}</b><br>AT RISK<extra></extra>" for b in risk_list
                    ],
                ))

            sw_closed = [s for s in switches if not s["is_open"]]
            if sw_closed:
                fig.add_trace(go.Scatter(
                    x=[s["x"] for s in sw_closed],
                    y=[s["y"] for s in sw_closed],
                    mode="markers",
                    marker=dict(symbol="square", size=11, color="#1B5E20",
                                line=dict(color="white", width=1.5)),
                    name="Switch (closed — candidate)",
                    hovertemplate=[
                        f"<b>Switch {s['name']}</b><extra></extra>" for s in sw_closed
                    ],
                ))

            fig.update_layout(
                uirevision=f"manual_zone_{feeder_name}_{scenario}",
                height=580,
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"scrollZoom": True, "displayModeBar": True})

    # ── Tab 2 — Optimize ──────────────────────────────────────────────────────
    with tab_opt:
        risk_buses_opt  = st.session_state.get(f"psps_risk_buses_{feeder_name}", set())
        risk_edge_pairs = st.session_state.get(f"psps_risk_edges_{feeder_name}", [])
        risk_edges      = [(b1, b2) for b1, b2 in risk_edge_pairs]

        if not risk_buses_opt:
            st.info("No risk zone defined — go to the **Risk Zone** tab first.")
            return

        o1, o2 = st.columns(2)
        with o1:
            st.subheader("Candidate Switches")
            sw_lookup = {s["name"]: s for s in switches}
            st.dataframe(
                pd.DataFrame([{
                    "Switch": name,
                    "Phases": sw_lookup[name]["nph"] if name in sw_lookup else "—",
                    "State":  "Closed (can open)",
                } for name in candidates]),
                hide_index=True,
            )
            if not candidates:
                st.error("No normally-closed switches found.")
            if not risk_edges:
                st.warning("No segments in risk zone — expand zone or add buses.")

        with o2:
            st.subheader("Risk Zone Summary")
            st.metric("At-risk buses",    len(risk_buses_opt))
            st.metric("At-risk segments", len(risk_edges))
            st.metric("Total feeder load", f"{total_load_kw:.0f} kW")

        run_btn = st.button(
            "🔥 Run PSPS Optimizer", type="primary",
            disabled=(not candidates or not risk_edges),
            key=f"run_manual_{feeder_name}",
        )
        opt_key = f"psps_manual_result_{feeder_name}"

        if run_btn:
            with st.spinner("Optimizing switching plan…"):
                result = optimize_isolation(G, source_bus, risk_edges, candidates)
                st.session_state[opt_key] = result

        result = st.session_state.get(opt_key)
        if result is None:
            st.info("Press **Run PSPS Optimizer** to find the optimal switching plan.")
            return

        st.divider()
        shed   = result["load_shed_kw"]
        served = total_load_kw - shed
        pct_shed = shed / total_load_kw * 100 if total_load_kw else 0
        n_de = len([b for b in result["de_energized_buses"] if b in buses])

        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("Switches opened", len(result["switches_to_open"]))
        rm2.metric("Load de-energized", f"{shed:.0f} kW",
                   delta=f"{pct_shed:.1f}% of total", delta_color="inverse")
        rm3.metric("Load served",      f"{served:.0f} kW",
                   delta=f"{100-pct_shed:.1f}% of total", delta_color="normal")
        rm4.metric("Buses de-energized", str(n_de))

        if result["switches_to_open"]:
            st.dataframe(
                pd.DataFrame([{"Switch": s, "Action": "OPEN"}
                              for s in result["switches_to_open"]]),
                hide_index=True,
            )
        if result.get("note"):
            st.warning(result["note"])

        # Result map
        rfig = go.Figure()
        energized_set = set(result["energized_buses"])
        de_set        = set(result["de_energized_buses"])

        en_lx, en_ly, de_lx, de_ly, rk_lx, rk_ly = [], [], [], [], [], []
        for ln in lines:
            b1, b2 = ln["b1"], ln["b2"]
            if b1 in risk_buses_opt and b2 in risk_buses_opt:
                rk_lx += [ln["x1"], ln["x2"], None]
                rk_ly += [ln["y1"], ln["y2"], None]
            elif b1 in energized_set and b2 in energized_set:
                en_lx += [ln["x1"], ln["x2"], None]
                en_ly += [ln["y1"], ln["y2"], None]
            else:
                de_lx += [ln["x1"], ln["x2"], None]
                de_ly += [ln["y1"], ln["y2"], None]

        for lx, ly, clr, w in [(en_lx, en_ly, "#2E7D32", 2), (de_lx, de_ly, "#CFD8DC", 1), (rk_lx, rk_ly, "#B71C1C", 3)]:
            if lx:
                rfig.add_trace(go.Scatter(x=lx, y=ly, mode="lines",
                                          line=dict(color=clr, width=w),
                                          hoverinfo="skip", showlegend=False))

        for b_list, label, clr, size in [
            ([b for b in energized_set if b in buses and b not in risk_buses_opt],
             "Energized", "#2E7D32", 8),
            ([b for b in de_set if b in buses and b not in risk_buses_opt],
             "De-energized", "#78909C", 8),
            ([b for b in risk_buses_opt if b in buses],
             "At-risk (isolated)", "#B71C1C", 10),
        ]:
            if b_list:
                rfig.add_trace(go.Scatter(
                    x=[buses[b]["x"] for b in b_list],
                    y=[buses[b]["y"] for b in b_list],
                    mode="markers",
                    marker=dict(size=size, color=clr, line=dict(color="white", width=1)),
                    name=label,
                    hovertemplate=[f"<b>Bus {b}</b><extra></extra>" for b in b_list],
                ))

        for sw_name in result["switches_to_open"]:
            sw = sw_lookup.get(sw_name)
            if sw:
                rfig.add_trace(go.Scatter(
                    x=[sw["x"]], y=[sw["y"]], mode="markers",
                    marker=dict(symbol="x", size=18, color="#E65100",
                                line=dict(color="#E65100", width=3)),
                    name=f"Open: {sw_name}",
                    hovertemplate=f"<b>{sw_name}</b><br>OPENED<extra></extra>",
                ))

        src = feeder_cfg["source_bus"]
        if src in buses:
            rfig.add_trace(go.Scatter(
                x=[buses[src]["x"]], y=[buses[src]["y"]], mode="markers",
                marker=dict(symbol="star", size=18, color="#FFD600",
                            line=dict(color="#333", width=1.5)),
                name="Feeder head",
                hovertemplate=f"<b>Bus {src}</b><extra></extra>",
            ))

        xs = [d["x"] for d in buses.values()]
        ys = [d["y"] for d in buses.values()]
        pad_x = (max(xs) - min(xs)) * 0.05
        pad_y = (max(ys) - min(ys)) * 0.05
        rfig.update_layout(
            uirevision=opt_key,
            xaxis=dict(visible=False, range=[min(xs)-pad_x, max(xs)+pad_x]),
            yaxis=dict(visible=False, range=[min(ys)-pad_y, max(ys)+pad_y],
                       scaleanchor="x", scaleratio=1),
            plot_bgcolor="white", paper_bgcolor="white",
            height=580, margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.85)"),
            title=dict(
                text="Green = energized  |  Gray = de-energized  |  Red = at-risk  |  ✕ = opened",
                font=dict(size=11), x=0.5, xanchor="center",
            ),
        )
        st.plotly_chart(rfig, use_container_width=True,
                        config={"scrollZoom": True, "displayModeBar": True})


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_base_map(
    buses: dict,
    lines: list[dict],
    feeder_cfg: dict,
    load_xfmr_lines: list[dict] | None = None,
    sub_transformers: list[dict] | None = None,
) -> go.Figure:
    """Feeder network base layer (lines + buses + feeder head star)."""
    fig = go.Figure()

    lx, ly = [], []
    for ln in lines:
        lx += [ln["x1"], ln["x2"], None]
        ly += [ln["y1"], ln["y2"], None]
    if lx:
        fig.add_trace(go.Scatter(
            x=lx, y=ly, mode="lines",
            line=dict(color="#B0BEC5", width=0.8),
            hoverinfo="skip", showlegend=False,
        ))

    bus_list = list(buses.keys())
    _large   = feeder_cfg.get("large_feeder", False)
    BusTrace = go.Scattergl if _large else go.Scatter

    if load_xfmr_lines:
        XfmrTrace = go.Scattergl if _large else go.Scatter
        fig.add_trace(XfmrTrace(
            x=[(ln["x1"] + ln["x2"]) / 2 for ln in load_xfmr_lines],
            y=[(ln["y1"] + ln["y2"]) / 2 for ln in load_xfmr_lines],
            mode="markers",
            marker=dict(symbol="triangle-up", size=9 if _large else 13,
                        color="#546E7A", line=dict(color="white", width=0.5)),
            name="Distribution transformer",
            hovertemplate="<b>Distribution transformer</b><extra></extra>",
        ))

    if sub_transformers:
        fig.add_trace(go.Scatter(
            x=[t["x"] for t in sub_transformers],
            y=[t["y"] for t in sub_transformers],
            mode="markers",
            marker=dict(symbol="square", size=18,
                        color="#6A1B9A",
                        line=dict(color="white", width=2)),
            name="Substation transformer (T&D)",
            hovertemplate=[
                f"<b>Substation: {t['name']}</b><br>"
                f"{t['kva']/1000:.1f} MVA · {t['kv1']:.1f} kV → {t['kv2']:.1f} kV"
                "<extra></extra>"
                for t in sub_transformers
            ],
        ))

    fig.add_trace(BusTrace(
        x=[buses[b]["x"] for b in bus_list],
        y=[buses[b]["y"] for b in bus_list],
        mode="markers",
        marker=dict(size=3 if _large else 7, color="#78909C",
                    line=dict(color="white", width=0.3 if _large else 0.8),
                    opacity=0.6 if _large else 1.0),
        hovertemplate=[f"<b>{b}</b><extra></extra>" for b in bus_list],
        showlegend=False,
    ))

    src = feeder_cfg["source_bus"]
    if src in buses:
        fig.add_trace(go.Scatter(
            x=[buses[src]["x"]], y=[buses[src]["y"]],
            mode="markers",
            marker=dict(symbol="star", size=16, color="#FFD600",
                        line=dict(color="#333", width=1.5)),
            name=f"Source (Bus {src})",
            hovertemplate=f"<b>Bus {src}</b> (feeder head)<extra></extra>",
        ))

    xs = [d["x"] for d in buses.values()]
    ys = [d["y"] for d in buses.values()]
    pad_x = (max(xs) - min(xs)) * 0.06
    pad_y = (max(ys) - min(ys)) * 0.06

    fig.update_layout(
        xaxis=dict(visible=False, range=[min(xs)-pad_x, max(xs)+pad_x]),
        yaxis=dict(visible=False, range=[min(ys)-pad_y, max(ys)+pad_y],
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.85)",
                    itemsizing="constant"),
    )
    return fig


def _build_result_map(
    buses: dict,
    lines: list[dict],
    feeder_cfg: dict,
    bus_risk: dict[str, dict],
    result: dict,
    new_locs: list[dict],
    switches: list[dict],
    psps_tier: int,
    load_xfmr_lines: list[dict] | None = None,
    sub_transformers: list[dict] | None = None,
) -> go.Figure:
    """Result overlay map for the geographic PSPS optimizer."""
    fig = go.Figure()

    # ── Risk zone polygons (bottom layer) ──────────────────────────────────────
    risk_zones = get_risk_zones_for_plotly()
    seen_tier_labels: set[int] = set()
    for rz in risk_zones:
        if rz["tier"] < psps_tier:
            continue
        show_legend = rz["tier"] not in seen_tier_labels
        seen_tier_labels.add(rz["tier"])
        fig.add_trace(go.Scatter(
            x=rz["lons"], y=rz["lats"],
            fill="toself",
            fillcolor=rz["fill_color"],
            line=dict(color=rz["line_color"], width=1.0),
            name=rz["label"],
            showlegend=show_legend,
            legendgroup=f"tier_{rz['tier']}",
            hovertemplate=f"<b>{rz['name']}</b><extra></extra>",
            mode="lines",
        ))

    # ── Base feeder map — WITHOUT transformer markers (we add them colored) ────
    base_fig = _build_base_map(buses, lines, feeder_cfg, None)
    for trace in base_fig.data:
        fig.add_trace(trace)
    fig.update_layout(base_fig.layout)

    energized_set = set(result["energized_buses"])
    de_set        = set(result["de_energized_buses"])
    at_risk_set   = {b for b, info in bus_risk.items() if info["tier"] >= psps_tier}

    # ── Colored conductors ─────────────────────────────────────────────────────
    en_lx, en_ly, de_lx, de_ly, rk_lx, rk_ly = [], [], [], [], [], []
    for ln in lines:
        b1, b2 = ln["b1"], ln["b2"]
        in_risk = (b1 in at_risk_set or b2 in at_risk_set)
        both_en = b1 in energized_set and b2 in energized_set
        if in_risk and not both_en:
            rk_lx += [ln["x1"], ln["x2"], None]
            rk_ly += [ln["y1"], ln["y2"], None]
        elif both_en and not in_risk:
            en_lx += [ln["x1"], ln["x2"], None]
            en_ly += [ln["y1"], ln["y2"], None]
        elif not both_en:
            de_lx += [ln["x1"], ln["x2"], None]
            de_ly += [ln["y1"], ln["y2"], None]

    for lx, ly, clr, w in [
        (en_lx, en_ly, "#2E7D32", 2.0),
        (de_lx, de_ly, "#78909C", 0.8),
        (rk_lx, rk_ly, "#B71C1C", 2.5),
    ]:
        if lx:
            fig.add_trace(go.Scattergl(x=lx, y=ly, mode="lines",
                                       line=dict(color=clr, width=w),
                                       hoverinfo="skip", showlegend=False))

    # ── Colored transformer markers ────────────────────────────────────────────
    if load_xfmr_lines:
        xfmr_en, xfmr_de, xfmr_rk = [], [], []
        for ln in load_xfmr_lines:
            b1 = ln["b1"]
            mx = (ln["x1"] + ln["x2"]) / 2
            my = (ln["y1"] + ln["y2"]) / 2
            if b1 in at_risk_set:
                xfmr_rk.append((mx, my, b1))
            elif b1 in energized_set:
                xfmr_en.append((mx, my, b1))
            else:
                xfmr_de.append((mx, my, b1))

        for pts, clr, label in [
            (xfmr_en, "#2E7D32", "Transformer — energized"),
            (xfmr_de, "#78909C", "Transformer — de-energized"),
            (xfmr_rk, "#B71C1C", "Transformer — at-risk"),
        ]:
            if pts:
                fig.add_trace(go.Scattergl(
                    x=[p[0] for p in pts],
                    y=[p[1] for p in pts],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=9,
                                color=clr,
                                line=dict(color="white", width=0.5)),
                    name=label,
                    hovertemplate=[
                        f"<b>Transformer at {p[2]}</b><extra></extra>" for p in pts
                    ],
                ))

    # ── Substation (T&D) transformer markers — large squares, colored ─────────
    if sub_transformers:
        sub_groups: dict[str, list] = {"en": [], "de": [], "rk": []}
        for t in sub_transformers:
            b = t.get("bus", "")
            if b in at_risk_set:
                sub_groups["rk"].append(t)
            elif b in energized_set:
                sub_groups["en"].append(t)
            else:
                sub_groups["de"].append(t)

        for key, clr, label in [
            ("en", "#1B5E20", "Substation xfmr — energized"),
            ("de", "#37474F", "Substation xfmr — de-energized"),
            ("rk", "#B71C1C", "Substation xfmr — at-risk"),
        ]:
            pts = sub_groups[key]
            if pts:
                fig.add_trace(go.Scatter(
                    x=[t["x"] for t in pts],
                    y=[t["y"] for t in pts],
                    mode="markers",
                    marker=dict(symbol="square", size=20, color=clr,
                                line=dict(color="white", width=2.5)),
                    name=label,
                    hovertemplate=[
                        f"<b>Substation: {t['name']}</b><br>"
                        f"{t['kva']/1000:.1f} MVA · {t['kv1']:.1f} kV → {t['kv2']:.1f} kV"
                        "<extra></extra>"
                        for t in pts
                    ],
                ))

    # ── New switch markers — placed at safe-side bus (edge_u), large gold star ──
    if new_locs:
        # Resolve marker coordinates: prefer edge_u (safe side of the cut),
        # fall back to midpoint, then edge_v if only one end has coords.
        def _sw_xy(c: dict) -> tuple[float, float] | None:
            eu, ev = c.get("edge_u", ""), c["edge_v"]
            has_u = eu in buses
            has_v = ev in buses
            if has_u and has_v:
                return (buses[eu]["x"] + buses[ev]["x"]) / 2, (buses[eu]["y"] + buses[ev]["y"]) / 2
            if has_u:
                return buses[eu]["x"], buses[eu]["y"]
            if has_v:
                return buses[ev]["x"], buses[ev]["y"]
            return None

        valid = [(c, _sw_xy(c)) for c in new_locs if _sw_xy(c) is not None]
        if valid:
            xs = [xy[0] for _, xy in valid]
            ys = [xy[1] for _, xy in valid]
            # Outer halo for contrast
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(symbol="star", size=32, color="#1A237E",
                            line=dict(color="#1A237E", width=0)),
                showlegend=False, hoverinfo="skip",
            ))
            # Inner bright star
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(symbol="star", size=26, color="#AEEA00",
                            line=dict(color="#212121", width=1.5)),
                name="★ New switch (recommended)",
                hovertemplate=[
                    f"<b>New switch</b><br>{c['edge_name']}<br>"
                    f"{c['benefit_service_pts']:,} customers saved<extra></extra>"
                    for c, _ in valid
                ],
            ))

    # ── Opened switch markers ──────────────────────────────────────────────────
    sw_lookup = {s["name"]: s for s in switches}
    seen_opened = False
    for sw_name in result.get("switches_to_open", []):
        sw = sw_lookup.get(sw_name)
        if sw:
            fig.add_trace(go.Scatter(
                x=[sw["x"]], y=[sw["y"]], mode="markers",
                marker=dict(symbol="x", size=16, color="#E65100",
                            line=dict(color="#BF360C", width=3.0)),
                name="✕ Opened switch" if not seen_opened else None,
                showlegend=not seen_opened,
                legendgroup="opened_sw",
                hovertemplate=f"<b>{sw_name}</b><br>OPENED<extra></extra>",
            ))
            seen_opened = True

    fig.update_layout(
        height=620,
        title=dict(
            text=(
                "Green = energized  |  Gray = de-energized  |  "
                "Red = at-risk (isolated)  |  ★ = new switch  |  ✕ = opened"
            ),
            font=dict(size=11), x=0.5, xanchor="center",
        ),
    )
    return fig
