"""modules/psps/optimizer.py — PSPS isolation point optimizer.

Two layers of analysis:
  1. optimize_isolation()        — which EXISTING switches to open (greedy/exhaustive).
  2. find_new_switch_locations() — where to ADD new switches to reduce collateral PSPS impact.
  3. run_sensitivity_analysis()  — marginal benefit curve for a switch investment budget.
"""
from __future__ import annotations
from collections import deque
from itertools import combinations
import networkx as nx

_EXHAUSTIVE_LIMIT = 20
_DIST_KV_MAX      = 15.0   # kV — edges above this are transmission, not distribution


def build_graph(
    topology_lines: list[dict],
    loads: list[dict],
    buses: dict | None = None,
) -> nx.Graph:
    """NetworkX graph from topology_lines + per-bus load kW (for shed calculation).

    When `buses` is supplied, each node is annotated with `kv_base` so the
    new-switch optimizer can restrict recommendations to distribution lines.
    """
    load_kw_by_bus: dict[str, float] = {}
    for ld in loads:
        load_kw_by_bus[ld["bus"]] = load_kw_by_bus.get(ld["bus"], 0.0) + ld["kw"]

    G = nx.Graph()
    for ln in topology_lines:
        b1, b2 = ln["b1"], ln["b2"]
        for b in (b1, b2):
            if b not in G:
                kv = buses[b]["kv"] if buses and b in buses else 0.0
                G.add_node(b, load_kw=load_kw_by_bus.get(b, 0.0), kv_base=kv)
        G.add_edge(
            b1, b2,
            name=ln["name"],
            nph=ln.get("nph", 3),
            is_switch=ln.get("is_switch", False),
            is_open=ln.get("is_open", False),
        )
    return G


def get_candidate_switches(G: nx.Graph) -> list[str]:
    """Names of normally-closed switch elements (can be opened by the optimizer)."""
    return [
        data["name"]
        for _, _, data in G.edges(data=True)
        if data.get("is_switch") and not data.get("is_open", False)
    ]


def get_energized_buses(G: nx.Graph, source_bus: str, extra_open: set[str]) -> set[str]:
    """Buses reachable from source_bus given base-case open switches + extra_open set."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        if data.get("is_open", False):
            continue
        if data.get("name") in extra_open:
            continue
        H.add_edge(u, v)
    if source_bus not in H:
        return set()
    return set(nx.node_connected_component(H, source_bus))


def calc_load_shed_kw(G: nx.Graph, energized_buses: set[str]) -> float:
    de_energized = set(G.nodes()) - energized_buses
    return sum(G.nodes[b].get("load_kw", 0.0) for b in de_energized)


def _risk_isolated(G: nx.Graph, source: str, risk_edges: list[tuple], extra_open: set[str]) -> bool:
    energized = get_energized_buses(G, source, extra_open)
    return not any(b1 in energized and b2 in energized for b1, b2 in risk_edges)


def _exhaustive(
    G: nx.Graph,
    source: str,
    risk_edges: list[tuple],
    candidates: list[str],
) -> dict:
    best_shed = float("inf")
    best_combo: set[str] | None = None

    for r in range(1, len(candidates) + 1):
        for combo in combinations(candidates, r):
            open_set = set(combo)
            if _risk_isolated(G, source, risk_edges, open_set):
                energized = get_energized_buses(G, source, open_set)
                shed = calc_load_shed_kw(G, energized)
                if shed < best_shed:
                    best_shed = shed
                    best_combo = open_set

    if best_combo is None:
        return _infeasible(G, source, set())

    energized = get_energized_buses(G, source, best_combo)
    return _result(G, source, best_combo, energized)


def _greedy(
    G: nx.Graph,
    source: str,
    risk_edges: list[tuple],
    candidates: list[str],
) -> dict:
    open_set: set[str] = set()
    remaining = list(risk_edges)

    while remaining:
        best_sw: str | None = None
        best_shed = float("inf")

        for sw in candidates:
            if sw in open_set:
                continue
            trial = open_set | {sw}
            energized = get_energized_buses(G, source, trial)
            newly = sum(
                1 for b1, b2 in remaining
                if not (b1 in energized and b2 in energized)
            )
            if newly == 0:
                continue
            shed = calc_load_shed_kw(G, energized)
            # Minimise load shed — keep the most customers on.
            # Any switch that makes forward progress is eligible;
            # prefer the one with the smallest collateral impact.
            if shed < best_shed:
                best_sw = sw
                best_shed = shed

        if best_sw is None:
            break
        open_set.add(best_sw)
        energized = get_energized_buses(G, source, open_set)
        remaining = [
            (b1, b2) for b1, b2 in remaining
            if b1 in energized and b2 in energized
        ]

    if remaining:
        return _infeasible(G, source, open_set, note=f"{len(remaining)} risk segments could not be isolated with existing switches")

    energized = get_energized_buses(G, source, open_set)
    return _result(G, source, open_set, energized)


def _result(G: nx.Graph, source: str, open_set: set[str], energized: set[str]) -> dict:
    de_energized = set(G.nodes()) - energized
    return dict(
        feasible=True,
        switches_to_open=sorted(open_set),
        load_shed_kw=calc_load_shed_kw(G, energized),
        energized_buses=list(energized),
        de_energized_buses=list(de_energized),
        note="",
    )


def _infeasible(G: nx.Graph, source: str, open_set: set[str], note: str = "") -> dict:
    energized = get_energized_buses(G, source, open_set)
    de_energized = set(G.nodes()) - energized
    return dict(
        feasible=False,
        switches_to_open=sorted(open_set),
        load_shed_kw=calc_load_shed_kw(G, energized),
        energized_buses=list(energized),
        de_energized_buses=list(de_energized),
        note=note or "No feasible isolation plan found with existing switches.",
    )


def optimize_isolation(
    G: nx.Graph,
    source_bus: str,
    risk_edges: list[tuple[str, str]],
    candidate_switches: list[str],
) -> dict:
    """Find minimum-load-shed switching plan that isolates all risk edges from source.

    Returns a dict with keys:
      feasible, switches_to_open, load_shed_kw,
      energized_buses, de_energized_buses, note
    """
    if not risk_edges:
        energized = get_energized_buses(G, source_bus, set())
        return dict(
            feasible=True,
            switches_to_open=[],
            load_shed_kw=0.0,
            energized_buses=list(energized),
            de_energized_buses=[],
            note="No risk zone defined.",
        )

    # Try virtual (newly-recommended) switches first using exhaustive search.
    # Exhaustive finds the minimum-collateral combination — a finer downstream
    # cut will beat a coarser upstream cut when both achieve isolation, so the
    # optimizer naturally converges on the tightest available set.
    virtual = [s for s in candidate_switches if s.startswith("vsw:")]
    if virtual:
        vresult = (
            _exhaustive(G, source_bus, risk_edges, virtual)
            if len(virtual) <= _EXHAUSTIVE_LIMIT
            else _greedy(G, source_bus, risk_edges, virtual)
        )
        if vresult.get("feasible"):
            return vresult

    # Fall back to all candidates when virtual switches alone aren't enough.
    if len(candidate_switches) <= _EXHAUSTIVE_LIMIT:
        return _exhaustive(G, source_bus, risk_edges, candidate_switches)
    return _greedy(G, source_bus, risk_edges, candidate_switches)


# ── New switch placement ───────────────────────────────────────────────────────

def annotate_graph_with_risk(G: nx.Graph, bus_risk: dict[str, dict]) -> None:
    """Attach risk_tier and fuel_type attributes to graph nodes in-place."""
    for b, info in bus_risk.items():
        if b in G:
            G.nodes[b]["risk_tier"]  = info.get("tier", 0)
            G.nodes[b]["fuel_type"]  = info.get("fuel_type", "")


def get_at_risk_edges(G: nx.Graph, min_tier: int = 2) -> list[tuple[str, str]]:
    """Edges where EITHER endpoint is in a risk zone at or above min_tier.
    'Either' is conservative — a line passing through risk territory is at risk."""
    return [
        (u, v)
        for u, v in G.edges()
        if (G.nodes[u].get("risk_tier", 0) >= min_tier or
            G.nodes[v].get("risk_tier", 0) >= min_tier)
    ]


def _bfs_tree(G: nx.Graph, source: str) -> tuple[dict[str, str | None], set[str]]:
    """BFS spanning tree from source, skipping base-case open switches.
    Returns (parent_map, energized_set).
    parent_map[node] = parent node (None for source).
    """
    parent: dict[str, str | None] = {source: None}
    visited: set[str] = {source}
    queue: deque[str] = deque([source])

    while queue:
        node = queue.popleft()
        for nb in G.neighbors(node):
            if nb in visited:
                continue
            edata = G.get_edge_data(node, nb)
            if edata and edata.get("is_open", False):
                continue          # skip base-case open switches
            parent[nb] = node
            visited.add(nb)
            queue.append(nb)

    return parent, visited


def _subtree_nodes(parent: dict, child: str) -> set[str]:
    """All nodes in the subtree rooted at `child` (inclusive) in the BFS tree."""
    children_of: dict[str, list[str]] = {}
    for node, par in parent.items():
        if par is not None:
            children_of.setdefault(par, []).append(node)

    result: set[str] = set()
    stack = [child]
    while stack:
        n = stack.pop()
        result.add(n)
        stack.extend(children_of.get(n, []))
    return result


def _nearest_upstream_switch(G: nx.Graph, parent: dict, node: str) -> str | None:
    """Walk up the BFS parent chain from `node`; return the first switch edge name found."""
    current = node
    while True:
        par = parent.get(current)
        if par is None:
            return None   # reached source without finding a switch
        edata = G.get_edge_data(par, current)
        if edata and edata.get("is_switch") and not edata.get("is_open", False):
            return edata["name"]
        current = par


def _find_one_best_switch(
    G: nx.Graph,
    source_bus: str,
    at_risk_buses: set[str],
    all_at_risk_buses: set[str],
    service_buses: set[str],
) -> dict | None:
    """Return the single best non-switch edge on which to place a new switch.

    at_risk_buses     — uncovered at-risk buses this round should address.
    all_at_risk_buses — full at-risk set (used for the boundary/safe checks).

    Only considers edges at the safe→at-risk boundary: par must NOT itself be
    at risk, otherwise opening a switch there still leaves the par→source
    segment energised and doesn't achieve isolation.
    """
    parent, energized = _bfs_tree(G, source_bus)
    _subtree_cache: dict[str, set[str]] = {}

    def subtree(node: str) -> set[str]:
        if node not in _subtree_cache:
            _subtree_cache[node] = _subtree_nodes(parent, node)
        return _subtree_cache[node]

    best: dict | None = None

    for node in list(energized):
        par = parent.get(node)
        if par is None:
            continue
        edata = G.get_edge_data(par, node)
        if edata is None or edata.get("is_switch"):
            continue

        # Distribution only — skip transmission / sub-transmission edges
        kv_u = G.nodes[par].get("kv_base", 0.0)
        kv_v = G.nodes[node].get("kv_base", 0.0)
        if max(kv_u, kv_v) > _DIST_KV_MAX:
            continue

        # Boundary check: the upstream node must be safe.  A switch inside
        # the risk zone doesn't isolate the at-risk edge above it.
        if par in all_at_risk_buses:
            continue

        sub = subtree(node)
        at_risk_in_sub = {b for b in sub if b in at_risk_buses and b in service_buses}
        if not at_risk_in_sub:
            continue

        safe_in_sub = {b for b in sub if b in service_buses and b not in all_at_risk_buses}
        upstream_sw = _nearest_upstream_switch(G, parent, par)

        if upstream_sw is None:
            collateral_saved = len(safe_in_sub)
        else:
            us_child = _find_switch_child(G, parent, par, upstream_sw)
            if us_child:
                us_sub = subtree(us_child)
                collateral_saved = len(
                    {b for b in (us_sub - sub)
                     if b in service_buses and b not in all_at_risk_buses}
                )
            else:
                collateral_saved = len(safe_in_sub)

        safe_count  = len(safe_in_sub)
        subtree_sz  = len(sub)
        # Primary:   minimise safe_count (fewest safe customers de-energized).
        # Secondary: minimise subtree_sz  (smallest subtree = closest to T3 boundary).
        # Tertiary:  maximise at_risk_isolated (wider risk coverage among equal cuts).
        is_better = (
            best is None
            or safe_count < best["safe_collateral"]
            or (safe_count == best["safe_collateral"]
                and subtree_sz < best["subtree_sz"])
            or (safe_count == best["safe_collateral"]
                and subtree_sz == best["subtree_sz"]
                and len(at_risk_in_sub) > best["at_risk_isolated"])
        )
        if is_better:
            best = dict(
                edge_u=par,
                edge_v=node,
                edge_name=edata.get("name", f"{par}--{node}"),
                benefit_service_pts=len(at_risk_in_sub),
                at_risk_isolated=len(at_risk_in_sub),
                safe_collateral=safe_count,
                subtree_sz=subtree_sz,
                subtree_nodes=sub,
            )

    return best


def find_new_switch_locations(
    G: nx.Graph,
    source_bus: str,
    at_risk_buses: set[str],
    service_buses: set[str],
    budget_n: int,
    risk_edges: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Greedy incremental switch placement with two stopping conditions.

    Primary stop — covered_at_risk: once every at-risk bus cluster has been
    assigned a virtual switch via the spanning-tree subtree, no more picks
    address new clusters, preventing redundant switches on parallel line
    segments to the same zone.

    Early-exit stop — isolation achieved: if risk_edges is supplied the loop
    exits as soon as the placed virtual switches already isolate all at-risk
    segments in the full network graph, regardless of remaining budget.

    Returns up to budget_n entries (best first), each containing:
      edge_u, edge_v, edge_name,
      benefit_service_pts  (safe customers saved vs. nearest upstream switch),
      at_risk_isolated     (at-risk service points isolated by this switch),
      safe_collateral      (safe service points inside the subtree),
      subtree_nodes        (buses downstream of this edge at time of selection).
    """
    G_aug = G.copy()
    selected: list[dict] = []
    covered_at_risk: set[str] = set()

    for _ in range(budget_n):
        # Early exit: current virtual switches already achieve full isolation
        if risk_edges is not None and selected:
            open_set = {c["edge_name"] for c in selected}
            if _risk_isolated(G_aug, source_bus, risk_edges, open_set):
                break

        remaining = at_risk_buses - covered_at_risk
        if not remaining:
            break

        best = _find_one_best_switch(G_aug, source_bus, remaining, at_risk_buses, service_buses)
        if best is None or best["benefit_service_pts"] == 0:
            break

        # Promote the chosen edge to a virtual switch so the next iteration
        # treats it as an available cut point.
        u, v = best["edge_u"], best["edge_v"]
        if G_aug.has_edge(u, v):
            edata = G_aug[u][v]
            vsw_name = f"vsw:{edata.get('name', u + '--' + v)}"
            edata["is_switch"] = True
            edata["is_open"]   = False
            edata["name"]      = vsw_name
            best["edge_name"]  = vsw_name

        # Mark at-risk buses in this subtree as covered so the next round
        # targets a different risk cluster instead of a parallel segment.
        covered_at_risk |= {b for b in best["subtree_nodes"] if b in at_risk_buses}

        selected.append(best)

    return selected


def _find_switch_child(G: nx.Graph, parent: dict, start: str, sw_name: str) -> str | None:
    """Walk up from `start`; return the child node of the edge named sw_name."""
    current = start
    while True:
        par = parent.get(current)
        if par is None:
            return None
        edata = G.get_edge_data(par, current)
        if edata and edata.get("name") == sw_name:
            return current
        current = par


def add_virtual_switches(G: nx.Graph, new_sw_candidates: list[dict]) -> nx.Graph:
    """Return a copy of G with virtual switches added at the specified edges.
    The new switches are normally-closed (is_open=False) and are named
    'vsw:<edge_name>' so the optimizer can open them.
    """
    H = G.copy()
    for c in new_sw_candidates:
        u, v = c["edge_u"], c["edge_v"]
        if H.has_edge(u, v):
            edata = H[u][v]
            if not edata.get("is_switch"):
                edata["is_switch"] = True
                edata["is_open"]   = False
                edata["name"]      = f"vsw:{edata.get('name', u + '--' + v)}"
    return H


def count_special_customers_affected(result: dict, registry) -> dict:
    """Break out key accounts and medical baseline customers in a result dict.

    Returns:
      key_accounts_affected, medical_customers_affected  (counts)
      key_accounts_detail, medical_detail                (CustomerRecord lists)
    """
    from .customer_registry import CustomerRegistry  # local import avoids circular dep
    de_en = set(result["de_energized_buses"])
    key_affected, med_affected = registry.affected(de_en)
    return dict(
        key_accounts_affected=len(key_affected),
        medical_customers_affected=len(med_affected),
        key_accounts_detail=key_affected,
        medical_detail=med_affected,
    )


def count_service_pts_affected(
    result: dict,
    service_buses: set[str],
    at_risk_buses: set[str],
) -> dict:
    """Summarise a optimize_isolation result in terms of service points."""
    de_en = set(result["de_energized_buses"])
    total_affected  = sum(1 for b in service_buses if b in de_en)
    at_risk_correct = sum(1 for b in service_buses if b in de_en and b in at_risk_buses)
    collateral      = total_affected - at_risk_correct
    return dict(
        total_affected=total_affected,
        at_risk_correct=at_risk_correct,
        collateral=collateral,
        pct_collateral=collateral / max(total_affected, 1) * 100,
    )


def run_sensitivity_analysis(
    G: nx.Graph,
    source_bus: str,
    at_risk_buses: set[str],
    service_buses: set[str],
    risk_edges: list[tuple[str, str]],
    max_new_switches: int = 15,
    cost_per_switch: float = 75_000,
) -> list[dict]:
    """Compute the marginal PSPS impact reduction for 0..max_new_switches new switches.

    Returns a list of dicts (one per step):
      n_new, cumulative_cost, total_affected, collateral,
      collateral_reduction_vs_baseline, switches_opened, new_switch_names
    """
    all_candidates = find_new_switch_locations(
        G, source_bus, at_risk_buses, service_buses, budget_n=max_new_switches
    )

    base_candidates = get_candidate_switches(G)

    rows = []
    for n in range(0, min(max_new_switches, len(all_candidates)) + 1):
        if n == 0:
            G_aug  = G
            sw_aug = base_candidates
            new_names = []
        else:
            G_aug  = add_virtual_switches(G, all_candidates[:n])
            sw_aug = get_candidate_switches(G_aug)
            new_names = [c["edge_name"] for c in all_candidates[:n]]

        result = optimize_isolation(G_aug, source_bus, risk_edges, sw_aug)
        svc    = count_service_pts_affected(result, service_buses, at_risk_buses)

        rows.append(dict(
            n_new=n,
            cumulative_cost=n * cost_per_switch,
            total_affected=svc["total_affected"],
            collateral=svc["collateral"],
            at_risk_correct=svc["at_risk_correct"],
            switches_opened=result["switches_to_open"],
            new_switch_names=new_names,
        ))

    # Annotate collateral reduction relative to baseline (n=0)
    baseline = rows[0]["collateral"]
    for r in rows:
        r["collateral_reduction"] = baseline - r["collateral"]
        r["pct_reduction"]        = r["collateral_reduction"] / max(baseline, 1) * 100

    return rows
