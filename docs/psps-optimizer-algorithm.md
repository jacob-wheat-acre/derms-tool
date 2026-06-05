# PSPS Switch Optimizer — Algorithm Description

This document describes the logic behind the Public Safety Power Shutoff (PSPS)
switch optimization module. It is written for a technical audience that does not
need to read the source code directly.

---

## Core Concepts

```
NETWORK is modeled as a graph:
  - NODES  = buses / substations / service points
  - EDGES  = line segments connecting them
  - Each edge has: name, is_switch (true/false), is_open (true/false)
  - Each node has: load_kw (power served at that point)

SOURCE_BUS = the feeder head (where power enters from transmission)

RISK_EDGES = line segments that must be de-energized during a PSPS event
  (determined by which buses fall inside a wildfire risk zone)

CANDIDATE_SWITCHES = all normally-closed switches that the optimizer
  is allowed to open
```

---

## Step 1 — Determine Which Buses Are Energized

```
FUNCTION get_energized_buses(network, source_bus, switches_to_open):

  Build a traversal graph from network,
    removing any edge that is already open (base-case opens)
    removing any edge whose switch name is in switches_to_open

  Starting from source_bus, do a breadth-first traversal
    (follow every connected edge outward)

  RETURN the set of all reachable nodes
    (everything NOT reachable is de-energized / load shed)
```

---

## Step 2 — Isolation Check

```
FUNCTION risk_is_isolated(network, source_bus, risk_edges, switches_to_open):

  energized = get_energized_buses(network, source_bus, switches_to_open)

  FOR each (node_A, node_B) in risk_edges:
    IF both node_A and node_B are in energized:
      RETURN false   ← at least one risk segment is still live

  RETURN true   ← all risk segments are successfully isolated
```

---

## Step 3 — Find the Optimal Switching Plan (Existing Switches)

Two strategies are used depending on how many candidate switches exist.

### 3a — Exhaustive Search (≤ 20 candidates)

```
FUNCTION exhaustive_search(network, source_bus, risk_edges, candidates):

  best_shed = infinity
  best_combination = none

  FOR each possible subset of candidates (size 1, then 2, then 3, ...):
    IF risk_is_isolated(network, source_bus, risk_edges, this_subset):
      energized = get_energized_buses(network, source_bus, this_subset)
      shed = total load at de-energized nodes
      IF shed < best_shed:
        best_shed = shed
        best_combination = this_subset

  RETURN best_combination   ← minimum collateral impact plan
```

### 3b — Greedy Search (> 20 candidates)

```
FUNCTION greedy_search(network, source_bus, risk_edges, candidates):

  open_set = {}   ← empty; switches we've decided to open
  remaining_risk = all risk_edges

  WHILE remaining_risk is not empty:

    best_switch = none
    best_shed = infinity

    FOR each switch in candidates not yet in open_set:
      trial = open_set + {this switch}
      energized = get_energized_buses(network, source_bus, trial)

      newly_isolated = count of risk_edges no longer live under trial
      IF newly_isolated == 0: skip   ← this switch makes no progress

      shed = total load at de-energized nodes
      IF shed < best_shed:
        best_switch = this switch
        best_shed = shed

    IF no switch makes progress: STOP (infeasible with existing switches)

    Add best_switch to open_set
    Remove newly-isolated edges from remaining_risk

  RETURN open_set
```

### Main Entry Point

```
FUNCTION optimize_isolation(network, source_bus, risk_edges, candidates):

  IF no risk_edges: RETURN "no action needed"

  IF candidates ≤ 20:
    RETURN exhaustive_search(...)
  ELSE:
    RETURN greedy_search(...)
```

---

## Step 4 — Recommend Where to Place New Switches

This runs before `optimize_isolation` when a capital investment budget is provided.

```
FUNCTION find_new_switch_locations(network, source_bus, at_risk_buses,
                                   service_buses, budget_n):

  selected = []   ← switches we've recommended so far
  covered  = {}   ← at-risk buses already addressed

  REPEAT up to budget_n times:

    IF all at-risk buses are already covered: STOP

    Build a spanning tree from source_bus outward
      (BFS tree — represents how power flows through the feeder)

    FOR each non-switch edge in the spanning tree:

      IF the upstream node of this edge is itself at risk: SKIP
        (opening here would not achieve isolation — the at-risk
         segment above it would still be live)

      IF this edge is on the sub-transmission / transmission system: SKIP
        (only distribution-voltage cuts are considered)

      subtree = all nodes downstream of this edge
      at_risk_in_subtree = at-risk buses inside the subtree
        that are actual service points and not yet covered
      safe_in_subtree    = non-at-risk service points inside the subtree
        (these are "collateral" customers who would lose power too)

      Score this candidate edge:
        PRIMARY   — minimize safe_in_subtree  (fewest collateral customers)
        SECONDARY — minimize subtree size     (tightest cut, closest to risk)
        TERTIARY  — maximize at_risk_in_subtree (broadest risk coverage)

    Pick the highest-scoring candidate edge

    Mark it as a virtual switch (available to the optimizer)
    Mark its at-risk subtree buses as covered

    Add to selected

  RETURN selected   ← ranked list: best first
```

---

## Step 5 — Sensitivity / Investment Analysis

```
FUNCTION run_sensitivity_analysis(network, source_bus, at_risk_buses,
                                  service_buses, risk_edges,
                                  max_new_switches, cost_per_switch):

  Pre-compute up to max_new_switches recommended new switch locations

  rows = []

  FOR n = 0 to max_new_switches:

    IF n == 0:
      Run optimize_isolation with existing switches only (baseline)
    ELSE:
      Add the top-n recommended new switches as virtual switches
      Run optimize_isolation with expanded candidate set

    Record:
      n_new_switches, cumulative_cost = n × cost_per_switch,
      total_customers_affected, collateral_customers (not in risk zone),
      at_risk_customers_correctly_isolated,
      reduction_vs_baseline

  RETURN rows   ← one row per investment level; used to plot the
                  "collateral customers saved vs. dollars spent" curve
```

---

## Key Trade-offs

| Decision | What the algorithm optimizes for |
|---|---|
| Which switches to open | Minimum customers de-energized (load shed in kW) while guaranteeing all risk segments are isolated |
| Where to add new switches | Fewest collateral (non-at-risk) customers per new switch, placed as close to the risk boundary as possible |
| Exhaustive vs. greedy | Exhaustive guarantees the global minimum; greedy is used when the search space is too large (> 20 candidates) |
| Switch placement boundary rule | A new switch must be placed where the upstream side is **not** in the risk zone — otherwise opening it doesn't prevent the energized path above from reaching the risk area |
