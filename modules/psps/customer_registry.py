"""modules/psps/customer_registry.py — Special customer data model and mock generator."""
from __future__ import annotations
import random
from dataclasses import dataclass

_KEY_ACCOUNT_NAMES = [
    "Tri-Cities Medical Center",
    "Mid-Columbia Regional Airport",
    "Richland Water Treatment Plant",
    "Kennewick Industrial Park",
    "Columbia River School District No. 1",
    "Pasco Cold Storage & Distribution",
    "Battelle Northwest Operations",
    "Benton County Public Works",
    "Hanford Site Operations Center",
    "Richland Fire Station 4",
    "Desert Wind Farm Control",
    "Columbia Basin College",
    "Franklin PUD Switching Center",
    "Ringold Fish Hatchery",
    "Pacific Coast Data Center",
    "Tri-Cities Commerce Park",
    "Yakima Valley Farm Bureau",
    "Columbia River Correctional Facility",
    "Kennewick General Hospital",
    "Mid-Columbia Wastewater Treatment",
]

_FIRST_NAMES = [
    "James", "Patricia", "Robert", "Linda", "Michael", "Barbara",
    "William", "Susan", "David", "Karen", "Richard", "Lisa",
    "Joseph", "Nancy", "Thomas", "Betty", "Charles", "Sandra",
    "Christopher", "Margaret", "Daniel", "Dorothy", "Matthew", "Jessica",
    "Anthony", "Sarah", "Donald", "Helen", "Steven", "Anna",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
    "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
    "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
]


@dataclass
class CustomerRecord:
    bus_id: str
    account_id: str
    account_name: str
    is_key_account: bool = False
    is_medical_baseline: bool = False


class CustomerRegistry:
    def __init__(self, records: list[CustomerRecord]):
        self._by_bus: dict[str, CustomerRecord] = {r.bus_id: r for r in records}

    def get(self, bus_id: str) -> CustomerRecord | None:
        return self._by_bus.get(bus_id)

    def all_key_accounts(self) -> list[CustomerRecord]:
        return [r for r in self._by_bus.values() if r.is_key_account]

    def all_medical(self) -> list[CustomerRecord]:
        return [r for r in self._by_bus.values() if r.is_medical_baseline]

    def affected(
        self, de_energized_buses: set[str]
    ) -> tuple[list[CustomerRecord], list[CustomerRecord]]:
        """Returns (key_accounts_affected, medical_affected)."""
        key = [r for r in self.all_key_accounts() if r.bus_id in de_energized_buses]
        med = [r for r in self.all_medical() if r.bus_id in de_energized_buses]
        return key, med

    @classmethod
    def mock(
        cls,
        service_buses: set[str],
        seed: int = 42,
        at_risk_buses: set[str] | None = None,
    ) -> "CustomerRegistry":
        """Deterministic mock registry for demonstration against any feeder.

        When at_risk_buses is supplied, ~40 % of key accounts and ~30 % of
        medical customers are placed on at-risk buses so the feature is always
        visible regardless of which buses the random seed happens to pick.
        """
        buses = sorted(service_buses)
        rng   = random.Random(seed)

        n_key = min(len(_KEY_ACCOUNT_NAMES), max(1, len(buses) // 80))
        n_med = max(2, len(buses) // 40)

        if at_risk_buses and (at_risk_svc := sorted(at_risk_buses & service_buses)):
            safe_svc = sorted(service_buses - at_risk_buses)

            n_key_risk = max(1, n_key * 2 // 5)
            n_key_safe = n_key - n_key_risk
            key_risk = rng.sample(at_risk_svc, min(n_key_risk, len(at_risk_svc)))
            key_safe = rng.sample(safe_svc,    min(n_key_safe, len(safe_svc)))
            key_buses = key_risk + key_safe

            used = set(key_buses)
            rem_risk = [b for b in at_risk_svc if b not in used]
            rem_safe = [b for b in safe_svc    if b not in used]
            n_med_risk = max(1, n_med * 3 // 10)
            n_med_safe = n_med - n_med_risk
            med_buses = (
                rng.sample(rem_risk, min(n_med_risk, len(rem_risk))) +
                rng.sample(rem_safe, min(n_med_safe, len(rem_safe)))
            )
        else:
            key_buses = rng.sample(buses, min(n_key, len(buses)))
            remaining = [b for b in buses if b not in set(key_buses)]
            med_buses = rng.sample(remaining, min(n_med, len(remaining)))

        records: list[CustomerRecord] = []
        for i, bus in enumerate(key_buses):
            records.append(CustomerRecord(
                bus_id=bus,
                account_id=f"KA-{1000 + i:04d}",
                account_name=_KEY_ACCOUNT_NAMES[i % len(_KEY_ACCOUNT_NAMES)],
                is_key_account=True,
            ))
        for i, bus in enumerate(med_buses):
            first = _FIRST_NAMES[i % len(_FIRST_NAMES)]
            last  = _LAST_NAMES[(i * 7) % len(_LAST_NAMES)]
            records.append(CustomerRecord(
                bus_id=bus,
                account_id=f"MB-{2000 + i:04d}",
                account_name=f"{first} {last}",
                is_medical_baseline=True,
            ))

        return cls(records)
