"""Role- and territory-based access policy.

The same policy object produces the SQL predicate used *inside* retrieval and the
per-record check used by the deterministic tools, so nothing outside the caller's
scope ever reaches the prompt.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import SENSITIVITY_LEVEL, AccessScope, Role, User

ALL_TYPES = ("account", "contact", "opportunity", "activity", "email", "meeting", "product_doc")


@dataclass(frozen=True)
class RolePolicy:
    record_types: tuple[str, ...]
    max_sensitivity: str


POLICIES: dict[Role, RolePolicy] = {
    Role.SDR: RolePolicy(("account", "contact", "activity", "meeting", "product_doc"), "internal"),
    Role.AE: RolePolicy(ALL_TYPES, "confidential"),
    Role.MANAGER: RolePolicy(ALL_TYPES, "restricted"),
}

GLOBAL_TERRITORY = "GLOBAL"


class AccessDenied(Exception):
    pass


def scope_for(user: User) -> AccessScope:
    p = POLICIES[user.role]
    return AccessScope(role=user.role, territories=list(user.territories), record_types=list(p.record_types),
                       max_sensitivity=p.max_sensitivity)  # type: ignore[arg-type]


def can_read(user: User, *, territory: str, record_type: str, sensitivity: str) -> bool:
    p = POLICIES[user.role]
    return (
        (territory == GLOBAL_TERRITORY or territory in user.territories)
        and record_type in p.record_types
        and SENSITIVITY_LEVEL[sensitivity] <= SENSITIVITY_LEVEL[p.max_sensitivity]
    )


def sql_predicate(user: User) -> tuple[str, list]:
    """WHERE fragment applied inside the KNN and keyword queries (pre-filtering)."""
    p = POLICIES[user.role]
    territories = [*user.territories, GLOBAL_TERRITORY]
    t_marks = ",".join("?" * len(territories))
    r_marks = ",".join("?" * len(p.record_types))
    clause = f"territory IN ({t_marks}) AND record_type IN ({r_marks}) AND sensitivity_level <= ?"
    return clause, [*territories, *p.record_types, SENSITIVITY_LEVEL[p.max_sensitivity]]
