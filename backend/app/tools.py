"""Deterministic tools the agent can call. Each enforces the caller's access policy itself;
the model chooses tools and arguments but never the identity they run as."""
from __future__ import annotations

from .data_store import DataStore
from .embeddings import tokenize
from .models import Evidence, ToolCall, User
from .security import can_read

TOOL_SPECS = {
    "crm_lookup": "Account profile, key contacts and recent activity notes. args: {account_id}",
    "opportunity_data": "Open opportunities for an account: stage, amount, close date, competitor, next step. "
                        "args: {account_id}",
    "product_catalogue": "Product documentation sections matching a topic. args: {query} or {product_id}",
    "calendar": "Upcoming meetings, optionally for one account. args: {account_id?}",
}


class Tools:
    def __init__(self, store: DataStore):
        self.store = store

    def _visible(self, user: User, ev: Evidence) -> bool:
        return can_read(user, territory=ev.territory, record_type=ev.source_type, sensitivity=ev.sensitivity)

    def accessible_accounts(self, user: User) -> dict[str, str]:
        return {a.id: a.name for a in self.store.accounts.values()
                if can_read(user, territory=a.territory, record_type="account", sensitivity=a.sensitivity)}

    def crm_lookup(self, user: User, account_id: str) -> list[Evidence]:
        acc = self.store.accounts.get(account_id)
        if acc is None:
            return []
        out = [self.store.account_evidence(acc)]
        out += [self.store.contact_evidence(c) for c in self.store.contacts.values() if c.account_id == account_id]
        acts = sorted((a for a in self.store.activities.values() if a.account_id == account_id),
                      key=lambda a: a.date, reverse=True)
        out += [self.store.activity_evidence(a) for a in acts]
        return [e for e in out if self._visible(user, e)][:10]

    def opportunity_data(self, user: User, account_id: str) -> list[Evidence]:
        evs = [self.store.opportunity_evidence(o) for o in self.store.opportunities.values()
               if o.account_id == account_id]
        return [e for e in evs if self._visible(user, e)]

    def product_catalogue(self, user: User, query: str = "", product_id: str = "") -> list[Evidence]:
        if product_id:
            prod = self.store.products.get(product_id)
            evs = self.store.product_evidence(prod) if prod else []
        else:
            terms = set(tokenize(query))
            evs = []
            for prod in self.store.products.values():
                for ev in self.store.product_evidence(prod):
                    if terms & set(tokenize(f"{ev.title} {ev.text}")):
                        evs.append(ev)
            evs = sorted(evs, key=lambda e: -len(terms & set(tokenize(f"{e.title} {e.text}"))))[:4]
        return [e for e in evs if self._visible(user, e)]

    def calendar(self, user: User, account_id: str = "") -> list[Evidence]:
        meetings = [m for m in self.store.meetings.values() if not account_id or m.account_id == account_id]
        evs = [self.store.meeting_evidence(m) for m in sorted(meetings, key=lambda m: m.start)]
        return [e for e in evs if self._visible(user, e)]

    def run(self, user: User, call: ToolCall) -> list[Evidence]:
        a = call.args
        if call.name == "crm_lookup":
            return self.crm_lookup(user, a.get("account_id", ""))
        if call.name == "opportunity_data":
            return self.opportunity_data(user, a.get("account_id", ""))
        if call.name == "product_catalogue":
            return self.product_catalogue(user, a.get("query", ""), a.get("product_id", ""))
        return self.calendar(user, a.get("account_id", ""))
