"""Loads the synthetic CRM JSON files and turns records into citable evidence units."""
from __future__ import annotations

import json
from pathlib import Path

from .models import (Account, AccountSummary, Activity, Contact, Email, Evidence, Meeting, Opportunity,
                     ProductDoc, User)


class DataStore:
    def __init__(self, data_dir: Path):
        def load(name: str) -> list[dict]:
            return json.loads((data_dir / name).read_text(encoding="utf-8"))

        self.meta: dict = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
        self.users = {u.id: u for u in map(User.model_validate, load("users.json"))}
        self.accounts = {a.id: a for a in map(Account.model_validate, load("accounts.json"))}
        self.contacts = {c.id: c for c in map(Contact.model_validate, load("contacts.json"))}
        self.opportunities = {o.id: o for o in map(Opportunity.model_validate, load("opportunities.json"))}
        self.activities = {a.id: a for a in map(Activity.model_validate, load("activities.json"))}
        self.emails = {e.id: e for e in map(Email.model_validate, load("emails.json"))}
        self.meetings = {m.id: m for m in map(Meeting.model_validate, load("calendar.json"))}
        self.products = {p.id: p for p in map(ProductDoc.model_validate, load("products.json"))}
        self._chunks = self._build_chunks()
        contact_evs = [self.contact_evidence(c) for c in self.contacts.values()]
        self._by_source = {c.source_id: c for c in [*self._chunks, *contact_evs]}

    @staticmethod
    def summary(acc: Account) -> AccountSummary:
        return AccountSummary.model_validate(acc.model_dump())

    def account_evidence(self, acc: Account) -> Evidence:
        return Evidence(
            source_id=acc.id, record_id=acc.id, source_type="account", account_id=acc.id, territory=acc.territory,
            sensitivity=acc.sensitivity, title=acc.name,
            text=(f"{acc.name} is a {acc.tier} {acc.industry} account based in {acc.city} ({acc.territory}), "
                  f"owned by {acc.owner_name}, with about {acc.employees:,} employees. Use case: {acc.use_case}."))

    def contact_evidence(self, c: Contact) -> Evidence:
        acc = self.accounts[c.account_id]
        return Evidence(
            source_id=c.id, record_id=c.id, source_type="contact", account_id=c.account_id, territory=acc.territory,
            sensitivity="internal", title=f"{c.name} ({c.title})",
            text=f"Contact at {acc.name}: {c.name}, {c.title}, email {c.email}, phone {c.phone}.")

    def opportunity_evidence(self, o: Opportunity) -> Evidence:
        products = ", ".join(self.products[p].name for p in o.products if p in self.products)
        return Evidence(
            source_id=o.id, record_id=o.id, source_type="opportunity", account_id=o.account_id,
            territory=o.territory, sensitivity=o.sensitivity, title=o.name,
            text=(f"Opportunity {o.name}: stage {o.stage}, amount USD {o.amount_usd:,}, close date {o.close_date}, "
                  f"probability {o.probability} percent. Products: {products}. "
                  f"Competitor: {o.competitor or 'none recorded'}. Next step: {o.next_step}."))

    def activity_evidence(self, a: Activity) -> Evidence:
        acc_name = self.accounts[a.account_id].name
        return Evidence(
            source_id=a.id, record_id=a.id, source_type="activity", account_id=a.account_id, territory=a.territory,
            sensitivity=a.sensitivity, title=f"{a.type.replace('_', ' ').title()} {a.date} - {acc_name}",
            text=f"{a.text}")

    def email_evidence(self, e: Email) -> Evidence:
        return Evidence(
            source_id=e.id, record_id=e.id, source_type="email", account_id=e.account_id, territory=e.territory,
            sensitivity=e.sensitivity, title=e.subject,
            text=f"Email on {e.date} from {e.sender}, subject '{e.subject}': {e.body}")

    def meeting_evidence(self, m: Meeting) -> Evidence:
        return Evidence(
            source_id=m.id, record_id=m.id, source_type="meeting", account_id=m.account_id, territory=m.territory,
            sensitivity=m.sensitivity, title=m.title,
            text=f"Meeting '{m.title}' starts {m.start} for {m.duration_minutes} minutes. Agenda: {m.agenda}")

    def product_evidence(self, p: ProductDoc) -> list[Evidence]:
        return [Evidence(source_id=f"{p.id}:s{i}", record_id=p.id, source_type="product_doc", account_id=None,
                         territory=p.territory, sensitivity=p.sensitivity, title=f"{p.name} - {s.title}", text=s.text)
                for i, s in enumerate(p.sections, start=1)]

    def _build_chunks(self) -> list[Evidence]:
        chunks = [self.account_evidence(a) for a in self.accounts.values()]
        chunks += [self.opportunity_evidence(o) for o in self.opportunities.values()]
        chunks += [self.activity_evidence(a) for a in self.activities.values()]
        chunks += [self.email_evidence(e) for e in self.emails.values()]
        chunks += [self.meeting_evidence(m) for m in self.meetings.values()]
        for p in self.products.values():
            chunks += self.product_evidence(p)
        return chunks

    def chunks(self) -> list[Evidence]:
        return list(self._chunks)

    def by_source_id(self, source_id: str) -> Evidence | None:
        return self._by_source.get(source_id)
