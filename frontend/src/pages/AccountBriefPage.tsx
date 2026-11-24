import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { buildIndex, CitationChips, CitationList, ClaimList, SourceDrawer } from "../components/Citations";
import { MetaPanel } from "../components/MetaPanel";
import { useUser } from "../context/UserContext";
import { useApi } from "../hooks/useApi";

export function AccountBriefPage() {
  const { accountId = "" } = useParams();
  const { userId } = useUser();
  const [openSource, setOpenSource] = useState<string | null>(null);
  const brief = useApi((s) => api.brief(userId, accountId, s), [userId, accountId]);
  const nba = useApi((s) => api.nextBestActions(userId, accountId, s), [userId, accountId]);

  if (brief.loading) return <p className="muted">Building account brief...</p>;
  if (brief.error) return <p className="error">{brief.error.message}</p>;
  if (!brief.data) return null;

  const { account, brief: b, citations, meta } = brief.data;
  // One numbering for brief and next-best-action citations so chips match the single Sources list.
  const seen = new Set(citations.map((c) => c.source_id));
  const allCitations = [...citations, ...(nba.data?.citations ?? []).filter((c) => !seen.has(c.source_id))];
  const index = buildIndex(allCitations);

  return (
    <>
      <div className="page-head">
        <div>
          <h2>{account.name}</h2>
          <p className="muted">
            {account.industry} · {account.city} · {account.territory} · owner {account.owner_name}
          </p>
        </div>
        <div className="actions">
          <Link className="button" to={`/accounts/${account.id}/meeting`}>Meeting prep</Link>
          <Link className="button secondary" to={`/ask?account=${account.id}`}>Ask about this account</Link>
        </div>
      </div>

      <section className="card">
        <h3>Summary</h3>
        <p>
          {b.summary} <CitationChips ids={b.summary_source_ids} index={index} onOpen={setOpenSource} />
        </p>
      </section>
      <div className="grid-2">
        <ClaimList title="Key facts" claims={b.key_facts} index={index} onOpen={setOpenSource} />
        <ClaimList title="Risks" claims={b.risks} index={index} onOpen={setOpenSource} />
      </div>
      <ClaimList title="Recent activity" claims={b.recent_activity} index={index} onOpen={setOpenSource} />

      <section className="card">
        <h3>Next best actions</h3>
        {nba.loading && <p className="muted">Working out next best actions...</p>}
        {nba.error && <p className="error">{nba.error.message}</p>}
        {nba.data && (
          <ol className="actions-list">
            {nba.data.result.actions.map((a, i) => (
              <li key={i}>
                <span className={`badge prio-${a.priority}`}>{a.priority}</span>{" "}
                <span className="badge">{a.action_type.replace("_", " ")}</span> <strong>{a.action}</strong>
                <div className="muted small">
                  {a.rationale} <CitationChips ids={a.source_ids} index={index} onOpen={setOpenSource} />
                </div>
              </li>
            ))}
          </ol>
        )}
      </section>

      <CitationList citations={allCitations} onOpen={setOpenSource} />
      <MetaPanel meta={meta} label="Brief trace" />
      {nba.data && <MetaPanel meta={nba.data.meta} label="Next-best-action trace" />}
      {openSource && <SourceDrawer sourceId={openSource} onClose={() => setOpenSource(null)} />}
    </>
  );
}
