import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { buildIndex, CitationList, ClaimList, SourceDrawer } from "../components/Citations";
import { MetaPanel } from "../components/MetaPanel";
import { useUser } from "../context/UserContext";
import { useApi } from "../hooks/useApi";

export function MeetingPrepPage() {
  const { accountId = "" } = useParams();
  const { userId } = useUser();
  const [openSource, setOpenSource] = useState<string | null>(null);
  const prep = useApi((s) => api.meetingPrep(userId, accountId, s), [userId, accountId]);

  if (prep.loading) return <p className="muted">Preparing meeting brief...</p>;
  if (prep.error) return <p className="error">{prep.error.message}</p>;
  if (!prep.data) return null;

  const { account, meeting, prep: p, citations, meta } = prep.data;
  const index = buildIndex(citations);

  return (
    <>
      <div className="page-head">
        <div>
          <h2>Meeting prep: {account.name}</h2>
          <p className="muted">
            <Link to={`/accounts/${account.id}`}>Back to account brief</Link>
          </p>
        </div>
      </div>
      <section className="card">
        {meeting ? (
          <>
            <h3>{meeting.title}</h3>
            <p className="muted">
              {meeting.start.replace("T", " ").slice(0, 16)} · {meeting.duration_minutes} minutes
            </p>
            <p>{meeting.agenda}</p>
            <p className="small">
              Attendees: {meeting.attendees.map((a) => `${a.name} (${a.title})`).join(", ") || "none listed"}
            </p>
          </>
        ) : (
          <p className="muted">No upcoming meeting is visible to you for this account.</p>
        )}
      </section>
      <div className="grid-2">
        <ClaimList title="Objectives" claims={p.objectives} index={index} onOpen={setOpenSource} />
        <ClaimList title="Talking points" claims={p.talking_points} index={index} onOpen={setOpenSource} />
      </div>
      <div className="grid-2">
        <section className="card">
          <h3>Questions to ask</h3>
          {p.questions_to_ask.length ? (
            <ul className="claims">
              {p.questions_to_ask.map((q) => (
                <li key={q}>{q}</li>
              ))}
            </ul>
          ) : (
            <p className="muted">No open questions found.</p>
          )}
        </section>
        <ClaimList title="Risks" claims={p.risks} index={index} onOpen={setOpenSource} />
      </div>
      <CitationList citations={citations} onOpen={setOpenSource} />
      <MetaPanel meta={meta} />
      {openSource && <SourceDrawer sourceId={openSource} onClose={() => setOpenSource(null)} />}
    </>
  );
}
