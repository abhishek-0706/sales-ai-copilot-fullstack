import { useEffect } from "react";
import { api } from "../api/client";
import type { Citation, Claim } from "../api/types";
import { useApi } from "../hooks/useApi";
import { useUser } from "../context/UserContext";

export type CitationIndex = Map<string, number>;

export function buildIndex(citations: Citation[]): CitationIndex {
  return new Map(citations.map((c, i) => [c.source_id, i + 1]));
}

interface ChipProps {
  ids: string[];
  index: CitationIndex;
  onOpen: (sourceId: string) => void;
}

export function CitationChips({ ids, index, onOpen }: ChipProps) {
  return (
    <span className="chips">
      {ids.map((id) => (
        <button key={id} type="button" className="chip" title={`Open source ${id}`} onClick={() => onOpen(id)}>
          {index.get(id) ?? id}
        </button>
      ))}
    </span>
  );
}

interface ClaimListProps {
  title: string;
  claims: Claim[];
  index: CitationIndex;
  onOpen: (sourceId: string) => void;
  empty?: string;
}

export function ClaimList({ title, claims, index, onOpen, empty = "Nothing found in the records you can access." }: ClaimListProps) {
  return (
    <section className="card">
      <h3>{title}</h3>
      {claims.length === 0 ? (
        <p className="muted">{empty}</p>
      ) : (
        <ul className="claims">
          {claims.map((c, i) => (
            <li key={i}>
              {c.text} <CitationChips ids={c.source_ids} index={index} onOpen={onOpen} />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function CitationList({ citations, onOpen }: { citations: Citation[]; onOpen: (id: string) => void }) {
  if (citations.length === 0) return null;
  return (
    <section className="card">
      <h3>Sources</h3>
      <ol className="sources">
        {citations.map((c) => (
          <li key={c.source_id}>
            <button type="button" className="link" onClick={() => onOpen(c.source_id)}>
              {c.title}
            </button>{" "}
            <span className="badge">{c.source_type}</span> <code>{c.source_id}</code>
            <div className="muted small">{c.snippet}</div>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function SourceDrawer({ sourceId, onClose }: { sourceId: string; onClose: () => void }) {
  const { userId } = useUser();
  const { data, error, loading } = useApi((s) => api.source(userId, sourceId, s), [userId, sourceId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" role="dialog" aria-label={`Source ${sourceId}`} onClick={(e) => e.stopPropagation()}>
        <header className="drawer-head">
          <code>{sourceId}</code>
          <button type="button" className="ghost" onClick={onClose} aria-label="Close source">Close</button>
        </header>
        {loading && <p className="muted">Loading source...</p>}
        {error && <p className="error">{error.message}</p>}
        {data && (
          <>
            <h3>{data.title}</h3>
            <p>
              <span className="badge">{data.source_type}</span> <span className="badge">{data.territory}</span>{" "}
              <span className={`badge sens-${data.sensitivity}`}>{data.sensitivity}</span>
            </p>
            {data.flagged_injection && (
              <p className="warning">
                Guardrail: this record contains prompt-injection patterns. It was withheld from the model.
              </p>
            )}
            <p className="source-text">{data.text}</p>
          </>
        )}
      </aside>
    </div>
  );
}
