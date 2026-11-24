import type { ResponseMeta } from "../api/types";

export function MetaPanel({ meta, label = "Trace" }: { meta: ResponseMeta; label?: string }) {
  const stages = Object.entries(meta.stages_ms);
  const total = stages.reduce((sum, [, ms]) => sum + ms, 0);
  const injections = meta.guardrail_events.filter((e) => e.type === "prompt_injection");

  return (
    <details className="card meta">
      <summary>
        {label} <code>{meta.trace_id.slice(0, 12)}</code> · {meta.provider} · {total.toFixed(0)} ms ·
        groundedness {(meta.groundedness * 100).toFixed(0)}%
        {meta.redactions > 0 && <> · {meta.redactions} PII redacted</>}
        {injections.length > 0 && <span className="warn-inline"> · {injections.length} injection blocked</span>}
        {meta.fallback_reason && <span className="warn-inline"> · fallback</span>}
      </summary>
      <div className="meta-grid">
        <div>
          <h4>Stage latency</h4>
          <ul className="plain">
            {stages.map(([name, ms]) => (
              <li key={name}>
                <span className="stage-name">{name}</span>
                <span className="bar" style={{ width: `${Math.max(4, (ms / (total || 1)) * 120)}px` }} />
                {ms.toFixed(1)} ms
              </li>
            ))}
          </ul>
          <p className="small">
            Model: <code>{meta.model}</code>
            {meta.tokens && (
              <> · tokens {meta.tokens.total_tokens}{meta.tokens.estimated ? " (estimated)" : ""}</>
            )}
          </p>
          {meta.fallback_reason && <p className="small warning">Fallback: {meta.fallback_reason}</p>}
        </div>
        <div>
          <h4>Guardrails</h4>
          {meta.guardrail_events.length === 0 ? (
            <p className="muted small">No events.</p>
          ) : (
            <ul className="plain small">
              {meta.guardrail_events.map((e, i) => (
                <li key={i}>
                  <span className={`badge ev-${e.type}`}>{e.type}</span> {e.source_id && <code>{e.source_id}</code>}{" "}
                  {e.detail}
                </li>
              ))}
            </ul>
          )}
          <h4>Tools called</h4>
          <ul className="plain small">
            {meta.tool_calls.map((t, i) => (
              <li key={i}>
                <code>{t.name}</code> {Object.values(t.args).join(", ")} → {t.result_ids.length} records
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4>Access scope</h4>
          <p className="small">
            {meta.access_scope.role} · {meta.access_scope.territories.join(", ")} · up to{" "}
            {meta.access_scope.max_sensitivity}
          </p>
          <p className="small muted">Record types: {meta.access_scope.record_types.join(", ")}</p>
          <p className="small muted">{meta.retrieved_ids.length} evidence items supplied to the model.</p>
        </div>
      </div>
    </details>
  );
}
