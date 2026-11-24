import { useState, type FormEvent } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { AskResponse } from "../api/types";
import { buildIndex, CitationChips, CitationList, SourceDrawer } from "../components/Citations";
import { MetaPanel } from "../components/MetaPanel";
import { useUser } from "../context/UserContext";
import { useApi } from "../hooks/useApi";

const SUGGESTIONS = [
  "Which competitor is Velmora Logistics evaluating?",
  "What is the maximum approved discount for Velmora Logistics?",
  "How does Orbitra Vector DB enforce access control?",
  "Summarise the latest call notes and pilot timelines for Velmora Logistics",
];

export function AskPage() {
  const { userId } = useUser();
  const [params, setParams] = useSearchParams();
  const accountId = params.get("account") ?? "";
  const accounts = useApi((s) => api.accounts(userId, s), [userId]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<AskResponse[]>([]);
  const [openSource, setOpenSource] = useState<string | null>(null);

  async function submit(e?: FormEvent, q = question) {
    e?.preventDefault();
    if (q.trim().length < 3) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.ask(userId, q.trim(), accountId || undefined);
      setHistory((h) => [res, ...h]);
      setQuestion("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <h2>Ask</h2>
      <form className="card ask-form" onSubmit={submit}>
        <label htmlFor="question">Question</label>
        <textarea
          id="question"
          rows={3}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about an account, a deal or a product"
        />
        <div className="row">
          <label htmlFor="account">Account</label>
          <select
            id="account"
            value={accountId}
            onChange={(e) => setParams(e.target.value ? { account: e.target.value } : {})}
          >
            <option value="">Any account I can access</option>
            {accounts.data?.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name}
              </option>
            ))}
          </select>
          <button type="submit" disabled={busy || question.trim().length < 3}>
            {busy ? "Thinking..." : "Ask"}
          </button>
        </div>
        <div className="suggestions">
          {SUGGESTIONS.map((s) => (
            <button key={s} type="button" className="ghost small" onClick={() => submit(undefined, s)} disabled={busy}>
              {s}
            </button>
          ))}
        </div>
        {error && <p className="error">{error}</p>}
      </form>

      {history.map((res) => {
        const index = buildIndex(res.citations);
        return (
          <article key={res.meta.trace_id} className="answer">
            <h3>{res.question}</h3>
            <section className="card">
              <p>{res.result.answer}</p>
              {res.result.claims.length > 0 && (
                <ul className="claims">
                  {res.result.claims.map((c, i) => (
                    <li key={i}>
                      {c.text} <CitationChips ids={c.source_ids} index={index} onOpen={setOpenSource} />
                    </li>
                  ))}
                </ul>
              )}
            </section>
            <CitationList citations={res.citations} onOpen={setOpenSource} />
            <MetaPanel meta={res.meta} />
          </article>
        );
      })}
      {openSource && <SourceDrawer sourceId={openSource} onClose={() => setOpenSource(null)} />}
    </>
  );
}
