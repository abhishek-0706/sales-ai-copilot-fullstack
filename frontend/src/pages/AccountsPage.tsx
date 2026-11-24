import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useUser } from "../context/UserContext";
import { useApi } from "../hooks/useApi";

export function AccountsPage() {
  const { userId } = useUser();
  const accounts = useApi((s) => api.accounts(userId, s), [userId]);
  const me = useApi((s) => api.me(userId, s), [userId]);

  return (
    <>
      <h2>Accounts</h2>
      {me.data && (
        <p className="muted">
          Signed in as {me.data.user.name} ({me.data.user.role}). You can see {me.data.scope.territories.join(", ")}{" "}
          records up to <strong>{me.data.scope.max_sensitivity}</strong> sensitivity. Filtering happens inside
          retrieval, so the assistant never sees anything outside this scope.
        </p>
      )}
      {accounts.loading && <p className="muted">Loading accounts...</p>}
      {accounts.error && <p className="error">{accounts.error.message}</p>}
      {accounts.data && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Account</th>
                <th>Industry</th>
                <th>Territory</th>
                <th>Stage</th>
                <th>Next meeting</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {accounts.data.map((a) => (
                <tr key={a.id}>
                  <td>
                    <Link to={`/accounts/${a.id}`}>{a.name}</Link>
                    <div className="muted small">{a.use_case}</div>
                  </td>
                  <td>{a.industry}</td>
                  <td>{a.territory}</td>
                  <td>
                    {a.opportunity_stage ?? (
                      <span className="muted" title="Opportunity data is not visible to your role">hidden</span>
                    )}
                  </td>
                  <td>{a.next_meeting ? a.next_meeting.slice(0, 16).replace("T", " ") : "-"}</td>
                  <td className="actions">
                    <Link to={`/accounts/${a.id}/meeting`}>Meeting prep</Link>
                    <Link to={`/ask?account=${a.id}`}>Ask</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
