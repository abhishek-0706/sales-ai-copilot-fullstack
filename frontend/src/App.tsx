import { NavLink, Route, Routes } from "react-router-dom";
import { useUser } from "./context/UserContext";
import { AccountBriefPage } from "./pages/AccountBriefPage";
import { AccountsPage } from "./pages/AccountsPage";
import { AskPage } from "./pages/AskPage";
import { MeetingPrepPage } from "./pages/MeetingPrepPage";

export function App() {
  const { users, userId, setUserId } = useUser();
  return (
    <>
      <header className="topbar">
        <h1>Sales AI Copilot</h1>
        <nav>
          <NavLink to="/" end>
            Accounts
          </NavLink>
          <NavLink to="/ask">Ask</NavLink>
        </nav>
        <label className="user-switch">
          <span>Acting as</span>
          <select value={userId} onChange={(e) => setUserId(e.target.value)} aria-label="Acting as user">
            {users.map((u) => (
              <option key={u.id} value={u.id}>
                {u.name} · {u.role} · {u.territories.join(", ")}
              </option>
            ))}
          </select>
        </label>
      </header>
      <div className="notice">Synthetic demo data: every company, person, email and product here is fictional.</div>
      <main>
        <Routes>
          <Route path="/" element={<AccountsPage />} />
          <Route path="/accounts/:accountId" element={<AccountBriefPage />} />
          <Route path="/accounts/:accountId/meeting" element={<MeetingPrepPage />} />
          <Route path="/ask" element={<AskPage />} />
          <Route path="*" element={<p>Page not found.</p>} />
        </Routes>
      </main>
    </>
  );
}
