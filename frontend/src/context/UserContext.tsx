import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { User } from "../api/types";

interface UserState {
  users: User[];
  userId: string;
  user?: User;
  setUserId: (id: string) => void;
}

const KEY = "sales-ai-user";
const DEFAULT_USER = "u-ae-01";
const UserContext = createContext<UserState | null>(null);

function readStored(): string {
  try {
    return localStorage.getItem(KEY) ?? DEFAULT_USER;
  } catch {
    return DEFAULT_USER;
  }
}

export function UserProvider({ children }: { children: ReactNode }) {
  const [users, setUsers] = useState<User[]>([]);
  const [userId, setUserIdState] = useState<string>(readStored);

  useEffect(() => {
    const ctrl = new AbortController();
    api.users(ctrl.signal).then(setUsers).catch(() => setUsers([]));
    return () => ctrl.abort();
  }, []);

  const setUserId = (id: string) => {
    setUserIdState(id);
    try {
      localStorage.setItem(KEY, id);
    } catch { /* storage unavailable: keep in memory only */ }
  };

  return (
    <UserContext.Provider value={{ users, userId, user: users.find((u) => u.id === userId), setUserId }}>
      {children}
    </UserContext.Provider>
  );
}

export function useUser(): UserState {
  const ctx = useContext(UserContext);
  if (!ctx) throw new Error("useUser must be used inside <UserProvider>");
  return ctx;
}
