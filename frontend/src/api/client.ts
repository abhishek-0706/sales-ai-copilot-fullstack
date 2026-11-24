import type {
  AccountListItem, AccountSummary, AskResponse, BriefResponse, Me, MeetingPrepResponse,
  NextBestActionsResponse, SourceRecord, User,
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string, public traceId?: string) {
    super(message);
  }
}

interface Options {
  method?: "GET" | "POST";
  body?: unknown;
  signal?: AbortSignal;
}

async function request<T>(path: string, userId: string | null, { method = "GET", body, signal }: Options = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (userId) headers["X-User-Id"] = userId;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(`/api${path}`, {
    method, headers, signal, body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") detail = data.detail;
    } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, detail || `HTTP ${res.status}`, res.headers.get("X-Trace-Id") ?? undefined);
  }
  return (await res.json()) as T;
}

export const api = {
  users: (signal?: AbortSignal) => request<User[]>("/users", null, { signal }),
  me: (u: string, signal?: AbortSignal) => request<Me>("/me", u, { signal }),
  accounts: (u: string, signal?: AbortSignal) => request<AccountListItem[]>("/accounts", u, { signal }),
  account: (u: string, id: string, signal?: AbortSignal) => request<AccountSummary>(`/accounts/${id}`, u, { signal }),
  brief: (u: string, id: string, signal?: AbortSignal) =>
    request<BriefResponse>(`/accounts/${id}/brief`, u, { method: "POST", signal }),
  meetingPrep: (u: string, id: string, signal?: AbortSignal) =>
    request<MeetingPrepResponse>(`/accounts/${id}/meeting-prep`, u, { method: "POST", body: {}, signal }),
  nextBestActions: (u: string, id: string, signal?: AbortSignal) =>
    request<NextBestActionsResponse>(`/accounts/${id}/next-best-actions`, u, { method: "POST", signal }),
  ask: (u: string, question: string, accountId?: string, signal?: AbortSignal) =>
    request<AskResponse>("/ask", u, { method: "POST", body: { question, account_id: accountId || null }, signal }),
  source: (u: string, id: string, signal?: AbortSignal) =>
    request<SourceRecord>(`/sources/${encodeURIComponent(id)}`, u, { signal }),
};
