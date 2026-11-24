import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AskResponse, Citation } from "../api/types";
import { buildIndex, ClaimList } from "../components/Citations";
import { UserProvider } from "../context/UserContext";
import { AskPage } from "../pages/AskPage";

const citations: Citation[] = [
  { source_id: "OPP-0011", record_id: "OPP-0011", source_type: "opportunity", title: "Velmora deal", snippet: "..." },
  { source_id: "ACT-0003", record_id: "ACT-0003", source_type: "activity", title: "Competitive note", snippet: "..." },
];

const askResponse: AskResponse = {
  question: "Which competitor is Velmora Logistics evaluating?",
  result: {
    answer: "Velmora Logistics is also evaluating Kordon Cloud.",
    claims: [{ text: "Velmora Logistics is also evaluating Kordon Cloud.", source_ids: ["ACT-0003"] }],
  },
  citations: [citations[1]],
  meta: {
    trace_id: "4bf92f3577b34da6a3ce929d0e0e4736",
    provider: "offline",
    model: "deterministic-extractive-v1",
    fallback_reason: null,
    stages_ms: { plan: 1, tools: 1, retrieval: 3, guardrails: 1, llm: 1 },
    tokens: { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120, estimated: true },
    redactions: 2,
    guardrail_events: [{ type: "prompt_injection", source_id: "ACT-0046", detail: "evidence quarantined" }],
    groundedness: 1,
    ungrounded_claims: [],
    retrieved_ids: ["ACT-0003", "ACT-0046"],
    tool_calls: [{ name: "crm_lookup", args: { account_id: "ACC-001" }, result_ids: ["ACC-001"], ms: 0.5 }],
    access_scope: { role: "AE", territories: ["APAC-West"], record_types: ["account"], max_sensitivity: "confidential" },
  },
};

function json(body: unknown) {
  return Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } }));
}

afterEach(() => vi.unstubAllGlobals());

describe("ClaimList", () => {
  it("numbers citation chips by source order and opens the clicked source", async () => {
    const onOpen = vi.fn();
    render(
      <ClaimList
        title="Key facts"
        claims={[{ text: "Competitor is Kordon Cloud", source_ids: ["ACT-0003", "OPP-0011"] }]}
        index={buildIndex(citations)}
        onOpen={onOpen}
      />,
    );
    expect(screen.getByText("Key facts")).toBeInTheDocument();
    const chips = screen.getAllByRole("button");
    expect(chips.map((c) => c.textContent)).toEqual(["2", "1"]);
    await userEvent.click(chips[0]);
    expect(onOpen).toHaveBeenCalledWith("ACT-0003");
  });

  it("shows an empty state when nothing is accessible", () => {
    render(<ClaimList title="Risks" claims={[]} index={new Map()} onOpen={() => {}} />);
    expect(screen.getByText(/Nothing found in the records you can access/)).toBeInTheDocument();
  });
});

describe("AskPage", () => {
  it("sends the question with the acting user's id and renders the cited answer", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, _init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/users")) return json([{ id: "u-ae-01", name: "Ravi Menon", role: "AE", territories: ["APAC-West"] }]);
      if (url.endsWith("/api/accounts")) return json([]);
      if (url.endsWith("/api/ask")) return json(askResponse);
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <MemoryRouter initialEntries={["/ask"]}>
        <UserProvider>
          <AskPage />
        </UserProvider>
      </MemoryRouter>,
    );

    await userEvent.type(screen.getByLabelText("Question"), "Which competitor is Velmora Logistics evaluating?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));

    await waitFor(() => expect(screen.getAllByText(/evaluating Kordon Cloud/).length).toBeGreaterThan(0));
    const askCall = fetchMock.mock.calls.find(([u]) => String(u).endsWith("/api/ask"))!;
    const init = askCall[1] as RequestInit;
    expect((init.headers as Record<string, string>)["X-User-Id"]).toBe("u-ae-01");
    expect(JSON.parse(String(init.body))).toEqual({
      question: "Which competitor is Velmora Logistics evaluating?",
      account_id: null,
    });
    expect(screen.getByText("Competitive note")).toBeInTheDocument();
    expect(screen.getByText(/1 injection blocked/)).toBeInTheDocument();
  });
});
