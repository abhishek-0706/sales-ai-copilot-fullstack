import type { components } from "./schema";

type S = components["schemas"];
export type User = S["User"];
export type Me = S["Me"];
export type AccountListItem = S["AccountListItem"];
export type AccountSummary = S["AccountSummary"];
export type BriefResponse = S["BriefResponse"];
export type MeetingPrepResponse = S["MeetingPrepResponse"];
export type NextBestActionsResponse = S["NextBestActionsResponse"];
export type AskResponse = S["AskResponse"];
export type SourceRecord = S["SourceRecord"];
export type Citation = S["Citation"];
export type Claim = S["Claim"];
export type ResponseMeta = S["ResponseMeta"];
