// Types and helpers for the take-up (promise -> bill) forecast endpoints
// (backend/src/westminster_weather/takeup/api.py).

export type Verdict = "likely" | "unlikely" | "uncertain" | string;

export interface LinkedBill {
  bill_id: number;
  title: string | null;
  first_reading_date: string;
}

export interface IdeaSummary {
  signal_id: string;
  idea_id: string;
  topic: string | null;
  trigger_date: string;
  first_source: string | null;
  taken_up: boolean;
  verdict: Verdict;
  pi: number;
  pi_now: number;
  p26: number;
  p52: number;
  weeks_elapsed: number;
  t_50: number | null;
  t_50_date: string | null;
  linked_bill: LinkedBill | null;
  forecast_source: string;
}

export interface IdeasResponse {
  as_of: string;
  run_id: number | null;
  count: number;
  ideas: IdeaSummary[];
}

export interface IdeaEvent {
  event_type: string;
  event_date: string;
  title: string | null;
  source_url: string | null;
}

export interface Mention {
  mention_id: string;
  group_id: string | null;
  date: string;
  source: string;
  firmness: number | null;
  sentence: string | null;
  title: string | null;
}

export interface ForecastResponse {
  as_of: string;
  run_id: number | null;
  idea_id: string;
  signal_id: string;
  signal_ids: string[];
  topic: string | null;
  trigger_date: string;
  first_source: string | null;
  taken_up: boolean;
  pi: number;
  pi_lo: number;
  pi_hi: number;
  verdict: Verdict;
  weeks_elapsed: number;
  pi_now: number;
  p26: number;
  p52: number;
  t_10: number | null;
  t_50: number | null;
  t_90: number | null;
  t_10_date: string | null;
  t_50_date: string | null;
  t_90_date: string | null;
  t_10_sitting: number | null;
  t_50_sitting: number | null;
  t_90_sitting: number | null;
  events: IdeaEvent[];
  mentions: Mention[];
  linked_bill: LinkedBill | null;
  forecast_source: string;
}

export interface HistoryPoint {
  week: number;
  date: string;
  pi_now: number;
  p52: number;
}

export interface HistoryMarker {
  week: number;
  date: string;
  kind: string;
  label: string | null;
}

export interface HistoryResponse {
  as_of: string;
  run_id: number | null;
  idea_id: string;
  signal_id: string;
  trigger_date: string;
  taken_up: boolean;
  points: HistoryPoint[];
  markers: HistoryMarker[];
}

/** Error carrying the HTTP status so pages can special-case 503 / 404. */
export class TakeupApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export async function fetchTakeup<T>(
  path: string,
  params: Record<string, string | undefined>,
  signal?: AbortSignal
): Promise<T> {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v) qs.set(k, v);
  }
  const res = await fetch(`/api/v1/takeup/${path}?${qs.toString()}`, { signal });
  if (!res.ok) {
    let detail = `API returned status ${res.status}`;
    try {
      const body = await res.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      // non-JSON error body
    }
    throw new TakeupApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export function errorMessage(err: unknown): string {
  if (err instanceof TakeupApiError) {
    if (err.status === 503) {
      return "Take-up data not loaded yet: the take-up database is not configured on the backend.";
    }
    if (err.status === 404) {
      return `Promise not found at this date (${err.message}).`;
    }
    return err.message;
  }
  return err instanceof Error ? err.message : "Failed to load take-up data";
}

export function todayIso(): string {
  const d = new Date();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${mm}-${dd}`;
}

function parseIso(iso: string): Date {
  return new Date(`${iso.slice(0, 10)}T00:00:00Z`);
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return parseIso(iso).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function fmtMonth(iso: string | null | undefined): string {
  if (!iso) return "—";
  return parseIso(iso).toLocaleDateString("en-GB", {
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function pct(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return "—";
  const v = p * 100;
  if (v > 0 && v < 1) return "<1%";
  if (v < 100 && v > 99) return ">99%";
  return `${Math.round(v)}%`;
}

export const VERDICT_BADGES: Record<string, { label: string; className: string }> = {
  likely: {
    label: "Likely",
    className: "bg-emerald-100 text-emerald-800 border-emerald-200",
  },
  unlikely: {
    label: "Unlikely",
    className: "bg-red-100 text-red-800 border-red-200",
  },
  uncertain: {
    label: "Uncertain",
    className: "bg-slate-100 text-slate-700 border-slate-200",
  },
};

export function verdictBadge(v: string): { label: string; className: string } {
  return (
    VERDICT_BADGES[v] ?? {
      label: v,
      className: "bg-slate-100 text-slate-700 border-slate-200",
    }
  );
}

export function humanSource(s: string | null | undefined): string {
  if (!s) return "—";
  return s.replace(/_/g, " ");
}
