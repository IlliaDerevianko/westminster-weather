// Types and helpers for GET /api/v1/bills/{bill_id}/outlook
// (backend/src/westminster_weather/schemas/outlook.py).

export type WeatherStatus = "sunny" | "partly_cloudy" | "rain" | "storm";

export interface OutlookDeadline {
  kind: string;
  label: string;
  date: string;
  estimated: boolean;
  p: number;
  sentence: string;
}

export interface OutlookWeather {
  status: WeatherStatus;
  label: string;
  target_date: string;
  target_label: string;
  target_estimated: boolean;
  p_by_target: number;
  p_fail: number;
  headline: string;
}

export interface CurvePoint {
  date: string;
  p_law_by_date: number;
}

export interface OutlookCurve {
  points: CurvePoint[];
  month_50: string | null;
  month_90: string | null;
  sentence: string;
}

export interface OutlookPace {
  current_stage: string;
  current_stage_name: string;
  sitting_days_so_far: number;
  typical_sitting_days: number | null;
  weeks_ahead_or_behind: number | null;
  pace_percentile: number | null;
  n_comparison_bills: number;
  sentence: string;
}

export interface OutlookNextMilestone {
  stage: string;
  stage_name: string;
  date: string;
  date_kind: "scheduled" | "expected";
  p_by_target_without: number | null;
  p_by_target_with: number | null;
  sentence: string;
  what_if_sentence: string | null;
}

export interface BillOutlook {
  bill_id: number;
  short_title: string;
  generated_at: string;
  forecast_date: string;
  model_version: string;
  recalibrated: boolean;
  current_stage: string;
  current_stage_name: string;
  distribution: {
    method: string;
    n_paths: number;
    p_royal_assent: number;
    p_fail: number;
    royal_assent_date_quantiles: Record<string, string | null>;
  };
  deadlines: OutlookDeadline[];
  weather: OutlookWeather;
  curve: OutlookCurve;
  pace: OutlookPace;
  next_milestone: OutlookNextMilestone | null;
  sentences: string[];
}

export const WEATHER_ICON: Record<WeatherStatus, string> = {
  sunny: "☀️",
  partly_cloudy: "⛅",
  rain: "🌧️",
  storm: "⛈️",
};

export const WEATHER_STYLE: Record<WeatherStatus, string> = {
  sunny: "border-emerald-300 bg-emerald-50 text-emerald-900",
  partly_cloudy: "border-amber-300 bg-amber-50 text-amber-900",
  rain: "border-sky-300 bg-sky-50 text-sky-900",
  storm: "border-red-300 bg-red-50 text-red-900",
};

export const pct = (p: number) => `${Math.round(p * 100)}%`;

export async function fetchOutlook(
  billId: string,
  asOf?: string,
  targetDate?: string
): Promise<BillOutlook | null> {
  const params = new URLSearchParams();
  if (asOf) params.set("as_of", asOf);
  if (targetDate) params.set("target_date", targetDate);
  const query = params.toString() ? `?${params}` : "";
  const res = await fetch(`/api/v1/bills/${encodeURIComponent(billId)}/outlook${query}`);
  if (res.status === 404) return null; // not in Parliament on that date
  if (!res.ok) throw new Error(`Outlook API returned status ${res.status}`);
  return res.json();
}
