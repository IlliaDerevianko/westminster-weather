"use client";

import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import React, { Suspense, useEffect, useState } from "react";

import {
  type ForecastResponse,
  type HistoryMarker,
  type HistoryResponse,
  TakeupApiError,
  errorMessage,
  fetchTakeup,
  fmtDate,
  fmtMonth,
  humanSource,
  pct,
  todayIso,
  verdictBadge,
} from "@/lib/takeup";

const MARKER_COLORS: Record<string, string> = {
  mention: "#2563eb",
  first_reading: "#059669",
};
const EVENT_COLOR = "#d97706";

function markerColor(kind: string): string {
  return MARKER_COLORS[kind] ?? EVENT_COLOR;
}

function headline(f: ForecastResponse): string {
  if (f.taken_up) {
    const b = f.linked_bill;
    return b
      ? `Bill introduced ${fmtDate(b.first_reading_date)}: ${b.title ?? `Bill #${b.bill_id}`}`
      : "Bill introduced";
  }
  const chance = `${pct(f.pi_now)} chance a bill is introduced`;
  if (!f.t_50_date) return `${chance}; no likely date within the model horizon`;
  const window =
    f.t_10_date && f.t_90_date
      ? ` (80%: ${fmtMonth(f.t_10_date)} – ${fmtMonth(f.t_90_date)})`
      : "";
  return `${chance}; most likely around ${fmtMonth(f.t_50_date)}${window}`;
}

function TakeupChart({
  history,
  forecast,
}: {
  history: HistoryResponse;
  forecast: ForecastResponse;
}) {
  const W = 800;
  const H = 300;
  const m = { top: 16, right: 20, bottom: 56, left: 44 };
  const iw = W - m.left - m.right;
  const ih = H - m.top - m.bottom;

  const pts = history.points;
  const lastWeek = pts.length ? pts[pts.length - 1].week : 0;
  const showWindow =
    !forecast.taken_up && forecast.t_10 !== null && forecast.t_90 !== null;
  const maxWeek = Math.max(
    lastWeek,
    showWindow ? (forecast.t_90 as number) : 0,
    ...history.markers.map((mk) => mk.week),
    4
  );
  const x = (w: number) => m.left + (w / maxWeek) * iw;
  const y = (p: number) => m.top + (1 - Math.max(0, Math.min(1, p))) * ih;

  const linePath = pts
    .map((p, k) => `${k === 0 ? "M" : "L"}${x(p.week).toFixed(1)},${y(p.pi_now).toFixed(1)}`)
    .join(" ");
  const p52Path = pts
    .map((p, k) => `${k === 0 ? "M" : "L"}${x(p.week).toFixed(1)},${y(p.p52).toFixed(1)}`)
    .join(" ");

  // ~6 x ticks in whole weeks
  const step = Math.max(1, Math.ceil(maxWeek / 6 / 4) * 4);
  const xTicks: number[] = [];
  for (let w = 0; w <= maxWeek; w += step) xTicks.push(w);

  // Stack markers that share a week so they stay visible
  const stackCount: Record<number, number> = {};
  const placed = history.markers.map((mk: HistoryMarker) => {
    const n = stackCount[mk.week] ?? 0;
    stackCount[mk.week] = n + 1;
    return { mk, n };
  });

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full h-auto"
      role="img"
      aria-label="Chance a bill is introduced, by week since the promise"
    >
      {showWindow && (
        <g>
          <rect
            x={x(forecast.t_10 as number)}
            y={m.top}
            width={Math.max(1, x(forecast.t_90 as number) - x(forecast.t_10 as number))}
            height={ih}
            fill="#10b981"
            fillOpacity={0.1}
          >
            <title>
              80% timing window: {fmtMonth(forecast.t_10_date)} –{" "}
              {fmtMonth(forecast.t_90_date)}
            </title>
          </rect>
          {forecast.t_50 !== null && (
            <line
              x1={x(forecast.t_50)}
              x2={x(forecast.t_50)}
              y1={m.top}
              y2={m.top + ih}
              stroke="#059669"
              strokeDasharray="4 3"
              strokeWidth={1}
            >
              <title>Most likely: {fmtMonth(forecast.t_50_date)}</title>
            </line>
          )}
          <text
            x={x(forecast.t_10 as number) + 4}
            y={m.top + 12}
            fontSize={11}
            fill="#047857"
          >
            80% window
          </text>
        </g>
      )}

      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line
            x1={m.left}
            x2={m.left + iw}
            y1={y(t)}
            y2={y(t)}
            stroke="#e2e8f0"
            strokeWidth={1}
          />
          <text
            x={m.left - 6}
            y={y(t) + 4}
            fontSize={11}
            fill="#64748b"
            textAnchor="end"
          >
            {Math.round(t * 100)}%
          </text>
        </g>
      ))}

      {xTicks.map((w) => (
        <g key={w}>
          <line
            x1={x(w)}
            x2={x(w)}
            y1={m.top + ih}
            y2={m.top + ih + 4}
            stroke="#94a3b8"
          />
          <text
            x={x(w)}
            y={m.top + ih + 16}
            fontSize={11}
            fill="#64748b"
            textAnchor="middle"
          >
            {w}
          </text>
        </g>
      ))}
      <text
        x={m.left + iw / 2}
        y={H - 6}
        fontSize={11}
        fill="#64748b"
        textAnchor="middle"
      >
        Weeks since the promise ({fmtDate(history.trigger_date)})
      </text>

      {/* as_of line */}
      {!history.taken_up && (
        <line
          x1={x(lastWeek)}
          x2={x(lastWeek)}
          y1={m.top}
          y2={m.top + ih}
          stroke="#94a3b8"
          strokeDasharray="2 3"
        >
          <title>Today (as of {fmtDate(history.as_of)})</title>
        </line>
      )}

      {pts.length > 1 && (
        <path d={p52Path} fill="none" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="5 4" />
      )}
      {pts.length > 1 && (
        <path d={linePath} fill="none" stroke="#0f172a" strokeWidth={2.25} />
      )}
      {pts.length > 0 && (
        <circle
          cx={x(lastWeek)}
          cy={y(pts[pts.length - 1].pi_now)}
          r={4}
          fill="#0f172a"
        >
          <title>
            Week {lastWeek}: {pct(pts[pts.length - 1].pi_now)}
          </title>
        </circle>
      )}

      {/* Markers along the bottom of the plot */}
      {placed.map(({ mk, n }, k) => {
        const cx = x(mk.week);
        const cy = m.top + ih + 28 + n * 7;
        const color = markerColor(mk.kind);
        const tip = `${fmtDate(mk.date)} · ${humanSource(mk.kind)}${mk.label ? ` · ${mk.label}` : ""}`;
        return (
          <g key={k}>
            {mk.kind === "first_reading" ? (
              <line
                x1={cx}
                x2={cx}
                y1={m.top}
                y2={m.top + ih}
                stroke={color}
                strokeWidth={2}
              >
                <title>{tip}</title>
              </line>
            ) : null}
            {mk.kind === "mention" ? (
              <circle cx={cx} cy={cy} r={3.5} fill={color}>
                <title>{tip}</title>
              </circle>
            ) : (
              <rect x={cx - 3.5} y={cy - 3.5} width={7} height={7} fill={color}>
                <title>{tip}</title>
              </rect>
            )}
          </g>
        );
      })}
    </svg>
  );
}

function PromiseDetailContent() {
  const params = useParams<{ ideaId: string }>();
  const searchParams = useSearchParams();
  const asOf = searchParams.get("as_of") || todayIso();
  let ideaId = params.ideaId ?? "";
  try {
    ideaId = decodeURIComponent(ideaId);
  } catch {
    // keep raw value
  }

  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [notLoaded, setNotLoaded] = useState<boolean>(false);

  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    const q = { idea_id: ideaId, as_of: asOf };
    Promise.all([
      fetchTakeup<ForecastResponse>("forecast", q, ctrl.signal),
      fetchTakeup<HistoryResponse>("history", q, ctrl.signal),
    ])
      .then(([f, h]) => {
        setForecast(f);
        setHistory(h);
        setError(null);
        setNotLoaded(false);
      })
      .catch((err) => {
        if (ctrl.signal.aborted) return;
        setForecast(null);
        setHistory(null);
        setNotLoaded(err instanceof TakeupApiError && err.status === 503);
        setError(errorMessage(err));
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
    return () => ctrl.abort();
  }, [ideaId, asOf]);

  const backHref = searchParams.get("as_of")
    ? `/promises?as_of=${asOf}`
    : "/promises";
  const badge = forecast ? verdictBadge(forecast.verdict) : null;

  return (
    <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">
      <div>
        <Link
          href={backHref}
          className="inline-flex items-center rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-800 hover:bg-slate-100 transition-colors"
        >
          ← All promises
        </Link>
      </div>

      {loading && (
        <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-blue-600 shadow-sm">
          Loading take-up forecast...
        </div>
      )}

      {error && (
        <div
          className={`rounded-lg border p-4 text-sm ${
            notLoaded
              ? "border-amber-300 bg-amber-50 text-amber-900"
              : "border-red-300 bg-red-50 text-red-800"
          }`}
        >
          <strong>{notLoaded ? "Not available:" : "Error:"}</strong> {error}
        </div>
      )}

      {forecast && badge && (
        <>
          <header className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
            <div className="flex flex-wrap items-center gap-2 text-xs font-semibold uppercase tracking-wider">
              <span className="rounded-full bg-blue-100 px-2.5 py-0.5 text-blue-800">
                Promise → Bill
              </span>
              <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-slate-700">
                As of {fmtDate(forecast.as_of)}
              </span>
              {forecast.taken_up ? (
                <span className="rounded-full border border-blue-200 bg-blue-100 px-2.5 py-0.5 text-blue-800 normal-case">
                  Bill introduced
                </span>
              ) : (
                <span
                  className={`rounded-full border px-2.5 py-0.5 normal-case ${badge.className}`}
                >
                  {badge.label}
                </span>
              )}
            </div>
            <h1 className="text-2xl font-bold text-slate-900">
              {forecast.topic ?? forecast.idea_id}
            </h1>
            <p className="text-lg font-semibold text-slate-800">
              {headline(forecast)}
            </p>
            {forecast.taken_up && forecast.linked_bill && (
              <Link
                href={`/?bill_id=${forecast.linked_bill.bill_id}`}
                className="inline-block text-sm font-semibold text-blue-700 hover:underline"
              >
                View bill on the tracker →
              </Link>
            )}

            {!forecast.taken_up && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="text-xs font-medium text-slate-500">
                    Chance now
                  </div>
                  <div className="text-xl font-bold text-slate-900">
                    {pct(forecast.pi_now)}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="text-xs font-medium text-slate-500">
                    At the promise (range)
                  </div>
                  <div className="text-xl font-bold text-slate-900">
                    {pct(forecast.pi)}{" "}
                    <span className="text-sm font-medium text-slate-500">
                      ({pct(forecast.pi_lo)}–{pct(forecast.pi_hi)})
                    </span>
                  </div>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="text-xs font-medium text-slate-500">
                    Within 26 / 52 weeks
                  </div>
                  <div className="text-xl font-bold text-slate-900">
                    {pct(forecast.p26)} / {pct(forecast.p52)}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="text-xs font-medium text-slate-500">
                    Weeks since promise
                  </div>
                  <div className="text-xl font-bold text-slate-900">
                    {forecast.weeks_elapsed}
                  </div>
                </div>
              </div>
            )}
            <p className="text-xs text-slate-500">
              First promised {fmtDate(forecast.trigger_date)}
              {forecast.first_source
                ? ` (${humanSource(forecast.first_source)})`
                : ""}
              . &ldquo;Chance now&rdquo; is the chance the commitment ever
              becomes a bill (first reading), given the weeks that have already
              passed without one.
            </p>
          </header>

          {history && (
            <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-3">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-600">
                Chance over time
              </h2>
              {history.points.length === 0 ? (
                <p className="text-sm text-slate-500">
                  No weekly history yet for this promise.
                </p>
              ) : (
                <TakeupChart history={history} forecast={forecast} />
              )}
              <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-600">
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-0.5 w-5 bg-slate-900" />
                  Chance now
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block w-5 border-t-2 border-dashed border-slate-400" />
                  Chance within 52 weeks
                </span>
                {!forecast.taken_up && forecast.t_10 !== null && (
                  <span className="inline-flex items-center gap-1.5">
                    <span className="inline-block h-3 w-5 bg-emerald-500/20" />
                    80% timing window
                  </span>
                )}
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-2 w-2 rounded-full bg-blue-600" />
                  Minister mention
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-2 w-2 bg-amber-600" />
                  Event
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-3 w-0.5 bg-emerald-600" />
                  First reading
                </span>
              </div>
            </section>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <section className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
              <div className="border-b border-slate-200 px-6 py-3 bg-slate-50 text-xs font-semibold uppercase tracking-wider text-slate-600">
                Minister quotes ({forecast.mentions.length})
              </div>
              <div className="divide-y divide-slate-200">
                {forecast.mentions.map((mn) => (
                  <div key={mn.mention_id} className="px-6 py-3 space-y-1">
                    <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                      <span className="font-semibold text-slate-700">
                        {fmtDate(mn.date)}
                      </span>
                      <span className="rounded bg-slate-100 px-2 py-0.5 capitalize">
                        {humanSource(mn.source)}
                      </span>
                      {mn.firmness !== null && (
                        <span className="rounded bg-slate-100 px-2 py-0.5">
                          Firmness {mn.firmness}
                        </span>
                      )}
                    </div>
                    {mn.sentence && (
                      <blockquote className="border-l-2 border-blue-300 pl-3 text-sm text-slate-800 italic">
                        &ldquo;{mn.sentence}&rdquo;
                      </blockquote>
                    )}
                    {mn.title && (
                      <div className="text-xs text-slate-500">{mn.title}</div>
                    )}
                  </div>
                ))}
                {forecast.mentions.length === 0 && (
                  <div className="p-6 text-center text-sm text-slate-500">
                    No minister quotes recorded.
                  </div>
                )}
              </div>
            </section>

            <section className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
              <div className="border-b border-slate-200 px-6 py-3 bg-slate-50 text-xs font-semibold uppercase tracking-wider text-slate-600">
                Events ({forecast.events.length})
              </div>
              <div className="divide-y divide-slate-200">
                {forecast.events.map((ev, k) => (
                  <div key={k} className="px-6 py-3 space-y-1">
                    <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                      <span className="font-semibold text-slate-700">
                        {fmtDate(ev.event_date)}
                      </span>
                      <span className="rounded bg-amber-50 border border-amber-200 px-2 py-0.5 text-amber-900 capitalize">
                        {humanSource(ev.event_type)}
                      </span>
                    </div>
                    <div className="text-sm text-slate-800">
                      {ev.source_url ? (
                        <a
                          href={ev.source_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-700 hover:underline"
                        >
                          {ev.title ?? ev.source_url}
                        </a>
                      ) : (
                        ev.title ?? "—"
                      )}
                    </div>
                  </div>
                ))}
                {forecast.linked_bill && (
                  <div className="px-6 py-3 space-y-1">
                    <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                      <span className="font-semibold text-slate-700">
                        {fmtDate(forecast.linked_bill.first_reading_date)}
                      </span>
                      <span className="rounded bg-emerald-50 border border-emerald-200 px-2 py-0.5 text-emerald-900">
                        First reading
                      </span>
                    </div>
                    <Link
                      href={`/?bill_id=${forecast.linked_bill.bill_id}`}
                      className="text-sm text-blue-700 hover:underline"
                    >
                      {forecast.linked_bill.title ??
                        `Bill #${forecast.linked_bill.bill_id}`}
                    </Link>
                  </div>
                )}
                {forecast.events.length === 0 && !forecast.linked_bill && (
                  <div className="p-6 text-center text-sm text-slate-500">
                    No events recorded.
                  </div>
                )}
              </div>
            </section>
          </div>

          {forecast.signal_ids.length > 1 && (
            <p className="text-xs text-slate-500">
              This idea currently has {forecast.signal_ids.length} separate
              topic groups:{" "}
              {forecast.signal_ids.map((sid, k) => (
                <React.Fragment key={sid}>
                  {k > 0 && ", "}
                  <Link
                    href={`/promises/${encodeURIComponent(sid)}${
                      searchParams.get("as_of") ? `?as_of=${asOf}` : ""
                    }`}
                    className="text-blue-700 hover:underline"
                  >
                    {sid}
                  </Link>
                </React.Fragment>
              ))}
              .
            </p>
          )}
        </>
      )}
    </main>
  );
}

export default function PromiseDetailPage() {
  return (
    <Suspense
      fallback={
        <main className="max-w-6xl mx-auto px-6 py-8 text-sm text-slate-600">
          Loading take-up forecast...
        </main>
      }
    >
      <PromiseDetailContent />
    </Suspense>
  );
}
