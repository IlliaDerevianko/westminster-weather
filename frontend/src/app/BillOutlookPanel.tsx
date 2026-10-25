"use client";

import React, { useEffect, useState } from "react";
import {
  BillOutlook,
  CurvePoint,
  WEATHER_ICON,
  WEATHER_STYLE,
  fetchOutlook,
  pct,
} from "@/lib/outlook";

// Weather outlook for a bill in Parliament (GET /api/v1/bills/{id}/outlook).
// Every sentence comes from the API ready to show; this component only lays them out.

const fmtDate = (iso: string) =>
  new Date(`${iso}T00:00:00`).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });

function Curve({
  points,
  targetDate,
}: {
  points: CurvePoint[];
  targetDate: string;
}) {
  const W = 640;
  const H = 180;
  const pad = { l: 36, r: 12, t: 10, b: 24 };
  const t0 = new Date(points[0].date).getTime();
  const t1 = new Date(points[points.length - 1].date).getTime();
  const x = (d: string) =>
    pad.l + ((new Date(d).getTime() - t0) / (t1 - t0)) * (W - pad.l - pad.r);
  const y = (p: number) => pad.t + (1 - p) * (H - pad.t - pad.b);
  const path = points
    .map((p, i) => `${i ? "L" : "M"}${x(p.date).toFixed(1)},${y(p.p_law_by_date).toFixed(1)}`)
    .join(" ");
  const years = Array.from(
    new Set(points.map((p) => p.date.slice(0, 4)))
  ).filter((yr) => new Date(`${yr}-01-01`).getTime() > t0);
  const tx = x(targetDate);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Chance of becoming law by date">
      {[0, 0.5, 0.9, 1].map((g) => (
        <g key={g}>
          <line x1={pad.l} x2={W - pad.r} y1={y(g)} y2={y(g)} stroke="#e2e8f0" strokeDasharray={g === 0.5 || g === 0.9 ? "4 4" : undefined} />
          <text x={pad.l - 6} y={y(g) + 4} textAnchor="end" fontSize="10" fill="#64748b">
            {pct(g)}
          </text>
        </g>
      ))}
      {years.map((yr) => (
        <g key={yr}>
          <line x1={x(`${yr}-01-01`)} x2={x(`${yr}-01-01`)} y1={pad.t} y2={H - pad.b} stroke="#f1f5f9" />
          <text x={x(`${yr}-01-01`)} y={H - 6} textAnchor="middle" fontSize="10" fill="#64748b">
            {yr}
          </text>
        </g>
      ))}
      {tx >= pad.l && tx <= W - pad.r && (
        <g>
          <line x1={tx} x2={tx} y1={pad.t} y2={H - pad.b} stroke="#f59e0b" strokeWidth="1.5" />
          <text x={tx + 4} y={pad.t + 10} fontSize="10" fill="#b45309">
            target
          </text>
        </g>
      )}
      <path d={path} fill="none" stroke="#2563eb" strokeWidth="2.5" />
    </svg>
  );
}

export default function BillOutlookPanel({
  billId,
  asOf,
}: {
  billId: string;
  asOf?: string;
}) {
  const [outlook, setOutlook] = useState<BillOutlook | null>(null);
  const [targetDate, setTargetDate] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchOutlook(billId, asOf, targetDate || undefined)
      .then((o) => {
        if (!cancelled) {
          setOutlook(o);
          setError(null);
        }
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [billId, asOf, targetDate]);

  useEffect(() => setTargetDate(""), [billId]);

  if (!loading && !error && outlook === null) return null; // not in Parliament on that date

  return (
    <section className="mb-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-wider text-slate-500">
            Parliamentary weather outlook
            {outlook && ` · as of ${fmtDate(outlook.forecast_date)}`}
          </p>
          {loading && !outlook && (
            <p className="mt-2 text-sm text-slate-500">Running 10,000 simulations…</p>
          )}
          {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
        </div>
        <label className="flex items-center gap-2 text-xs font-semibold text-slate-600">
          Target date
          <input
            type="date"
            value={targetDate}
            onChange={(e) => setTargetDate(e.target.value)}
            className="rounded-lg border border-slate-300 px-2 py-1 text-xs"
          />
          {targetDate && (
            <button onClick={() => setTargetDate("")} className="text-slate-500 underline">
              next recess
            </button>
          )}
        </label>
      </div>

      {outlook && (
        <div className={loading ? "opacity-60 transition-opacity" : ""}>
          {/* Weather headline */}
          <div className={`mt-4 flex items-center gap-4 rounded-xl border p-4 ${WEATHER_STYLE[outlook.weather.status]}`}>
            <span className="text-4xl" aria-hidden="true">
              {WEATHER_ICON[outlook.weather.status]}
            </span>
            <div>
              <p className="text-lg font-bold">{outlook.weather.label}</p>
              <p className="text-sm">{outlook.weather.headline}</p>
            </div>
            <div className="ml-auto text-right text-xs">
              <p>
                Chance it ever becomes law:{" "}
                <span className="font-bold">{pct(outlook.distribution.p_royal_assent)}</span>
              </p>
              <p>
                Chance it fails: <span className="font-bold">{pct(outlook.distribution.p_fail)}</span>
              </p>
            </div>
          </div>

          <div className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Curve */}
            <div>
              <h3 className="text-sm font-bold text-slate-800">Chance of becoming law by date</h3>
              <p className="mb-2 text-sm text-slate-600">{outlook.curve.sentence}</p>
              <Curve points={outlook.curve.points} targetDate={outlook.weather.target_date} />
            </div>

            {/* Deadlines */}
            <div>
              <h3 className="text-sm font-bold text-slate-800">Deadlines</h3>
              <ul className="mt-2 space-y-1.5">
                {outlook.deadlines.map((d) => (
                  <li key={d.kind} className="text-sm text-slate-700">
                    <div className="flex items-center gap-3">
                      <span className="w-11 shrink-0 text-right font-bold tabular-nums text-slate-900">
                        {pct(d.p)}
                      </span>
                      <span className="h-1.5 w-24 shrink-0 rounded-full bg-slate-100">
                        <span
                          className="block h-1.5 rounded-full bg-blue-600"
                          style={{ width: `${Math.round(d.p * 100)}%` }}
                        />
                      </span>
                      <span>
                        {d.label} <span className="text-slate-500">({fmtDate(d.date)}{d.estimated ? ", estimated" : ""})</span>
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Pace and next milestone */}
          <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="rounded-xl bg-slate-50 p-4">
              <h3 className="text-sm font-bold text-slate-800">Pace</h3>
              <p className="mt-1 text-sm text-slate-700">{outlook.pace.sentence}</p>
            </div>
            {outlook.next_milestone && (
              <div className="rounded-xl bg-slate-50 p-4">
                <h3 className="text-sm font-bold text-slate-800">Next milestone</h3>
                <p className="mt-1 text-sm text-slate-700">{outlook.next_milestone.sentence}</p>
                {outlook.next_milestone.what_if_sentence && (
                  <p className="mt-1 text-sm text-slate-700">{outlook.next_milestone.what_if_sentence}</p>
                )}
              </div>
            )}
          </div>

          <p className="mt-4 text-xs text-slate-500">
            From {outlook.distribution.n_paths.toLocaleString("en-GB")} simulated paths of the passage
            model ({outlook.model_version})
            {outlook.recalibrated &&
              "; deadline chances recalibrated on 2014–2019 bills (when we said about 70%, it happened 70% of the time on 2019–2026 bills)"}
            . Recess dates marked estimated are typical dates from past years.
          </p>
        </div>
      )}
    </section>
  );
}
