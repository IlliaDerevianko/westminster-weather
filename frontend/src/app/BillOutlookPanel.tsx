"use client";
import React, { useEffect, useState } from "react";
import { BillOutlook, CurvePoint, fetchOutlook, pct, } from "@/lib/outlook";
// Weather outlook for a bill in Parliament (GET /api/v1/bills/{id}/outlook).
// Every sentence comes from the API ready to show; this component only lays them out.
const fmtDate = (iso: string) => new Date(`${iso}T00:00:00`).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
});
function Curve({ points: allPoints, targetDate, }: {
    points: CurvePoint[];
    targetDate: string;
}) {
    if (!allPoints.length)
        return <p className="muted">No forecast curve available.</p>;
    const W = 1000;
    const H = 300;
    const pad = { l: 56, r: 16, t: 16, b: 52 };
    const axisFont = 15;
    // Show one year from the forecast date, with a tick at the start of each month
    const start = new Date(allPoints[0].date);
    const end = new Date(start);
    end.setUTCFullYear(end.getUTCFullYear() + 1);
    const points = allPoints.filter((p) => new Date(p.date) <= end);
    const t0 = start.getTime();
    const t1 = end.getTime();
    const xt = (t: number) => pad.l + ((t - t0) / (t1 - t0)) * (W - pad.l - pad.r);
    const x = (d: string) => xt(new Date(d).getTime());
    const y = (p: number) => pad.t + (1 - p) * (H - pad.t - pad.b);
    const path = points
        .map((p, i) => `${i ? "L" : "M"}${x(p.date).toFixed(1)},${y(p.p_law_by_date).toFixed(1)}`)
        .join(" ");
    const months: Date[] = [];
    for (let m = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth() + 1, 1)); m <= end; m = new Date(Date.UTC(m.getUTCFullYear(), m.getUTCMonth() + 1, 1))) {
        months.push(m);
    }
    const tx = x(targetDate);
    return (<svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Chance of becoming law by date">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (<g key={g}>
          <line x1={pad.l} x2={W - pad.r} y1={y(g)} y2={y(g)} stroke="#E5E7EB" strokeDasharray={g > 0 && g < 1 ? "4 4" : undefined}/>
          <text x={pad.l - 6} y={y(g) + 4} textAnchor="end" fontSize={axisFont} fill="#6B7280">
            {pct(g)}
          </text>
        </g>))}
      {months.map((m, i) => {
            const mx = xt(m.getTime());
            const showYear = i === 0 || m.getUTCMonth() === 0;
            return (<g key={m.toISOString()}>
            <line x1={mx} x2={mx} y1={pad.t} y2={H - pad.b} stroke="#F3F5F7"/>
            <line x1={mx} x2={mx} y1={H - pad.b} y2={H - pad.b + 4} stroke="#6B7280"/>
            <text x={mx} y={H - pad.b + 21} textAnchor="middle" fontSize={axisFont} fill="#6B7280">
              {m.toLocaleDateString("en-GB", { month: "short", timeZone: "UTC" })}
            </text>
            {showYear && (<text x={mx} y={H - pad.b + 41} textAnchor="middle" fontSize={axisFont} fontWeight="600" fill="#1F2933">
                {m.getUTCFullYear()}
              </text>)}
          </g>);
        })}
      {tx >= pad.l && tx <= W - pad.r && (<g>
          <line x1={tx} x2={tx} y1={pad.t} y2={H - pad.b} stroke="#6B7280" strokeWidth="2"/>
          <text x={tx + 6} y={pad.t + 14} fontSize={axisFont} fontWeight="600" fill="#6B7280">
            target
          </text>
        </g>)}
      <path d={path} fill="none" stroke="#12A39A" strokeWidth="3.5" strokeLinejoin="round"/>
    </svg>);
}
export default function BillOutlookPanel({ billId, asOf, }: {
    billId: string;
    asOf?: string;
}) {
    const [outlook, setOutlook] = useState<BillOutlook | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        fetchOutlook(billId, asOf)
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
    }, [billId, asOf]);
    if (!loading && !error && outlook === null)
        return null; // not in Parliament on that date
    return (<section className="outlook-panel">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-wider text-[#6B7280]">
            Weather outlook
            {outlook && ` · as of ${fmtDate(outlook.forecast_date)}`}
          </p>
          {loading && !outlook && (<p className="mt-2 text-sm text-[#6B7280]">Loading forecast…</p>)}
          {error && <p className="mt-2 text-sm text-[#D14343]">{error}</p>}
        </div>
      </div>

      {outlook && (<div className={loading ? "opacity-60 transition-opacity" : ""}>
          {/* Weather headline */}
          <div className="weather-headline">
            <svg className="weather-symbol" viewBox="0 0 40 40" fill="none" stroke={{ sunny: "#E0A526", partly_cloudy: "#6B7280", rain: "#4B7BA7", storm: "#D14343" }[outlook.weather.status]} strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              {outlook.weather.status === "sunny" ? <><circle cx="20" cy="20" r="7"/><path d="M20 3v4m0 26v4M3 20h4m26 0h4M8 8l3 3m18 18 3 3M8 32l3-3M29 11l3-3"/></> : <><path d="M10 27a7 7 0 0 1-1-14 10 10 0 0 1 19-1 7.5 7.5 0 1 1 2 15Z"/>{outlook.weather.status === "rain" && <path d="m13 31-2 5m10-5-2 5m10-5-2 5"/>}{outlook.weather.status === "storm" && <path d="m21 25-5 8h7l-4 6"/>}</>}
            </svg>
            <div><h2>{outlook.weather.label}</h2><p className="text-sm text-[#6B7280]">{outlook.weather.headline}</p></div>
            <div className="weather-probability"><strong>{pct(outlook.distribution.p_royal_assent)}</strong>chance of becoming law<p>{pct(outlook.distribution.p_fail)} chance of failure</p></div>
          </div>
          <details className="forecast-details"><summary>Explore forecast and deadlines</summary>
          <div className="mt-5 grid grid-cols-1 gap-8">
            {/* Curve */}
            <div>
              <h3 className="text-sm font-bold text-[#1F2933]">Chance of becoming law by date</h3>
              <p className="mb-2 text-sm text-[#6B7280]">{outlook.curve.sentence}</p>
              <div className="chart-scroll"><Curve points={outlook.curve.points} targetDate={outlook.weather.target_date}/></div>
            </div>

            {/* Deadlines */}
            <div>
              <h3 className="text-sm font-bold text-[#1F2933]">Deadlines</h3>
              <ul className="mt-2 grid grid-cols-1 gap-x-10 gap-y-1.5 md:grid-cols-2">
                {outlook.deadlines.map((d) => (<li key={d.kind} className="text-sm text-[#1F2933]">
                    <div className="flex items-center gap-3">
                      <span className="w-11 shrink-0 text-right font-bold tabular-nums text-[#1F2933]">
                        {pct(d.p)}
                      </span>
                      <span className="h-1.5 w-24 shrink-0 rounded-full bg-[#F3F5F7]">
                        <span className="block h-1.5 rounded-full bg-[#12A39A]" style={{ width: `${Math.round(d.p * 100)}%` }}/>
                      </span>
                      <span>
                        {d.label} <span className="text-[#6B7280]">({fmtDate(d.date)}{d.estimated ? ", estimated" : ""})</span>
                      </span>
                    </div>
                  </li>))}
              </ul>
            </div>
          </div>

          </details>
          {/* Pace and next milestone */}
          <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="rounded-md bg-[#F3F5F7] p-4">
              <h3 className="text-sm font-bold text-[#1F2933]">Pace</h3>
              <p className="mt-1 text-sm text-[#1F2933]">{outlook.pace.sentence}</p>
            </div>
            {outlook.next_milestone && (<div className="rounded-md bg-[#F3F5F7] p-4">
                <h3 className="text-sm font-bold text-[#1F2933]">Next milestone</h3>
                <p className="mt-1 text-sm text-[#1F2933]">{outlook.next_milestone.sentence}</p>
                {outlook.next_milestone.what_if_sentence && (<p className="mt-1 text-sm text-[#1F2933]">{outlook.next_milestone.what_if_sentence}</p>)}
              </div>)}
          </div>

        </div>)}
    </section>);
}
