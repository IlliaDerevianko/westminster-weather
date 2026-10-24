"use client";

import React, { useEffect, useState } from "react";

type StageStatus = "completed" | "current" | "upcoming" | "stopped";

interface ApiEvidenceEvent {
  event_date: string;
  title: string;
  note: string;
  source_api_url: string;
  web_url: string;
  document_type: string;
}

interface TrackedStageNode {
  stage: string;
  phase: "before_parliament";
  order: number;
  label: string;
  description: string;
  stop_risks: string[];
  status: StageStatus;
  entered_date: string | null;
  completed_date: string | null;
  summary: string | null;
  source_reference: string | null;
  source_api_url: string | null;
  web_url: string | null;
  visible_events: ApiEvidenceEvent[];
  active_stop_reason: string | null;
}

interface GovernmentBillTracking {
  id: string;
  parliament_bill_id: number;
  short_title: string;
  long_title: string;
  sponsoring_department: string;
  as_of_date: string;
  earliest_signal_date: string;
  first_reading_date: string;
  current_phase: "before_parliament";
  current_stage: string | null;
  current_stage_order: number;
  total_stages: number;
  stages: TrackedStageNode[];
}

const STOP_RISK_DESCRIPTIONS: Record<string, { title: string; subtitle: string }> = {
  never_adopted: {
    title: "Never adopted",
    subtitle: "Low priority or costly",
  },
  shelved: {
    title: "Shelved",
    subtitle: "Backlash or rethink",
  },
  stuck_in_drafting: {
    title: "Stuck in drafting",
    subtitle: "Too complex, delayed",
  },
  no_slot_this_session: {
    title: "No slot this session",
    subtitle: "May return next year",
  },
};

const HISTORICAL_DATE_PRESETS: { label: string; date: string; note: string }[] = [
  {
    label: "Nov 2017 (Stage 1: Signal)",
    date: "2017-11-15",
    note: "After Internet Safety Strategy Green Paper published",
  },
  {
    label: "Jul 2018 (Stage 2: Commitment)",
    date: "2018-07-01",
    note: "After Govt Response commits to legislation",
  },
  {
    label: "Jun 2019 (Stage 3: White Paper Open)",
    date: "2019-06-01",
    note: "Online Harms White Paper consultation open",
  },
  {
    label: "Mar 2020 (Stage 3: Initial Response)",
    date: "2020-03-01",
    note: "Initial consultation response published, awaiting full outcome",
  },
  {
    label: "Jan 2021 (Stage 4: Drafting)",
    date: "2021-01-15",
    note: "Full response out; Parliamentary Counsel drafting Bill",
  },
  {
    label: "Jun 2021 (Stage 5: Programme Slot)",
    date: "2021-06-01",
    note: "Queen's Speech 2021 & Draft Online Safety Bill published",
  },
  {
    label: "17 Mar 2022 (Stage 6: Ready for 1st Reading)",
    date: "2022-03-17",
    note: "Joint Committee Govt response & Commons 1st Reading",
  },
  {
    label: "Oct 2026 (Present Day)",
    date: "2026-10-24",
    note: "All pre-parliament stages completed",
  },
];

export default function PreParliamentTrackerPage() {
  const [selectedDate, setSelectedDate] = useState<string>("2021-01-15");
  const [bill, setBill] = useState<GovernmentBillTracking | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  async function fetchBillAtDate(asOfDate: string) {
    try {
      setLoading(true);
      const res = await fetch(
        `/api/v1/bills/online-safety-bill?as_of_date=${encodeURIComponent(
          asOfDate
        )}`
      );
      if (!res.ok) {
        throw new Error(`API returned status ${res.status}`);
      }
      const data: GovernmentBillTracking = await res.json();
      setBill(data);
      setError(null);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to fetch historical bill data from backend API"
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchBillAtDate(selectedDate);
  }, [selectedDate]);

  const currentStageObj = bill?.stages.find(
    (s) => s.stage === bill.current_stage
  );
  const isPhaseOneComplete =
    bill !== null && bill.current_stage === "ready_for_first_reading";

  return (
    <main className="max-w-5xl mx-auto px-6 py-8 space-y-6">
      {/* Header & Historical Date Selector */}
      <header className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-5">
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2 text-xs font-semibold uppercase tracking-wider">
              <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-amber-800">
                1. Before Parliament (Pre-Legislative Tracker)
              </span>
              <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-slate-700">
                Live GOV.UK &amp; Parliament API Data
              </span>
            </div>
            <h1 className="mt-2 text-2xl font-bold text-slate-900">
              {bill?.short_title ?? "Online Safety Bill"}{" "}
              {bill && (
                <span className="text-base font-normal text-slate-500">
                  (Parliament Bill #{bill.parliament_bill_id})
                </span>
              )}
            </h1>
            {bill && (
              <>
                <p className="mt-1 text-sm text-slate-600">{bill.long_title}</p>
                <p className="mt-1.5 text-xs text-slate-500">
                  <span className="font-semibold text-slate-700">
                    Sponsoring Department:
                  </span>{" "}
                  {bill.sponsoring_department}
                </p>
              </>
            )}
          </div>

          {/* Current Position Badge */}
          {bill && (
            <div
              className={`rounded-lg border px-4 py-3 text-center shrink-0 min-w-[220px] ${
                isPhaseOneComplete
                  ? "border-emerald-400 bg-emerald-50"
                  : "border-amber-300 bg-amber-50"
              }`}
            >
              <div
                className={`text-xs font-medium ${
                  isPhaseOneComplete ? "text-emerald-800" : "text-amber-800"
                }`}
              >
                State as of {bill.as_of_date}
              </div>
              <div className="mt-0.5 text-lg font-bold text-slate-900">
                {bill.current_stage_order === 0
                  ? "Pre-Signal (Not Started)"
                  : isPhaseOneComplete
                  ? "All 6 Stages Completed ✓"
                  : `Stage ${bill.current_stage_order} of ${bill.total_stages}`}
              </div>
              <div
                className={`text-xs font-semibold ${
                  isPhaseOneComplete ? "text-emerald-900" : "text-amber-900"
                }`}
              >
                {currentStageObj ? currentStageObj.label : "Before first signal"}
              </div>
            </div>
          )}
        </div>

        {/* Date Picker Controls */}
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div>
              <label
                htmlFor="historical-date-picker"
                className="block text-sm font-semibold text-slate-800"
              >
                Select Historical Snapshot Date (`as_of_date`)
              </label>
              <p className="text-xs text-slate-500">
                Evaluates where the Online Safety Bill stood on that date using
                real timestamps from the GOV.UK Content API and UK Parliament
                Bills API ({bill?.earliest_signal_date ?? "2017-10-10"} →{" "}
                {bill?.first_reading_date ?? "2022-03-17"}).
              </p>
            </div>

            <div className="flex items-center gap-2">
              <input
                id="historical-date-picker"
                type="date"
                value={selectedDate}
                onChange={(e) => {
                  if (e.target.value) {
                    setSelectedDate(e.target.value);
                  }
                }}
                className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-900 shadow-sm focus:border-amber-500 focus:outline-none focus:ring-2 focus:ring-amber-200"
              />
              {loading && (
                <span className="text-xs font-medium text-amber-700">
                  Updating...
                </span>
              )}
            </div>
          </div>

          {/* Preset Historical Dates */}
          <div className="space-y-1.5 pt-1">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              Jump to Historical Dates:
            </div>
            <div className="flex flex-wrap gap-1.5">
              {HISTORICAL_DATE_PRESETS.map((preset) => {
                const isActive = selectedDate === preset.date;
                return (
                  <button
                    key={preset.date}
                    type="button"
                    title={preset.note}
                    onClick={() => setSelectedDate(preset.date)}
                    className={`rounded-md border px-2.5 py-1 text-xs font-medium transition-colors ${
                      isActive
                        ? "border-amber-500 bg-amber-500 text-white shadow-sm"
                        : "border-slate-300 bg-white text-slate-700 hover:bg-slate-100"
                    }`}
                  >
                    {preset.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </header>

      {/* Error State */}
      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-800">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Phase 1: Before Parliament Flowchart Column */}
      {bill && (
        <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="mb-6 border-b border-slate-200 pb-3 flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold text-slate-900">
                1. Before Parliament
              </h2>
              <p className="text-xs text-slate-500">
                Main path runs down the left column. Dashed red boxes on the
                right show where a bill can stop at that step.
              </p>
            </div>
            <div className="flex items-center gap-4 text-xs text-slate-600">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-3 w-3 rounded-full bg-emerald-500"></span>
                Completed
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-3 w-3 rounded-full bg-amber-500 ring-4 ring-amber-200"></span>
                <strong>Current Stage</strong>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-3 w-3 rounded-full bg-slate-300"></span>
                Upcoming
              </span>
            </div>
          </div>

          <div className="space-y-4">
            {bill.stages.map((stage, idx) => {
              const isCompleted = stage.status === "completed";
              const isCurrent = stage.status === "current";
              const stopRiskKey = stage.stop_risks[0];
              const stopMeta = stopRiskKey
                ? STOP_RISK_DESCRIPTIONS[stopRiskKey]
                : null;

              return (
                <div key={stage.stage}>
                  <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
                    {/* Main Stage Node (Left 8 columns) */}
                    <div
                      className={`md:col-span-8 rounded-lg border p-4 transition-all ${
                        isCurrent
                          ? "border-2 border-amber-500 bg-amber-50/90 shadow-md ring-4 ring-amber-200/70"
                          : isCompleted
                          ? "border-emerald-300 bg-emerald-50/70"
                          : "border-slate-200 bg-slate-50/60 opacity-75"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex items-center gap-2.5">
                          <span
                            className={`inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
                              isCurrent
                                ? "bg-amber-500 text-white"
                                : isCompleted
                                ? "bg-emerald-600 text-white"
                                : "bg-slate-200 text-slate-700"
                            }`}
                          >
                            {isCompleted ? "✓" : stage.order}
                          </span>
                          <div>
                            <div className="text-base font-bold text-slate-900">
                              {stage.label}
                            </div>
                            <div className="text-xs text-slate-600">
                              {stage.description}
                            </div>
                          </div>
                        </div>

                        <span
                          className={`shrink-0 rounded px-2.5 py-0.5 text-xs font-semibold uppercase ${
                            isCurrent
                              ? "bg-amber-500 text-white"
                              : isCompleted
                              ? "bg-emerald-100 text-emerald-800"
                              : "bg-slate-200 text-slate-600"
                          }`}
                        >
                          {isCurrent
                            ? `Current Stage (on ${bill.as_of_date})`
                            : isCompleted
                            ? "Completed"
                            : "Upcoming"}
                        </span>
                      </div>

                      {/* Real API Evidence Box (Only shown if reached by as_of_date) */}
                      {stage.entered_date && (
                        <div className="mt-3 rounded-md border border-slate-200 bg-white p-3 text-xs space-y-2">
                          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-1.5">
                            <span className="font-semibold text-slate-800">
                              Stage Period: {stage.entered_date} →{" "}
                              {stage.completed_date
                                ? stage.completed_date
                                : `In Progress as of ${bill.as_of_date}`}
                            </span>
                            {stage.source_api_url && (
                              <a
                                href={stage.source_api_url}
                                target="_blank"
                                rel="noreferrer"
                                className="text-[11px] font-medium text-blue-600 hover:underline"
                              >
                                View Raw API JSON ↗
                              </a>
                            )}
                          </div>

                          {stage.visible_events.length > 0 && (
                            <div className="space-y-1.5">
                              <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                                Official API Events Published ≤{" "}
                                {bill.as_of_date}:
                              </div>
                              <ul className="space-y-1.5">
                                {stage.visible_events.map((ev, evIdx) => (
                                  <li
                                    key={`${ev.event_date}-${evIdx}`}
                                    className="rounded bg-slate-50 p-2 border border-slate-100"
                                  >
                                    <div className="flex items-center justify-between gap-2">
                                      <span className="font-semibold text-slate-800">
                                        {ev.event_date} — {ev.title}
                                      </span>
                                      <span className="rounded bg-slate-200 px-1.5 py-0.5 text-[10px] font-mono text-slate-700">
                                        {ev.document_type}
                                      </span>
                                    </div>
                                    <p className="mt-0.5 text-slate-600">
                                      {ev.note}
                                    </p>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Dashed Red Stop Node (Right 4 columns, matching flowchart) */}
                    <div className="md:col-span-4">
                      {stopMeta && (
                        <div className="flex items-center gap-2">
                          <span className="text-red-400 font-mono text-sm shrink-0">
                            ---▸
                          </span>
                          <div
                            className={`w-full rounded-lg border border-dashed p-3 text-center ${
                              isCurrent
                                ? "border-red-400 bg-red-50/90 shadow-sm"
                                : "border-red-300 bg-red-50/40 opacity-75"
                            }`}
                          >
                            <div className="text-sm font-bold text-slate-900">
                              {stopMeta.title}
                            </div>
                            <div className="text-xs text-slate-600 mt-0.5">
                              {stopMeta.subtitle}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Downward Arrow between stages */}
                  {idx < bill.stages.length - 1 && (
                    <div className="grid grid-cols-1 md:grid-cols-12">
                      <div className="md:col-span-8 flex justify-center py-1.5 text-slate-400 text-sm leading-none">
                        ↓
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}
    </main>
  );
}
