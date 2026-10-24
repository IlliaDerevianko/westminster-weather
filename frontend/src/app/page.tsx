"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import React, { Suspense, useEffect, useState } from "react";

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
  phase: "before_parliament" | "inside_parliament" | "after_royal_assent";
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

interface MilestoneDate {
  label: string;
  date: string;
  note: string;
}

interface GovernmentBillTracking {
  id: string;
  parliament_bill_id: number;
  short_title: string;
  long_title: string;
  sponsoring_department: string;
  originating_house: string;
  introduced_session_id: number | null;
  outcome: string;
  outcome_date: string | null;
  stop_reason: string | null;
  as_of_date: string;
  earliest_signal_date: string;
  first_reading_date: string;
  current_phase: "before_parliament" | "inside_parliament" | "after_royal_assent";
  current_stage: string | null;
  current_stage_order: number;
  total_stages: number;
  stages: TrackedStageNode[];
  milestone_dates: MilestoneDate[];
}

interface BillSummary {
  bill_id: number;
  short_title: string;
  originating_house: string;
  introduced_session_id: number;
  introduced_date: string | null;
  department: string | null;
  outcome: string;
  has_pre_parliament_data: boolean;
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
  session_runs_out: {
    title: "Session runs out",
    subtitle: "Prorogation ends bill without carry-over",
  },
  election_called: {
    title: "Election called",
    subtitle: "Dissolution ends all unfinished bills",
  },
  defeated: {
    title: "Defeated",
    subtitle: "Lost vote in Commons or Lords",
  },
  paused_or_withdrawn: {
    title: "Paused or withdrawn",
    subtitle: "Government withdraws or pauses bill",
  },
  deadlock: {
    title: "Deadlock",
    subtitle: "Commons and Lords fail to agree in ping-pong",
  },
};

const PHASE_LABELS: Record<string, string> = {
  before_parliament: "1. Before Parliament (Pre-Legislative Tracker)",
  inside_parliament: "2. Inside Parliament (Parliamentary Passage)",
  after_royal_assent: "3. Royal Assent (Enacted into Law)",
};

function BillTrackerContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const queryBillId = searchParams.get("bill_id");

  const [allBills, setAllBills] = useState<BillSummary[]>([]);
  const [activeBillId, setActiveBillId] = useState<string | null>(queryBillId);
  const [selectedDate, setSelectedDate] = useState<string>("");
  const [bill, setBill] = useState<GovernmentBillTracking | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Load directory of bills from Neon Postgres
  useEffect(() => {
    async function loadDirectory() {
      try {
        const res = await fetch("/api/v1/bills");
        if (!res.ok) return;
        const data: BillSummary[] = await res.json();
        setAllBills(data);
        if (!queryBillId && data.length > 0) {
          setActiveBillId(String(data[0].bill_id));
        }
      } catch {
        // Handled by main fetch
      }
    }
    loadDirectory();
  }, [queryBillId]);

  // Sync activeBillId when URL query parameter changes
  useEffect(() => {
    if (queryBillId) {
      setActiveBillId(queryBillId);
      setSelectedDate("");
    }
  }, [queryBillId]);

  // Fetch bill state from Neon Lakebase Postgres
  useEffect(() => {
    if (!activeBillId) return;
    let cancelled = false;

    async function fetchBill() {
      try {
        setLoading(true);
        const query = selectedDate
          ? `?as_of_date=${encodeURIComponent(selectedDate)}`
          : "";
        const res = await fetch(
          `/api/v1/bills/${encodeURIComponent(activeBillId!)}${query}`
        );
        if (!res.ok) {
          throw new Error(`API returned status ${res.status}`);
        }
        const data: GovernmentBillTracking = await res.json();
        if (!cancelled) {
          setBill(data);
          if (!selectedDate) {
            setSelectedDate(data.as_of_date);
          }
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Failed to fetch bill data from Neon Lakebase Postgres"
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchBill();
    return () => {
      cancelled = true;
    };
  }, [activeBillId, selectedDate]);

  const currentStageObj = bill?.stages.find(
    (s) => s.stage === bill.current_stage
  );
  const isComplete =
    bill !== null &&
    (bill.current_stage === "ready_for_first_reading" ||
      bill.current_stage === "royal_assent");
  const isStopped =
    currentStageObj !== undefined && currentStageObj.status === "stopped";

  return (
    <main className="max-w-5xl mx-auto px-6 py-8 space-y-6">
      {/* Top Navigation & Bill Switcher Bar */}
      <nav className="rounded-xl border border-slate-200 bg-slate-900 text-white px-5 py-3.5 shadow-sm flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs font-bold uppercase tracking-wider text-amber-400">
            Neon Database Bill Selector
          </span>
          <select
            aria-label="Select a bill from Neon Postgres"
            value={
              activeBillId === "online-safety-bill"
                ? "3137"
                : activeBillId ?? ""
            }
            onChange={(e) => {
              const nextId = e.target.value;
              if (nextId) {
                setSelectedDate("");
                router.push(`/?bill_id=${encodeURIComponent(nextId)}`);
              }
            }}
            className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-white focus:border-amber-400 focus:outline-none max-w-xs sm:max-w-md"
          >
            {allBills.map((b) => (
              <option key={b.bill_id} value={String(b.bill_id)}>
                #{b.bill_id} — {b.short_title} (Session {b.introduced_session_id},{" "}
                {b.outcome})
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {bill?.parliament_bill_id === 3137 && (
            <button
              type="button"
              onClick={() => {
                setSelectedDate("");
                router.push(
                  activeBillId === "online-safety-bill"
                    ? "/?bill_id=3137"
                    : "/?bill_id=online-safety-bill"
                );
              }}
              className="rounded-lg border border-amber-400/50 bg-amber-500/20 px-3 py-1.5 text-xs font-semibold text-amber-200 hover:bg-amber-500/30 transition-colors"
            >
              {activeBillId === "online-safety-bill"
                ? "Switch to Parliamentary Stages →"
                : "Switch to Phase 1 (Before Parliament) →"}
            </button>
          )}
          <Link
            href="/bills"
            className="rounded-lg bg-amber-500 px-3.5 py-1.5 text-xs font-bold text-slate-950 hover:bg-amber-400 transition-colors"
          >
            Browse All {allBills.length || ""} Bills →
          </Link>
        </div>
      </nav>

      {/* Header & Historical Date Selector */}
      <header className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-5">
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2 text-xs font-semibold uppercase tracking-wider">
              <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-amber-800">
                {bill
                  ? PHASE_LABELS[bill.current_phase] ?? bill.current_phase
                  : "Loading Pipeline..."}
              </span>
              {bill && (
                <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-slate-700">
                  {bill.originating_house}
                  {bill.introduced_session_id
                    ? ` · Session ${bill.introduced_session_id}`
                    : ""}
                </span>
              )}
              {bill && (
                <span className="rounded-full bg-blue-50 px-2.5 py-0.5 text-blue-800">
                  Outcome: {bill.outcome}
                </span>
              )}
            </div>
            <h1 className="mt-2 text-2xl font-bold text-slate-900">
              {bill?.short_title ?? "Loading Bill from Database..."}{" "}
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
                isStopped
                  ? "border-red-400 bg-red-50"
                  : isComplete
                  ? "border-emerald-400 bg-emerald-50"
                  : "border-amber-300 bg-amber-50"
              }`}
            >
              <div
                className={`text-xs font-medium ${
                  isStopped
                    ? "text-red-800"
                    : isComplete
                    ? "text-emerald-800"
                    : "text-amber-800"
                }`}
              >
                State as of {bill.as_of_date}
              </div>
              <div className="mt-0.5 text-lg font-bold text-slate-900">
                {bill.current_stage_order === 0
                  ? "Not Yet Entered"
                  : isStopped
                  ? `Stopped at Stage ${bill.current_stage_order}`
                  : isComplete
                  ? `All ${bill.total_stages} Stages Completed ✓`
                  : `Stage ${bill.current_stage_order} of ${bill.total_stages}`}
              </div>
              <div
                className={`text-xs font-semibold ${
                  isStopped
                    ? "text-red-900"
                    : isComplete
                    ? "text-emerald-900"
                    : "text-amber-900"
                }`}
              >
                {currentStageObj ? currentStageObj.label : "Before first stage"}
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
                Evaluates where{" "}
                <strong>{bill?.short_title ?? "this bill"}</strong> stood on
                that date using stage sittings stored in Neon Lakebase Postgres
                ({bill?.earliest_signal_date ?? "—"} →{" "}
                {bill?.outcome_date ?? bill?.as_of_date ?? "—"}).
              </p>
            </div>

            <div className="flex items-center gap-2">
              <input
                id="historical-date-picker"
                type="date"
                value={selectedDate || bill?.as_of_date || ""}
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

          {/* Dynamic Milestone Dates from Database */}
          {bill && bill.milestone_dates.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                Jump to Bill Milestone Dates (from Neon Database):
              </div>
              <div className="flex flex-wrap gap-1.5">
                {bill.milestone_dates.map((milestone) => {
                  const isActive =
                    (selectedDate || bill.as_of_date) === milestone.date;
                  return (
                    <button
                      key={`${milestone.date}-${milestone.label}`}
                      type="button"
                      title={milestone.note}
                      onClick={() => setSelectedDate(milestone.date)}
                      className={`rounded-md border px-2.5 py-1 text-xs font-medium transition-colors ${
                        isActive
                          ? "border-amber-500 bg-amber-500 text-white shadow-sm"
                          : "border-slate-300 bg-white text-slate-700 hover:bg-slate-100"
                      }`}
                    >
                      {milestone.label}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Error State */}
      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-800">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Stage Flowchart Column */}
      {bill && (
        <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="mb-6 border-b border-slate-200 pb-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
            <div>
              <h2 className="text-lg font-bold text-slate-900">
                {PHASE_LABELS[bill.current_phase] ?? bill.current_phase}
              </h2>
              <p className="text-xs text-slate-500">
                Main path runs down the left column. Dashed red boxes on the
                right show where a bill can stop at that step.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-4 text-xs text-slate-600">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-3 w-3 rounded-full bg-emerald-500"></span>
                Completed
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-3 w-3 rounded-full bg-amber-500 ring-4 ring-amber-200"></span>
                <strong>Current Stage</strong>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-3 w-3 rounded-full bg-red-500"></span>
                Stopped
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
              const isStageStopped = stage.status === "stopped";
              const stopRiskKey =
                stage.active_stop_reason ?? stage.stop_risks[0];
              const stopMeta = stopRiskKey
                ? STOP_RISK_DESCRIPTIONS[stopRiskKey]
                : null;

              return (
                <div key={stage.stage}>
                  <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
                    {/* Main Stage Node (Left 8 columns) */}
                    <div
                      className={`md:col-span-8 rounded-lg border p-4 transition-all ${
                        isStageStopped
                          ? "border-2 border-red-500 bg-red-50/90 shadow-md ring-4 ring-red-200/70"
                          : isCurrent
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
                              isStageStopped
                                ? "bg-red-600 text-white"
                                : isCurrent
                                ? "bg-amber-500 text-white"
                                : isCompleted
                                ? "bg-emerald-600 text-white"
                                : "bg-slate-200 text-slate-700"
                            }`}
                          >
                            {isCompleted ? "✓" : isStageStopped ? "✕" : stage.order}
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
                            isStageStopped
                              ? "bg-red-600 text-white"
                              : isCurrent
                              ? "bg-amber-500 text-white"
                              : isCompleted
                              ? "bg-emerald-100 text-emerald-800"
                              : "bg-slate-200 text-slate-600"
                          }`}
                        >
                          {isStageStopped
                            ? `Stopped (${stage.active_stop_reason ?? bill.outcome})`
                            : isCurrent
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
                                Official Sittings / Events Recorded ≤{" "}
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
                              isStageStopped
                                ? "border-2 border-red-600 bg-red-100 shadow-md"
                                : isCurrent
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

export default function BillTrackerPage() {
  return (
    <Suspense
      fallback={
        <main className="max-w-5xl mx-auto px-6 py-8 text-sm text-slate-600">
          Loading tracker from Neon Lakebase Postgres...
        </main>
      }
    >
      <BillTrackerContent />
    </Suspense>
  );
}
