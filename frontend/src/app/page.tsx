"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import React, { Suspense, useEffect, useMemo, useState } from "react";
import BillOutlookPanel from "./BillOutlookPanel";

type StageStatus =
  | "completed"
  | "fast_tracked"
  | "current"
  | "upcoming"
  | "stopped"
  | "skipped";
type PhaseKey = "before_parliament" | "inside_parliament" | "after_royal_assent";

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
  phase: PhaseKey;
  order: number;
  label: string;
  description: string;
  stop_risks: string[];
  status: StageStatus;
  sitting_count: number;
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
  current_phase: PhaseKey;
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

// `bills.outcome` values (DATA_CONTRACT §3.1)
const OUTCOME_LABELS: Record<string, string> = {
  royal_assent: "Royal Assent",
  fell_session_end: "Fell at session end",
  fell_dissolution: "Fell at dissolution",
  withdrawn: "Withdrawn",
  defeated: "Defeated",
  in_progress: "In progress",
};

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
  never_commenced: {
    title: "Never commenced",
    subtitle: "Passed into law, never switched on",
  },
  annulled: {
    title: "Annulled",
    subtitle: "Rare: Parliament blocks a statutory instrument",
  },
};

const PHASE_CONFIG: {
  key: PhaseKey;
  number: number;
  title: string;
  subtitle: string;
  accentBg: string;
  accentBorder: string;
  badgeClass: string;
}[] = [
  {
    key: "before_parliament",
    number: 1,
    title: "1. Before Parliament",
    subtitle: "Signal, commitment, consultation, drafting & King's Speech slot",
    accentBg: "bg-amber-50/50",
    accentBorder: "border-amber-200",
    badgeClass: "bg-amber-100 text-amber-900",
  },
  {
    key: "inside_parliament",
    number: 2,
    title: "2. Inside Parliament",
    subtitle: "Commons & Lords readings, committees, report, 3rd reading & ping-pong",
    accentBg: "bg-blue-50/40",
    accentBorder: "border-blue-200",
    badgeClass: "bg-blue-100 text-blue-900",
  },
  {
    key: "after_royal_assent",
    number: 3,
    title: "3. After Royal Assent",
    subtitle: "Royal Assent, commencement orders (SIs), regulations & law in force",
    accentBg: "bg-emerald-50/40",
    accentBorder: "border-emerald-200",
    badgeClass: "bg-emerald-100 text-emerald-900",
  },
];

function BillTrackerContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const queryBillId = searchParams.get("bill_id");

  const [allBills, setAllBills] = useState<BillSummary[]>([]);
  const [activeBillId, setActiveBillId] = useState<string | null>(
    queryBillId === "online-safety-bill" ? "3137" : queryBillId
  );
  const [selectedDate, setSelectedDate] = useState<string>("");
  // The snapshot date the page opened on (latest data); the outlook then forecasts as of today.
  const [latestDate, setLatestDate] = useState<string>("");
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
          const onlineSafety = data.find((b) => b.bill_id === 3137);
          setActiveBillId(String((onlineSafety ?? data[0]).bill_id));
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
      setActiveBillId(
        queryBillId === "online-safety-bill" ? "3137" : queryBillId
      );
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
            setLatestDate(data.as_of_date);
          }
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Failed to fetch bill data"
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

  const stagesByPhase = useMemo(() => {
    const grouped: Record<PhaseKey, TrackedStageNode[]> = {
      before_parliament: [],
      inside_parliament: [],
      after_royal_assent: [],
    };
    if (!bill) return grouped;
    for (const s of bill.stages) {
      grouped[s.phase]?.push(s);
    }
    return grouped;
  }, [bill]);

  const currentStageObj = bill?.stages.find(
    (s) => s.stage === bill.current_stage
  );
  const isStopped =
    currentStageObj !== undefined && currentStageObj.status === "stopped";
  const hasRoyalAssent =
    bill?.stages.find((s) => s.stage === "royal_assent")?.status ===
    "completed";
  // `outcome` stops at Royal Assent; once the Act is fully in force (as of the selected
  // date) that is the furthest point reached, so show it instead
  const isLawFullyInForce =
    bill?.stages.find((s) => s.stage === "law_fully_in_force")?.status ===
    "completed";
  const overallOutcome = !bill
    ? null
    : isLawFullyInForce
    ? "Law fully in force"
    : OUTCOME_LABELS[bill.outcome] ?? bill.outcome.replace(/_/g, " ");

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">
      {/* Top Navigation & Bill Switcher Bar */}
      <nav className="rounded-xl border border-slate-200 bg-slate-900 text-white px-5 py-3.5 shadow-sm flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs font-bold uppercase tracking-wider text-amber-400">
            Bill Selector
          </span>
          <select
            aria-label="Select a bill"
            value={activeBillId ?? ""}
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

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <Link
            href="/promises"
            className="rounded-lg border border-amber-500 px-3.5 py-1.5 text-xs font-bold text-amber-400 hover:bg-amber-500 hover:text-slate-950 transition-colors"
          >
            Promises → Bills
          </Link>
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
              {bill && (
                <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-slate-700">
                  Originating House: {bill.originating_house}
                  {bill.introduced_session_id
                    ? ` · Session ${bill.introduced_session_id}`
                    : ""}
                </span>
              )}
              {bill && (
                <span className="rounded-full bg-blue-50 px-2.5 py-0.5 text-blue-800">
                  Overall Outcome: {overallOutcome}
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
                <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
                  <span>
                    <strong className="text-slate-700">First Signal:</strong>{" "}
                    {bill.earliest_signal_date}
                  </span>
                  {bill.outcome_date && (
                    <span>
                      <strong className="text-slate-700">Outcome Date:</strong>{" "}
                      {bill.outcome_date}
                    </span>
                  )}
                </div>
              </>
            )}
          </div>

          {/* Current Position Badge */}
          {bill && (
            <div
              className={`rounded-lg border px-4 py-3 text-center shrink-0 min-w-[230px] ${
                isStopped
                  ? "border-red-400 bg-red-50"
                  : hasRoyalAssent
                  ? "border-emerald-400 bg-emerald-50"
                  : "border-amber-300 bg-amber-50"
              }`}
            >
              <div
                className={`text-xs font-medium ${
                  isStopped
                    ? "text-red-800"
                    : hasRoyalAssent
                    ? "text-emerald-800"
                    : "text-amber-800"
                }`}
              >
                Snapshot as of {bill.as_of_date}
              </div>
              <div className="mt-0.5 text-lg font-bold text-slate-900">
                {bill.current_stage_order === 0
                  ? "Pre-Signal"
                  : isStopped
                  ? `Stopped at Stage ${bill.current_stage_order}/${bill.total_stages}`
                  : `Stage ${bill.current_stage_order} of ${bill.total_stages}`}
              </div>
              <div
                className={`text-xs font-semibold ${
                  isStopped
                    ? "text-red-900"
                    : hasRoyalAssent
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
                Time-travel across all phases of{" "}
                <strong>{bill?.short_title ?? "this bill"}</strong>.
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
                Jump to Bill Milestone Dates:
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

        {/* Legend */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-1 text-xs text-slate-600 border-t border-slate-100">
          <span>
            Each phase column shows the main legislative path, with a{" "}
            <strong className="text-red-700">red stop branch</strong> on the
            right only when a bill actually stopped at that stage.
          </span>
          <div className="flex flex-wrap items-center gap-4">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded-full bg-emerald-500"></span>
              Completed
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded-full bg-sky-500"></span>
              Fast-Tracked / Bypassed
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded-full bg-teal-500"></span>
              Skipped (0 rounds/SIs)
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
      </header>

      {/* Error State */}
      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-800">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Weather outlook (bills in Parliament on the snapshot date only) */}
      {activeBillId && (
        <BillOutlookPanel
          billId={activeBillId}
          asOf={selectedDate && selectedDate !== latestDate ? selectedDate : undefined}
        />
      )}

      {/* 3-Phase Pipeline Grid (1. Before Parliament | 2. Inside Parliament | 3. After Royal Assent) */}
      {bill && (
        <div
          className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start"
        >
          {PHASE_CONFIG.map((phaseCfg) => {
            const phaseStages = stagesByPhase[phaseCfg.key] ?? [];
            const completedCount = phaseStages.filter(
              (s) =>
                s.status === "completed" ||
                s.status === "fast_tracked" ||
                s.status === "skipped"
            ).length;

            return (
              <section
                key={phaseCfg.key}
                className={`rounded-xl border ${phaseCfg.accentBorder} bg-white p-4 sm:p-5 shadow-sm space-y-4`}
              >
                <div className="border-b border-slate-200 pb-3 flex items-start justify-between gap-2">
                  <div>
                    <span
                      className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-bold ${phaseCfg.badgeClass}`}
                    >
                      {phaseCfg.title}
                    </span>
                    <p className="mt-1 text-xs text-slate-500">
                      {phaseCfg.subtitle}
                    </p>
                  </div>
                  <span className="shrink-0 rounded bg-slate-100 px-2 py-0.5 text-[11px] font-mono font-semibold text-slate-700">
                    {completedCount}/{phaseStages.length}
                  </span>
                </div>

                <div className="space-y-3">
                  {phaseStages.map((stage, idx) => {
                    const isCompleted = stage.status === "completed";
                    const isFastTracked = stage.status === "fast_tracked";
                    const isSkipped = stage.status === "skipped";
                    const isCurrent = stage.status === "current";
                    const isStageStopped = stage.status === "stopped";
                    const isPingPong = stage.stage === "ping_pong";
                    const isPostRoyalAssent =
                      stage.phase === "after_royal_assent";
                    // Upcoming after Royal Assent with none recorded yet: the backend sends a
                    // summary "<badge> — <detail>"; a stage not reached yet has no summary
                    const noneYetLabel =
                      stage.status === "upcoming" && isPostRoyalAssent && stage.summary
                        ? stage.summary.split(" — ")[0]
                        : null;
                    const countUnit =
                      stage.stage === "ping_pong"
                        ? stage.sitting_count === 1
                          ? "round"
                          : "rounds"
                        : stage.stage === "commencement_orders"
                        ? stage.sitting_count === 1
                          ? "order"
                          : "orders"
                        : stage.stage === "regulations_drafted"
                        ? stage.sitting_count === 1
                          ? "draft SI"
                          : "draft SIs"
                        : stage.stage === "regulations_laid"
                        ? stage.sitting_count === 1
                          ? "SI"
                          : "SIs"
                        : null;
                    const countLabel = countUnit
                      ? `${stage.sitting_count} ${countUnit}`
                      : null;
                    const stopRiskKey = isStageStopped
                      ? stage.active_stop_reason
                      : null;
                    const stopMeta = stopRiskKey
                      ? STOP_RISK_DESCRIPTIONS[stopRiskKey] ?? {
                          title: stopRiskKey.replace(/_/g, " "),
                          subtitle: "Bill stopped at this stage",
                        }
                      : null;

                    return (
                      <div key={stage.stage}>
                        <div className="grid grid-cols-12 gap-2.5 items-stretch">
                          {/* Main Stage Card */}
                          <div
                            className={`${
                              stopMeta ? "col-span-8" : "col-span-12"
                            } rounded-lg border p-3 transition-all ${
                              isStageStopped
                                ? "border-2 border-red-500 bg-red-50/90 shadow-md ring-2 ring-red-200"
                                : isCurrent
                                ? "border-2 border-amber-500 bg-amber-50/90 shadow-md ring-2 ring-amber-200"
                                : isCompleted
                                ? "border-emerald-300 bg-emerald-50/70"
                                : isFastTracked
                                ? "border-sky-300 bg-sky-50/60"
                                : isSkipped
                                ? "border-teal-300 bg-teal-50/60"
                                : noneYetLabel
                                ? "border-dashed border-slate-300 bg-white"
                                : "border-slate-200 bg-slate-50/60 opacity-75"
                            }`}
                          >
                            <div className="flex items-start justify-between gap-2">
                              <div className="flex items-start gap-2">
                                <span
                                  className={`mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-bold ${
                                    isStageStopped
                                      ? "bg-red-600 text-white"
                                      : isCurrent
                                      ? "bg-amber-500 text-white"
                                      : isCompleted
                                      ? "bg-emerald-600 text-white"
                                      : isFastTracked
                                      ? "bg-sky-600 text-white"
                                      : isSkipped
                                      ? "bg-teal-600 text-white"
                                      : "bg-slate-200 text-slate-700"
                                  }`}
                                >
                                  {isCompleted
                                    ? "✓"
                                    : isFastTracked
                                    ? "»"
                                    : isSkipped
                                    ? "0"
                                    : isStageStopped
                                    ? "✕"
                                    : stage.order}
                                </span>
                                <div>
                                  <div className="text-sm font-bold text-slate-900 leading-snug">
                                    {stage.label}
                                  </div>
                                  <div className="text-[11px] text-slate-600 mt-0.5">
                                    {stage.description}
                                  </div>
                                </div>
                              </div>

                              <span
                                className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${
                                  isStageStopped
                                    ? "bg-red-600 text-white"
                                    : isCurrent
                                    ? "bg-amber-500 text-white"
                                    : isCompleted
                                    ? "bg-emerald-100 text-emerald-800"
                                    : isFastTracked
                                    ? "bg-sky-100 text-sky-800"
                                    : isSkipped
                                    ? "bg-teal-100 text-teal-800"
                                    : "bg-slate-200 text-slate-600"
                                }`}
                              >
                                {isStageStopped
                                  ? "Stopped"
                                  : isFastTracked
                                  ? stage.phase === "inside_parliament"
                                    ? "Fast-Tracked (0 sittings)"
                                    : "Fast-Tracked"
                                  : isSkipped
                                  ? countLabel
                                    ? `Skipped (${countLabel})`
                                    : "Skipped (0)"
                                  : isCurrent
                                  ? countLabel
                                    ? `Current (${countLabel})`
                                    : "Current"
                                  : isCompleted
                                  ? countLabel
                                    ? `Done (${countLabel})`
                                    : "Done"
                                  : noneYetLabel ?? "Upcoming"}
                              </span>
                            </div>

                            {/* Entered/Completed Dates & Collapsible Sittings */}
                            {(stage.entered_date ||
                              stage.visible_events.length > 0 ||
                              ((stage.phase === "before_parliament" ||
                                isPingPong ||
                                isPostRoyalAssent ||
                                isSkipped ||
                                isFastTracked) &&
                                stage.summary)) && (
                              <div className="mt-2.5 rounded border border-slate-200/80 bg-white p-2 text-[11px] space-y-1.5">
                                {(stage.phase === "before_parliament" ||
                                  isPingPong ||
                                  isPostRoyalAssent ||
                                  isSkipped ||
                                  isFastTracked) &&
                                  stage.summary && (
                                    <div
                                      className={`font-medium ${
                                        isFastTracked
                                          ? "text-sky-900"
                                          : stage.phase === "before_parliament"
                                          ? "text-slate-800"
                                          : "text-teal-900"
                                      }`}
                                    >
                                      {stage.summary}
                                    </div>
                                  )}
                                {stage.entered_date &&
                                  !isSkipped &&
                                  !isFastTracked && (
                                  <div className="flex flex-wrap items-center justify-between gap-1 text-slate-700 font-medium">
                                    <span>
                                      {stage.entered_date}
                                      {stage.completed_date &&
                                      stage.completed_date !== stage.entered_date
                                        ? ` → ${stage.completed_date}`
                                        : ""}
                                    </span>
                                  </div>
                                )}

                                {stage.visible_events.length > 0 && (
                                  <details className="group">
                                    <summary className="cursor-pointer font-semibold text-slate-700 hover:text-slate-900 select-none">
                                      {stage.visible_events.length} recorded{" "}
                                      {stage.visible_events.length === 1
                                        ? "event/sitting"
                                        : "events/sittings"}{" "}
                                      ≤ {bill.as_of_date}
                                    </summary>
                                    <ul className="mt-1.5 space-y-1 max-h-48 overflow-y-auto pr-1">
                                      {stage.visible_events.map((ev, evIdx) => (
                                        <li
                                          key={`${ev.event_date}-${evIdx}`}
                                          className="rounded bg-slate-50 p-1.5 border border-slate-100"
                                        >
                                          <div className="font-semibold text-slate-800">
                                            {ev.event_date} — {ev.title}
                                          </div>
                                          <div className="text-[10px] text-slate-600 mt-0.5">
                                            {ev.note}
                                          </div>
                                        </li>
                                      ))}
                                    </ul>
                                  </details>
                                )}
                              </div>
                            )}
                          </div>

                          {/* Dashed Red Stop Risk Box on Right (4 columns) */}
                          {stopMeta && (
                            <div className="col-span-4 flex items-center gap-1">
                              <span className="text-red-400 font-mono text-xs shrink-0">
                                ▸
                              </span>
                              <div
                                className={`w-full rounded-lg border border-dashed p-2 text-center flex flex-col justify-center ${
                                  isStageStopped
                                    ? "border-2 border-red-600 bg-red-100 shadow-sm"
                                    : isCurrent
                                    ? "border-red-400 bg-red-50/90"
                                    : "border-red-300 bg-red-50/40 opacity-75"
                                }`}
                              >
                                <div className="text-[11px] font-bold text-slate-900 leading-tight">
                                  {stopMeta.title}
                                </div>
                                <div className="text-[10px] text-slate-600 mt-0.5 leading-tight">
                                  {stopMeta.subtitle}
                                </div>
                              </div>
                            </div>
                          )}
                        </div>

                        {/* Downward Connector Arrow */}
                        {idx < phaseStages.length - 1 && (
                          <div className="grid grid-cols-12">
                            <div
                              className={`${
                                stopMeta ? "col-span-8" : "col-span-12"
                              } flex justify-center py-1 text-slate-400 text-xs leading-none`}
                            >
                              ↓
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </section>
            );
          })}
        </div>
      )}
    </main>
  );
}

export default function BillTrackerPage() {
  return (
    <Suspense
      fallback={
        <main className="max-w-7xl mx-auto px-6 py-8 text-sm text-slate-600">
          Loading bill tracker...
        </main>
      }
    >
      <BillTrackerContent />
    </Suspense>
  );
}
