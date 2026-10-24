"use client";

import React, { useEffect, useState } from "react";

type LifecyclePhase =
  | "before_parliament"
  | "inside_parliament"
  | "after_royal_assent";

type StageStatus = "completed" | "current" | "upcoming" | "stopped";

interface TrackedStageNode {
  stage: string;
  phase: LifecyclePhase;
  order: number;
  label: string;
  description: string;
  stop_risks: string[];
  status: StageStatus;
  entered_date: string | null;
  completed_date: string | null;
  summary: string | null;
  source_reference: string | null;
  active_stop_reason: string | null;
}

interface GovernmentBillTracking {
  id: string;
  short_title: string;
  long_title: string;
  sponsoring_department: string;
  simulated_as_of_date: string;
  current_phase: LifecyclePhase;
  current_stage: string;
  current_stage_order: number;
  total_stages: number;
  stopped_reason: string | null;
  passage_probability: number;
  estimated_months_to_commencement: number | null;
  stages: TrackedStageNode[];
}

const PHASE_META: {
  key: LifecyclePhase;
  title: string;
  subtitle: string;
}[] = [
  {
    key: "before_parliament",
    title: "1. Before Parliament",
    subtitle: "Signals, consultations, drafting & King's Speech slot",
  },
  {
    key: "inside_parliament",
    title: "2. Inside Parliament",
    subtitle: "Commons & Lords readings, committees, report & ping-pong",
  },
  {
    key: "after_royal_assent",
    title: "3. After Royal Assent",
    subtitle: "Commencement orders, secondary legislation & full force",
  },
];

function formatStopRisk(risk: string): string {
  return risk.replace(/_/g, " ");
}

export default function BillTrackerPage() {
  const [bill, setBill] = useState<GovernmentBillTracking | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [updatingStage, setUpdatingStage] = useState<boolean>(false);

  async function fetchBill() {
    try {
      setLoading(true);
      const res = await fetch("/api/v1/bills/online-safety-bill");
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
          : "Failed to connect to backend API"
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleSelectStage(stageKey: string) {
    if (!bill || updatingStage) return;
    try {
      setUpdatingStage(true);
      const res = await fetch(`/api/v1/bills/${bill.id}/stage`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          current_stage: stageKey,
          stopped_reason: null,
        }),
      });
      if (!res.ok) {
        throw new Error(`Failed to update stage: ${res.status}`);
      }
      const updated: GovernmentBillTracking = await res.json();
      setBill(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update stage");
    } finally {
      setUpdatingStage(false);
    }
  }

  useEffect(() => {
    fetchBill();
  }, []);

  if (loading) {
    return (
      <main className="max-w-7xl mx-auto p-8">
        <div className="rounded-lg border border-slate-200 bg-white p-8 text-center text-slate-600">
          Loading UK Government Bill lifecycle tracker from backend...
        </div>
      </main>
    );
  }

  if (error || !bill) {
    return (
      <main className="max-w-7xl mx-auto p-8">
        <div className="rounded-lg border border-red-300 bg-red-50 p-6 text-red-800">
          <h2 className="text-lg font-semibold">Backend Connection Error</h2>
          <p className="mt-1 text-sm">{error}</p>
          <button
            onClick={fetchBill}
            className="mt-4 rounded bg-red-700 px-4 py-2 text-sm font-medium text-white hover:bg-red-800"
          >
            Retry Connection
          </button>
        </div>
      </main>
    );
  }

  const currentStageObj = bill.stages.find((s) => s.status === "current");

  return (
    <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      {/* Header Banner */}
      <header className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-amber-700">
              <span className="rounded-full bg-amber-100 px-2.5 py-0.5">
                Historical Pre-Parliament Simulation (As of{" "}
                {bill.simulated_as_of_date})
              </span>
              <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-slate-700">
                UK Government Bill
              </span>
            </div>
            <h1 className="mt-2 text-2xl font-bold text-slate-900">
              {bill.short_title}
            </h1>
            <p className="mt-1 text-sm text-slate-600 max-w-3xl">
              {bill.long_title}
            </p>
            <p className="mt-2 text-xs text-slate-500">
              <span className="font-semibold text-slate-700">
                Sponsoring Department:
              </span>{" "}
              {bill.sponsoring_department}
            </p>
          </div>

          {/* Forecast Summary Cards */}
          <div className="flex flex-wrap gap-3 shrink-0">
            <div className="rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-center">
              <div className="text-xs font-medium text-blue-700">
                Current Position
              </div>
              <div className="mt-0.5 text-base font-bold text-blue-950">
                Stage {bill.current_stage_order} / {bill.total_stages}
              </div>
              <div className="text-xs text-blue-800">
                {currentStageObj?.label}
              </div>
            </div>

            <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-center">
              <div className="text-xs font-medium text-emerald-700">
                Passage Probability
              </div>
              <div className="mt-0.5 text-xl font-bold text-emerald-900">
                {Math.round(bill.passage_probability * 100)}%
              </div>
              <div className="text-xs text-emerald-800">To Royal Assent</div>
            </div>

            <div className="rounded-lg border border-purple-200 bg-purple-50 px-4 py-3 text-center">
              <div className="text-xs font-medium text-purple-700">
                Est. Time to Commencement
              </div>
              <div className="mt-0.5 text-xl font-bold text-purple-900">
                {bill.estimated_months_to_commencement ?? "—"} mos
              </div>
              <div className="text-xs text-purple-800">From snapshot date</div>
            </div>
          </div>
        </div>

        {/* Interactive Stage Switcher Bar */}
        <div className="mt-5 flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-4 text-xs text-slate-600">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded-full bg-emerald-500"></span>
              Completed Stage
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded-full bg-amber-500 ring-4 ring-amber-200"></span>
              <strong>Current Highlighted Stage</strong>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded-full bg-slate-300"></span>
              Upcoming Stage
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded border border-dashed border-red-400 bg-red-50"></span>
              Where a bill can stop
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-500">
              Click any stage card below to test moving the bill across stages, or:
            </span>
            <button
              onClick={() => handleSelectStage("drafting")}
              disabled={updatingStage || bill.current_stage === "drafting"}
              className="rounded border border-slate-300 bg-slate-50 px-2.5 py-1 font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
            >
              Reset to Pre-Parliament Drafting (Stage 4)
            </button>
          </div>
        </div>
      </header>

      {/* 3-Column Flowchart Pipeline */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        {PHASE_META.map((phaseInfo) => {
          const phaseStages = bill.stages.filter(
            (s) => s.phase === phaseInfo.key
          );
          const isCurrentPhase = bill.current_phase === phaseInfo.key;

          return (
            <div
              key={phaseInfo.key}
              className={`rounded-xl border p-4 transition-all ${
                isCurrentPhase
                  ? "border-amber-400 bg-amber-50/30 shadow-md"
                  : "border-slate-200 bg-white shadow-sm"
              }`}
            >
              <div className="mb-4 border-b border-slate-200 pb-3">
                <div className="flex items-center justify-between">
                  <h2 className="text-base font-bold text-slate-900">
                    {phaseInfo.title}
                  </h2>
                  {isCurrentPhase && (
                    <span className="rounded-full bg-amber-500 px-2.5 py-0.5 text-xs font-bold text-white">
                      ACTIVE PHASE
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-500 mt-0.5">
                  {phaseInfo.subtitle}
                </p>
              </div>

              <div className="space-y-3">
                {phaseStages.map((stage, idx) => {
                  const isCompleted = stage.status === "completed";
                  const isCurrent = stage.status === "current";

                  return (
                    <div key={stage.stage} className="relative">
                      <button
                        type="button"
                        onClick={() => handleSelectStage(stage.stage)}
                        disabled={updatingStage}
                        className={`w-full text-left rounded-lg border p-3.5 transition-all ${
                          isCurrent
                            ? "border-2 border-amber-500 bg-amber-50 shadow-md ring-4 ring-amber-200/70"
                            : isCompleted
                            ? "border-emerald-300 bg-emerald-50/70 hover:bg-emerald-50"
                            : "border-slate-200 bg-slate-50/60 opacity-80 hover:opacity-100 hover:bg-slate-100/70"
                        }`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span
                              className={`inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
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
                              <div className="text-sm font-semibold text-slate-900">
                                {stage.label}
                              </div>
                              <div className="text-xs text-slate-600">
                                {stage.description}
                              </div>
                            </div>
                          </div>

                          <span
                            className={`shrink-0 rounded px-2 py-0.5 text-[11px] font-semibold uppercase ${
                              isCurrent
                                ? "bg-amber-500 text-white"
                                : isCompleted
                                ? "bg-emerald-100 text-emerald-800"
                                : "bg-slate-200 text-slate-600"
                            }`}
                          >
                            {isCurrent
                              ? "Current Stage"
                              : isCompleted
                              ? "Done"
                              : "Upcoming"}
                          </span>
                        </div>

                        {/* Historical dates & evidence if present */}
                        {(stage.entered_date || stage.summary) && (
                          <div className="mt-2.5 rounded border border-slate-200/80 bg-white/90 p-2.5 text-xs space-y-1">
                            {stage.entered_date && (
                              <div className="font-medium text-slate-700">
                                {stage.entered_date}
                                {stage.completed_date
                                  ? ` → ${stage.completed_date}`
                                  : " → Present (In Progress)"}
                              </div>
                            )}
                            {stage.summary && (
                              <p className="text-slate-600 leading-relaxed">
                                {stage.summary}
                              </p>
                            )}
                            {stage.source_reference && (
                              <div className="text-[11px] text-slate-400 italic">
                                Source: {stage.source_reference}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Dashed Red Stop Risks ("Where it stops") */}
                        {stage.stop_risks.length > 0 && (
                          <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                            <span className="text-[11px] font-medium text-red-700">
                              Stop risks:
                            </span>
                            {stage.stop_risks.map((risk) => (
                              <span
                                key={risk}
                                className="rounded border border-dashed border-red-400 bg-red-50 px-2 py-0.5 text-[11px] font-medium text-red-800"
                              >
                                ⇢ {formatStopRisk(risk)}
                              </span>
                            ))}
                          </div>
                        )}
                      </button>

                      {/* Connector arrow between steps in the column */}
                      {idx < phaseStages.length - 1 && (
                        <div className="flex justify-center py-1 text-slate-300 text-xs leading-none">
                          ↓
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </section>
    </main>
  );
}
