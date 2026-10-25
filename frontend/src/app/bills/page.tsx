"use client";

import Link from "next/link";
import React, { useEffect, useMemo, useState } from "react";

interface BillSummary {
  bill_id: number;
  short_title: string;
  originating_house: string;
  introduced_session_id: number;
  earliest_signal_date?: string | null;
  introduced_date: string | null;
  department: string | null;
  outcome: string;
  outcome_date: string | null;
  stop_reason: string | null;
  is_act: boolean;
  is_defeated: boolean;
  withdrawn_date: string | null;
  has_pre_parliament_data: boolean;
}

const OUTCOME_BADGES: Record<string, { label: string; className: string }> = {
  royal_assent: {
    label: "Royal Assent (Act)",
    className: "bg-emerald-100 text-emerald-800 border-emerald-200",
  },
  in_progress: {
    label: "In Progress",
    className: "bg-amber-100 text-amber-800 border-amber-200",
  },
  fell_dissolution: {
    label: "Fell (Dissolution)",
    className: "bg-red-100 text-red-800 border-red-200",
  },
  fell_session_end: {
    label: "Fell (Session End)",
    className: "bg-rose-100 text-rose-800 border-rose-200",
  },
  withdrawn: {
    label: "Withdrawn",
    className: "bg-slate-200 text-slate-700 border-slate-300",
  },
  defeated: {
    label: "Defeated",
    className: "bg-red-200 text-red-900 border-red-300",
  },
};

export default function BillsDirectoryPage() {
  const [bills, setBills] = useState<BillSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState<string>("");
  const [outcomeFilter, setOutcomeFilter] = useState<string>("all");
  const [houseFilter, setHouseFilter] = useState<string>("all");

  useEffect(() => {
    async function loadBills() {
      try {
        setLoading(true);
        const res = await fetch("/api/v1/bills");
        if (!res.ok) {
          throw new Error(`API returned status ${res.status}`);
        }
        const data: BillSummary[] = await res.json();
        setBills(data);
        setError(null);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to load bills"
        );
      } finally {
        setLoading(false);
      }
    }
    loadBills();
  }, []);

  const filteredBills = useMemo(() => {
    const q = search.trim().toLowerCase();
    return bills.filter((b) => {
      if (outcomeFilter !== "all" && b.outcome !== outcomeFilter) {
        return false;
      }
      if (houseFilter !== "all" && b.originating_house !== houseFilter) {
        return false;
      }
      if (!q) {
        return true;
      }
      return (
        b.short_title.toLowerCase().includes(q) ||
        (b.department ?? "").toLowerCase().includes(q) ||
        String(b.bill_id).includes(q)
      );
    });
  }, [bills, search, outcomeFilter, houseFilter]);

  const counts = useMemo(() => {
    return {
      total: bills.length,
      acts: bills.filter((b) => b.outcome === "royal_assent").length,
      inProgress: bills.filter((b) => b.outcome === "in_progress").length,
      stopped: bills.filter(
        (b) => b.outcome !== "royal_assent" && b.outcome !== "in_progress"
      ).length,
    };
  }, [bills]);

  return (
    <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">
      <header className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider">
              <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-slate-700">
                {counts.total} UK Government Bills
              </span>
            </div>
            <h1 className="mt-2 text-2xl font-bold text-slate-900">
              Select a UK Government Bill
            </h1>
            <p className="mt-1 text-sm text-slate-600">
              Choose any bill to inspect its
              stage-by-stage timeline and historical snapshot dates on the
              tracker page.
            </p>
          </div>
          <Link
            href="/"
            className="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-slate-50 px-4 py-2 text-sm font-semibold text-slate-800 hover:bg-slate-100 transition-colors shrink-0"
          >
            ← Back to Tracker Page
          </Link>
        </div>

        {/* Summary Counters */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="text-xs font-medium text-slate-500">
              Total Government Bills
            </div>
            <div className="text-xl font-bold text-slate-900">
              {counts.total}
            </div>
          </div>
          <div className="rounded-lg border border-emerald-200 bg-emerald-50/60 p-3">
            <div className="text-xs font-medium text-emerald-700">
              Royal Assent (Acts)
            </div>
            <div className="text-xl font-bold text-emerald-900">
              {counts.acts}
            </div>
          </div>
          <div className="rounded-lg border border-amber-200 bg-amber-50/60 p-3">
            <div className="text-xs font-medium text-amber-700">
              Currently In Progress
            </div>
            <div className="text-xl font-bold text-amber-900">
              {counts.inProgress}
            </div>
          </div>
          <div className="rounded-lg border border-rose-200 bg-rose-50/60 p-3">
            <div className="text-xs font-medium text-rose-700">
              Fell / Withdrawn
            </div>
            <div className="text-xl font-bold text-rose-900">
              {counts.stopped}
            </div>
          </div>
        </div>

        {/* Search & Filters */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-3 pt-2">
          <div className="md:col-span-6">
            <label
              htmlFor="bill-search"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1"
            >
              Search by Title, Department, or Bill ID
            </label>
            <input
              id="bill-search"
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="e.g. Online Safety, Railways, Home Office, 3137..."
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
            />
          </div>
          <div className="md:col-span-3">
            <label
              htmlFor="outcome-filter"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1"
            >
              Outcome
            </label>
            <select
              id="outcome-filter"
              value={outcomeFilter}
              onChange={(e) => setOutcomeFilter(e.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-blue-500 focus:outline-none"
            >
              <option value="all">All Outcomes</option>
              <option value="in_progress">In Progress</option>
              <option value="royal_assent">Royal Assent (Act)</option>
              <option value="fell_dissolution">Fell (Dissolution)</option>
              <option value="fell_session_end">Fell (Session End)</option>
              <option value="withdrawn">Withdrawn</option>
            </select>
          </div>
          <div className="md:col-span-3">
            <label
              htmlFor="house-filter"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1"
            >
              Originating House
            </label>
            <select
              id="house-filter"
              value={houseFilter}
              onChange={(e) => setHouseFilter(e.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-blue-500 focus:outline-none"
            >
              <option value="all">Both Houses</option>
              <option value="Commons">Commons</option>
              <option value="Lords">Lords</option>
            </select>
          </div>
        </div>
      </header>

      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-800">
          <strong>Error:</strong> {error}
        </div>
      )}

      <section className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
        <div className="border-b border-slate-200 px-6 py-3 flex items-center justify-between bg-slate-50">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-600">
            Showing {filteredBills.length} of {bills.length} Government Bills
          </span>
          {loading && (
            <span className="text-xs font-medium text-blue-600">
              Loading bills...
            </span>
          )}
        </div>

        <div className="divide-y divide-slate-200">
          {filteredBills.map((b) => {
            const badge = OUTCOME_BADGES[b.outcome] ?? {
              label: b.outcome,
              className: "bg-slate-100 text-slate-700 border-slate-200",
            };
            return (
              <div
                key={b.bill_id}
                className="p-4 sm:px-6 hover:bg-slate-50 transition-colors flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
              >
                <div className="space-y-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-xs font-semibold text-slate-500">
                      #{b.bill_id}
                    </span>
                    <span className="text-base font-bold text-slate-900">
                      {b.short_title}
                    </span>
                    <span
                      className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold ${badge.className}`}
                    >
                      {badge.label}
                    </span>
                    <span className="rounded bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700">
                      {b.originating_house} (Session {b.introduced_session_id})
                    </span>
                  </div>
                  <div className="text-xs text-slate-600 flex flex-wrap gap-x-4 gap-y-1">
                    <span>
                      <strong>Department:</strong>{" "}
                      {b.department ?? "HM Government"}
                    </span>
                    {b.earliest_signal_date && (
                      <span>
                        <strong>First Signal:</strong> {b.earliest_signal_date}
                      </span>
                    )}
                    {b.outcome_date && (
                      <span>
                        <strong>Outcome Date:</strong> {b.outcome_date}
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2 shrink-0">
                  {b.has_pre_parliament_data && (
                    <span className="rounded-full border border-amber-300 bg-amber-50 px-2.5 py-0.5 text-[11px] font-semibold text-amber-900">
                      Includes GOV.UK Pre-Parliament Events
                    </span>
                  )}
                  <Link
                    href={`/?bill_id=${b.bill_id}`}
                    className="rounded-lg bg-slate-900 px-3.5 py-1.5 text-xs font-semibold text-white hover:bg-slate-800 transition-colors"
                  >
                    Display on 3-Phase Tracker →
                  </Link>
                </div>

              </div>
            );
          })}

          {!loading && filteredBills.length === 0 && (
            <div className="p-8 text-center text-sm text-slate-500">
              No bills match your filter criteria.
            </div>
          )}
        </div>
      </section>
    </main>
  );
}
