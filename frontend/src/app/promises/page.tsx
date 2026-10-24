"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import React, { Suspense, useEffect, useMemo, useState } from "react";

import {
  type IdeaSummary,
  type IdeasResponse,
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

type SortKey = "pi_now_desc" | "pi_now_asc" | "date_desc" | "date_asc";

function sortValue(i: IdeaSummary): number {
  return i.taken_up ? 1 : i.pi_now;
}

function PromisesContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const asOf = searchParams.get("as_of") || todayIso();

  const [data, setData] = useState<IdeasResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [notLoaded, setNotLoaded] = useState<boolean>(false);
  const [search, setSearch] = useState<string>("");
  const [verdictFilter, setVerdictFilter] = useState<string>("all");
  const [sortKey, setSortKey] = useState<SortKey>("pi_now_desc");

  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    fetchTakeup<IdeasResponse>("ideas", { as_of: asOf }, ctrl.signal)
      .then((d) => {
        setData(d);
        setError(null);
        setNotLoaded(false);
      })
      .catch((err) => {
        if (ctrl.signal.aborted) return;
        setData(null);
        setNotLoaded(err instanceof TakeupApiError && err.status === 503);
        setError(errorMessage(err));
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
    return () => ctrl.abort();
  }, [asOf]);

  const ideas = useMemo(() => data?.ideas ?? [], [data]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const rows = ideas.filter((i) => {
      if (verdictFilter === "taken_up") {
        if (!i.taken_up) return false;
      } else if (verdictFilter !== "all") {
        if (i.taken_up || i.verdict !== verdictFilter) return false;
      }
      if (!q) return true;
      return (
        (i.topic ?? "").toLowerCase().includes(q) ||
        i.signal_id.toLowerCase().includes(q) ||
        (i.linked_bill?.title ?? "").toLowerCase().includes(q)
      );
    });
    const sorted = [...rows];
    sorted.sort((a, b) => {
      switch (sortKey) {
        case "pi_now_asc":
          return sortValue(a) - sortValue(b);
        case "date_desc":
          return b.trigger_date.localeCompare(a.trigger_date);
        case "date_asc":
          return a.trigger_date.localeCompare(b.trigger_date);
        default:
          return sortValue(b) - sortValue(a);
      }
    });
    return sorted;
  }, [ideas, search, verdictFilter, sortKey]);

  const counts = useMemo(
    () => ({
      total: ideas.length,
      takenUp: ideas.filter((i) => i.taken_up).length,
      likely: ideas.filter((i) => !i.taken_up && i.verdict === "likely").length,
      unlikely: ideas.filter((i) => !i.taken_up && i.verdict === "unlikely")
        .length,
    }),
    [ideas]
  );

  function setAsOf(value: string) {
    const params = new URLSearchParams(searchParams.toString());
    if (value) params.set("as_of", value);
    else params.delete("as_of");
    router.replace(`/promises?${params.toString()}`);
  }

  return (
    <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      <header className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2 text-xs font-semibold uppercase tracking-wider">
              <span className="rounded-full bg-blue-100 px-2.5 py-0.5 text-blue-800">
                Take-up Forecast
              </span>
              <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-slate-700">
                {counts.total} Government Promises
              </span>
            </div>
            <h1 className="mt-2 text-2xl font-bold text-slate-900">
              Will the promise become a bill?
            </h1>
            <p className="mt-1 text-sm text-slate-600 max-w-3xl">
              For each commitment made by a minister, the chance that the
              Government introduces a bill on it (first reading). &ldquo;Chance
              now&rdquo; accounts for the weeks that have already passed without
              a bill.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 shrink-0">
            <Link
              href="/"
              className="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-slate-50 px-4 py-2 text-sm font-semibold text-slate-800 hover:bg-slate-100 transition-colors"
            >
              ← Tracker
            </Link>
            <Link
              href="/bills"
              className="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-slate-50 px-4 py-2 text-sm font-semibold text-slate-800 hover:bg-slate-100 transition-colors"
            >
              All Bills
            </Link>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="text-xs font-medium text-slate-500">Promises</div>
            <div className="text-xl font-bold text-slate-900">{counts.total}</div>
          </div>
          <div className="rounded-lg border border-blue-200 bg-blue-50/60 p-3">
            <div className="text-xs font-medium text-blue-700">
              Bill Introduced
            </div>
            <div className="text-xl font-bold text-blue-900">
              {counts.takenUp}
            </div>
          </div>
          <div className="rounded-lg border border-emerald-200 bg-emerald-50/60 p-3">
            <div className="text-xs font-medium text-emerald-700">
              Likely (pending)
            </div>
            <div className="text-xl font-bold text-emerald-900">
              {counts.likely}
            </div>
          </div>
          <div className="rounded-lg border border-rose-200 bg-rose-50/60 p-3">
            <div className="text-xs font-medium text-rose-700">
              Unlikely (pending)
            </div>
            <div className="text-xl font-bold text-rose-900">
              {counts.unlikely}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-3 pt-2">
          <div className="md:col-span-5">
            <label
              htmlFor="promise-search"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1"
            >
              Search by topic or bill
            </label>
            <input
              id="promise-search"
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="e.g. water, planning, renters..."
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
            />
          </div>
          <div className="md:col-span-2">
            <label
              htmlFor="verdict-filter"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1"
            >
              Verdict
            </label>
            <select
              id="verdict-filter"
              value={verdictFilter}
              onChange={(e) => setVerdictFilter(e.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-blue-500 focus:outline-none"
            >
              <option value="all">All</option>
              <option value="likely">Likely</option>
              <option value="uncertain">Uncertain</option>
              <option value="unlikely">Unlikely</option>
              <option value="taken_up">Bill introduced</option>
            </select>
          </div>
          <div className="md:col-span-3">
            <label
              htmlFor="sort-key"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1"
            >
              Sort
            </label>
            <select
              id="sort-key"
              value={sortKey}
              onChange={(e) => setSortKey(e.target.value as SortKey)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-blue-500 focus:outline-none"
            >
              <option value="pi_now_desc">Chance now (high → low)</option>
              <option value="pi_now_asc">Chance now (low → high)</option>
              <option value="date_desc">First promised (newest)</option>
              <option value="date_asc">First promised (oldest)</option>
            </select>
          </div>
          <div className="md:col-span-2">
            <label
              htmlFor="as-of"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1"
            >
              As of
            </label>
            <input
              id="as-of"
              type="date"
              value={asOf}
              onChange={(e) => setAsOf(e.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-blue-500 focus:outline-none"
            />
          </div>
        </div>
      </header>

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

      <section className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
        <div className="border-b border-slate-200 px-6 py-3 flex items-center justify-between bg-slate-50">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-600">
            Showing {filtered.length} of {ideas.length} promises
            {data ? ` · as of ${fmtDate(data.as_of)}` : ""}
          </span>
          {loading && (
            <span className="text-xs font-medium text-blue-600">
              Loading take-up forecasts...
            </span>
          )}
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-white text-left text-xs font-semibold uppercase tracking-wider text-slate-500 border-b border-slate-200">
              <tr>
                <th className="px-4 py-2">Topic</th>
                <th className="px-4 py-2 whitespace-nowrap">First promised</th>
                <th className="px-4 py-2">Verdict</th>
                <th className="px-4 py-2 whitespace-nowrap text-right">
                  Chance now
                </th>
                <th className="px-4 py-2 whitespace-nowrap text-right">
                  Within 52 wks
                </th>
                <th className="px-4 py-2 whitespace-nowrap">Expected bill</th>
                <th className="px-4 py-2">Bill</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {filtered.map((i) => {
                const badge = verdictBadge(i.verdict);
                const href = `/promises/${encodeURIComponent(i.signal_id)}${
                  searchParams.get("as_of") ? `?as_of=${asOf}` : ""
                }`;
                return (
                  <tr
                    key={i.signal_id}
                    className="hover:bg-slate-50 transition-colors align-top"
                  >
                    <td className="px-4 py-3">
                      <Link
                        href={href}
                        className="font-semibold text-slate-900 hover:text-blue-700 hover:underline"
                      >
                        {i.topic ?? i.signal_id}
                      </Link>
                      <div className="text-xs text-slate-500 capitalize">
                        {humanSource(i.first_source)}
                      </div>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-slate-700">
                      {fmtDate(i.trigger_date)}
                    </td>
                    <td className="px-4 py-3">
                      {i.taken_up ? (
                        <span className="rounded-full border border-blue-200 bg-blue-100 px-2.5 py-0.5 text-xs font-semibold text-blue-800 whitespace-nowrap">
                          Bill introduced
                        </span>
                      ) : (
                        <span
                          className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold ${badge.className}`}
                        >
                          {badge.label}
                        </span>
                      )}
                    </td>
                    {i.taken_up ? (
                      <td
                        colSpan={3}
                        className="px-4 py-3 text-slate-700 whitespace-nowrap"
                      >
                        Bill introduced{" "}
                        {fmtDate(i.linked_bill?.first_reading_date)}
                      </td>
                    ) : (
                      <>
                        <td className="px-4 py-3 text-right font-bold text-slate-900 tabular-nums">
                          {pct(i.pi_now)}
                        </td>
                        <td className="px-4 py-3 text-right text-slate-700 tabular-nums">
                          {pct(i.p52)}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-slate-700">
                          {fmtMonth(i.t_50_date)}
                        </td>
                      </>
                    )}
                    <td className="px-4 py-3">
                      {i.linked_bill ? (
                        <Link
                          href={`/?bill_id=${i.linked_bill.bill_id}`}
                          className="text-blue-700 hover:underline"
                        >
                          {i.linked_bill.title ??
                            `Bill #${i.linked_bill.bill_id}`}
                        </Link>
                      ) : (
                        <span className="text-slate-400">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {!loading && !error && filtered.length === 0 && (
            <div className="p-8 text-center text-sm text-slate-500">
              {ideas.length === 0
                ? "No promises are visible at this date."
                : "No promises match your filter criteria."}
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

export default function PromisesPage() {
  return (
    <Suspense
      fallback={
        <main className="max-w-7xl mx-auto px-6 py-8 text-sm text-slate-600">
          Loading take-up forecasts...
        </main>
      }
    >
      <PromisesContent />
    </Suspense>
  );
}
