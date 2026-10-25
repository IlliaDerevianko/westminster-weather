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
    latest_stage?: string | null; // Acts only: furthest stage completed after Royal Assent
    latest_stage_label?: string | null;
}
const OUTCOME_LABELS: Record<string, string> = { royal_assent: "Royal Assent", in_progress: "In progress", fell_dissolution: "Fell at dissolution", fell_session_end: "Fell at session end", withdrawn: "Withdrawn", defeated: "Defeated" };
const OUTCOME_GROUPS = [
    { key: "all", label: "All bills", matches: () => true },
    { key: "in_progress", label: "In progress", matches: (outcome: string) => outcome === "in_progress" },
    { key: "acts", label: "Acts", matches: (outcome: string) => outcome === "royal_assent" },
    { key: "stopped", label: "Stopped", matches: (outcome: string) => outcome !== "royal_assent" && outcome !== "in_progress" },
] as const;
type OutcomeGroupKey = (typeof OUTCOME_GROUPS)[number]["key"];
export default function BillsDirectoryPage() {
    const [bills, setBills] = useState<BillSummary[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const [search, setSearch] = useState<string>("");
    const [outcomeGroup, setOutcomeGroup] = useState<OutcomeGroupKey>("all");
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
            }
            catch (err) {
                setError(err instanceof Error
                    ? err.message
                    : "Failed to load bills");
            }
            finally {
                setLoading(false);
            }
        }
        loadBills();
    }, []);
    const filteredBills = useMemo(() => {
        const q = search.trim().toLowerCase();
        const group = OUTCOME_GROUPS.find((g) => g.key === outcomeGroup) ?? OUTCOME_GROUPS[0];
        return bills.filter((b) => {
            if (!group.matches(b.outcome)) {
                return false;
            }
            if (houseFilter !== "all" && b.originating_house !== houseFilter) {
                return false;
            }
            if (!q) {
                return true;
            }
            return (b.short_title.toLowerCase().includes(q) ||
                (b.department ?? "").toLowerCase().includes(q) ||
                String(b.bill_id).includes(q));
        });
    }, [bills, search, outcomeGroup, houseFilter]);
    const counts = useMemo(() => Object.fromEntries(OUTCOME_GROUPS.map((g) => [g.key, bills.filter((b) => g.matches(b.outcome)).length])) as Record<OutcomeGroupKey, number>, [bills]);
    const resetFilters = () => { setSearch(""); setHouseFilter("all"); setOutcomeGroup("all"); };
    return (<main className="page-shell directory">
      <header className="directory-heading"><p className="eyebrow">UK legislation</p><h1>Government bills</h1><p className="intro">Follow a bill from its first proposal to a law in force.</p></header>
      <section aria-label="Find a bill" className="directory-controls">
        <div className="search-row"><div><label htmlFor="bill-search">Find a bill</label><input id="bill-search" type="search" value={search} onChange={e => setSearch(e.target.value)} placeholder="Search by title, department or bill number"/></div><div><label htmlFor="house-filter">Originating House</label><select id="house-filter" value={houseFilter} onChange={e => setHouseFilter(e.target.value)}><option value="all">Both Houses</option><option value="Commons">Commons</option><option value="Lords">Lords</option></select></div></div>
        <div className="filter-tabs" aria-label="Filter by outcome">{OUTCOME_GROUPS.map(group => <button key={group.key} aria-pressed={outcomeGroup === group.key} onClick={() => setOutcomeGroup(group.key)}>{group.label}<span>{loading ? "—" : counts[group.key]}</span></button>)}</div>
      </section>
      {error ? <p role="alert" className="error-message">Couldn’t load bills. Please refresh to try again.</p> : loading ? <p role="status" className="empty-state">Loading government bills…</p> : <section aria-label="Government bills">
        <div className="results-heading"><p aria-live="polite">{filteredBills.length} {filteredBills.length === 1 ? "bill" : "bills"}{filteredBills.length !== bills.length && ` of ${bills.length}`}</p>{(search || houseFilter !== "all" || outcomeGroup !== "all") && <button className="text-link" onClick={resetFilters}>Clear filters</button>}</div>
        <ul className="bill-list">{filteredBills.map(bill => <li key={bill.bill_id}>
          <Link href={`/?bill_id=${bill.bill_id}`} className="bill-row"><div><h2>{bill.short_title}</h2><p>{bill.department ?? "HM Government"}<span className="meta-divider">·</span>{bill.originating_house}</p></div><span className={`directory-status ${bill.outcome === "in_progress" ? "active" : ""}`}>{bill.latest_stage && bill.latest_stage !== "royal_assent" && bill.latest_stage_label ? bill.latest_stage_label : OUTCOME_LABELS[bill.outcome] ?? bill.outcome.replace(/_/g, " ")}</span><span className="row-arrow" aria-hidden="true">↗</span></Link>
        </li>)}</ul>
        {filteredBills.length === 0 && <div className="empty-state"><h2>No matching bills</h2><p>Try a different title or department, or clear your filters.</p><button className="text-link" onClick={resetFilters}>Clear filters</button></div>}
      </section>}
    </main>);
}
