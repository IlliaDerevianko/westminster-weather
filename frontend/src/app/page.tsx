"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import React, { Suspense, useEffect, useMemo, useState } from "react";
import BillOutlookPanel from "./BillOutlookPanel";
type StageStatus = "completed" | "fast_tracked" | "current" | "upcoming" | "stopped" | "skipped";
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
    full_text_url?: string | null;
    full_text_label?: string | null;
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
const STOP_RISK_DESCRIPTIONS: Record<string, {
    title: string;
    subtitle: string;
}> = {
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
    title: string;
    subtitle: string;
}[] = [
    { key: "before_parliament", title: "Before Parliament", subtitle: "From the first signal to a place on the legislative agenda." },
    { key: "inside_parliament", title: "In Parliament", subtitle: "Readings, scrutiny and agreement in the Commons and Lords." },
    { key: "after_royal_assent", title: "After Royal Assent", subtitle: "From an Act of Parliament to a law in force." },
];
const formatDate = (date: string) => new Date(`${date}T00:00:00`).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
function BillTrackerContent() {
    const searchParams = useSearchParams();
    const queryBillId = searchParams.get("bill_id");
    const [selectedPhase, setSelectedPhase] = useState<PhaseKey | null>(null);
    const [activeBillId, setActiveBillId] = useState<string | null>(queryBillId);
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
                if (!res.ok)
                    throw new Error("Unable to load the bill directory.");
                const data: BillSummary[] = await res.json();
                // No bill in the URL: open the most recently introduced bill (the list's first)
                if (!queryBillId && data.length > 0) {
                    setActiveBillId(String(data[0].bill_id));
                }
                else if (!queryBillId) {
                    setError("No government bills are available yet.");
                    setLoading(false);
                }
            }
            catch {
                if (!queryBillId) {
                    setError("Unable to load bills. Please try again.");
                    setLoading(false);
                }
            }
        }
        loadDirectory();
    }, [queryBillId]);
    // Sync activeBillId when URL query parameter changes
    useEffect(() => {
        if (queryBillId) {
            setActiveBillId(queryBillId);
            setSelectedDate("");
            setSelectedPhase(null);
        }
    }, [queryBillId]);
    // Fetch bill state from Neon Lakebase Postgres
    useEffect(() => {
        if (!activeBillId)
            return;
        let cancelled = false;
        async function fetchBill() {
            try {
                setLoading(true);
                const query = selectedDate
                    ? `?as_of_date=${encodeURIComponent(selectedDate)}`
                    : "";
                const res = await fetch(`/api/v1/bills/${encodeURIComponent(activeBillId!)}${query}`);
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
            }
            catch (err) {
                if (!cancelled) {
                    setError(err instanceof Error
                        ? err.message
                        : "Failed to fetch bill data");
                }
            }
            finally {
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
        if (!bill)
            return grouped;
        for (const s of bill.stages) {
            grouped[s.phase]?.push(s);
        }
        return grouped;
    }, [bill]);
    const currentStageObj = bill?.stages.find((s) => s.stage === bill.current_stage);
    const isStopped = currentStageObj !== undefined && currentStageObj.status === "stopped";
    // `outcome` stops at Royal Assent; once the Act is fully in force (as of the selected
    // date) that is the furthest point reached, so show it instead
    const isLawFullyInForce = bill?.stages.find((s) => s.stage === "law_fully_in_force")?.status ===
        "completed";
    const overallOutcome = !bill
        ? null
        : isLawFullyInForce
            ? "Law fully in force"
            : OUTCOME_LABELS[bill.outcome] ?? bill.outcome.replace(/_/g, " ");
    const visiblePhase = selectedPhase ?? bill?.current_phase ?? "inside_parliament";
    return (<main className="page-shell">
      <Link href="/bills" className="back-link">← All government bills</Link>
      {error && <p role="alert" className="error-message">{error}</p>}
      {loading && !bill && <p role="status" className="empty-state">Loading bill…</p>}
      {bill && <>
        <header className="bill-heading">
          <p className="eyebrow">{bill.originating_house} · {bill.sponsoring_department}</p>
          <h1>{bill.short_title}</h1>
          <div className="heading-meta"><span className="status-label">{overallOutcome}</span>
            <a className="text-link" href={bill.full_text_url ?? `https://bills.parliament.uk/bills/${bill.parliament_bill_id}/publications`} target="_blank" rel="noopener noreferrer">Read the bill ↗</a>
          </div>
          <details className="bill-description"><summary>About this bill</summary><p>{bill.long_title}</p><p>Parliament Bill {bill.parliament_bill_id}</p><p>First signal: {formatDate(bill.earliest_signal_date)}{bill.outcome_date && ` · Outcome: ${formatDate(bill.outcome_date)}`}{bill.introduced_session_id && ` · Session ${bill.introduced_session_id}`}</p></details>
        </header>
        <section className="snapshot-bar" aria-label="Bill position and date">
          <div><p className="eyebrow">{isStopped ? "Stopped at" : "Current position"}</p><strong>{currentStageObj?.label ?? "Before first stage"}</strong><p className="muted">As of {formatDate(bill.as_of_date)}</p></div>
          <details className="snapshot-controls"><summary>View a historical snapshot</summary>
            <label htmlFor="historical-date-picker">Snapshot date</label>
            <input id="historical-date-picker" type="date" value={selectedDate} max={latestDate || undefined} onChange={(e) => { if (e.target.value)
            setSelectedDate(e.target.value); }}/>
            <label htmlFor="milestone-picker">Or jump to a milestone</label>
            <select id="milestone-picker" value="" onChange={(e) => { if (e.target.value)
            setSelectedDate(e.target.value); }}><option value="">Choose a milestone</option>{bill.milestone_dates.map((m) => <option key={`${m.date}-${m.label}`} value={m.date}>{m.label} · {formatDate(m.date)}</option>)}</select>
            <button className="text-link" onClick={() => setSelectedDate(latestDate)}>Back to latest</button>
          </details>
          {loading && <span role="status" className="muted">Updating…</span>}
        </section>
        {activeBillId && <BillOutlookPanel billId={activeBillId} asOf={selectedDate && selectedDate !== latestDate ? selectedDate : undefined}/>}
        <section className="journey" aria-labelledby="journey-title">
          <div className="section-heading"><h2 id="journey-title">The bill’s journey</h2><span className="muted">Stage {bill.current_stage_order} of {bill.total_stages}</span></div>
          <div className="phase-tabs" aria-label="Timeline phase">
            {PHASE_CONFIG.map((phase, index) => <button key={phase.key} aria-pressed={visiblePhase === phase.key} onClick={() => setSelectedPhase(phase.key)}><span className="phase-number">0{index + 1}</span>{phase.title}</button>)}
          </div>
          <details key={visiblePhase} className="phase-description">
            <summary>About this phase</summary>
            <p>{PHASE_CONFIG.find(p => p.key === visiblePhase)?.subtitle}</p>
          </details>
          <ol className="timeline">
            {stagesByPhase[visiblePhase].map(stage => <li key={stage.stage} className={`timeline-stage stage-${stage.status}`}>
              <span className="stage-marker" aria-hidden="true">{stage.status === "completed" ? "✓" : stage.order}</span>
              <details key={`${bill.id}-${stage.stage}-${stage.status}`} open={stage.status === "current" || stage.status === "stopped"}>
                <summary><span className="stage-title">{stage.label}</span><span className="stage-status">{{ completed: "Completed", current: "Current", upcoming: "Upcoming", stopped: "Stopped", skipped: "Skipped", fast_tracked: "Fast-tracked" }[stage.status]}</span></summary>
                <div className="stage-content"><p>{stage.description}</p>
                  {stage.summary && <p>{stage.summary}</p>}
                  {stage.entered_date && <p className="muted">{formatDate(stage.entered_date)}{stage.completed_date && stage.completed_date !== stage.entered_date && ` – ${formatDate(stage.completed_date)}`}</p>}
                  {stage.active_stop_reason && <p className="stop-reason">{STOP_RISK_DESCRIPTIONS[stage.active_stop_reason]?.title ?? stage.active_stop_reason.replace(/_/g, " ")}: {STOP_RISK_DESCRIPTIONS[stage.active_stop_reason]?.subtitle}</p>}
                  {stage.visible_events.length > 0 && <details className="evidence"><summary>{stage.visible_events.length} recorded {stage.visible_events.length === 1 ? "event" : "events"}</summary><ul>{stage.visible_events.map((event, i) => <li key={`${event.event_date}-${i}`}><span className="muted">{formatDate(event.event_date)}</span><p><strong>{event.title}</strong></p><p>{event.note}</p>{event.web_url && <a className="text-link" href={event.web_url} target="_blank" rel="noopener noreferrer">View source ↗</a>}</li>)}</ul></details>}
                </div>
              </details>
            </li>)}
          </ol>
        </section>
      </>}
    </main>);
}
export default function BillTrackerPage() {
    return (<Suspense fallback={<main className="max-w-7xl mx-auto px-6 py-8 text-sm text-slate-600">
          Loading bill tracker...
        </main>}>
      <BillTrackerContent />
    </Suspense>);
}
