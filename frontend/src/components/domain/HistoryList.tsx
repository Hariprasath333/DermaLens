import { useState } from "react";
import type { HistoryEntry } from "../../types/lesioniq";
import { pct } from "../../lib/format";
import { StatusBadge } from "../primitives/StatusBadge";

export type HistoryFilter = "patient" | "all" | "high";

export function HistoryList({ entries, onCompare }: { entries: HistoryEntry[]; onCompare?: (id: string) => void }) {
  const [selected, setSelected] = useState(entries[0]?.id);
  return (
    <div className="control-surface overflow-hidden rounded-clinical border border-clinical-line shadow-clinical">
      <div className="grid grid-cols-[1.1fr_1fr_.8fr_.9fr_.9fr] gap-3 border-b border-clinical-line px-4 py-3 text-xs font-semibold uppercase tracking-[0.12em] text-clinical-muted max-lg:hidden">
        <span>Date</span><span>Prediction</span><span>Confidence</span><span>Status</span><span>Action</span>
      </div>
      <div className="divide-y divide-clinical-line">
        {entries.map((entry) => (
          <button
            key={entry.id}
            type="button"
            onClick={() => setSelected(entry.id)}
            className={`grid w-full gap-3 px-4 py-3 text-left outline-none transition hover:bg-clinical-soft focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-clinical-accent/50 lg:grid-cols-[1.1fr_1fr_.8fr_.9fr_.9fr] ${selected === entry.id ? "bg-clinical-accentSoft" : ""}`}
          >
            <span className="font-medium text-clinical-ink">{entry.date}</span>
            <span className="text-clinical-ink">{entry.predictedClassCode} · {entry.predictedClassLabel}</span>
            <span className="tabular-nums text-clinical-ink">{pct(entry.confidence)}</span>
            <span><StatusBadge label={entry.urgency} tone={entry.urgency === "High concern" ? "danger" : entry.urgency === "Expedited review" ? "warning" : "success"} /></span>
            <span
              onClick={(event) => { event.stopPropagation(); onCompare?.(entry.id); }}
              className="font-semibold text-clinical-accent"
            >
              View in Compare
            </span>
            <span className="lg:col-span-5 text-sm text-clinical-muted">{entry.note}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

