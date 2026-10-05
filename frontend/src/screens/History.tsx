import { useMemo, useState } from "react";
import type { CaseRecord } from "../types/lesioniq";
import type { HistoryFilter } from "../components/domain/HistoryList";
import { HistoryList } from "../components/domain/HistoryList";
import { Card } from "../components/primitives/Card";
import { PageHeader } from "../components/primitives/PageHeader";
import { SectionTabs } from "../components/primitives/SectionTabs";

const filters: Array<{ id: HistoryFilter; label: string }> = [
  { id: "patient", label: "This Patient Only" },
  { id: "all", label: "All Cases" },
  { id: "high", label: "High Concern" }
];

export function History({ caseRecord, onNavigateCompare }: { caseRecord: CaseRecord; onNavigateCompare: () => void }) {
  const [filter, setFilter] = useState<HistoryFilter>("patient");
  const entries = useMemo(() => {
    if (filter === "patient") return caseRecord.historyEntries.filter((entry) => entry.samePatient);
    if (filter === "high") return caseRecord.historyEntries.filter((entry) => entry.urgency === "High concern");
    return caseRecord.historyEntries;
  }, [caseRecord.historyEntries, filter]);

  return (
    <>
      <PageHeader title="History" subtitle={`Current filter: ${filters.find((item) => item.id === filter)?.label}. Prior analyses are de-identified and scoped to review context.`} />
      <div className="mb-4"><SectionTabs tabs={filters} value={filter} onChange={setFilter} ariaLabel="History filters" /></div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.3fr)_380px]">
        <HistoryList entries={entries} onCompare={onNavigateCompare} />
        <Card title="Supporting context" eyebrow="Review stream">
          <div className="space-y-3 text-sm text-clinical-muted">
            <p><strong className="text-clinical-ink">Recent actions:</strong> selected case is awaiting dermatologist verification before report export.</p>
            <p><strong className="text-clinical-ink">High concern summary:</strong> melanoma-like outputs are routed to senior review in this prototype workflow.</p>
            <p><strong className="text-clinical-ink">Quick compare:</strong> use View in Compare to inspect current vs previous evidence panels.</p>
            <p><strong className="text-clinical-ink">Clinician note:</strong> {caseRecord.clinicianNotesPreview ?? "No note preview attached."}</p>
          </div>
        </Card>
      </div>
    </>
  );
}

