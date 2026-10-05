import type { CaseRecord } from "../../types/lesioniq";

export function CaseSelector({
  cases,
  selectedId,
  onSelect
}: {
  cases: CaseRecord[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-sm font-medium text-clinical-muted">
      Case
      <select
        value={selectedId ?? "__intake__"}
        onChange={(event) => onSelect(event.target.value)}
        className="min-w-[220px] rounded-md border border-clinical-line bg-clinical-surface px-3 py-2 text-sm font-semibold text-clinical-ink outline-none hover:border-clinical-accent/45 focus-visible:ring-2 focus-visible:ring-clinical-accent/50"
      >
        <option value="__intake__">Upload intake - no analysis</option>
        {cases.map((caseRecord) => (
          <option key={caseRecord.id} value={caseRecord.id}>
            {caseRecord.caseId} · {caseRecord.predictedClassCode}
          </option>
        ))}
      </select>
    </label>
  );
}

