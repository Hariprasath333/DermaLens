import type { CaseRecord } from "../../types/lesioniq";
import { Card } from "../primitives/Card";

export function MetadataCard({ caseRecord }: { caseRecord: CaseRecord }) {
  const rows = [
    ["Age", `${caseRecord.metadata.ageYears} years`],
    ["Sex", caseRecord.metadata.sex],
    ["Anatomical site", caseRecord.metadata.anatomicalSite],
    ["Fitzpatrick type", caseRecord.metadata.FitzpatrickType ?? "Unknown"]
  ];
  return (
    <Card title="Metadata" eyebrow="Fusion branch context">
      <dl className="grid gap-3 sm:grid-cols-2">
        {rows.map(([label, value]) => (
          <div key={label} className="rounded-md border border-clinical-line bg-clinical-raised p-3">
            <dt className="text-xs font-semibold uppercase tracking-[0.12em] text-clinical-muted">{label}</dt>
            <dd className="mt-1 font-semibold text-clinical-ink">{value}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-3 space-y-2">
        {caseRecord.lesionMetrics.map((metric) => (
          <div key={metric.label} className="flex justify-between gap-3 border-t border-clinical-line pt-2 text-sm">
            <span className="text-clinical-muted">{metric.label}</span>
            <span className="text-right font-medium text-clinical-ink">{metric.value}</span>
          </div>
        ))}
      </div>
      {caseRecord.clinicianNotesPreview && <p className="mt-3 text-sm text-clinical-muted">{caseRecord.clinicianNotesPreview}</p>}
    </Card>
  );
}

