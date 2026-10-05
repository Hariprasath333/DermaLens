import type { CompareEntry, OverlayMode } from "../../types/lesioniq";
import { pct } from "../../lib/format";
import { Card } from "../primitives/Card";
import { StatusBadge } from "../primitives/StatusBadge";
import { DermoscopyMock } from "./DermoscopyMock";

export function ComparePanel({ entry, overlay, imageUrl }: { entry: CompareEntry; overlay: OverlayMode; imageUrl?: string }) {
  return (
    <Card title={entry.label} eyebrow={entry.date}>
      <DermoscopyMock compact overlay={overlay} imageUrl={imageUrl} />
      <div className="mt-3 grid gap-2 text-sm">
        <div className="flex justify-between gap-3"><span className="text-clinical-muted">Predicted class</span><strong className="text-clinical-ink">{entry.predictedClassCode} · {entry.predictedClassLabel}</strong></div>
        <div className="flex justify-between gap-3"><span className="text-clinical-muted">Calibrated confidence</span><strong className="tabular-nums text-clinical-ink">{pct(entry.calibratedConfidence)}</strong></div>
        <div className="flex justify-between gap-3"><span className="text-clinical-muted">Review status</span><StatusBadge label={entry.reviewStatus} tone={entry.reviewStatus === "Reviewed" ? "success" : "warning"} /></div>
      </div>
      <p className="mt-3 text-sm text-clinical-muted">{entry.summary}</p>
    </Card>
  );
}

