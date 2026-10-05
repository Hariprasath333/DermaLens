import { useEffect, useMemo, useState } from "react";
import type { CaseRecord, CompareEntry, OverlayMode } from "../types/lesioniq";
import { resolveLesionIQArtifactUrl } from "../lib/lesioniqApi";
import { ComparePanel } from "../components/domain/ComparePanel";
import { OverlayToggle } from "../components/domain/OverlayToggle";
import { Card } from "../components/primitives/Card";
import { PageHeader } from "../components/primitives/PageHeader";
import { SectionTabs } from "../components/primitives/SectionTabs";

type CompareMode = "previous" | "gradcam" | "preprocessed";
const modes: Array<{ id: CompareMode; label: string }> = [
  { id: "previous", label: "Current vs Previous" },
  { id: "gradcam", label: "Raw vs Grad-CAM" },
  { id: "preprocessed", label: "Raw vs Preprocessed" }
];

const reviewStatusSeverity: Record<CompareEntry["reviewStatus"], number> = {
  "Senior review": 3,
  "In review": 2,
  "Needs review": 1,
  Reviewed: 0
};

function entryDateValue(entry: Pick<CompareEntry, "date">): number | null {
  const value = entry.date.trim();
  if (!value) return null;

  const structured = value.match(/^(\d{4})-(\d{2})-(\d{2})(?:[T\s](\d{1,2}):(\d{2})(?::(\d{2}))?)?/);
  if (structured) {
    const [, year, month, day, hour = "0", minute = "0", second = "0"] = structured;
    const parsed = new Date(
      Number(year),
      Number(month) - 1,
      Number(day),
      Number(hour),
      Number(minute),
      Number(second)
    ).getTime();
    return Number.isNaN(parsed) ? null : parsed;
  }

  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : parsed;
}

function compareEntriesRecentFirst(a: CompareEntry, b: CompareEntry): number {
  const dateA = entryDateValue(a);
  const dateB = entryDateValue(b);
  if (dateA !== dateB) {
    if (dateA === null) return 1;
    if (dateB === null) return -1;
    return dateB - dateA;
  }

  const severityDelta = reviewStatusSeverity[b.reviewStatus] - reviewStatusSeverity[a.reviewStatus];
  if (severityDelta !== 0) return severityDelta;

  const confidenceDelta = b.calibratedConfidence - a.calibratedConfidence;
  if (confidenceDelta !== 0) return confidenceDelta;

  return a.id.localeCompare(b.id);
}

export function Compare({ caseRecord }: { caseRecord: CaseRecord }) {
  const currentEntry = useMemo<CompareEntry>(() => ({
    id: `${caseRecord.id}-active`,
    date: caseRecord.visitDate || caseRecord.acquisitionTimestamp,
    label: "Current analyzed case",
    predictedClassCode: caseRecord.predictedClassCode,
    predictedClassLabel: caseRecord.predictedClassLabel,
    calibratedConfidence: caseRecord.calibratedConfidence,
    reviewStatus: caseRecord.reviewStatus,
    summary: caseRecord.recommendation
  }), [caseRecord]);

  const compareEntries = useMemo<CompareEntry[]>(() => {
    const entries = caseRecord.compareEntries.length > 0 ? caseRecord.compareEntries : caseRecord.historyEntries.map((entry) => ({
      id: entry.id,
      date: entry.date,
      label: entry.samePatient ? "Previous same-patient review" : "Historical review",
      predictedClassCode: entry.predictedClassCode,
      predictedClassLabel: entry.predictedClassLabel,
      calibratedConfidence: entry.confidence,
      reviewStatus: entry.status,
      summary: entry.note
    }));

    return [...entries].sort(compareEntriesRecentFirst);
  }, [caseRecord.compareEntries, caseRecord.historyEntries]);

  const [mode, setMode] = useState<CompareMode>("previous");
  const [selectedId, setSelectedId] = useState(compareEntries[0]?.id ?? "");
  const [overlay, setOverlay] = useState<OverlayMode>("raw");
  const bundle = caseRecord.inferenceBundle;
  const resolvedArtifacts = useMemo(() => {
    const artifact = (url?: string) => resolveLesionIQArtifactUrl(url, bundle?.outputDirectory);

    return {
      rawArtifact: artifact(bundle?.rawArtifact ?? caseRecord.uploadedImageUrl),
      gradcamArtifact: artifact(bundle?.gradcamArtifact),
      finalPreprocessedArtifact: artifact(bundle?.finalPreprocessedArtifact ?? bundle?.originalArtifact)
    };
  }, [bundle, caseRecord.uploadedImageUrl]);
  const current = currentEntry;
  const selected = compareEntries.find((entry) => entry.id === selectedId) ?? compareEntries[0] ?? currentEntry;
  const imageForCompareEntry = (entry: CompareEntry) => {
    const descriptor = `${entry.id} ${entry.label} ${entry.summary}`.toLowerCase();
    if (descriptor.includes("gradcam") || descriptor.includes("grad-cam")) return resolvedArtifacts.gradcamArtifact;
    if (descriptor.includes("preprocess") || descriptor.includes("inference") || descriptor.includes("final")) return resolvedArtifacts.finalPreprocessedArtifact;
    if (descriptor.includes("raw") || entry.id === currentEntry.id) return resolvedArtifacts.rawArtifact;
    return undefined;
  };
  const selectedImageUrl = mode === "previous" ? imageForCompareEntry(selected) : undefined;
  const rightOverlay = mode === "gradcam" ? "raw" : mode === "preprocessed" ? "raw" : selectedImageUrl ? "raw" : overlay;
  const leftImageUrl = resolvedArtifacts.rawArtifact;
  const rightImageUrl = mode === "gradcam" ? resolvedArtifacts.gradcamArtifact : mode === "preprocessed" ? resolvedArtifacts.finalPreprocessedArtifact : selectedImageUrl;

  useEffect(() => {
    setSelectedId(compareEntries.find((entry) => entry.id !== currentEntry.id)?.id ?? compareEntries[0]?.id ?? "");
  }, [caseRecord.id, compareEntries, currentEntry.id]);

  if (!current) {
    return (
      <>
        <PageHeader title="Compare" subtitle="Compare the current case against prior evidence or alternate evidence views." />
        <Card title="No comparison records" eyebrow="Compare unavailable">
          <p className="text-sm leading-6 text-clinical-muted">This case does not include compare entries or history entries yet.</p>
        </Card>
      </>
    );
  }

  return (
    <>
      <PageHeader title="Compare" subtitle="Compare the current case against prior evidence or alternate evidence views." />
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <SectionTabs tabs={modes} value={mode} onChange={setMode} ariaLabel="Compare mode" />
        <OverlayToggle value={overlay} onChange={setOverlay} />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <ComparePanel entry={current} overlay={mode === "previous" ? overlay : "raw"} imageUrl={leftImageUrl} />
        <ComparePanel entry={mode === "previous" ? selected : current} overlay={rightOverlay} imageUrl={rightImageUrl} />
      </div>
      <Card title="Compare items" eyebrow="Selectable records" className="mt-4">
        <div className="grid gap-3 md:grid-cols-2">
          {compareEntries.map((entry) => (
            <button
              key={entry.id}
              type="button"
              onClick={() => setSelectedId(entry.id)}
              className={`rounded-clinical border p-3 text-left outline-none transition focus-visible:ring-2 focus-visible:ring-clinical-accent/50 ${selectedId === entry.id ? "feature-panel border-clinical-accent/45" : "border-clinical-line bg-clinical-raised hover:bg-clinical-soft"}`}
            >
              <p className="font-semibold text-clinical-ink">{entry.label}</p>
              <p className="text-sm text-clinical-muted">{entry.date} · {entry.predictedClassCode} · {Math.round(entry.calibratedConfidence * 100)}%</p>
            </button>
          ))}
        </div>
      </Card>
    </>
  );
}

