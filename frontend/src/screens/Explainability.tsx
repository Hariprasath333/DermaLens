import { useMemo, useState } from "react";
import { Download } from "lucide-react";
import type { CaseRecord } from "../types/lesioniq";
import { reviewActions } from "../data/system";
import { resolveLesionIQArtifactUrl } from "../lib/lesioniqApi";
import { buildExplainabilityReport, downloadExplainabilityReport } from "../lib/explainabilityReport";
import { ActionPanel } from "../components/domain/ActionPanel";
import { AttributionBars } from "../components/domain/AttributionBars";
import { AuditChecklist } from "../components/domain/AuditChecklist";
import { AuditNoteList } from "../components/domain/AuditNoteList";
import { DermoscopyMock } from "../components/domain/DermoscopyMock";
import { Card } from "../components/primitives/Card";
import { PageHeader } from "../components/primitives/PageHeader";
import { SectionTabs } from "../components/primitives/SectionTabs";
import { StatusBadge } from "../components/primitives/StatusBadge";

type ExplainTab = "gradcam" | "attention" | "metadata" | "slm";

const tabs: Array<{ id: ExplainTab; label: string }> = [
  { id: "gradcam", label: "Grad-CAM" },
  { id: "attention", label: "Attention" },
  { id: "metadata", label: "Metadata Attribution" },
  { id: "slm", label: "SLM Summary" }
];

export function Explainability({
  caseRecord,
  uploadedPreviewUrl,
  analysisReady
}: {
  caseRecord: CaseRecord | null;
  uploadedPreviewUrl: string | null;
  analysisReady: boolean;
}) {
  const [tab, setTab] = useState<ExplainTab>("gradcam");
  const bundle = caseRecord?.inferenceBundle;
  const resolvedArtifacts = useMemo(() => {
    const artifact = (url?: string) => resolveLesionIQArtifactUrl(url, bundle?.outputDirectory);

    return {
      finalPreprocessedArtifact: artifact(bundle?.finalPreprocessedArtifact ?? bundle?.originalArtifact ?? caseRecord?.uploadedImageUrl),
      gradcamArtifact: artifact(bundle?.gradcamArtifact),
      attentionArtifact: artifact(bundle?.attentionArtifact),
      diagnosisArtifact: artifact(bundle?.diagnosisArtifact)
    };
  }, [bundle, caseRecord?.uploadedImageUrl]);
  const explainabilityReport = useMemo(
    () => (caseRecord ? buildExplainabilityReport(caseRecord) : ""),
    [caseRecord]
  );

  if (!caseRecord || !analysisReady) {
    return (
      <>
        <PageHeader
          title="Explainability"
          subtitle={uploadedPreviewUrl ? "Image uploaded. Run analysis to generate explanation outputs." : "Upload a dermoscopy image before reviewing explanations."}
          meta={<StatusBadge label={uploadedPreviewUrl ? "Analysis pending" : "Awaiting image"} tone={uploadedPreviewUrl ? "warning" : "neutral"} />}
        />
        <Card title="Explanation placeholder" eyebrow="Evidence appears after analysis">
          {uploadedPreviewUrl ? <div className="max-w-[340px]"><DermoscopyMock square overlay="raw" imageUrl={uploadedPreviewUrl} /></div> : null}
          <p className="mt-3 text-sm leading-6 text-clinical-muted">
            Grad-CAM, transformer attention, metadata attribution, SLM explanation, and audit checks appear only after analysis is ready.
          </p>
        </Card>
      </>
    );
  }

  return (
    <>
      <PageHeader
        title="Explainability"
        subtitle={`${caseRecord.caseId}: ${caseRecord.predictedClassCode} - ${caseRecord.predictedClassLabel} at ${Math.round(caseRecord.calibratedConfidence * 100)}% calibrated confidence`}
        meta={<StatusBadge label="Clinician verification required" tone="warning" />}
      />
      <div className="mb-4"><SectionTabs tabs={tabs} value={tab} onChange={setTab} ariaLabel="Explainability evidence tabs" /></div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.45fr)_420px]">
        <div className="space-y-4">
          <Card title="Visual evidence" eyebrow="Image branch audit">
            <div className="grid gap-4 lg:grid-cols-3">
              <div><DermoscopyMock compact overlay="raw" imageUrl={resolvedArtifacts.finalPreprocessedArtifact} /><p className="mt-2 text-sm font-medium text-clinical-ink">Final preprocessed input</p><p className="text-xs text-clinical-muted">{resolvedArtifacts.finalPreprocessedArtifact ?? "final_preprocessed.png"}</p></div>
              <div><DermoscopyMock compact overlay="raw" imageUrl={resolvedArtifacts.gradcamArtifact} /><p className="mt-2 text-sm font-medium text-clinical-ink">Grad-CAM++ output</p><p className="text-xs text-clinical-muted">{resolvedArtifacts.gradcamArtifact ?? "gradcam.png"}</p></div>
              <div><DermoscopyMock compact overlay="raw" imageUrl={resolvedArtifacts.attentionArtifact} /><p className="mt-2 text-sm font-medium text-clinical-ink">Graded attention weights</p><p className="text-xs text-clinical-muted">{resolvedArtifacts.attentionArtifact ?? "attention.png"}</p></div>
            </div>
            <div className="mt-4 rounded-md border border-clinical-line bg-clinical-raised p-3 text-sm leading-5 text-clinical-muted">
              {tab === "gradcam" && caseRecord.explainability.gradcamSummary}
              {tab === "attention" && caseRecord.explainability.attentionSummary}
              {tab === "metadata" && "Metadata attribution is displayed below with direction and relative contribution. Image evidence remains the primary classification branch."}
              {tab === "slm" && "The generated narrative is constrained by structured evidence but still requires clinician verification."}
            </div>
          </Card>
          <AttributionBars signals={caseRecord.explainability.metadataSignals} title="Metadata attribution" />
          <Card
            title="Generated SLM explanation"
            eyebrow="Natural-language rationale"
            action={
              <button
                type="button"
                onClick={() => downloadExplainabilityReport(caseRecord)}
                className="inline-flex items-center gap-2 rounded-md border border-clinical-line bg-clinical-raised px-3 py-2 text-xs font-semibold text-clinical-ink outline-none transition hover:border-clinical-accent/35 hover:bg-clinical-accentSoft focus-visible:ring-2 focus-visible:ring-clinical-accent/50"
                aria-label="Download explainability report"
              >
                <Download className="h-4 w-4" aria-hidden="true" />
                Download
              </button>
            }
          >
            <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-6 text-clinical-ink">
              {explainabilityReport}
            </pre>
          </Card>
          <Card title="Local SLM handoff" eyebrow="Docker bridge">
            <div className="grid gap-3 md:grid-cols-2">
              {[
                ["Container", caseRecord.inferenceBundle?.slmContainer ?? "lesioniq_ollama"],
                ["Model", caseRecord.inferenceBundle?.slmModel ?? "gemma3:4b-it-qat"],
                ["Visual inputs", "final preprocessed + Grad-CAM + attention"],
                ["Structured input", resolvedArtifacts.diagnosisArtifact ?? "diagnosis.json"]
              ].map(([label, value]) => (
                <div key={label} className="rounded-md border border-clinical-line bg-clinical-raised p-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.11em] text-clinical-muted">{label}</p>
                  <p className="mt-1 text-sm font-medium text-clinical-ink">{value}</p>
                </div>
              ))}
            </div>
            <p className="mt-3 text-sm leading-5 text-clinical-muted">
              The frontend expects the backend to package artifact URLs and metadata into the CaseRecord response. The SLM call stays behind the backend boundary.
            </p>
          </Card>
        </div>
        <aside className="space-y-4">
          <AuditChecklist checks={caseRecord.explainability.auditChecks} />
          <ActionPanel actions={reviewActions.filter((action) => ["explain", "senior", "follow-up"].includes(action.id)).concat([
            { id: "confirm", label: "Confirm AI rationale", tone: "success" },
            { id: "correct", label: "Correct highlighted region", tone: "secondary" },
            { id: "reject", label: "Reject artifact-driven cue", tone: "danger" },
            { id: "note", label: "Add clinician note", tone: "secondary" }
          ])} />
          <AuditNoteList notes={caseRecord.explainability.auditNotes} />
        </aside>
      </div>
    </>
  );
}
