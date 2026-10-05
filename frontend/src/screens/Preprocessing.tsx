import { useMemo } from "react";
import type { CaseRecord } from "../types/lesioniq";
import { systemStatus } from "../data/system";
import { resolveLesionIQArtifactUrl } from "../lib/lesioniqApi";
import { DermoscopyMock } from "../components/domain/DermoscopyMock";
import { Card } from "../components/primitives/Card";
import { MetricCard } from "../components/primitives/MetricCard";
import { PageHeader } from "../components/primitives/PageHeader";
import { StatusBadge } from "../components/primitives/StatusBadge";

export function Preprocessing({
  caseRecord,
  uploadedPreviewUrl,
  analysisReady
}: {
  caseRecord: CaseRecord | null;
  uploadedPreviewUrl: string | null;
  analysisReady: boolean;
}) {
  const comparisonSteps = caseRecord?.preprocessingSteps.filter((step) => step.id !== "raw") ?? [];
  const bundle = caseRecord?.inferenceBundle;
  const resolvedArtifacts = useMemo(() => {
    const artifact = (url?: string) => resolveLesionIQArtifactUrl(url, bundle?.outputDirectory);
    const finalPreprocessedArtifact = artifact(bundle?.finalPreprocessedArtifact ?? bundle?.originalArtifact);

    return {
      rawArtifact: artifact(bundle?.rawArtifact ?? caseRecord?.uploadedImageUrl),
      finalPreprocessedArtifact,
      artifactByStepId: {
        dullrazor: artifact(bundle?.dullRazorArtifact),
        shadesofgrey: artifact(bundle?.normalizedArtifact),
        clahe: artifact(bundle?.claheArtifact),
        borderremoved: finalPreprocessedArtifact
      } as Record<string, string | undefined>
    };
  }, [bundle, caseRecord?.uploadedImageUrl]);

  if (!caseRecord || !analysisReady) {
    return (
      <>
        <PageHeader
          title="Preprocessing"
          subtitle={uploadedPreviewUrl ? "Image uploaded. Run analysis to complete preprocessing provenance." : "Upload a dermoscopy image before preprocessing provenance is available."}
          meta={<StatusBadge label={uploadedPreviewUrl ? "Analysis pending" : "Awaiting image"} tone={uploadedPreviewUrl ? "warning" : "neutral"} />}
        />
        <Card title="Pipeline placeholder" eyebrow="Stages appear after analysis">
          {uploadedPreviewUrl ? <div className="max-w-[340px]"><img src={uploadedPreviewUrl} alt="Uploaded dermoscopy preview" className="aspect-square w-full rounded-clinical border border-clinical-line object-cover" /></div> : null}
          <p className="mt-3 text-sm leading-6 text-clinical-muted">
            DullRazor hair removal, Shades-of-Gray normalization, LAB CLAHE, and border removal cards appear only after analysis is ready.
          </p>
        </Card>
      </>
    );
  }

  return (
    <>
      <PageHeader
        title="Preprocessing"
        subtitle={`${caseRecord.caseId}: transparent image cleanup provenance before hybrid inference.`}
        meta={<StatusBadge label={systemStatus.preprocessingVersion} tone="accent" />}
      />
      <Card title="Pipeline overview" eyebrow="LesionIQ 4-step preprocessing">
        <div className="grid gap-3 md:grid-cols-4">
          {["Raw input", "DullRazor hair removal", "Shades-of-Gray + CLAHE", "Border removal"].map((step, index) => (
            <div key={step} className="rounded-md border border-clinical-line bg-clinical-raised p-3">
              <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-clinical-muted">Step {index + 1}</p>
              <p className="mt-1 text-sm font-medium text-clinical-ink">{step}</p>
            </div>
          ))}
        </div>
      </Card>
      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        {comparisonSteps.map((step, index) => (
          <Card
            key={step.id}
            title={`${index + 1}. Raw vs ${step.shortTitle}`}
            eyebrow="Layer comparison"
            action={<StatusBadge label={step.status} tone={step.status === "Complete" ? "success" : "warning"} />}
          >
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <DermoscopyMock square imageUrl={resolvedArtifacts.rawArtifact} label="Raw dermoscopy input" />
                <p className="mt-2 text-[11px] font-semibold uppercase tracking-[0.11em] text-clinical-muted">Raw image</p>
              </div>
              <div>
                <DermoscopyMock square imageUrl={resolvedArtifacts.artifactByStepId[step.id]} stepTone={step.previewTone} label={step.title} />
                <p className="mt-2 text-[11px] font-semibold uppercase tracking-[0.11em] text-clinical-muted">{step.shortTitle}</p>
              </div>
            </div>
            <p className="mt-3 text-sm leading-5 text-clinical-ink">{step.description}</p>
          </Card>
        ))}
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_1fr]">
        <Card title="Final preprocessed input" eyebrow="Fed to inference">
          <div className="grid gap-4 md:grid-cols-[240px_minmax(0,1fr)] md:items-center">
            <DermoscopyMock compact imageUrl={resolvedArtifacts.finalPreprocessedArtifact} stepTone="border" label="Final preprocessed image sent to inference" />
            <div className="space-y-3 text-sm leading-5 text-clinical-muted">
              <p className="font-medium text-clinical-ink">This final 384x384 view is the model input for EfficientNet-B4, SwinV2, and metadata fusion.</p>
              <p>Backend artifact target: <span className="font-medium text-clinical-ink">{resolvedArtifacts.finalPreprocessedArtifact ?? "final_preprocessed.png"}</span></p>
              <p>The raw image remains visible for audit comparison; inference consumes only the cleaned, normalized image bundle.</p>
            </div>
          </div>
        </Card>
        <Card title="Backend handoff contract" eyebrow="Single API result">
          <div className="space-y-2 text-sm">
            <div className="flex justify-between gap-3"><span className="text-clinical-muted">Endpoint</span><span className="font-semibold text-clinical-ink">POST /cases/analyze</span></div>
            <div className="flex justify-between gap-3"><span className="text-clinical-muted">Input</span><span className="text-right font-semibold text-clinical-ink">image + age + sex + site + mode</span></div>
            <div className="flex justify-between gap-3"><span className="text-clinical-muted">Returns</span><span className="text-right font-semibold text-clinical-ink">CaseRecord + artifact URLs</span></div>
            <div className="flex justify-between gap-3"><span className="text-clinical-muted">Output dir</span><span className="text-right font-semibold text-clinical-ink">{caseRecord.inferenceBundle?.outputDirectory}</span></div>
          </div>
        </Card>
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-[1.2fr_.8fr]">
        <Card title="Technical rationale" eyebrow="Robustness and transparency">
          <p className="text-sm leading-6 text-clinical-ink">
            LesionIQ uses preprocessing to reduce non-biological variance before the image and metadata fusion model runs.
            Hair artifacts, device color shifts, local contrast loss, and dermoscope vignette borders are handled as explicit,
            auditable stages so reviewers can inspect whether the model received a clinically faithful lesion view.
          </p>
        </Card>
        <Card title="Processing trace" eyebrow="Audit record">
          <div className="grid gap-3">
            <MetricCard label="Preprocessing version" value={systemStatus.preprocessingVersion} />
            <MetricCard label="Pipeline timestamp" value={caseRecord.acquisitionTimestamp} />
            <MetricCard label="Image size" value="1022x767 -> 900x675 -> 384x384" detail="Border crop followed by model resize." />
            <MetricCard label="Integrity" value="All stages completed" tone="accent" />
          </div>
        </Card>
      </div>
    </>
  );
}
