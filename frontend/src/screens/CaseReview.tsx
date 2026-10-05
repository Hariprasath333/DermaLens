import { useMemo } from "react";
import { AlertTriangle, Download, Loader2 } from "lucide-react";
import type { CaseRecord, OverlayMode, UploadMetadataInput } from "../types/lesioniq";
import { pct } from "../lib/format";
import { resolveLesionIQArtifactUrl } from "../lib/lesioniqApi";
import { buildExplainabilityReport, downloadExplainabilityReport } from "../lib/explainabilityReport";
import { ImageViewerCard } from "../components/domain/ImageViewerCard";
import { UploadInferenceCard } from "../components/domain/UploadInferenceCard";
import { Card } from "../components/primitives/Card";
import { StatusBadge } from "../components/primitives/StatusBadge";

export function CaseReview({
  caseRecord,
  uploadedImage,
  uploadedPreviewUrl,
  analysisReady,
  analysisPending,
  analysisError,
  hasUploadedImage,
  onNavigateExplainability,
  onImageSelected,
  onRunAnalysis,
  onUseSampleCase,
  uploadMetadata,
  onUploadMetadataChange
}: {
  caseRecord: CaseRecord | null;
  uploadedImage: File | null;
  uploadedPreviewUrl: string | null;
  analysisReady: boolean;
  analysisPending: boolean;
  analysisError: string | null;
  hasUploadedImage: boolean;
  onNavigateExplainability: () => void;
  onImageSelected: (file: File, metadata?: UploadMetadataInput) => void;
  onRunAnalysis: (metadata: UploadMetadataInput) => void | Promise<void>;
  onUseSampleCase?: () => void;
  uploadMetadata: UploadMetadataInput;
  onUploadMetadataChange: (metadata: UploadMetadataInput) => void;
}) {
  const viewerArtifactUrls = useMemo<Partial<Record<OverlayMode, string>>>(() => {
    if (!analysisReady || !caseRecord?.inferenceBundle) return {};

    const bundle = caseRecord.inferenceBundle;
    const resolve = (url?: string) => resolveLesionIQArtifactUrl(url, bundle.outputDirectory);
    const raw = resolve(bundle.rawArtifact ?? caseRecord.uploadedImageUrl);

    return {
      raw,
      gradcam: resolve(bundle.gradcamArtifact),
      attention: resolve(bundle.attentionArtifact),
      metadata: raw
    };
  }, [
    analysisReady,
    caseRecord?.inferenceBundle,
    caseRecord?.uploadedImageUrl
  ]);
  const explainabilityReport = useMemo(
    () => (caseRecord ? buildExplainabilityReport(caseRecord) : ""),
    [caseRecord]
  );

  if (analysisPending) {
    return (
      <div className="grid min-h-[calc(100vh-120px)] place-items-center">
        <div className="w-full max-w-2xl rounded-clinical border border-clinical-line bg-clinical-surface p-6 text-center shadow-clinical">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full border border-clinical-accent/40 bg-clinical-accentSoft text-clinical-accent">
            <Loader2 className="h-7 w-7 animate-spin" aria-hidden="true" />
          </div>
          <p className="mt-5 text-xs font-semibold uppercase tracking-[0.14em] text-clinical-muted">Live analysis running</p>
          <h1 className="mt-2 text-2xl font-semibold text-clinical-ink">Processing dermoscopy evidence</h1>
          <p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-clinical-muted">
            LesionIQ is waiting for the backend response from preprocessing, inference, explainability artifacts, and the local SLM bundle.
          </p>
          <div className="mt-5 h-1.5 overflow-hidden rounded-full bg-clinical-raised">
            <div className="h-full w-1/2 animate-[pulse_1.4s_ease-in-out_infinite] rounded-full bg-clinical-accent" />
          </div>
        </div>
      </div>
    );
  }

  if (analysisError) {
    return (
      <div className="grid min-h-[calc(100vh-120px)] place-items-center">
        <div className="w-full max-w-2xl rounded-clinical border border-clinical-danger/40 bg-clinical-surface p-6 text-center shadow-clinical">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full border border-clinical-danger/40 bg-clinical-danger/10 text-clinical-danger">
            <AlertTriangle className="h-7 w-7" aria-hidden="true" />
          </div>
          <p className="mt-5 text-xs font-semibold uppercase tracking-[0.14em] text-clinical-muted">Analysis failed</p>
          <h1 className="mt-2 text-2xl font-semibold text-clinical-ink">Could not complete analysis</h1>
          <p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-clinical-muted">{analysisError}</p>
          <button
            type="button"
            onClick={() => onRunAnalysis(uploadMetadata)}
            className="mt-5 rounded-md border border-clinical-accent bg-clinical-accent px-4 py-2 text-sm font-semibold text-clinical-canvas outline-none transition hover:bg-clinical-accentHover focus-visible:ring-2 focus-visible:ring-clinical-accent/50"
          >
            Retry analysis
          </button>
        </div>
      </div>
    );
  }

  if (!hasUploadedImage && !caseRecord) {
    return (
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="relative">
          <div className="relative space-y-4">
            <div className="rounded-clinical border border-clinical-line bg-clinical-surface px-5 py-4 shadow-clinical">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-clinical-muted">Case intake</p>
              <h1 className="mt-1 text-xl font-semibold text-clinical-ink">Upload dermoscopy evidence</h1>
              <p className="mt-1 max-w-2xl text-sm leading-6 text-clinical-muted">
                Start with a lesion image, then complete age, sex, and anatomical site before running analysis.
              </p>
            </div>

            <UploadInferenceCard
              caseRecord={caseRecord}
              uploadedImage={uploadedImage}
              uploadedPreviewUrl={uploadedPreviewUrl}
              analysisReady={analysisReady}
              onImageSelected={onImageSelected}
              onRunAnalysis={onRunAnalysis}
              onUseSampleCase={onUseSampleCase}
              uploadMetadata={uploadMetadata}
              onUploadMetadataChange={onUploadMetadataChange}
            />

            <div className="grid gap-2 rounded-clinical border border-clinical-line bg-clinical-raised p-3 sm:grid-cols-3">
              {["Upload", "Metadata", "Analyze"].map((stage, index) => (
                <div key={stage} className="flex items-center gap-2 rounded-md border border-clinical-line bg-clinical-surface px-3 py-2">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full border border-clinical-accent/45 bg-clinical-accentSoft text-xs font-semibold text-clinical-accent">
                    {index + 1}
                  </span>
                  <span className="text-sm font-semibold text-clinical-ink">{stage}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <aside className="space-y-4">
          <Card title="What happens next" eyebrow="Guided workflow">
            <div className="space-y-3 text-sm leading-6 text-clinical-muted">
              <p>After upload, analysis waits until required metadata is complete.</p>
              <p>Evidence, confidence, threshold margin, and recommendation appear in one compact review workspace.</p>
            </div>
          </Card>
          <Card title="Supported input" eyebrow="Image requirements">
            <div className="grid gap-2 text-sm">
              <div className="flex justify-between gap-3"><span className="text-clinical-muted">Formats</span><span className="font-semibold text-clinical-ink">PNG, JPG, WebP</span></div>
              <div className="flex justify-between gap-3"><span className="text-clinical-muted">View</span><span className="font-semibold text-clinical-ink">Square crop review</span></div>
              <div className="flex justify-between gap-3"><span className="text-clinical-muted">Mode</span><span className="font-semibold text-clinical-ink">Full Hybrid</span></div>
            </div>
          </Card>
          <Card title="Review focus" eyebrow="Decision support">
            <p className="text-sm leading-6 text-clinical-muted">
              Use the output to inspect image evidence and calibrated support. Dermatologist verification remains required.
            </p>
          </Card>
        </aside>
      </div>
    );
  }

  if (!analysisReady || !caseRecord) {
    return (
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-4">
          <div className="rounded-clinical border border-clinical-line bg-clinical-surface px-5 py-4 shadow-clinical">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-clinical-muted">Metadata-gated intake</p>
            <h1 className="mt-1 text-xl font-semibold text-clinical-ink">Complete metadata to run analysis</h1>
            <p className="mt-1 max-w-2xl text-sm leading-6 text-clinical-muted">
              Image preview is loaded. Full Hybrid analysis stays pending until age, sex, and anatomical site are entered.
            </p>
          </div>
          <UploadInferenceCard
            caseRecord={caseRecord}
            uploadedImage={uploadedImage}
            uploadedPreviewUrl={uploadedPreviewUrl}
            analysisReady={analysisReady}
            onImageSelected={onImageSelected}
            onRunAnalysis={onRunAnalysis}
            onUseSampleCase={onUseSampleCase}
            uploadMetadata={uploadMetadata}
            onUploadMetadataChange={onUploadMetadataChange}
          />
        </div>
        <aside className="space-y-4">
          <Card title="Metadata gate" eyebrow="Hybrid model input">
            <p className="text-sm leading-6 text-clinical-muted">
              LesionIQ Full Hybrid mode fuses the image branch with normalized age plus encoded sex and anatomical site.
            </p>
          </Card>
          <Card title="Analysis status" eyebrow="Pending">
            <p className="text-sm leading-6 text-clinical-muted">
              The run control unlocks only after all required metadata is complete.
            </p>
          </Card>
        </aside>
      </div>
    );
  }

  return (
    <>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-clinical border border-clinical-line bg-clinical-surface px-5 py-4 shadow-clinical">
        <div>
          <h1 className="text-xl font-semibold text-clinical-ink">Case Review</h1>
          <p className="mt-0.5 text-sm text-clinical-muted">
            <span className="font-bold text-clinical-ink">{caseRecord.caseId}</span> - {caseRecord.maskedPatientId} - {caseRecord.visitDate}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <StatusBadge label={caseRecord.reviewStatus} tone={caseRecord.reviewStatus === "Senior review" ? "danger" : "warning"} />
          <StatusBadge label={caseRecord.modelMode} tone="accent" />
        </div>
      </div>
      <div className="space-y-4">
        <div className="grid gap-4 md:grid-cols-2">
          <Card title="Primary output" eyebrow="Prediction summary">
            <div className="min-h-[112px]">
              <p className="text-[11px] font-bold uppercase tracking-[0.11em] text-clinical-muted">Predicted class</p>
              <p className="mt-2 text-[30px] font-bold leading-none tracking-[-0.02em] text-clinical-accent">{caseRecord.predictedClassCode}</p>
              <p className="mt-1 text-sm font-semibold text-clinical-ink">{caseRecord.predictedClassLabel}</p>
              <p className="mt-3 max-w-md text-sm leading-5 text-clinical-muted">Top-ranked class from the calibrated 8-class dermoscopy model.</p>
            </div>
          </Card>
          <Card title="Probability" eyebrow="Calibrated confidence">
            <div className="min-h-[112px]">
              <p className="text-[30px] font-bold leading-none tabular-nums tracking-[-0.02em] text-clinical-accent">{pct(caseRecord.calibratedConfidence)}</p>
              <p className="mt-3 max-w-md text-sm leading-5 text-clinical-muted">Temperature-scaled model probability for review support, not clinical urgency.</p>
            </div>
          </Card>
        </div>

        <ImageViewerCard imageUrl={uploadedPreviewUrl ?? viewerArtifactUrls.raw ?? caseRecord.uploadedImageUrl} artifactUrls={viewerArtifactUrls} />

        <Card
          title="SLM explanation"
          eyebrow="Generated rationale"
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

        <div className="grid gap-4 md:grid-cols-2">
          <Card title="Review explanation" eyebrow="Review note" action={<StatusBadge label={caseRecord.urgency} tone={caseRecord.urgency === "High concern" ? "danger" : caseRecord.urgency === "Routine" ? "success" : "warning"} />}>
            <p className="text-sm font-semibold leading-5 text-clinical-ink">{caseRecord.recommendation}</p>
            <button
              type="button"
              onClick={onNavigateExplainability}
              className="mt-3 rounded-md border border-clinical-line bg-clinical-raised px-3 py-2 text-sm font-medium text-clinical-ink outline-none transition hover:border-clinical-accent/35 hover:bg-clinical-accentSoft focus-visible:ring-2 focus-visible:ring-clinical-accent/50"
            >
              Review explanation
            </button>
          </Card>
          <Card title="Class threshold" eyebrow="Threshold margin">
            <p className="text-2xl font-bold tabular-nums text-clinical-accent">{(caseRecord.thresholdMargin * 100).toFixed(0)} pts</p>
            <p className="mt-2 text-sm leading-5 text-clinical-muted">Above the tuned threshold for {caseRecord.predictedClassCode}.</p>
          </Card>
        </div>
      </div>
    </>
  );
}
