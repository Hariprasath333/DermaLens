import { UploadCloud } from "lucide-react";
import type { ThemeMode } from "../types/lesioniq";
import { systemStatus } from "../data/system";
import { Card } from "../components/primitives/Card";
import { DisclaimerBanner } from "../components/primitives/DisclaimerBanner";
import { MetricCard } from "../components/primitives/MetricCard";
import { PageHeader } from "../components/primitives/PageHeader";
import { StatusBadge } from "../components/primitives/StatusBadge";
import { ThemeToggle } from "../components/layout/ThemeToggle";

export function Settings({
  theme,
  onToggleTheme,
  uploadedImage,
  uploadedPreviewUrl,
  analysisReady,
  onImageSelected
}: {
  theme: ThemeMode;
  onToggleTheme: () => void;
  uploadedImage: File | null;
  uploadedPreviewUrl: string | null;
  analysisReady: boolean;
  onImageSelected: (file: File) => void;
}) {
  const hasUploadedImage = Boolean(uploadedImage || uploadedPreviewUrl);

  return (
    <>
      <PageHeader title="Settings" subtitle="Prototype system configuration and future backend integration notes." />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Model and inference" eyebrow="LesionIQ runtime context">
          <div className="grid gap-3">
            <MetricCard label="Model mode" value={systemStatus.inferenceMode} />
            <MetricCard label="Model version" value={systemStatus.modelVersion} />
            <MetricCard label="Calibration" value={systemStatus.calibrationStatus} />
            <MetricCard label="Threshold tuning" value={systemStatus.thresholdTuningStatus} />
            <MetricCard label="Explainability" value={systemStatus.explainabilityStatus} />
          </div>
        </Card>
        <Card title="Application" eyebrow="Frontend prototype">
          <div className="grid gap-3">
            <MetricCard label="App version" value={systemStatus.appVersion} />
            <MetricCard label="Preprocessing version" value={systemStatus.preprocessingVersion} />
            <div className="rounded-clinical border border-clinical-line bg-clinical-raised p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-[0.12em] text-clinical-muted">Theme controls</p>
              <ThemeToggle theme={theme} onToggle={onToggleTheme} />
            </div>
            <div className="rounded-clinical border border-clinical-line bg-clinical-raised p-3">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-clinical-muted">Upload control</p>
                <StatusBadge label={hasUploadedImage && analysisReady ? "Analyzed" : hasUploadedImage ? "Image loaded" : "No image"} tone={hasUploadedImage && analysisReady ? "success" : hasUploadedImage ? "warning" : "neutral"} />
              </div>
              <div className="flex flex-wrap items-center gap-3">
                {uploadedPreviewUrl && (
                  <img src={uploadedPreviewUrl} alt="Current uploaded dermoscopy preview" className="h-14 w-20 rounded-md border border-clinical-line object-cover" />
                )}
                <label className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-clinical-accent bg-clinical-accent px-3 py-2 text-sm font-semibold text-clinical-canvas outline-none transition hover:bg-clinical-accentHover focus-within:ring-2 focus-within:ring-clinical-accent/50">
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    className="sr-only"
                    onChange={(event) => {
                      const nextFile = event.target.files?.[0] ?? null;
                      if (nextFile) onImageSelected(nextFile);
                      event.currentTarget.value = "";
                    }}
                  />
                  <UploadCloud className="h-4 w-4" aria-hidden="true" />
                  {hasUploadedImage ? "Re-upload image" : "Upload image"}
                </label>
                {uploadedImage && <span className="text-sm text-clinical-muted">{uploadedImage.name}</span>}
              </div>
            </div>
          </div>
        </Card>
        <Card title="Backend integration" eyebrow="Live API shape">
          <p className="leading-7 text-clinical-muted">
            Live analysis records use the CaseRecord bundle: case metadata, calibrated probabilities, per-class threshold margins,
            preprocessing trace, visual explainability artifacts, metadata attribution, audit checks, and SLM narrative. Demo seed
            cases stay isolated behind the explicit demo-case flag.
          </p>
        </Card>
        <Card title="Prototype caveats" eyebrow="Research use">
          <p className="leading-7 text-clinical-muted">
            This frontend can use de-identified demo cases and stylized dermoscopy previews when demo mode is enabled. It is suitable for demo workflows,
            product review, and backend contract discussion, but it is not externally validated clinical software.
          </p>
        </Card>
      </div>
      <div className="mt-4"><DisclaimerBanner /></div>
    </>
  );
}

