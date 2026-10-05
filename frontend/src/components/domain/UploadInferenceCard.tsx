import { FileImage, UploadCloud } from "lucide-react";
import type { CaseRecord, ModelMode, UploadMetadataInput } from "../../types/lesioniq";
import { useProxiedImage } from "../../hooks/useProxiedImage";
import { Card } from "../primitives/Card";
import { StatusBadge } from "../primitives/StatusBadge";

interface UploadInferenceCardProps {
  caseRecord?: CaseRecord | null;
  uploadedImage: File | null;
  uploadedPreviewUrl: string | null;
  analysisReady: boolean;
  onImageSelected: (file: File, metadata?: UploadMetadataInput) => void;
  onRunAnalysis: (metadata: UploadMetadataInput) => void | Promise<void>;
  onUseSampleCase?: () => void;
  uploadMetadata: UploadMetadataInput;
  onUploadMetadataChange: (metadata: UploadMetadataInput) => void;
}

const siteOptions = [
  "anterior torso",
  "head/neck",
  "lateral torso",
  "lower extremity",
  "palms/soles",
  "posterior torso",
  "upper extremity",
  "unknown"
];

const modelModes: ModelMode[] = ["Full Hybrid", "Image Only", "EffNet Only", "Swin Only"];

export function UploadInferenceCard({
  caseRecord,
  uploadedImage,
  uploadedPreviewUrl,
  analysisReady,
  onImageSelected,
  onRunAnalysis,
  onUseSampleCase,
  uploadMetadata,
  onUploadMetadataChange
}: UploadInferenceCardProps) {
  const hasUploadedImage = Boolean(uploadedImage || uploadedPreviewUrl || caseRecord?.uploadedImageUrl);
  const proxiedCaseImage = useProxiedImage(caseRecord?.uploadedImageUrl);
  const metadataComplete = uploadMetadata.ageYears !== null && uploadMetadata.sex !== "Unknown" && uploadMetadata.anatomicalSite !== "unknown" && uploadMetadata.anatomicalSite.trim().length > 0;
  const canRunAnalysis = Boolean(uploadedImage && uploadedPreviewUrl && metadataComplete);
  const recommendationLabel = analysisReady ? "Analyzed" : hasUploadedImage && metadataComplete ? "Ready to analyze" : hasUploadedImage ? "Complete metadata" : "Pending";
  const recommendationTone = analysisReady || (hasUploadedImage && metadataComplete) ? "success" : "accent";

  function runAnalysis() {
    if (!uploadedImage || !uploadedPreviewUrl) return;
    void onRunAnalysis(uploadMetadata);
  }

  if (!hasUploadedImage) {
    return (
      <Card title="Upload image" eyebrow="Case intake" className="relative overflow-hidden border-clinical-accent/35">
        <label className="relative mx-auto flex aspect-square w-full max-w-[420px] cursor-pointer flex-col items-center justify-center rounded-clinical border border-dashed border-clinical-accent/45 bg-clinical-raised p-6 text-center outline-none transition hover:border-clinical-accent/70 hover:bg-clinical-soft">
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            className="sr-only"
            onChange={(event) => {
              const nextFile = event.target.files?.[0] ?? null;
              if (nextFile) onImageSelected(nextFile, uploadMetadata);
              event.currentTarget.value = "";
            }}
          />
          <UploadCloud className="h-9 w-9 text-clinical-accent" aria-hidden="true" />
          <span className="mt-3 text-base font-semibold text-clinical-ink">Upload lesion image</span>
          <span className="mt-1 text-sm text-clinical-muted">PNG, JPG, or WebP</span>
        </label>
        <div className="relative mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <label className="grid gap-1 text-sm font-medium text-clinical-muted">
            Age
            <input
              type="number"
              min="0"
              max="110"
              value={uploadMetadata.ageYears ?? ""}
              onChange={(event) => onUploadMetadataChange({ ...uploadMetadata, ageYears: event.target.value ? Number(event.target.value) : null })}
              className="clinical-field rounded-md border px-3 py-2 outline-none"
            />
          </label>
          <label className="grid gap-1 text-sm font-medium text-clinical-muted">
            Sex
            <select
              value={uploadMetadata.sex}
              onChange={(event) => onUploadMetadataChange({ ...uploadMetadata, sex: event.target.value as UploadMetadataInput["sex"] })}
              className="clinical-field rounded-md border px-3 py-2 outline-none"
            >
              <option>Unknown</option>
              <option>Female</option>
              <option>Male</option>
            </select>
          </label>
          <label className="grid gap-1 text-sm font-medium text-clinical-muted">
            Site
            <select
              value={uploadMetadata.anatomicalSite}
              onChange={(event) => onUploadMetadataChange({ ...uploadMetadata, anatomicalSite: event.target.value })}
              className="clinical-field rounded-md border px-3 py-2 outline-none"
            >
              {siteOptions.map((site) => <option key={site}>{site}</option>)}
            </select>
          </label>
          <label className="grid gap-1 text-sm font-medium text-clinical-muted">
            Mode
            <select
              value={uploadMetadata.modelMode}
              onChange={(event) => onUploadMetadataChange({ ...uploadMetadata, modelMode: event.target.value as ModelMode })}
              className="clinical-field rounded-md border px-3 py-2 outline-none"
            >
              {modelModes.map((mode) => <option key={mode}>{mode}</option>)}
            </select>
          </label>
        </div>
        <div className="relative mt-4 rounded-clinical border border-clinical-accent/25 bg-clinical-accentSoft/70 p-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-clinical-muted">Recommendation control</p>
              <p className="mt-1 text-sm font-semibold text-clinical-ink">Clinical review tier</p>
            </div>
            <StatusBadge label={recommendationLabel} tone={recommendationTone} />
          </div>
          <select
            value={recommendationLabel}
            disabled
            className="clinical-field mt-3 w-full rounded-md border px-3 py-2 text-sm font-semibold outline-none"
          >
            <option>Pending</option>
            <option>Complete metadata</option>
            <option>Ready to analyze</option>
            <option>Routine review</option>
            <option>Needs review</option>
            <option>High-priority review</option>
            <option>Analyzed</option>
          </select>
          <p className="mt-2 text-xs leading-5 text-clinical-muted">Full Hybrid mode waits for age, sex, and anatomical site before inference.</p>
        </div>
      </Card>
    );
  }

  return (
    <Card title="Upload context" eyebrow="Workflow input">
      <div className="grid gap-3">
        <label className="mx-auto flex aspect-square w-full max-w-[360px] cursor-pointer flex-col items-center justify-center rounded-clinical border border-dashed border-clinical-line bg-clinical-raised p-2 text-center outline-none transition hover:border-clinical-accent/45 hover:bg-clinical-soft">
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            className="sr-only"
            onChange={(event) => {
              const nextFile = event.target.files?.[0] ?? null;
              if (nextFile) onImageSelected(nextFile, uploadMetadata);
              event.currentTarget.value = "";
            }}
          />
          {uploadedPreviewUrl || proxiedCaseImage ? (
            <img src={uploadedPreviewUrl ?? proxiedCaseImage} alt="Uploaded dermoscopy preview" className="h-full w-full rounded-md object-cover" />
          ) : (
            <>
              <UploadCloud className="h-8 w-8 text-clinical-accent" aria-hidden="true" />
              <span className="mt-2 text-sm font-semibold text-clinical-ink">Upload lesion image</span>
              <span className="mt-1 text-xs text-clinical-muted">PNG, JPG, or WebP</span>
            </>
          )}
        </label>

        <div className="flex flex-wrap items-center justify-between gap-3 rounded-clinical border border-clinical-line bg-clinical-raised p-3">
          <div>
            <p className="text-sm font-semibold text-clinical-ink">{caseRecord?.caseId ?? uploadedImage?.name ?? "Upload intake"}</p>
            <p className="text-xs text-clinical-muted">{caseRecord ? `${caseRecord.maskedPatientId} - ${caseRecord.visitDate}` : "No analysis loaded"}</p>
          </div>
          <StatusBadge label={caseRecord?.reviewStatus ?? "Awaiting analysis"} tone={caseRecord ? caseRecord.reviewStatus === "Senior review" ? "danger" : caseRecord.reviewStatus === "Reviewed" ? "success" : "warning" : "neutral"} />
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
            <label className="grid gap-1 text-sm font-medium text-clinical-muted">
              Age
              <input
                type="number"
                min="0"
                max="110"
                value={uploadMetadata.ageYears ?? ""}
                onChange={(event) => onUploadMetadataChange({ ...uploadMetadata, ageYears: event.target.value ? Number(event.target.value) : null })}
                className="clinical-field rounded-md border px-3 py-2 outline-none"
              />
            </label>
            <label className="grid gap-1 text-sm font-medium text-clinical-muted">
              Sex
              <select
                value={uploadMetadata.sex}
                onChange={(event) => onUploadMetadataChange({ ...uploadMetadata, sex: event.target.value as UploadMetadataInput["sex"] })}
                className="clinical-field rounded-md border px-3 py-2 outline-none"
              >
                <option>Unknown</option>
                <option>Female</option>
                <option>Male</option>
              </select>
            </label>
            <label className="grid gap-1 text-sm font-medium text-clinical-muted">
              Site
              <select
                value={uploadMetadata.anatomicalSite}
                onChange={(event) => onUploadMetadataChange({ ...uploadMetadata, anatomicalSite: event.target.value })}
                className="clinical-field rounded-md border px-3 py-2 outline-none"
              >
                {siteOptions.map((site) => <option key={site}>{site}</option>)}
              </select>
            </label>
            <label className="grid gap-1 text-sm font-medium text-clinical-muted">
              Mode
              <select
                value={uploadMetadata.modelMode}
                onChange={(event) => onUploadMetadataChange({ ...uploadMetadata, modelMode: event.target.value as ModelMode })}
                className="clinical-field rounded-md border px-3 py-2 outline-none"
              >
                {modelModes.map((mode) => <option key={mode}>{mode}</option>)}
              </select>
            </label>
        </div>

        <div className="rounded-clinical border border-clinical-accent/25 bg-clinical-accentSoft/70 p-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-clinical-muted">Recommendation control</p>
              <p className="mt-1 text-sm font-semibold text-clinical-ink">Clinical review tier</p>
            </div>
            <StatusBadge label={recommendationLabel} tone={recommendationTone} />
          </div>
          <p className="mt-2 text-xs leading-5 text-clinical-muted">
            {metadataComplete ? "Metadata gate is complete. Analysis can run." : "Complete age, sex, and site to enable analysis."}
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 rounded-clinical border border-clinical-line bg-clinical-raised p-3">
            <div className="flex items-start gap-3">
              <FileImage className="mt-0.5 h-4 w-4 text-clinical-accent" />
              <div>
                <p className="text-sm font-semibold text-clinical-ink">{metadataComplete ? "Ready for live analysis" : "Metadata required before analysis"}</p>
                <p className="text-xs text-clinical-muted">Input image + metadata to preprocessing, classifier, explainability, and SLM output bundle.</p>
              </div>
            </div>
            <StatusBadge label={analysisReady ? "Analyzed" : metadataComplete ? "Ready" : "Metadata required"} tone={analysisReady || metadataComplete ? "success" : "warning"} />
        </div>

        <div className="flex flex-wrap gap-2">
            <button
              type="button"
              disabled={!canRunAnalysis}
              onClick={runAnalysis}
              className="rounded-md border border-clinical-accent bg-clinical-accent px-3 py-2 text-left text-sm font-semibold text-clinical-canvas outline-none transition hover:bg-clinical-accentHover focus-visible:ring-2 focus-visible:ring-clinical-accent/50 disabled:cursor-not-allowed disabled:border-clinical-line disabled:bg-clinical-raised disabled:text-clinical-muted"
            >
              Run analysis
            </button>
            <label className="cursor-pointer rounded-md border border-clinical-line bg-clinical-raised px-3 py-2 text-sm font-semibold text-clinical-ink outline-none transition hover:border-clinical-accent/35 hover:bg-clinical-soft focus-within:ring-2 focus-within:ring-clinical-accent/50">
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                className="sr-only"
                onChange={(event) => {
                  const nextFile = event.target.files?.[0] ?? null;
                  if (nextFile) onImageSelected(nextFile, uploadMetadata);
                  event.currentTarget.value = "";
                }}
              />
              Replace image
            </label>
            {onUseSampleCase && (
              <button
                type="button"
                onClick={onUseSampleCase}
                className="rounded-md border border-clinical-line bg-clinical-raised px-3 py-2 text-sm font-semibold text-clinical-ink outline-none transition hover:border-clinical-accent/35 hover:bg-clinical-soft focus-visible:ring-2 focus-visible:ring-clinical-accent/50"
              >
                Use sample case
              </button>
            )}
        </div>
      </div>
    </Card>
  );
}
