export type LesionClassCode = "MEL" | "NV" | "BCC" | "BKL" | "AK" | "SCC" | "VASC" | "DF";

export type ReviewStatus = "Needs review" | "In review" | "Reviewed" | "Senior review";
export type ModelMode = "Full Hybrid" | "Image Only" | "EffNet Only" | "Swin Only";
export type Urgency = "Routine" | "Expedited review" | "High concern";
export type Direction = "supports" | "weakens";
export type AuditStatus = "pass" | "warning" | "review-needed";
export type OverlayMode = "raw" | "gradcam" | "attention" | "metadata";
export type ThemeMode = "light" | "dark";

export interface PredictionScore {
  classCode: LesionClassCode;
  classLabel: string;
  probability: number;
  threshold: number;
  thresholdMargin: number;
}

export interface PatientMetadata {
  ageYears: number;
  sex: "Female" | "Male" | "Unknown";
  anatomicalSite: string;
  FitzpatrickType?: "I" | "II" | "III" | "IV" | "V" | "VI" | "Unknown";
  metadataCompleteness: "Complete" | "Partial";
}

export interface LesionMetric {
  label: string;
  value: string;
  note?: string;
}

export interface AttributionSignal {
  id: string;
  label: string;
  value: number;
  direction: Direction;
  note: string;
}

export interface AuditCheck {
  id: string;
  label: string;
  status: AuditStatus;
  note: string;
}

export interface ExplainabilityBundle {
  gradcamSummary: string;
  attentionSummary: string;
  metadataSignals: AttributionSignal[];
  slmSummary: string;
  auditChecks: AuditCheck[];
  auditNotes: string[];
}

export interface PreprocessingStep {
  id: string;
  title: string;
  shortTitle: string;
  description: string;
  technicalNote: string;
  status: "Complete" | "Review";
  previewTone: "raw" | "hair" | "normalized" | "border";
}

export interface HistoryEntry {
  id: string;
  date: string;
  predictedClassCode: LesionClassCode;
  predictedClassLabel: string;
  confidence: number;
  status: ReviewStatus;
  urgency: Urgency;
  note: string;
  samePatient: boolean;
}

export interface CompareEntry {
  id: string;
  date: string;
  label: string;
  predictedClassCode: LesionClassCode;
  predictedClassLabel: string;
  calibratedConfidence: number;
  reviewStatus: ReviewStatus;
  summary: string;
}

export interface ReviewAction {
  id: string;
  label: string;
  tone: "primary" | "secondary" | "danger" | "success";
}

export interface SystemStatus {
  modelVersion: string;
  appVersion: string;
  calibrationStatus: string;
  thresholdTuningStatus: string;
  explainabilityStatus: string;
  preprocessingVersion: string;
  inferenceMode: ModelMode;
}

export interface CaseRecord {
  id: string;
  caseId: string;
  maskedPatientId: string;
  visitDate: string;
  acquisitionTimestamp: string;
  reviewStatus: ReviewStatus;
  modelMode: ModelMode;
  decisionSupportBanner: string;
  predictedClassCode: LesionClassCode;
  predictedClassLabel: string;
  calibratedConfidence: number;
  thresholdMargin: number;
  recommendation: string;
  urgency: Urgency;
  metadata: PatientMetadata;
  lesionMetrics: LesionMetric[];
  predictionScores: PredictionScore[];
  explainability: ExplainabilityBundle;
  preprocessingSteps: PreprocessingStep[];
  historyEntries: HistoryEntry[];
  compareEntries: CompareEntry[];
  clinicianNotesPreview?: string;
  uploadedImageUrl?: string;
  inferenceBundle?: InferenceBundle;
}

export interface InferenceBundle {
  sourceImageName: string;
  outputDirectory: string;
  rawArtifact?: string;
  dullRazorArtifact?: string;
  normalizedArtifact?: string;
  claheArtifact?: string;
  finalPreprocessedArtifact?: string;
  originalArtifact: string;
  gradcamArtifact: string;
  attentionArtifact: string;
  diagnosisArtifact: string;
  slmContainer?: string;
  slmModel?: string;
  slmStatus: "Generated" | "Pending clinician verification";
}

export interface UploadMetadataInput {
  ageYears: number | null;
  sex: "Female" | "Male" | "Unknown";
  anatomicalSite: string;
  modelMode: ModelMode;
}

