import type { ReviewAction, SystemStatus } from "../types/lesioniq";

export const systemStatus: SystemStatus = {
  modelVersion: "LesionIQ hybrid ensemble v3",
  appVersion: "Frontend prototype 0.1.0",
  calibrationStatus: "Temperature scaling active (T=0.75)",
  thresholdTuningStatus: "Clinical DiffEvo thresholds active with MEL safety review flag",
  explainabilityStatus: "Grad-CAM++, Swin attention rollout, metadata attribution, SLM narrative",
  preprocessingVersion: "lesioniq-preprocess 2026.04",
  inferenceMode: "Full Hybrid"
};

export const reviewActions: ReviewAction[] = [
  { id: "explain", label: "Review explanation", tone: "primary" },
  { id: "follow-up", label: "Flag for follow-up", tone: "secondary" },
  { id: "senior", label: "Request senior review", tone: "danger" },
  { id: "export", label: "Export report", tone: "secondary" },
  { id: "reviewed", label: "Mark reviewed", tone: "success" }
];

