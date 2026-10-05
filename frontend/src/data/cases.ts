import { lesionClasses } from "./classes";
import type { CaseRecord, LesionClassCode, PredictionScore, PreprocessingStep } from "../types/lesioniq";

export const preprocessingSteps: PreprocessingStep[] = [
  {
    id: "raw",
    title: "Raw input",
    shortTitle: "Raw input",
    description: "Original dermoscopy frame retained for provenance and visual comparison.",
    technicalNote: "Input is preserved before any cleanup so reviewers can inspect artifacts.",
    status: "Complete",
    previewTone: "raw"
  },
  {
    id: "dullrazor",
    title: "DullRazor hair removal",
    shortTitle: "DullRazor",
    description: "Hair-like occlusions are detected and inpainted before inference.",
    technicalNote: "DullRazor removes hair artifacts before inference using morphological detection and conservative inpainting.",
    status: "Complete",
    previewTone: "hair"
  },
  {
    id: "color-clahe",
    title: "Shades-of-Gray + CLAHE",
    shortTitle: "Color + CLAHE",
    description: "Color and contrast are normalized while preserving lesion color variation.",
    technicalNote:
      "Shades-of-Gray is conservative color normalization, and CLAHE is applied in LAB to enhance contrast without RGB distortion.",
    status: "Complete",
    previewTone: "normalized"
  },
  {
    id: "border",
    title: "Vignette border removal",
    shortTitle: "Border removal",
    description: "Dark circular dermoscope borders are removed to reduce hardware bias.",
    technicalNote: "Border removal reduces dermoscope vignette and border bias without cropping lesion content.",
    status: "Complete",
    previewTone: "border"
  }
];

function scores(entries: Array<[LesionClassCode, number, number]>): PredictionScore[] {
  return entries
    .map(([classCode, probability, threshold]) => ({
      classCode,
      classLabel: lesionClasses[classCode],
      probability,
      threshold,
      thresholdMargin: probability - threshold
    }))
    .sort((a, b) => b.probability - a.probability);
}

export const cases: CaseRecord[] = [
  {
    id: "case-mel-001",
    caseId: "LIQ-2026-0418-MEL",
    maskedPatientId: "PT-7A4-29X",
    visitDate: "2026-04-18",
    acquisitionTimestamp: "2026-04-18 10:42 IST",
    reviewStatus: "Senior review",
    modelMode: "Full Hybrid",
    decisionSupportBanner: "Decision support only — dermatologist review required.",
    predictedClassCode: "MEL",
    predictedClassLabel: "Melanoma",
    calibratedConfidence: 0.78,
    thresholdMargin: 0.21,
    recommendation: "Expedited dermatologist review; consider biopsy correlation if clinically concordant.",
    urgency: "High concern",
    metadata: {
      ageYears: 67,
      sex: "Male",
      anatomicalSite: "Upper back / posterior torso",
      FitzpatrickType: "III",
      metadataCompleteness: "Complete"
    },
    lesionMetrics: [
      { label: "Longest visible axis", value: "8.6 mm", note: "estimated from acquisition metadata" },
      { label: "Pigment pattern", value: "Asymmetric, variegated" },
      { label: "Image quality", value: "Adequate after preprocessing" }
    ],
    predictionScores: scores([
      ["MEL", 0.78, 0.57],
      ["NV", 0.1, 0.48],
      ["BKL", 0.05, 0.39],
      ["BCC", 0.03, 0.42],
      ["SCC", 0.02, 0.35],
      ["AK", 0.012, 0.31],
      ["DF", 0.005, 0.24],
      ["VASC", 0.003, 0.22]
    ]),
    explainability: {
      gradcamSummary: "EfficientNet activation concentrates over the darker eccentric pigment network and superior border.",
      attentionSummary: "Swin attention remains strongest on central and superior lesion patches, with minimal frame attention.",
      metadataSignals: [
        { id: "age", label: "Age 67 years", value: 0.16, direction: "supports", note: "Age increases support for malignant melanocytic concern." },
        { id: "site", label: "Posterior torso", value: 0.09, direction: "supports", note: "Anatomical site is plausible for melanoma-like presentation." },
        { id: "sex", label: "Male sex", value: 0.04, direction: "supports", note: "Small positive metadata contribution." },
        { id: "quality", label: "Adequate image quality", value: -0.03, direction: "weakens", note: "Clearer image reduces uncertainty but does not drive the class." }
      ],
      slmSummary:
        "Generated explanation: The leading melanoma prediction is primarily supported by image evidence, including asymmetric pigmentation and activation over the lesion center. Metadata contributes secondarily through age and posterior torso site. This output is decision support only and requires dermatologist verification before any clinical action.",
      auditChecks: [
        { id: "center", label: "Lesion-centered activation", status: "pass", note: "Heatmap overlaps lesion center and upper border." },
        { id: "artifact", label: "Artifact risk", status: "warning", note: "A small hair-removal trace is visible but not the dominant cue." },
        { id: "metadata", label: "Metadata overreliance", status: "pass", note: "Metadata contribution remains secondary to image branches." },
        { id: "agreement", label: "Cross-branch agreement", status: "pass", note: "CNN and transformer branches both support MEL." },
        { id: "slm", label: "Explanation consistency", status: "review-needed", note: "Narrative should be checked against the heatmap before report export." }
      ],
      auditNotes: [
        "Heatmap overlaps lesion center.",
        "Attention remains concentrated within lesion boundary.",
        "Metadata influence appears secondary.",
        "SLM explanation requires clinician verification."
      ]
    },
    preprocessingSteps,
    historyEntries: [
      { id: "h1", date: "2026-04-18", predictedClassCode: "MEL", predictedClassLabel: "Melanoma", confidence: 0.78, status: "Senior review", urgency: "High concern", note: "Current case, high-concern melanocytic differential.", samePatient: true },
      { id: "h2", date: "2025-11-03", predictedClassCode: "NV", predictedClassLabel: "Melanocytic nevus", confidence: 0.64, status: "Reviewed", urgency: "Routine", note: "Prior nevus on left shoulder reviewed as stable.", samePatient: true },
      { id: "h3", date: "2026-04-16", predictedClassCode: "SCC", predictedClassLabel: "Squamous cell carcinoma", confidence: 0.58, status: "In review", urgency: "Expedited review", note: "Different de-identified patient, keratinocyte lesion stream.", samePatient: false }
    ],
    compareEntries: [
      { id: "c1", date: "2026-04-18", label: "Current case", predictedClassCode: "MEL", predictedClassLabel: "Melanoma", calibratedConfidence: 0.78, reviewStatus: "Senior review", summary: "Current high-concern lesion with image-led MEL support." },
      { id: "c2", date: "2025-11-03", label: "Previous patient lesion", predictedClassCode: "NV", predictedClassLabel: "Melanocytic nevus", calibratedConfidence: 0.64, reviewStatus: "Reviewed", summary: "Prior reviewed nevus, lower confidence and different lesion site." }
    ],
    clinicianNotesPreview: "Irregular pigment network noted; senior dermatology review requested."
  },
  {
    id: "case-bcc-002",
    caseId: "LIQ-2026-0426-BCC",
    maskedPatientId: "PT-2Q8-11M",
    visitDate: "2026-04-26",
    acquisitionTimestamp: "2026-04-26 14:16 IST",
    reviewStatus: "In review",
    modelMode: "Full Hybrid",
    decisionSupportBanner: "Decision support only — dermatologist review required.",
    predictedClassCode: "BCC",
    predictedClassLabel: "Basal cell carcinoma",
    calibratedConfidence: 0.66,
    thresholdMargin: 0.17,
    recommendation: "Dermatologist review recommended; correlate with pearly border or arborizing vessel features.",
    urgency: "Expedited review",
    metadata: {
      ageYears: 58,
      sex: "Female",
      anatomicalSite: "Head / neck",
      FitzpatrickType: "II",
      metadataCompleteness: "Complete"
    },
    lesionMetrics: [
      { label: "Longest visible axis", value: "5.2 mm" },
      { label: "Visible structures", value: "Pink-white areas, focal vessels" },
      { label: "Image quality", value: "Good" }
    ],
    predictionScores: scores([
      ["BCC", 0.66, 0.49],
      ["SCC", 0.13, 0.38],
      ["AK", 0.08, 0.33],
      ["BKL", 0.05, 0.39],
      ["MEL", 0.04, 0.57],
      ["NV", 0.025, 0.48],
      ["VASC", 0.01, 0.22],
      ["DF", 0.005, 0.24]
    ]),
    explainability: {
      gradcamSummary: "CNN activation highlights the central pink-white region and adjacent vascular-looking structures.",
      attentionSummary: "Transformer patch attribution is distributed across the lesion body, with mild attention near the superior frame edge.",
      metadataSignals: [
        { id: "site", label: "Head / neck site", value: 0.13, direction: "supports", note: "Site contributes toward BCC-compatible context." },
        { id: "age", label: "Age 58 years", value: 0.08, direction: "supports", note: "Age modestly supports keratinocyte-origin concern." },
        { id: "sex", label: "Female sex", value: -0.01, direction: "weakens", note: "Minimal contribution." },
        { id: "fitz", label: "Fitzpatrick II", value: 0.05, direction: "supports", note: "Small contextual support in mock metadata branch." }
      ],
      slmSummary:
        "Generated explanation: The model favors basal cell carcinoma with moderate calibrated confidence. Visual evidence is centered on pale structureless zones and focal vascular cues, while head/neck site and age add secondary support. Dermatologist review is required because SCC and AK remain in the ranked differential.",
      auditChecks: [
        { id: "center", label: "Lesion-centered activation", status: "pass", note: "Activation is centered within the lesion." },
        { id: "artifact", label: "Artifact risk", status: "pass", note: "No dominant hair or border cue detected." },
        { id: "metadata", label: "Metadata overreliance", status: "warning", note: "Head/neck site is meaningful; inspect image evidence before accepting rationale." },
        { id: "agreement", label: "Cross-branch agreement", status: "warning", note: "CNN favors BCC; transformer also keeps SCC visible." },
        { id: "slm", label: "Explanation consistency", status: "pass", note: "Narrative matches ranked differential and attribution bundle." }
      ],
      auditNotes: [
        "Grad-CAM remains centered on visible lesion structures.",
        "Attention includes BCC-relevant patches but SCC remains a competing class.",
        "Metadata influence is clinically plausible but should not be treated as diagnostic.",
        "SLM explanation requires clinician verification."
      ]
    },
    preprocessingSteps,
    historyEntries: [
      { id: "h4", date: "2026-04-26", predictedClassCode: "BCC", predictedClassLabel: "Basal cell carcinoma", confidence: 0.66, status: "In review", urgency: "Expedited review", note: "Current moderate-risk keratinocyte lesion.", samePatient: true },
      { id: "h5", date: "2026-02-09", predictedClassCode: "BKL", predictedClassLabel: "Benign keratosis", confidence: 0.61, status: "Reviewed", urgency: "Routine", note: "Prior benign keratosis-like lesion reviewed.", samePatient: true },
      { id: "h6", date: "2026-04-18", predictedClassCode: "MEL", predictedClassLabel: "Melanoma", confidence: 0.78, status: "Senior review", urgency: "High concern", note: "High-concern reference case in review queue.", samePatient: false }
    ],
    compareEntries: [
      { id: "c3", date: "2026-04-26", label: "Current case", predictedClassCode: "BCC", predictedClassLabel: "Basal cell carcinoma", calibratedConfidence: 0.66, reviewStatus: "In review", summary: "Moderate confidence BCC with SCC visible in differential." },
      { id: "c4", date: "2026-02-09", label: "Previous patient lesion", predictedClassCode: "BKL", predictedClassLabel: "Benign keratosis", calibratedConfidence: 0.61, reviewStatus: "Reviewed", summary: "Prior lower-concern keratosis-like lesion." }
    ],
    clinicianNotesPreview: "Moderate BCC signal; compare SCC differential before final review."
  },
  {
    id: "case-nv-003",
    caseId: "LIQ-2026-0502-NV",
    maskedPatientId: "PT-4N6-03K",
    visitDate: "2026-05-02",
    acquisitionTimestamp: "2026-05-02 09:28 IST",
    reviewStatus: "Needs review",
    modelMode: "Full Hybrid",
    decisionSupportBanner: "Decision support only — dermatologist review required.",
    predictedClassCode: "NV",
    predictedClassLabel: "Melanocytic nevus",
    calibratedConfidence: 0.82,
    thresholdMargin: 0.34,
    recommendation: "Routine dermatologist review; no high-concern model flag in this mock case.",
    urgency: "Routine",
    metadata: {
      ageYears: 34,
      sex: "Female",
      anatomicalSite: "Lower extremity",
      FitzpatrickType: "IV",
      metadataCompleteness: "Complete"
    },
    lesionMetrics: [
      { label: "Longest visible axis", value: "3.9 mm" },
      { label: "Pigment pattern", value: "Symmetric reticular pattern" },
      { label: "Image quality", value: "Good" }
    ],
    predictionScores: scores([
      ["NV", 0.82, 0.48],
      ["BKL", 0.06, 0.39],
      ["MEL", 0.05, 0.57],
      ["DF", 0.025, 0.24],
      ["VASC", 0.018, 0.22],
      ["BCC", 0.012, 0.42],
      ["AK", 0.009, 0.31],
      ["SCC", 0.006, 0.35]
    ]),
    explainability: {
      gradcamSummary: "CNN activation is compact and centered on the symmetric pigment network.",
      attentionSummary: "Transformer attention follows the lesion patch grid with low off-lesion activation.",
      metadataSignals: [
        { id: "age", label: "Age 34 years", value: 0.1, direction: "supports", note: "Younger age supports benign nevus context in the mock branch." },
        { id: "site", label: "Lower extremity", value: 0.03, direction: "supports", note: "Small contextual contribution." },
        { id: "sex", label: "Female sex", value: 0.01, direction: "supports", note: "Minimal metadata influence." },
        { id: "fitz", label: "Fitzpatrick IV", value: -0.02, direction: "weakens", note: "Small countervailing signal; image evidence dominates." }
      ],
      slmSummary:
        "Generated explanation: The model favors melanocytic nevus with high calibrated confidence. Image evidence appears compact and lesion-centered, and metadata adds only minor support. This remains decision support; dermatologist review is required and melanoma remains listed in the differential.",
      auditChecks: [
        { id: "center", label: "Lesion-centered activation", status: "pass", note: "Activation is compact and centered." },
        { id: "artifact", label: "Artifact risk", status: "pass", note: "No artifact-driven cue detected." },
        { id: "metadata", label: "Metadata overreliance", status: "pass", note: "Metadata contribution is low." },
        { id: "agreement", label: "Cross-branch agreement", status: "pass", note: "Both image branches support NV." },
        { id: "slm", label: "Explanation consistency", status: "pass", note: "Generated summary remains cautious and review-oriented." }
      ],
      auditNotes: [
        "Heatmap is compact and lesion-centered.",
        "Attention remains concentrated within lesion boundary.",
        "Metadata influence appears secondary.",
        "SLM explanation requires clinician verification."
      ]
    },
    preprocessingSteps,
    historyEntries: [
      { id: "h7", date: "2026-05-02", predictedClassCode: "NV", predictedClassLabel: "Melanocytic nevus", confidence: 0.82, status: "Needs review", urgency: "Routine", note: "Current low-concern nevus-like case.", samePatient: true },
      { id: "h8", date: "2026-01-12", predictedClassCode: "NV", predictedClassLabel: "Melanocytic nevus", confidence: 0.79, status: "Reviewed", urgency: "Routine", note: "Prior similar lesion reviewed without high-concern flag.", samePatient: true },
      { id: "h9", date: "2026-04-26", predictedClassCode: "BCC", predictedClassLabel: "Basal cell carcinoma", confidence: 0.66, status: "In review", urgency: "Expedited review", note: "Moderate-risk reference case in queue.", samePatient: false }
    ],
    compareEntries: [
      { id: "c5", date: "2026-05-02", label: "Current case", predictedClassCode: "NV", predictedClassLabel: "Melanocytic nevus", calibratedConfidence: 0.82, reviewStatus: "Needs review", summary: "Low-concern nevus-like output; still requires review." },
      { id: "c6", date: "2026-01-12", label: "Previous patient lesion", predictedClassCode: "NV", predictedClassLabel: "Melanocytic nevus", calibratedConfidence: 0.79, reviewStatus: "Reviewed", summary: "Prior similar nevus-like case for context." }
    ],
    clinicianNotesPreview: "Low-concern model output; verify benign pattern before marking reviewed."
  }
];

