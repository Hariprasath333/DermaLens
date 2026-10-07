import { useState } from "react";
import { Download, FileText, Printer, X } from "lucide-react";
import type { CaseRecord, OverlayMode } from "../../types/lesioniq";
import type { AbcdeState } from "../../types/abcde";
import { downloadExplainabilityReport } from "../../lib/explainabilityReport";
import { pct } from "../../lib/format";

interface ClinicalReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  caseRecord: CaseRecord;
  abcdeState: AbcdeState;
  artifactUrls?: Partial<Record<OverlayMode, string>>;
}

export function ClinicalReportModal({
  isOpen,
  onClose,
  caseRecord,
  abcdeState,
  artifactUrls
}: ClinicalReportModalProps) {
  const [clinicianName, setClinicianName] = useState("Attending Dermatologist, MD");
  const [clinicalAction, setClinicalAction] = useState<"biopsy" | "monitoring" | "routine" | "referral">("biopsy");
  const [clinicianNote, setClinicianNote] = useState(
    caseRecord.recommendation ?? "Correlate with clinical history and dermoscopic structural criteria."
  );

  if (!isOpen) return null;

  const totalAbcde =
    abcdeState.asymmetry +
    abcdeState.border +
    abcdeState.color +
    abcdeState.diameter +
    abcdeState.evolution;

  const abcdeTier =
    totalAbcde >= 5
      ? "High Concern — Urgent Biopsy Indicated"
      : totalAbcde >= 3
      ? "Moderate Concern — Short-Term Monitoring"
      : "Low Concern — Routine Surveillance";

  const isMalignant = ["MEL", "BCC", "SCC"].includes(caseRecord.predictedClassCode);

  function handlePrint() {
    window.print();
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-xs print:p-0 print:bg-white print:static">
      <div className="relative flex max-h-[92vh] w-full max-w-4xl flex-col rounded-clinical border border-clinical-line bg-clinical-surface shadow-2xl print:max-h-none print:w-full print:border-none print:shadow-none print:rounded-none">
        {/* Top Control Bar (Hidden on print) */}
        <div className="no-print flex items-center justify-between border-b border-clinical-line px-5 py-3 bg-clinical-surface">
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-clinical-accent" />
            <h2 className="text-sm font-bold text-clinical-ink">Clinical Monograph Export</h2>
            <span className="font-mono text-xs text-clinical-muted">[{caseRecord.caseId}]</span>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handlePrint}
              className="inline-flex items-center gap-2 rounded-clinical border border-clinical-accent bg-clinical-accent px-3 py-1.5 font-mono text-xs font-semibold text-clinical-canvas outline-none transition hover:bg-clinical-accentHover focus-visible:ring-2 focus-visible:ring-clinical-accent/50"
            >
              <Printer className="h-3.5 w-3.5" />
              Print / Save as PDF
            </button>
            <button
              type="button"
              onClick={() => downloadExplainabilityReport(caseRecord, abcdeState)}
              className="inline-flex items-center gap-1.5 rounded-clinical border border-clinical-line bg-clinical-surface px-3 py-1.5 font-mono text-xs font-medium text-clinical-ink hover:border-clinical-accent/40"
            >
              <Download className="h-3.5 w-3.5" />
              .TXT
            </button>
            <button
              type="button"
              onClick={onClose}
              className="rounded-clinical border border-clinical-line p-1.5 text-clinical-muted hover:border-clinical-accent/40 hover:text-clinical-ink"
              aria-label="Close modal"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Printable Monograph Body */}
        <div className="overflow-y-auto p-6 text-clinical-ink print:overflow-visible print:p-0">
          <article id="clinical-monograph" className="space-y-5 text-sm">
            {/* Monograph Header */}
            <header className="border-b-2 border-clinical-ink pb-4">
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="h-3 w-3 rounded-sm bg-clinical-accent" />
                    <span className="font-mono text-xs font-bold uppercase tracking-widest text-clinical-ink">
                      DermaLens · Clinical Dermatopathology CDS
                    </span>
                  </div>
                  <h1 className="mt-1 text-xl font-bold tracking-tight text-clinical-ink">
                    DERMOSCOPY DECISION SUPPORT MONOGRAPH
                  </h1>
                </div>
                <div className="text-right font-mono text-[11px] text-clinical-muted">
                  <p>REPORT REF: <strong className="text-clinical-ink">DL-{caseRecord.caseId}</strong></p>
                  <p>GENERATED: {new Date().toISOString().replace("T", " ").slice(0, 19)} UTC</p>
                  <p>CALIBRATION: T=0.75 Scaling Active</p>
                </div>
              </div>
            </header>

            {/* Patient & Case Demographics */}
            <section className="rounded-clinical border border-clinical-line bg-clinical-surface p-3 print:bg-white">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 font-mono text-xs">
                <div>
                  <span className="text-[10px] uppercase text-clinical-muted block">Case ID</span>
                  <strong className="text-clinical-ink">{caseRecord.caseId}</strong>
                </div>
                <div>
                  <span className="text-[10px] uppercase text-clinical-muted block">Patient Identifier</span>
                  <strong className="text-clinical-ink">{caseRecord.maskedPatientId}</strong>
                </div>
                <div>
                  <span className="text-[10px] uppercase text-clinical-muted block">Acquisition Date</span>
                  <strong className="text-clinical-ink">{caseRecord.visitDate || caseRecord.acquisitionTimestamp}</strong>
                </div>
                <div>
                  <span className="text-[10px] uppercase text-clinical-muted block">Clinical Review Status</span>
                  <strong className={caseRecord.reviewStatus === "Senior review" ? "text-clinical-danger" : "text-clinical-ink"}>
                    {caseRecord.reviewStatus}
                  </strong>
                </div>
                <div>
                  <span className="text-[10px] uppercase text-clinical-muted block">Patient Age</span>
                  <strong className="text-clinical-ink">{caseRecord.metadata.ageYears ?? "Unknown"} years</strong>
                </div>
                <div>
                  <span className="text-[10px] uppercase text-clinical-muted block">Biological Sex</span>
                  <strong className="text-clinical-ink">{caseRecord.metadata.sex}</strong>
                </div>
                <div>
                  <span className="text-[10px] uppercase text-clinical-muted block">Anatomical Site</span>
                  <strong className="text-clinical-ink">{caseRecord.metadata.anatomicalSite}</strong>
                </div>
                <div>
                  <span className="text-[10px] uppercase text-clinical-muted block">Inference Mode</span>
                  <strong className="text-clinical-ink">{caseRecord.modelMode}</strong>
                </div>
              </div>
            </section>

            {/* Side-by-Side Photographic & Saliency Evidence */}
            <section className="space-y-2">
              <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-clinical-muted">
                Dermoscopy Photographic Evidence & Saliency Rollout
              </h3>
              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-clinical border border-clinical-line bg-clinical-canvas p-1.5 text-center">
                  <div className="aspect-square overflow-hidden rounded-sm bg-neutral-900">
                    <img
                      src={artifactUrls?.raw ?? caseRecord.uploadedImageUrl}
                      alt="Raw dermoscopy input"
                      className="h-full w-full object-cover"
                    />
                  </div>
                  <p className="mt-1 font-mono text-[10px] font-semibold text-clinical-ink">1. Raw Input</p>
                  <p className="text-[9px] text-clinical-muted">Cleaned input view</p>
                </div>

                <div className="rounded-clinical border border-clinical-line bg-clinical-canvas p-1.5 text-center">
                  <div className="aspect-square overflow-hidden rounded-sm bg-neutral-900">
                    <img
                      src={artifactUrls?.gradcam ?? artifactUrls?.raw ?? caseRecord.uploadedImageUrl}
                      alt="Grad-CAM++ activation map"
                      className="h-full w-full object-cover"
                    />
                  </div>
                  <p className="mt-1 font-mono text-[10px] font-semibold text-clinical-ink">2. Grad-CAM++ Map</p>
                  <p className="text-[9px] text-clinical-muted">EfficientNet-B4 localization</p>
                </div>

                <div className="rounded-clinical border border-clinical-line bg-clinical-canvas p-1.5 text-center">
                  <div className="aspect-square overflow-hidden rounded-sm bg-neutral-900">
                    <img
                      src={artifactUrls?.attention ?? artifactUrls?.raw ?? caseRecord.uploadedImageUrl}
                      alt="Transformer attention rollout"
                      className="h-full w-full object-cover"
                    />
                  </div>
                  <p className="mt-1 font-mono text-[10px] font-semibold text-clinical-ink">3. SwinV2 Attention</p>
                  <p className="text-[9px] text-clinical-muted">Fine architectural weighting</p>
                </div>
              </div>
            </section>

            {/* Model Diagnostic Classification & Differential */}
            <section className="grid gap-3 sm:grid-cols-2">
              {/* Primary Diagnostic Summary */}
              <div className={`rounded-clinical border p-3.5 ${
                isMalignant
                  ? "border-clinical-danger/50 bg-clinical-danger/5"
                  : "border-clinical-accent/40 bg-clinical-accentSoft"
              }`}>
                <span className="font-mono text-[10px] uppercase tracking-wider text-clinical-muted block">
                  Primary Algorithmic Classification
                </span>
                <div className="mt-1 flex items-baseline justify-between">
                  <span className={`font-mono text-2xl font-bold ${isMalignant ? "text-clinical-danger" : "text-clinical-accent"}`}>
                    {caseRecord.predictedClassCode} · {caseRecord.predictedClassLabel}
                  </span>
                  <span className="font-mono text-lg font-bold tabular-nums text-clinical-ink">
                    {pct(caseRecord.calibratedConfidence)}
                  </span>
                </div>
                <div className="mt-2 flex flex-wrap items-center gap-2 font-mono text-xs text-clinical-muted">
                  <span>Threshold Margin: <strong>+{(caseRecord.thresholdMargin * 100).toFixed(0)} pts</strong></span>
                  <span>·</span>
                  <span>Urgency: <strong className={isMalignant ? "text-clinical-danger" : "text-clinical-accent"}>{caseRecord.urgency}</strong></span>
                </div>
              </div>

              {/* ABCDE Dermatological Summary */}
              <div className="rounded-clinical border border-clinical-line bg-clinical-surface p-3.5 print:bg-white">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px] uppercase tracking-wider text-clinical-muted">
                    Clinical ABCDE Scoring
                  </span>
                  <span className="font-mono text-xs font-bold text-clinical-ink">
                    {totalAbcde} / 8 pts
                  </span>
                </div>
                <div className="mt-2 grid grid-cols-5 gap-1 font-mono text-[11px] text-center">
                  <div className="rounded border border-clinical-line p-1">
                    <span className="text-[9px] text-clinical-muted block">A</span>
                    <strong>+{abcdeState.asymmetry}</strong>
                  </div>
                  <div className="rounded border border-clinical-line p-1">
                    <span className="text-[9px] text-clinical-muted block">B</span>
                    <strong>+{abcdeState.border}</strong>
                  </div>
                  <div className="rounded border border-clinical-line p-1">
                    <span className="text-[9px] text-clinical-muted block">C</span>
                    <strong>+{abcdeState.color}</strong>
                  </div>
                  <div className="rounded border border-clinical-line p-1">
                    <span className="text-[9px] text-clinical-muted block">D</span>
                    <strong>+{abcdeState.diameter}</strong>
                  </div>
                  <div className="rounded border border-clinical-line p-1">
                    <span className="text-[9px] text-clinical-muted block">E</span>
                    <strong>+{abcdeState.evolution}</strong>
                  </div>
                </div>
                <p className="mt-2 text-xs font-semibold text-clinical-ink">
                  {abcdeTier}
                </p>
              </div>
            </section>

            {/* Ranked ISIC 8-Class Differential Table */}
            <section className="space-y-1.5">
              <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-clinical-muted">
                Calibrated Differential Diagnosis (8 ISIC Classes)
              </h3>
              <table className="w-full border-collapse border border-clinical-line text-left font-mono text-xs">
                <thead>
                  <tr className="border-b border-clinical-line bg-clinical-surface text-clinical-muted">
                    <th className="p-2 text-center w-10">Rank</th>
                    <th className="p-2 w-16">Code</th>
                    <th className="p-2">Pathological Entity</th>
                    <th className="p-2 text-right">Probability</th>
                    <th className="p-2 text-right">Margin</th>
                    <th className="p-2 text-center w-24">Risk Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-clinical-line">
                  {(caseRecord.predictionScores ?? []).map((score, idx) => {
                    const isMalignantClass = ["MEL", "BCC", "SCC"].includes(score.classCode);
                    return (
                      <tr key={score.classCode} className={idx === 0 ? "bg-clinical-surface font-semibold" : ""}>
                        <td className="p-2 text-center text-clinical-muted">#{idx + 1}</td>
                        <td className="p-2 font-bold">{score.classCode}</td>
                        <td className="p-2">{score.classLabel}</td>
                        <td className="p-2 text-right tabular-nums">{pct(score.probability)}</td>
                        <td className="p-2 text-right tabular-nums text-clinical-muted">
                          +{(score.thresholdMargin * 100).toFixed(0)} pts
                        </td>
                        <td className="p-2 text-center">
                          {isMalignantClass ? (
                            <span className="rounded border border-clinical-danger/40 bg-clinical-danger/10 px-1.5 py-0.5 text-[10px] font-bold text-clinical-danger">
                              High Risk
                            </span>
                          ) : (
                            <span className="text-[10px] text-clinical-muted">Benign</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </section>

            {/* Natural-Language SLM Rationale */}
            {caseRecord.explainability.slmSummary && (
              <section className="rounded-clinical border border-clinical-line bg-clinical-surface p-3 print:bg-white space-y-1">
                <h4 className="font-mono text-[10px] font-bold uppercase tracking-wider text-clinical-muted">
                  Structured Model Rationale (SLM Synthesis)
                </h4>
                <p className="text-xs leading-5 text-clinical-ink">
                  {caseRecord.explainability.slmSummary}
                </p>
              </section>
            )}

            {/* Clinician Attestation & Sign-off Block */}
            <section className="border-t-2 border-clinical-ink pt-4 space-y-3">
              <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-clinical-ink">
                Attending Clinician Review & Disposition
              </h3>

              <div className="grid gap-3 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <label className="no-print font-mono text-[11px] text-clinical-muted block">
                    Clinician Name & Designation:
                  </label>
                  <input
                    type="text"
                    value={clinicianName}
                    onChange={(e) => setClinicianName(e.target.value)}
                    className="no-print w-full rounded-clinical border border-clinical-line px-2.5 py-1 font-mono text-xs text-clinical-ink"
                  />
                  <p className="print:block hidden font-mono text-xs font-bold">
                    Reviewed By: {clinicianName}
                  </p>

                  <div className="mt-2 space-y-1 font-mono text-xs">
                    <p className="text-clinical-muted text-[10px] uppercase">Disposition Checked:</p>
                    <label className="flex items-center gap-2">
                      <input
                        type="radio"
                        name="action"
                        checked={clinicalAction === "biopsy"}
                        onChange={() => setClinicalAction("biopsy")}
                      />
                      <span>Excisional Biopsy Scheduled (1–2 mm margin)</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input
                        type="radio"
                        name="action"
                        checked={clinicalAction === "monitoring"}
                        onChange={() => setClinicalAction("monitoring")}
                      />
                      <span>Short-Term Digital Dermoscopy Follow-up (3 Months)</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input
                        type="radio"
                        name="action"
                        checked={clinicalAction === "routine"}
                        onChange={() => setClinicalAction("routine")}
                      />
                      <span>Routine Annual Skin Examination</span>
                    </label>
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="no-print font-mono text-[11px] text-clinical-muted block">
                    Clinician Evaluation Note:
                  </label>
                  <textarea
                    rows={3}
                    value={clinicianNote}
                    onChange={(e) => setClinicianNote(e.target.value)}
                    className="no-print w-full rounded-clinical border border-clinical-line p-2 text-xs text-clinical-ink"
                  />
                  <p className="print:block hidden text-xs leading-5">
                    <strong>Note:</strong> {clinicianNote}
                  </p>

                  <div className="pt-4 flex items-end justify-between font-mono text-xs">
                    <div>
                      <span className="block border-t border-clinical-line pt-1 text-[10px] text-clinical-muted">
                        Clinician Signature & Medical License #
                      </span>
                    </div>
                    <div>
                      <span className="block border-t border-clinical-line pt-1 text-[10px] text-clinical-muted">
                        Date Signed
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* Mandatory Regulatory Disclaimer */}
            <footer className="border-t border-clinical-line pt-2 text-[10px] leading-4 text-clinical-muted">
              <p>
                <strong>Clinical Decision Support Notice:</strong> DermaLens is an investigational decision-support software. It does not replace histopathological analysis or the clinical judgment of a licensed dermatologist. Excisional biopsy remains the definitive gold standard for suspected melanocytic malignancies.
              </p>
            </footer>
          </article>
        </div>
      </div>
    </div>
  );
}
