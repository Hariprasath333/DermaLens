import { useMemo, useState } from "react";
import { AlertCircle, CheckCircle2, RotateCcw, ShieldAlert } from "lucide-react";
import { Card } from "../primitives/Card";
import { pct } from "../../lib/format";
import { defaultAbcdeState, type AbcdeState } from "../../types/abcde";

interface CriterionConfig<T extends number> {
  letter: string;
  name: string;
  question: string;
  options: Array<{
    value: T;
    points: number;
    label: string;
    description: string;
  }>;
}

const criteria: {
  asymmetry: CriterionConfig<0 | 1 | 2>;
  border: CriterionConfig<0 | 1 | 2>;
  color: CriterionConfig<0 | 1 | 2>;
  diameter: CriterionConfig<0 | 1>;
  evolution: CriterionConfig<0 | 1>;
} = {
  asymmetry: {
    letter: "A",
    name: "Asymmetry",
    question: "Do the two halves match when bisected?",
    options: [
      { value: 0, points: 0, label: "Symmetric", description: "Bilateral symmetry across both orthogonal axes" },
      { value: 1, points: 1, label: "1-Axis Asymmetric", description: "Asymmetry in contour, pigment, or structure across 1 axis" },
      { value: 2, points: 2, label: "2-Axis Asymmetric", description: "Marked asymmetry across both orthogonal axes" }
    ]
  },
  border: {
    letter: "B",
    name: "Border",
    question: "Are the borders smooth, sharp, or irregular?",
    options: [
      { value: 0, points: 0, label: "Regular & Sharp", description: "Well-circumscribed, uniform perimeter" },
      { value: 1, points: 1, label: "Mild Notching", description: "Focal scalloping or notched edge in 1–2 quadrants" },
      { value: 2, points: 2, label: "Marked Jaggedness", description: "Ill-defined, abrupt cutoff, or spiculated across 3+ quadrants" }
    ]
  },
  color: {
    letter: "C",
    name: "Color Variegation",
    question: "How many distinct colors or pigments are present?",
    options: [
      { value: 0, points: 0, label: "Uniform (1 color)", description: "Monochrome tan, light brown, or uniform flesh tone" },
      { value: 1, points: 1, label: "Dual (2 colors)", description: "Tan with areas of dark brown or uniform darkening" },
      { value: 2, points: 2, label: "Variegated (3+ colors)", description: "Black, blue-gray veil, dark brown, red, or white regression" }
    ]
  },
  diameter: {
    letter: "D",
    name: "Diameter",
    question: "Is the clinical diameter ≥ 6mm?",
    options: [
      { value: 0, points: 0, label: "< 6 mm", description: "Smaller than a pencil eraser head" },
      { value: 1, points: 1, label: "≥ 6 mm", description: "Equal to or exceeding 6 mm across the longest dimension" }
    ]
  },
  evolution: {
    letter: "E",
    name: "Evolution / Change",
    question: "Has the patient or clinician documented recent evolution?",
    options: [
      { value: 0, points: 0, label: "Stable / Static", description: "No documented changes in size, shape, color, or symptoms" },
      { value: 1, points: 1, label: "Evolving / Symptomatic", description: "Enlargement, shape alteration, darkening, itching, or bleeding" }
    ]
  }
};

export function AbcdeCalculator({
  melanomaProbability = 0,
  state,
  onChange
}: {
  melanomaProbability?: number;
  state?: AbcdeState;
  onChange?: (state: AbcdeState) => void;
}) {
  const [internalState, setInternalState] = useState<AbcdeState>(defaultAbcdeState);
  const current = state ?? internalState;

  function update(key: keyof AbcdeState, value: number) {
    const next = { ...current, [key]: value };
    if (onChange) {
      onChange(next as AbcdeState);
    } else {
      setInternalState(next as AbcdeState);
    }
  }

  const score = current.asymmetry + current.border + current.color + current.diameter + current.evolution;

  // Composite Risk Assessment (ABCDE + Calibrated Model Probability)
  const assessment = useMemo(() => {
    if (score >= 5 || (score >= 4 && melanomaProbability >= 0.25) || melanomaProbability >= 0.5) {
      return {
        tier: "High Concern — Biopsy Recommended",
        tone: "danger" as const,
        recommendation: "Strong clinical & algorithmic correlation for malignancy. Urgent in-person excisional biopsy (1–2 mm margins) recommended.",
        urgencyIcon: ShieldAlert
      };
    }

    if (score >= 3 || melanomaProbability >= 0.15) {
      return {
        tier: "Moderate Concern — Short-term Follow-up",
        tone: "warning" as const,
        recommendation: "Atypical structural cues detected. Recommend short-term digital dermoscopy follow-up (3 months) or diagnostic biopsy if evolution continues.",
        urgencyIcon: AlertCircle
      };
    }

    return {
      tier: "Low Concern — Routine Surveillance",
      tone: "success" as const,
      recommendation: "Clinically reassuring ABCDE pattern combined with low model probability. Routine annual skin surveillance.",
      urgencyIcon: CheckCircle2
    };
  }, [score, melanomaProbability]);

  return (
    <Card
      title="ABCDE Melanoma Rule Calculator"
      eyebrow="Clinical Dermatological Criteria"
      action={
        <button
          type="button"
          onClick={() => {
            if (onChange) onChange(defaultAbcdeState);
            else setInternalState(defaultAbcdeState);
          }}
          className="inline-flex items-center gap-1.5 rounded-clinical border border-clinical-line bg-clinical-surface px-2.5 py-1 font-mono text-[11px] font-medium text-clinical-muted outline-none transition hover:border-clinical-accent/40 hover:text-clinical-ink focus-visible:ring-2 focus-visible:ring-clinical-accent/50"
          aria-label="Reset ABCDE inputs"
          title="Reset to 0 points"
        >
          <RotateCcw className="h-3 w-3" />
          Reset
        </button>
      }
      className="space-y-4"
    >
      {/* Criteria Breakdown Rows */}
      <div className="space-y-3.5">
        {(Object.keys(criteria) as Array<keyof AbcdeState>).map((key) => {
          const cfg = criteria[key];
          const selectedValue = current[key];

          return (
            <div key={key} className="rounded-clinical border border-clinical-line bg-clinical-surface p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-sm border border-clinical-accent/40 bg-clinical-accentSoft font-mono text-xs font-bold text-clinical-accent">
                    {cfg.letter}
                  </span>
                  <span className="text-sm font-semibold text-clinical-ink">{cfg.name}</span>
                </div>
                <span className="font-mono text-xs font-semibold tabular-nums text-clinical-accent">
                  +{selectedValue} {selectedValue === 1 ? "pt" : "pts"}
                </span>
              </div>
              <p className="mt-1 text-xs text-clinical-muted">{cfg.question}</p>

              {/* Selection Options */}
              <div className="mt-2.5 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {cfg.options.map((opt) => {
                  const isSelected = selectedValue === opt.value;
                  const isHighPoint = opt.points >= 2 || (key === "evolution" && opt.points === 1);

                  return (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => update(key, opt.value)}
                      className={`flex flex-col items-start rounded-clinical border p-2.5 text-left outline-none transition focus-visible:ring-2 focus-visible:ring-clinical-accent/50 ${
                        isSelected
                          ? isHighPoint
                            ? "border-clinical-danger/60 bg-clinical-danger/10 text-clinical-ink"
                            : "border-clinical-accent/60 bg-clinical-accentSoft text-clinical-ink"
                          : "border-clinical-line bg-clinical-surface hover:border-clinical-line/80 hover:bg-clinical-raised text-clinical-muted hover:text-clinical-ink"
                      }`}
                    >
                      <div className="flex w-full items-center justify-between gap-1">
                        <span className="font-mono text-xs font-semibold text-clinical-ink">
                          {opt.label}
                        </span>
                        <span
                          className={`font-mono text-[10px] font-bold ${
                            isSelected
                              ? isHighPoint
                                ? "text-clinical-danger"
                                : "text-clinical-accent"
                              : "text-clinical-muted"
                          }`}
                        >
                          +{opt.points}
                        </span>
                      </div>
                      <span className="mt-1 line-clamp-2 text-[11px] leading-4 text-clinical-muted">
                        {opt.description}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {/* Quick Presets */}
      <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-clinical-line text-xs text-clinical-muted">
        <span className="font-mono text-[11px] uppercase tracking-wider font-semibold">Presets:</span>
        <button
          type="button"
          onClick={() => {
            const preset: AbcdeState = { asymmetry: 0, border: 0, color: 0, diameter: 0, evolution: 0 };
            if (onChange) onChange(preset);
            else setInternalState(preset);
          }}
          className="rounded-sm border border-clinical-line bg-clinical-raised px-2 py-0.5 font-mono text-[11px] hover:border-clinical-accent/40 hover:text-clinical-ink"
        >
          Benign Nevus (0 pts)
        </button>
        <button
          type="button"
          onClick={() => {
            const preset: AbcdeState = { asymmetry: 1, border: 1, color: 1, diameter: 1, evolution: 0 };
            if (onChange) onChange(preset);
            else setInternalState(preset);
          }}
          className="rounded-sm border border-clinical-line bg-clinical-raised px-2 py-0.5 font-mono text-[11px] hover:border-clinical-accent/40 hover:text-clinical-ink"
        >
          Dysplastic Atypical (4 pts)
        </button>
        <button
          type="button"
          onClick={() => {
            const preset: AbcdeState = { asymmetry: 2, border: 2, color: 2, diameter: 1, evolution: 1 };
            if (onChange) onChange(preset);
            else setInternalState(preset);
          }}
          className="rounded-sm border border-clinical-danger/30 bg-clinical-danger/5 px-2 py-0.5 font-mono text-[11px] text-clinical-danger hover:border-clinical-danger/60"
        >
          Melanoma Red Flags (8 pts)
        </button>
      </div>

      {/* Composite Clinical Score & Recommendation Card */}
      <div
        className={`rounded-clinical border p-3.5 transition-colors ${
          assessment.tone === "danger"
            ? "border-clinical-danger/50 bg-clinical-danger/5"
            : assessment.tone === "warning"
            ? "border-clinical-danger/30 bg-clinical-danger/5"
            : "border-clinical-accent/40 bg-clinical-accentSoft"
        }`}
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <assessment.urgencyIcon
                className={`h-4 w-4 shrink-0 ${
                  assessment.tone === "danger" || assessment.tone === "warning"
                    ? "text-clinical-danger"
                    : "text-clinical-accent"
                }`}
                aria-hidden="true"
              />
              <p className="font-semibold text-clinical-ink text-sm">{assessment.tier}</p>
            </div>
            <p className="mt-1 text-xs leading-5 text-clinical-muted">{assessment.recommendation}</p>
          </div>

          <div className="text-right">
            <div className="flex items-baseline justify-end gap-1">
              <span
                className={`font-mono text-2xl font-bold tabular-nums ${
                  score >= 4 ? "text-clinical-danger" : "text-clinical-accent"
                }`}
              >
                {score}
              </span>
              <span className="font-mono text-xs text-clinical-muted">/ 8 pts</span>
            </div>
            <p className="font-mono text-[11px] text-clinical-muted">
              Model MEL: <strong className="text-clinical-ink">{pct(melanomaProbability)}</strong>
            </p>
          </div>
        </div>
      </div>
    </Card>
  );
}
