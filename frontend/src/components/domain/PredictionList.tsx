import type { PredictionScore } from "../../types/lesioniq";
import { pct } from "../../lib/format";
import { Card } from "../primitives/Card";

export function PredictionRow({ score, rank }: { score: PredictionScore; rank: number }) {
  const top = rank === 1;
  const isMalignant = ["MEL", "BCC", "SCC"].includes(score.classCode);
  return (
    <li
      className={`rounded-clinical border px-3.5 py-2.5 transition ${
        top
          ? isMalignant
            ? "border-clinical-danger/45 bg-clinical-danger/5"
            : "border-clinical-accent/45 bg-clinical-accentSoft"
          : "border-clinical-line bg-clinical-surface hover:bg-clinical-raised"
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-semibold text-clinical-muted">#{rank}</span>
            <span
              className={`font-mono text-xs font-bold uppercase tracking-wider ${
                top && isMalignant ? "text-clinical-danger" : "text-clinical-ink"
              }`}
            >
              {score.classCode}
            </span>
            <span className="truncate text-sm text-clinical-muted">{score.classLabel}</span>
            {top && isMalignant && (
              <span className="rounded border border-clinical-danger/30 bg-clinical-danger/10 px-1.5 py-0.2 font-mono text-[10px] font-bold uppercase tracking-wider text-clinical-danger">
                High Risk
              </span>
            )}
          </div>
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-clinical-line/60">
            <div
              className={`h-full transition-all duration-300 ${
                top && isMalignant ? "bg-clinical-danger" : "bg-clinical-accent"
              }`}
              style={{ width: pct(score.probability) }}
            />
          </div>
        </div>
        <div className="text-right">
          <p
            className={`font-mono text-sm font-semibold tabular-nums ${
              top && isMalignant ? "text-clinical-danger" : "text-clinical-accent"
            }`}
          >
            {pct(score.probability)}
          </p>
          <p className="font-mono text-[11px] tabular-nums text-clinical-muted">
            margin {(score.thresholdMargin * 100).toFixed(0)} pts
          </p>
        </div>
      </div>
    </li>
  );
}

export function PredictionList({ scores }: { scores: PredictionScore[] }) {
  return (
    <Card title="Ranked differential diagnosis" eyebrow="All 8 ISIC classes">
      <ol className="space-y-2">
        {scores.map((score, index) => (
          <PredictionRow key={score.classCode} score={score} rank={index + 1} />
        ))}
      </ol>
    </Card>
  );
}

