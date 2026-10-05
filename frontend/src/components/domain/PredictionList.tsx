import type { PredictionScore } from "../../types/lesioniq";
import { pct } from "../../lib/format";
import { Card } from "../primitives/Card";

export function PredictionRow({ score, rank }: { score: PredictionScore; rank: number }) {
  const top = rank === 1;
  return (
    <li className={`rounded-md border px-3 py-2 transition ${top ? "feature-panel border-clinical-accent/40" : "border-clinical-line bg-clinical-raised hover:bg-clinical-soft"}`}>
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-clinical-muted">#{rank}</span>
            <span className="font-semibold text-clinical-ink">{score.classCode}</span>
            <span className="truncate text-sm text-clinical-muted">{score.classLabel}</span>
          </div>
          <div className="mt-2 h-2 rounded-full bg-clinical-line">
            <div className="h-2 rounded-full bg-clinical-accent" style={{ width: pct(score.probability) }} />
          </div>
        </div>
        <div className="text-right">
          <p className="font-semibold tabular-nums text-clinical-accent">{pct(score.probability)}</p>
          <p className="text-xs tabular-nums text-clinical-accent">margin {(score.thresholdMargin * 100).toFixed(0)} pts</p>
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

