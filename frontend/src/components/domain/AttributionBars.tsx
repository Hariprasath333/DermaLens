import type { AttributionSignal } from "../../types/lesioniq";
import { signed } from "../../lib/format";
import { Card } from "../primitives/Card";

export function AttributionBars({ signals, title = "Metadata signal attribution" }: { signals: AttributionSignal[]; title?: string }) {
  const sorted = [...signals].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  return (
    <Card title={title} eyebrow="SHAP-like contribution">
      <div className="space-y-3">
        {sorted.map((signal) => {
          const width = `${Math.min(Math.abs(signal.value) * 480, 100)}%`;
          const supporting = signal.direction === "supports";
          return (
            <div key={signal.id}>
              <div className="mb-1 flex items-center justify-between gap-3 text-sm">
                <span className="font-medium text-clinical-ink">{signal.label}</span>
                <span className={`font-semibold tabular-nums ${supporting ? "text-clinical-accent" : "text-clinical-danger"}`}>
                  {signed(signal.value)} {supporting ? "supports" : "weakens"}
                </span>
              </div>
              <div className="h-2 rounded-full bg-clinical-line">
                <div className={`h-2 rounded-full ${supporting ? "bg-clinical-accent" : "bg-clinical-danger"}`} style={{ width }} />
              </div>
              <p className="mt-1 text-xs text-clinical-muted">{signal.note}</p>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

