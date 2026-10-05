import type { ReviewAction } from "../../types/lesioniq";
import { Card } from "../primitives/Card";

const tones = {
  primary: "border-clinical-accent bg-clinical-accent text-clinical-canvas hover:bg-clinical-accentHover",
  secondary: "border-clinical-line bg-clinical-raised text-clinical-ink hover:border-clinical-accent/35 hover:bg-clinical-accentSoft",
  danger: "border-clinical-danger/40 bg-clinical-danger/10 text-clinical-danger hover:bg-clinical-danger/15",
  success: "border-clinical-success/40 bg-clinical-success/10 text-clinical-success hover:bg-clinical-success/15"
};

export function ActionPanel({ actions, onAction }: { actions: ReviewAction[]; onAction?: (id: string) => void }) {
  return (
    <Card title="Clinician actions" eyebrow="Review workflow">
      <div className="grid gap-2">
        {actions.map((action) => (
          <button
            key={action.id}
            type="button"
            onClick={() => onAction?.(action.id)}
            className={`rounded-md border px-3 py-2 text-left text-sm font-semibold outline-none transition focus-visible:ring-2 focus-visible:ring-clinical-accent/50 ${tones[action.tone]}`}
          >
            {action.label}
          </button>
        ))}
      </div>
    </Card>
  );
}

