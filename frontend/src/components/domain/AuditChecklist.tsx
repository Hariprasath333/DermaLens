import type { AuditCheck } from "../../types/lesioniq";
import { Card } from "../primitives/Card";
import { StatusBadge } from "../primitives/StatusBadge";

function tone(status: AuditCheck["status"]) {
  return status === "pass" ? "success" : status === "warning" ? "warning" : "danger";
}

export function AuditChecklist({ checks }: { checks: AuditCheck[] }) {
  return (
    <Card title="Audit checks" eyebrow="Heuristic review">
      <div className="space-y-3">
        {checks.map((check) => (
          <div key={check.id} className="rounded-md border border-clinical-line bg-clinical-raised p-3">
            <div className="flex items-center justify-between gap-3">
              <p className="font-semibold text-clinical-ink">{check.label}</p>
              <StatusBadge label={check.status.replace("-", " ")} tone={tone(check.status)} />
            </div>
            <p className="mt-1 text-sm text-clinical-muted">{check.note}</p>
          </div>
        ))}
      </div>
    </Card>
  );
}

