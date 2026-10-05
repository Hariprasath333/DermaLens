import type { PreprocessingStep } from "../../types/lesioniq";
import { Card } from "../primitives/Card";
import { StatusBadge } from "../primitives/StatusBadge";
import { DermoscopyMock } from "./DermoscopyMock";

export function PipelineStepCard({ step, index }: { step: PreprocessingStep; index: number }) {
  return (
    <Card
      title={`${index + 1}. ${step.title}`}
      action={<StatusBadge label={step.status} tone={step.status === "Complete" ? "success" : "warning"} />}
    >
      <DermoscopyMock compact stepTone={step.previewTone} label={step.title} />
      <p className="mt-3 text-sm text-clinical-ink">{step.description}</p>
      <p className="mt-2 rounded-md border border-clinical-line bg-clinical-raised p-3 text-sm text-clinical-muted">{step.technicalNote}</p>
    </Card>
  );
}

