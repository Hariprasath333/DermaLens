import { ShieldAlert } from "lucide-react";

export function DisclaimerBanner() {
  return (
    <div className="flex items-start gap-3 rounded-clinical border border-clinical-danger/35 bg-clinical-danger/5 px-4 py-3 text-sm text-clinical-ink">
      <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-clinical-danger" aria-hidden="true" />
      <p>
        <strong className="font-semibold text-clinical-danger">Clinical Decision Support:</strong> DermaLens is an assistive research and clinical decision support system, not an autonomous diagnostic replacement. All predicted differential probabilities, saliency maps, and exported summaries require evaluation by a qualified dermatologist.
      </p>
    </div>
  );
}

