import { ShieldAlert } from "lucide-react";

export function DisclaimerBanner() {
  return (
    <div className="warning-surface flex items-start gap-3 rounded-clinical border px-4 py-3 text-sm text-clinical-ink shadow-[0_12px_30px_rgba(0,0,0,0.18)]">
      <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-clinical-warning" aria-hidden="true" />
      <p>
        <strong>Decision support only.</strong> LesionIQ is not a diagnostic replacement. All predictions,
        explanations, and exported reports require qualified dermatologist review.
      </p>
    </div>
  );
}

