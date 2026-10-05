import { CheckCircle2, Clock3, ShieldAlert, TriangleAlert } from "lucide-react";
import { cx } from "../../lib/format";

interface StatusBadgeProps {
  label: string;
  tone?: "neutral" | "success" | "warning" | "danger" | "accent";
}

const toneMap = {
  neutral: "border-clinical-line bg-clinical-raised text-clinical-muted",
  success: "border-clinical-success/45 bg-clinical-success/20 text-clinical-ink",
  warning: "border-clinical-warning/45 bg-clinical-warning/25 text-clinical-ink",
  danger: "border-clinical-danger/40 bg-clinical-danger/15 text-clinical-ink",
  accent: "border-clinical-accent/45 bg-clinical-accentSoft text-clinical-ink"
};

export function StatusBadge({ label, tone = "neutral" }: StatusBadgeProps) {
  const Icon = tone === "danger" ? ShieldAlert : tone === "warning" ? TriangleAlert : tone === "success" ? CheckCircle2 : Clock3;
  return (
    <span className={cx("inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-semibold", toneMap[tone])}>
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />
      {label}
    </span>
  );
}

