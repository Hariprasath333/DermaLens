import { CheckCircle2, Clock3, ShieldAlert, TriangleAlert } from "lucide-react";
import { cx } from "../../lib/format";

interface StatusBadgeProps {
  label: string;
  tone?: "neutral" | "success" | "warning" | "danger" | "accent";
}

const toneMap = {
  neutral: "border-clinical-line bg-clinical-surface text-clinical-muted",
  success: "border-clinical-accent/40 bg-clinical-accentSoft text-clinical-accent",
  warning: "border-clinical-danger/30 bg-clinical-danger/10 text-clinical-danger",
  danger: "border-clinical-danger/50 bg-clinical-danger/15 text-clinical-danger",
  accent: "border-clinical-accent/40 bg-clinical-accentSoft text-clinical-accent"
};

export function StatusBadge({ label, tone = "neutral" }: StatusBadgeProps) {
  const Icon = tone === "danger" ? ShieldAlert : tone === "warning" ? TriangleAlert : tone === "success" ? CheckCircle2 : Clock3;
  return (
    <span className={cx("inline-flex items-center gap-1.5 rounded-sm border px-2 py-0.5 font-mono text-[11px] font-medium tracking-wide", toneMap[tone])}>
      <Icon className="h-3 w-3 shrink-0" aria-hidden="true" />
      {label}
    </span>
  );
}

