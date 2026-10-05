import { cx } from "../../lib/format";

interface MetricCardProps {
  label: string;
  value: string;
  detail?: string;
  tone?: "neutral" | "accent" | "warning" | "danger";
}

const tones = {
  neutral: "border-clinical-line quiet-panel",
  accent: "border-clinical-accent/30 bg-clinical-accentSoft shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]",
  warning: "border-clinical-warning/30 bg-clinical-warning/10",
  danger: "border-clinical-danger/30 bg-clinical-danger/10"
};

export function MetricCard({ label, value, detail, tone = "neutral" }: MetricCardProps) {
  const valueTone = tone === "accent" || tone === "warning" ? "text-clinical-accent" : "text-clinical-ink";

  return (
    <div className={cx("rounded-clinical border bg-clinical-raised p-3", tones[tone])}>
      <p className="text-[11px] font-bold uppercase tracking-[0.11em] text-clinical-muted">{label}</p>
      <p className={cx("mt-1 text-base font-semibold tabular-nums", valueTone)}>{value}</p>
      {detail && <p className="mt-1 text-sm text-clinical-muted">{detail}</p>}
    </div>
  );
}

