import type { ReactNode } from "react";

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  meta?: ReactNode;
}

export function PageHeader({ title, subtitle, meta }: PageHeaderProps) {
  return (
    <div className="feature-panel mb-5 flex flex-col gap-3 rounded-clinical border px-5 py-4 shadow-clinical lg:flex-row lg:items-end lg:justify-between">
      <div>
        <h1 className="text-2xl font-semibold tracking-[-0.01em] text-clinical-ink">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-sm leading-6 text-clinical-muted">{subtitle}</p>}
      </div>
      {meta && <div className="flex flex-wrap gap-2">{meta}</div>}
    </div>
  );
}

