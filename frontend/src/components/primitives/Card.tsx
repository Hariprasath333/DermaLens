import type { PropsWithChildren, ReactNode } from "react";
import { cx } from "../../lib/format";

interface CardProps extends PropsWithChildren {
  title?: string;
  eyebrow?: string;
  action?: ReactNode;
  className?: string;
}

export function Card({ title, eyebrow, action, className, children }: CardProps) {
  return (
    <section className={cx("panel-surface rounded-clinical border border-clinical-line shadow-clinical", className)}>
      {(title || eyebrow || action) && (
        <header className="flex items-start justify-between gap-4 border-b border-clinical-line px-4 py-3">
          <div>
            {eyebrow && <p className="text-[11px] font-bold uppercase tracking-[0.11em] text-clinical-muted">{eyebrow}</p>}
            {title && <h2 className="mt-0.5 text-[15px] font-semibold text-clinical-ink">{title}</h2>}
          </div>
          {action}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

