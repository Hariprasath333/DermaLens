import { cx } from "../../lib/format";

interface Tab<T extends string> {
  id: T;
  label: string;
}

interface SectionTabsProps<T extends string> {
  tabs: Tab<T>[];
  value: T;
  onChange: (value: T) => void;
  ariaLabel: string;
}

export function SectionTabs<T extends string>({ tabs, value, onChange, ariaLabel }: SectionTabsProps<T>) {
  return (
    <div className="control-surface flex flex-wrap gap-1 rounded-clinical border border-clinical-line p-1" role="tablist" aria-label={ariaLabel}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={value === tab.id}
          onClick={() => onChange(tab.id)}
          className={cx(
            "rounded-md px-3 py-2 text-sm font-semibold outline-none transition focus-visible:ring-2 focus-visible:ring-clinical-accent/50",
            value === tab.id ? "bg-clinical-accentSoft text-clinical-ink shadow-sm" : "text-clinical-muted hover:bg-clinical-raised hover:text-clinical-ink"
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

