import { Moon, Sun } from "lucide-react";
import type { ThemeMode } from "../../types/lesioniq";

export function ThemeToggle({ theme, onToggle }: { theme: ThemeMode; onToggle: () => void }) {
  const dark = theme === "dark";
  return (
    <button
      type="button"
      onClick={onToggle}
      className="inline-flex items-center gap-2 rounded-clinical border border-clinical-line bg-clinical-surface px-3 py-1.5 font-mono text-xs font-medium text-clinical-ink outline-none hover:border-clinical-accent/45 hover:bg-clinical-accentSoft focus-visible:ring-2 focus-visible:ring-clinical-accent/50"
      aria-label="Toggle review theme"
    >
      {dark ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
      {dark ? "Dark review" : "Light view"}
    </button>
  );
}

