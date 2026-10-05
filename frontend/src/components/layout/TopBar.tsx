import { Menu, UploadCloud } from "lucide-react";
import type { CaseRecord, ThemeMode } from "../../types/lesioniq";
import { CaseSelector } from "./CaseSelector";
import { ThemeToggle } from "./ThemeToggle";

export function TopBar({
  cases,
  selectedCase,
  onSelectCase,
  theme,
  onToggleTheme,
  onOpenNav,
  hasUploadedImage,
  onImageSelected
}: {
  cases: CaseRecord[];
  selectedCase: CaseRecord | null;
  onSelectCase: (id: string) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  onOpenNav: () => void;
  hasUploadedImage: boolean;
  onImageSelected: (file: File) => void;
}) {
  return (
    <header className="topbar-surface sticky top-0 z-30 border-b border-clinical-line shadow-[0_10px_30px_rgba(0,0,0,0.22)] backdrop-blur">
      <div className="mx-auto flex min-h-16 max-w-[1500px] flex-wrap items-center justify-between gap-3 px-4 py-3 lg:px-8">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onOpenNav}
            className="rounded-md border border-clinical-line bg-clinical-raised p-2 text-clinical-ink outline-none hover:border-clinical-accent/45 focus-visible:ring-2 focus-visible:ring-clinical-accent/50 lg:hidden"
            aria-label="Open navigation"
          >
            <Menu className="h-5 w-5" />
          </button>
          <p className="text-sm font-bold tracking-[0.01em] text-clinical-ink">DermaLens</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <CaseSelector cases={cases} selectedId={selectedCase?.id ?? null} onSelect={onSelectCase} />
          <label className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-clinical-accent bg-clinical-accent px-3 py-2 text-sm font-semibold text-clinical-canvas outline-none transition hover:bg-clinical-accentHover focus-within:ring-2 focus-within:ring-clinical-accent/50">
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              className="sr-only"
              onChange={(event) => {
                const nextFile = event.target.files?.[0] ?? null;
                if (nextFile) onImageSelected(nextFile);
                event.currentTarget.value = "";
              }}
            />
            <UploadCloud className="h-4 w-4" aria-hidden="true" />
            {hasUploadedImage ? "Re-upload" : "Upload"}
          </label>
          <ThemeToggle theme={theme} onToggle={onToggleTheme} />
        </div>
      </div>
    </header>
  );
}
