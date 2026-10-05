import { ClipboardCheck, GitCompare, History, Microscope, Settings, Workflow } from "lucide-react";

export type ScreenId = "review" | "explainability" | "preprocessing" | "history" | "compare" | "settings";

export const navItems = [
  { id: "review", label: "Case Review", icon: ClipboardCheck },
  { id: "explainability", label: "Explainability", icon: Microscope },
  { id: "preprocessing", label: "Preprocessing", icon: Workflow },
  { id: "history", label: "History", icon: History },
  { id: "compare", label: "Compare", icon: GitCompare },
  { id: "settings", label: "Settings", icon: Settings }
] as const satisfies ReadonlyArray<{ id: ScreenId; label: string; icon: typeof ClipboardCheck }>;

