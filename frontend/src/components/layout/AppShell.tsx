import type { PropsWithChildren } from "react";
import type { ScreenId } from "../../app/navigation";
import type { CaseRecord, ThemeMode } from "../../types/lesioniq";
import { SidebarNav } from "./SidebarNav";
import { TopBar } from "./TopBar";

interface AppShellProps extends PropsWithChildren {
  activeScreen: ScreenId;
  onNavigate: (screen: ScreenId) => void;
  cases: CaseRecord[];
  selectedCase: CaseRecord | null;
  onSelectCase: (id: string) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  mobileOpen: boolean;
  onOpenNav: () => void;
  onCloseNav: () => void;
  sidebarCollapsed: boolean;
  onToggleSidebarCollapsed: () => void;
  hasUploadedImage: boolean;
  onImageSelected: (file: File) => void;
}

export function AppShell(props: AppShellProps) {
  return (
    <div className="app-gradient min-h-screen text-clinical-ink">
      <div className="flex min-h-screen">
        <SidebarNav
          active={props.activeScreen}
          onNavigate={props.onNavigate}
          mobileOpen={props.mobileOpen}
          onClose={props.onCloseNav}
          collapsed={props.sidebarCollapsed}
          onToggleCollapsed={props.onToggleSidebarCollapsed}
        />
        <div className="min-w-0 flex-1">
          <TopBar
            cases={props.cases}
            selectedCase={props.selectedCase}
            onSelectCase={props.onSelectCase}
            theme={props.theme}
            onToggleTheme={props.onToggleTheme}
            onOpenNav={props.onOpenNav}
            hasUploadedImage={props.hasUploadedImage}
            onImageSelected={props.onImageSelected}
          />
          <main className="mx-auto w-full max-w-[1500px] px-4 py-6 lg:px-8">{props.children}</main>
        </div>
      </div>
    </div>
  );
}

