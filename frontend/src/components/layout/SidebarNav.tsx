import { ChevronLeft, ChevronRight, X } from "lucide-react";
import type { ScreenId } from "../../app/navigation";
import { navItems } from "../../app/navigation";
import { cx } from "../../lib/format";

export function SidebarNav({
  active,
  onNavigate,
  mobileOpen,
  onClose,
  collapsed,
  onToggleCollapsed
}: {
  active: ScreenId;
  onNavigate: (screen: ScreenId) => void;
  mobileOpen: boolean;
  onClose: () => void;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}) {
  const renderNav = (isCollapsed: boolean, mobile = false) => (
    <aside
      className={cx(
        "sidebar-surface flex h-full flex-col border-r border-clinical-line transition-[width] duration-200 ease-out",
        isCollapsed ? "w-[68px]" : "w-[252px]"
      )}
    >
      <div className={cx("flex items-center border-b border-clinical-line px-4 py-5", isCollapsed ? "justify-center" : "justify-between")}>
        <div className={cx("min-w-0", isCollapsed && "text-center")}>
          <p className={cx("font-bold tracking-[0.01em] text-clinical-ink", isCollapsed ? "text-sm" : "text-base")}>{isCollapsed ? "DL" : "DermaLens"}</p>
          {!isCollapsed && <p className="mt-0.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-clinical-muted">Dermoscopy CDS</p>}
        </div>
        <div className="flex items-center gap-1">
          {!mobile && (
            <button
              type="button"
              onClick={onToggleCollapsed}
              className="hidden rounded-md border border-clinical-line bg-clinical-surface p-2 text-clinical-muted outline-none hover:border-clinical-accent/45 hover:text-clinical-ink focus-visible:ring-2 focus-visible:ring-clinical-accent/50 lg:inline-flex"
              aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
              title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
            </button>
          )}
          {mobile && (
            <button type="button" onClick={onClose} className="rounded-md p-2 text-clinical-muted hover:text-clinical-ink lg:hidden" aria-label="Close navigation">
              <X className="h-5 w-5" />
            </button>
          )}
        </div>
      </div>
      <nav className={cx("flex-1 space-y-1.5 py-5", isCollapsed ? "px-2" : "px-4")} aria-label="Primary navigation">
        {navItems.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            title={isCollapsed ? label : undefined}
            onClick={() => {
              onNavigate(id);
              onClose();
            }}
            className={cx(
              "flex w-full items-center rounded-clinical py-2 text-[13px] outline-none transition focus-visible:ring-2 focus-visible:ring-clinical-accent/50",
              isCollapsed ? "justify-center px-2" : "gap-3 px-3 text-left",
              active === id ? "border border-clinical-accent/25 bg-clinical-accentSoft font-semibold text-clinical-ink" : "border border-transparent font-medium text-clinical-muted hover:border-clinical-line hover:bg-clinical-surface hover:text-clinical-ink"
            )}
          >
            <Icon className="h-4 w-4 shrink-0" />
            {!isCollapsed && label}
          </button>
        ))}
      </nav>
      <div className={cx("border-t border-clinical-line text-[11px] leading-5 text-clinical-muted", isCollapsed ? "px-2 py-3 text-center" : "p-4")}>
        {isCollapsed ? "Full" : "Full hybrid mode: EfficientNet-B4 + SwinV2 + metadata MLP."}
      </div>
    </aside>
  );

  return (
    <>
      <div className="hidden lg:block">{renderNav(collapsed)}</div>
      {mobileOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <button className="drawer-scrim absolute inset-0" onClick={onClose} aria-label="Close overlay" type="button" />
          <div className="relative h-full">{renderNav(false, true)}</div>
        </div>
      )}
    </>
  );
}

