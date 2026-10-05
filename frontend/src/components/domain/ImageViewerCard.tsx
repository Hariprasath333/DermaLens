import { Maximize2, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import { useMemo, useState } from "react";
import type { OverlayMode } from "../../types/lesioniq";
import { Card } from "../primitives/Card";
import { DermoscopyMock } from "./DermoscopyMock";
import { OverlayToggle } from "./OverlayToggle";

export function ImageViewerCard({ imageUrl, artifactUrls }: { imageUrl?: string; artifactUrls?: Partial<Record<OverlayMode, string>> }) {
  const [overlay, setOverlay] = useState<OverlayMode>("raw");
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [fitMode, setFitMode] = useState<"cover" | "contain">("cover");
  const minZoom = fitMode === "contain" ? 1 : 0.9;
  const maxZoom = 1.8;
  const canZoomIn = zoom < maxZoom;
  const canZoomOut = zoom > minZoom;
  const toolbar = [
    { icon: ZoomIn, label: "Zoom in", disabled: !canZoomIn, active: zoom > 1, action: () => { setFitMode("cover"); setZoom((value) => Math.min(Number((value + 0.15).toFixed(2)), maxZoom)); } },
    { icon: ZoomOut, label: "Zoom out", disabled: !canZoomOut, active: zoom < 1, action: () => setZoom((value) => Math.max(Number((value - 0.15).toFixed(2)), minZoom)) },
    { icon: Maximize2, label: "Fit image to square", disabled: fitMode === "contain" && zoom === 1 && pan.x === 0 && pan.y === 0, active: fitMode === "contain", action: () => { setFitMode("contain"); setZoom(1); setPan({ x: 0, y: 0 }); } },
    { icon: RotateCcw, label: "Reset viewer", disabled: fitMode === "cover" && zoom === 1 && pan.x === 0 && pan.y === 0 && overlay === "raw", active: false, action: () => { setZoom(1); setPan({ x: 0, y: 0 }); setFitMode("cover"); setOverlay("raw"); } }
  ];
  const activeImageUrl = useMemo(() => artifactUrls?.[overlay] ?? imageUrl, [artifactUrls, imageUrl, overlay]);
  const renderedOverlay = activeImageUrl && overlay !== "metadata" ? "raw" : overlay;

  return (
    <Card
      title="Image evidence"
      eyebrow="Evidence viewer"
      className="min-w-0"
    >
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <OverlayToggle value={overlay} onChange={setOverlay} />
        <div className="flex gap-1" aria-label="Image toolbar">
          {toolbar.map(({ icon: Icon, label, action, disabled, active }) => (
            <button
              key={label}
              type="button"
              disabled={disabled}
              onClick={action}
              className={`rounded-md border p-2.5 outline-none transition focus-visible:ring-2 focus-visible:ring-clinical-accent/50 ${
                active
                  ? "border-clinical-accent/55 bg-clinical-accentSoft text-clinical-accent"
                  : "border-clinical-line bg-clinical-raised text-clinical-muted hover:border-clinical-accent/35 hover:text-clinical-ink"
              } disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-clinical-line disabled:hover:text-clinical-muted`}
              aria-label={label}
              title={label}
            >
              <Icon className="h-4 w-4" />
            </button>
          ))}
        </div>
      </div>
      <div className="mx-auto aspect-square w-full max-w-[420px] overflow-hidden rounded-clinical border border-clinical-line bg-clinical-raised shadow-clinical">
        <div
          className="h-full w-full"
          style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: "center", transition: "transform 120ms ease-out" }}
        >
          <DermoscopyMock overlay={renderedOverlay} imageUrl={activeImageUrl} square objectFit={fitMode} className="h-full w-full" />
        </div>
      </div>
      <div className="mt-3 flex items-center justify-between gap-3 text-xs text-clinical-muted">
        <span>{fitMode === "contain" ? "Fit mode: full image visible" : "Review mode: square crop"}</span>
        <span className="tabular-nums">{Math.round(zoom * 100)}%</span>
      </div>
    </Card>
  );
}

