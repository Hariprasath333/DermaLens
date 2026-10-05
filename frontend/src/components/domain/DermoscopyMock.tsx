import { memo } from "react";
import type { OverlayMode, PreprocessingStep } from "../../types/lesioniq";
import { cx } from "../../lib/format";
import { useProxiedImage } from "../../hooks/useProxiedImage";

interface DermoscopyMockProps {
  overlay?: OverlayMode;
  stepTone?: PreprocessingStep["previewTone"];
  compact?: boolean;
  square?: boolean;
  label?: string;
  imageUrl?: string;
  objectFit?: "cover" | "contain";
  className?: string;
}

export const DermoscopyMock = memo(function DermoscopyMock({ overlay = "raw", stepTone = "raw", compact = false, square = false, label, imageUrl, objectFit = "cover", className }: DermoscopyMockProps) {
  const proxiedUrl = useProxiedImage(imageUrl);

  return (
    <div
      className={cx(
        "dermoscopy-frame relative overflow-hidden rounded-clinical border border-clinical-line bg-clinical-canvas",
        square ? "aspect-square" : compact ? "aspect-[4/3]" : "aspect-[5/4] min-h-[320px]",
        className
      )}
      aria-label={label ?? `Dermoscopy preview ${overlay}`}
      role="img"
    >
      {proxiedUrl ? (
        <img src={proxiedUrl} alt="" className={cx("h-full w-full", objectFit === "contain" ? "object-contain" : "object-cover")} />
      ) : imageUrl ? (
        <div className="flex h-full w-full items-center justify-center">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-clinical-accent border-t-transparent" />
        </div>
      ) : (
        <div className={cx("dermoscopy-lesion", `tone-${stepTone}`)} />
      )}
      <div className={cx("overlay-layer", `overlay-${overlay}`)} />
      {overlay === "metadata" && (
        <div className="metadata-chip absolute left-4 top-4 rounded-md border border-clinical-line px-3 py-2 text-xs font-semibold text-clinical-ink shadow-sm">
          Metadata branch: secondary influence
        </div>
      )}
    </div>
  );
});
