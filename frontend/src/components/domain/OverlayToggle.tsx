import type { OverlayMode } from "../../types/lesioniq";
import { SectionTabs } from "../primitives/SectionTabs";

const overlayTabs: Array<{ id: OverlayMode; label: string }> = [
  { id: "raw", label: "Raw" },
  { id: "gradcam", label: "Grad-CAM" },
  { id: "attention", label: "Attention" },
  { id: "metadata", label: "Metadata" }
];

export function OverlayToggle({ value, onChange }: { value: OverlayMode; onChange: (mode: OverlayMode) => void }) {
  return <SectionTabs tabs={overlayTabs} value={value} onChange={onChange} ariaLabel="Image overlay mode" />;
}

