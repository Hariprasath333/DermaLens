import { useEffect, useMemo, useState } from "react";
import { cases } from "../data/cases";
import { useTheme } from "../hooks/useTheme";
import type { ScreenId } from "./navigation";
import type { CaseRecord, UploadMetadataInput } from "../types/lesioniq";
import { runLesionIQAnalysis } from "../lib/lesioniqApi";
import { AppShell } from "../components/layout/AppShell";
import { CaseReview } from "../screens/CaseReview";
import { Compare } from "../screens/Compare";
import { Explainability } from "../screens/Explainability";
import { History } from "../screens/History";
import { Preprocessing } from "../screens/Preprocessing";
import { Settings } from "../screens/Settings";

const defaultUploadMetadata: UploadMetadataInput = {
  ageYears: null,
  sex: "Unknown",
  anatomicalSite: "unknown",
  modelMode: "Full Hybrid"
};

const envDemoVar =
  (import.meta as unknown as { env?: Record<string, string | undefined> }).env?.VITE_DERMALENS_ENABLE_DEMO_CASES ??
  (import.meta as unknown as { env?: Record<string, string | undefined> }).env?.VITE_LESIONIQ_ENABLE_DEMO_CASES;
const enableDemoCases = envDemoVar !== undefined ? envDemoVar === "true" : true;
const demoCases = enableDemoCases ? cases : [];

export function App() {
  const [screen, setScreen] = useState<ScreenId>("review");
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [uploadedPreviewUrl, setUploadedPreviewUrl] = useState<string | null>(null);
  const [analysisReady, setAnalysisReady] = useState(false);
  const [analysisPending, setAnalysisPending] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [uploadedCases, setUploadedCases] = useState<CaseRecord[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [uploadMetadata, setUploadMetadata] = useState<UploadMetadataInput>(defaultUploadMetadata);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const { theme, toggleTheme } = useTheme();
  const allCases = useMemo(() => [...uploadedCases, ...demoCases], [uploadedCases]);
  const selectedCase = useMemo(
    () => allCases.find((caseRecord) => caseRecord.id === selectedCaseId) ?? null,
    [allCases, selectedCaseId]
  );
  const hasUploadedImage = Boolean(uploadedImage || uploadedPreviewUrl);

  useEffect(() => {
    return () => {
      if (uploadedPreviewUrl) URL.revokeObjectURL(uploadedPreviewUrl);
    };
  }, [uploadedPreviewUrl]);

  function selectUploadedImage(file: File) {
    const previewUrl = URL.createObjectURL(file);
    setUploadedPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return previewUrl;
    });
    setUploadedImage(file);
    setSelectedCaseId(null);
    setAnalysisReady(false);
    setAnalysisPending(false);
    setScreen("review");
  }

  function handleImageSelected(file: File) {
    selectUploadedImage(file);
  }

  async function handleRunAnalysis(metadata: UploadMetadataInput) {
    if (!uploadedImage || !uploadedPreviewUrl) return;
    setUploadMetadata(metadata);
    setScreen("review");
    setAnalysisReady(false);
    setAnalysisPending(true);
    setAnalysisError(null);

    try {
      const caseRecord = await runLesionIQAnalysis({ image: uploadedImage, previewUrl: uploadedPreviewUrl, metadata });
      setUploadedCases((current) => [caseRecord, ...current.filter((item) => item.id !== caseRecord.id)]);
      setSelectedCaseId(caseRecord.id);
      setUploadedPreviewUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
      setAnalysisReady(true);
    } catch (err) {
      setAnalysisError(err instanceof Error ? err.message : "Analysis failed. Check that the backend is running.");
    } finally {
      setAnalysisPending(false);
    }
  }

  function handleSelectCase(id: string) {
    setSelectedCaseId(id === "__intake__" ? null : id);
    setAnalysisReady(id !== "__intake__");
    setAnalysisPending(false);
    if (id === "__intake__") setScreen("review");
  }

  function handleUseSampleCase() {
    if (!enableDemoCases) return;
    setSelectedCaseId(cases[0].id);
    setAnalysisReady(true);
    setAnalysisPending(false);
    setScreen("review");
  }

  function handleNavigate(nextScreen: ScreenId) {
    if (!selectedCaseId && !uploadedImage && !["review", "settings"].includes(nextScreen)) {
      setScreen("review");
      return;
    }

    setScreen(nextScreen);
  }

  return (
    <AppShell
      activeScreen={screen}
      onNavigate={handleNavigate}
      cases={allCases}
      selectedCase={selectedCase}
      onSelectCase={handleSelectCase}
      theme={theme}
      onToggleTheme={toggleTheme}
      mobileOpen={mobileNavOpen}
      onOpenNav={() => setMobileNavOpen(true)}
      onCloseNav={() => setMobileNavOpen(false)}
      sidebarCollapsed={sidebarCollapsed}
      onToggleSidebarCollapsed={() => setSidebarCollapsed((current) => !current)}
      hasUploadedImage={hasUploadedImage}
      onImageSelected={selectUploadedImage}
    >
      {screen === "review" && (
        <CaseReview
          caseRecord={selectedCase}
          uploadedImage={uploadedImage}
          uploadedPreviewUrl={uploadedPreviewUrl}
          analysisReady={analysisReady}
          analysisPending={analysisPending}
          analysisError={analysisError}
          hasUploadedImage={hasUploadedImage}
          onNavigateExplainability={() => setScreen("explainability")}
          onImageSelected={handleImageSelected}
          onRunAnalysis={handleRunAnalysis}
          onUseSampleCase={enableDemoCases ? handleUseSampleCase : undefined}
          uploadMetadata={uploadMetadata}
          onUploadMetadataChange={setUploadMetadata}
        />
      )}
      {screen === "explainability" && <Explainability caseRecord={selectedCase} uploadedPreviewUrl={uploadedPreviewUrl} analysisReady={analysisReady} />}
      {screen === "preprocessing" && <Preprocessing caseRecord={selectedCase} uploadedPreviewUrl={uploadedPreviewUrl} analysisReady={analysisReady} />}
      {screen === "history" && selectedCase && <History caseRecord={selectedCase} onNavigateCompare={() => setScreen("compare")} />}
      {screen === "compare" && selectedCase && <Compare caseRecord={selectedCase} />}
      {screen === "settings" && (
        <Settings
          theme={theme}
          onToggleTheme={toggleTheme}
          uploadedImage={uploadedImage}
          uploadedPreviewUrl={uploadedPreviewUrl}
          analysisReady={analysisReady}
          onImageSelected={selectUploadedImage}
        />
      )}
    </AppShell>
  );
}
