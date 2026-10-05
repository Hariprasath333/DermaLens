import type { CaseRecord, UploadMetadataInput } from "../types/lesioniq";

interface AnalyzeCaseInput {
  image: File;
  previewUrl: string;
  metadata: UploadMetadataInput;
}

const env = (import.meta as unknown as { env?: Record<string, string | undefined> }).env || {};
const RAW_API_BASE = (
  env.VITE_DERMALENS_API_BASE_URL ||
  env.VITE_LESIONIQ_API_BASE_URL ||
  env.VITE_API_BASE_URL ||
  "/api"
).trim();

const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

export function resolveLesionIQArtifactUrl(url?: string, outputDirectory?: string): string | undefined {
  if (!url) return undefined;
  if (/^(https?:|blob:|data:)/i.test(url)) return url;

  const originBase = API_BASE.startsWith("http") ? API_BASE : "";

  if (url.startsWith("/artifacts") || url.startsWith("/")) {
    return originBase ? `${originBase}${url}` : url;
  }

  if (outputDirectory && /^(https?:)/i.test(outputDirectory)) {
    return new URL(url, outputDirectory.endsWith("/") ? outputDirectory : `${outputDirectory}/`).toString();
  }

  return originBase ? `${originBase}/${url.replace(/^\//, "")}` : `/${url.replace(/^\//, "")}`;
}

export async function runLesionIQAnalysis({ image, metadata }: AnalyzeCaseInput): Promise<CaseRecord> {
  const payload = new FormData();
  payload.append("image", image);
  payload.append("metadata", JSON.stringify(metadata));
  payload.append("slm_container", "lesioniq_ollama");
  payload.append("slm_model", "gemma3:4b-it-qat");

  const response = await fetch(`${API_BASE}/cases/analyze`, {
    method: "POST",
    body: payload
  });

  if (!response.ok) {
    throw new Error(`LesionIQ analysis failed with status ${response.status}`);
  }

  return response.json() as Promise<CaseRecord>;
}
