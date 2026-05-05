/**
 * Backend API client.
 *
 * Two base URLs are tracked so server components (running inside the Docker
 * network) talk to `backend:8000` while browser code uses `localhost:8000`.
 */

const INTERNAL_BASE =
  process.env.INTERNAL_API_BASE ?? "http://localhost:8000/api/v1";
const PUBLIC_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000/api/v1";

export const FEATURE_NAMES = ["hsv", "cm", "lbp", "glcm", "hog", "hu"] as const;
export type FeatureName = (typeof FEATURE_NAMES)[number];

export const VISUALIZATION_NAMES = [
  "preprocess",
  ...FEATURE_NAMES,
] as const;
export type VisualizationName = (typeof VISUALIZATION_NAMES)[number];

export interface PingResponse {
  pong: true;
  service: string;
  version: string;
}

export interface ImageRead {
  id: number;
  sha256: string;
  filename: string;
  storage_path: string;
  animal_type: string;
  width: number;
  height: number;
  size_bytes: number;
  role: "corpus" | "query";
  uploaded_at: string;
}

export interface FeatureSetRead {
  image_id: number;
  extractor_ver: string;
}

export interface ImageWithFeatures {
  image: ImageRead;
  features: FeatureSetRead;
}

export interface SearchResultItem {
  image_id: number;
  rank: number;
  score: number;
  per_feature: Record<string, number>;
  image: ImageRead | null;
}

export interface SearchTraceStage {
  name: string;
  elapsed_ms: number;
  detail: Record<string, unknown>;
}

export interface SearchResponse {
  run_id: number;
  weights: Record<string, number>;
  results: SearchResultItem[];
  pipeline_trace: SearchTraceStage[];
  elapsed_ms: number;
  corpus_size: number;
  query_dims: Record<string, number>;
}

export interface EvaluationMetricsRead {
  precision_at_k: number;
  map_at_k: number;
  n_queries: number;
}

export interface EvaluationReportRead {
  method: string;
  weights: Record<string, number>;
  top_k: number;
  overall: EvaluationMetricsRead;
  per_class: Record<string, EvaluationMetricsRead>;
}

export interface AblationReportRead {
  top_k: number;
  base: EvaluationReportRead;
  variants: Record<string, EvaluationReportRead>;
}

export interface EvaluationResponse {
  run_id: number;
  method: string;
  top_k: number;
  corpus_size: number;
  report: EvaluationReportRead | null;
  ablation: AblationReportRead | null;
  elapsed_ms: number;
}

export interface SearchOptions {
  topK?: number;
  weights?: Record<string, number>;
  streamId?: string | null;
}

export interface RerankOptions {
  topK?: number;
}

export interface CompareResponse {
  fused_score: number;
  per_feature: Record<string, number>;
  left_dims: Record<string, number>;
  right_dims: Record<string, number>;
  elapsed_ms: number;
}

export interface EvaluateOptions {
  method?: "default" | "ablation";
  topK?: number;
  weights?: Record<string, number>;
}

export interface ApiError {
  status: number;
  message: string;
}

function activeBase(): string {
  return typeof window === "undefined" ? INTERNAL_BASE : PUBLIC_BASE;
}

export function apiUrl(path: string): string {
  const base = activeBase().replace(/\/$/, "");
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${base}${suffix}`;
}

/** Build a public URL for a visualisation PNG (always uses the browser base). */
export function visualizationUrl(
  imageId: number,
  feature: VisualizationName,
): string {
  const base = (
    process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000/api/v1"
  ).replace(/\/$/, "");
  return `${base}/visualize/${imageId}/${feature}`;
}

/** Build a public URL for the original uploaded image bytes. */
export function originalImageUrl(imageId: number): string {
  const base = (
    process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000/api/v1"
  ).replace(/\/$/, "");
  return `${base}/images/${imageId}/raw`;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    if (typeof body.detail === "string") {
      return body.detail;
    }
  } catch {
    // fallthrough — body wasn't JSON
  }
  return response.statusText || `HTTP ${response.status}`;
}

async function ensureOk(response: Response): Promise<void> {
  if (response.ok) {
    return;
  }
  const message = await readError(response);
  const err = new Error(message) as Error & { status: number };
  err.status = response.status;
  throw err;
}

export async function fetchPing(): Promise<PingResponse> {
  const response = await fetch(apiUrl("/ping"), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`ping returned HTTP ${response.status}`);
  }
  return (await response.json()) as PingResponse;
}

export async function uploadImage(
  file: File,
  animalType: string,
  role: "corpus" | "query" = "corpus",
): Promise<ImageWithFeatures> {
  const form = new FormData();
  form.append("file", file);
  form.append("animal_type", animalType);
  form.append("role", role);
  const response = await fetch(apiUrl("/images"), {
    method: "POST",
    body: form,
  });
  await ensureOk(response);
  return (await response.json()) as ImageWithFeatures;
}

export async function allocateStreamId(): Promise<string> {
  const response = await fetch(apiUrl("/search/streams"), { method: "POST" });
  await ensureOk(response);
  const body = (await response.json()) as { stream_id: string };
  return body.stream_id;
}

export async function searchImages(
  file: File,
  options: SearchOptions = {},
): Promise<SearchResponse> {
  const form = new FormData();
  form.append("file", file);
  if (options.topK !== undefined) {
    form.append("top_k", String(options.topK));
  }
  if (options.weights !== undefined) {
    form.append("weights", JSON.stringify(options.weights));
  }
  if (options.streamId) {
    form.append("stream_id", options.streamId);
  }
  const response = await fetch(apiUrl("/search"), {
    method: "POST",
    body: form,
  });
  await ensureOk(response);
  return (await response.json()) as SearchResponse;
}

export async function rerankSearch(
  runId: number,
  weights: Record<string, number>,
  options: RerankOptions = {},
): Promise<SearchResponse> {
  const body = {
    weights,
    top_k: options.topK ?? 5,
  };
  const response = await fetch(apiUrl(`/search/${runId}/weights`), {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  await ensureOk(response);
  return (await response.json()) as SearchResponse;
}

export async function evaluateCorpus(
  options: EvaluateOptions = {},
): Promise<EvaluationResponse> {
  const body: Record<string, unknown> = {
    method: options.method ?? "default",
    top_k: options.topK ?? 10,
  };
  if (options.weights) {
    body.weights = options.weights;
  }
  const response = await fetch(apiUrl("/evaluate"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  await ensureOk(response);
  return (await response.json()) as EvaluationResponse;
}

export async function compareTwoImages(
  left: File,
  right: File,
): Promise<CompareResponse> {
  const form = new FormData();
  form.append("left", left);
  form.append("right", right);
  const response = await fetch(apiUrl("/compare"), {
    method: "POST",
    body: form,
  });
  await ensureOk(response);
  return (await response.json()) as CompareResponse;
}

export async function invalidateCache(): Promise<void> {
  const response = await fetch(apiUrl("/admin/cache/invalidate"), {
    method: "POST",
  });
  await ensureOk(response);
}

export async function visualizeQueryFeature(
  file: File,
  feature: string = "preprocess",
): Promise<Blob> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(
    apiUrl(`/visualize/query?feature=${encodeURIComponent(feature)}`),
    {
      method: "POST",
      body: form,
    },
  );
  await ensureOk(response);
  return response.blob();
}
