"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useMemo, useState } from "react";

import { Dropzone } from "@/components/Dropzone";
import {
  type CompareResponse,
  type VisualizationName,
  VISUALIZATION_NAMES,
  compareTwoImages,
  visualizationUrl,
} from "@/lib/api";

const TITLES: Record<VisualizationName, string> = {
  preprocess: "Tiền xử lý",
  hsv: "HSV histogram",
  cm: "Color moments",
  lbp: "LBP",
  glcm: "GLCM",
  hog: "HOG",
  hu: "Hu moments",
};

const FEATURE_ORDER = ["hsv", "cm", "lbp", "glcm", "hog", "hu"] as const;

function parseId(raw: string | null): number | null {
  if (raw === null) {
    return null;
  }
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null;
  }
  return parsed;
}

/* ------------------------------------------------------------------ */
/*  Upload comparison panel                                           */
/* ------------------------------------------------------------------ */

interface UploadPanelProps {
  file: File | null;
  previewUrl: string | null;
  label: string;
  onFile: (file: File | null) => void;
  disabled?: boolean;
}

function UploadPanel({ file, previewUrl, label, onFile, disabled }: UploadPanelProps) {
  return (
    <div className="flex flex-col gap-3">
      {previewUrl ? (
        <div className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="flex items-baseline justify-between">
            <h2 className="text-base font-semibold text-slate-800">
              Ảnh {label.toUpperCase()}
            </h2>
            <button
              type="button"
              onClick={() => onFile(null)}
              className="text-xs text-rose-600 hover:text-rose-700"
            >
              Xóa
            </button>
          </div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={previewUrl}
            alt={`Preview ${label}`}
            className="aspect-square w-full rounded-md object-cover"
          />
          <p className="text-xs text-slate-500">{file?.name}</p>
        </div>
      ) : (
        <Dropzone
          accept="image/*"
          disabled={disabled}
          label={`Tải ảnh ${label.toUpperCase()}`}
          hint="JPG, PNG, WebP — tối đa 8 MB"
          onFiles={(files) => {
            if (files[0]) onFile(files[0]);
          }}
        />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Comparison result bars                                            */
/* ------------------------------------------------------------------ */

function FeatureBars({ perFeature }: { perFeature: Record<string, number> }) {
  return (
    <div className="flex flex-col gap-3">
      {FEATURE_ORDER.map((name) => {
        const value = perFeature[name] ?? 0;
        const pct = Math.max(0, Math.min(100, value * 100));
        return (
          <div key={name} className="flex flex-col gap-1">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-slate-700">{TITLES[name]}</span>
              <span className="font-mono text-xs text-slate-500">
                {value.toFixed(4)}
              </span>
            </div>
            <div className="h-2 w-full rounded-full bg-slate-100">
              <div
                className="h-2 rounded-full bg-sky-500 transition-all"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  ID-based compare panel (legacy)                                   */
/* ------------------------------------------------------------------ */

function IdComparePanel({ id, label }: { id: number | null; label: string }) {
  if (id === null) {
    return (
      <div
        data-testid={`compare-empty-${label}`}
        className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 p-6 text-sm text-slate-500"
      >
        <p>Chọn ảnh {label.toUpperCase()} qua chuỗi truy vấn URL.</p>
        <p className="mt-1 font-mono text-xs">e.g. ?{label}=42</p>
      </div>
    );
  }
  return (
    <article
      data-testid={`compare-panel-${label}`}
      className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
    >
      <header className="flex items-baseline justify-between">
        <h2 className="text-base font-semibold text-slate-800">
          Ảnh {label.toUpperCase()} · #{id}
        </h2>
      </header>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={visualizationUrl(id, "preprocess")}
        alt={`Preprocess of image ${id}`}
        className="aspect-square w-full rounded-md object-cover"
        loading="lazy"
      />
      <div className="grid grid-cols-3 gap-2">
        {VISUALIZATION_NAMES.filter((f) => f !== "preprocess").map(
          (feature) => (
            <figure
              key={feature}
              className="flex flex-col gap-1 rounded-md border border-slate-200 p-1"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={visualizationUrl(id, feature)}
                alt={`${TITLES[feature]} of image ${id}`}
                className="aspect-square w-full rounded-sm object-contain"
                loading="lazy"
              />
              <figcaption className="text-center text-xs text-slate-600">
                {TITLES[feature]}
              </figcaption>
            </figure>
          )
        )}
      </div>
    </article>
  );
}

/* ------------------------------------------------------------------ */
/*  Main content                                                      */
/* ------------------------------------------------------------------ */

type CompareMode = "upload" | "id";

function CompareContent() {
  const params = useSearchParams();
  const a = useMemo(() => parseId(params.get("a")), [params]);
  const b = useMemo(() => parseId(params.get("b")), [params]);

  const [mode, setMode] = useState<CompareMode>(
    a !== null || b !== null ? "id" : "upload"
  );

  const [leftFile, setLeftFile] = useState<File | null>(null);
  const [rightFile, setRightFile] = useState<File | null>(null);
  const [leftPreview, setLeftPreview] = useState<string | null>(null);
  const [rightPreview, setRightPreview] = useState<string | null>(null);

  const [result, setResult] = useState<CompareResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLeft = useCallback((file: File | null) => {
    setLeftFile(file);
    setLeftPreview(file ? URL.createObjectURL(file) : null);
    setResult(null);
    setError(null);
  }, []);

  const handleRight = useCallback((file: File | null) => {
    setRightFile(file);
    setRightPreview(file ? URL.createObjectURL(file) : null);
    setResult(null);
    setError(null);
  }, []);

  const handleCompare = useCallback(async () => {
    if (!leftFile || !rightFile) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const data = await compareTwoImages(leftFile, rightFile);
      setResult(data);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setBusy(false);
    }
  }, [leftFile, rightFile]);

  return (
    <main className="mx-auto flex max-w-6xl flex-col gap-6 p-6 lg:p-10">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">So sánh ảnh</h1>
          <p className="mt-1 text-sm text-slate-600">
            So sánh độ tương đồng của hai ảnh dựa trên 6 đặc trưng thị giác.
          </p>
        </div>
        <a href="/" className="text-sm text-slate-500 hover:text-slate-700">
          ← Trang chủ
        </a>
      </header>

      {/* Mode toggle */}
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setMode("upload")}
          className={`rounded-md px-4 py-2 text-sm font-medium transition ${
            mode === "upload"
              ? "bg-slate-800 text-white"
              : "bg-slate-100 text-slate-600 hover:bg-slate-200"
          }`}
        >
          Tải ảnh lên
        </button>
        <button
          type="button"
          onClick={() => setMode("id")}
          className={`rounded-md px-4 py-2 text-sm font-medium transition ${
            mode === "id"
              ? "bg-slate-800 text-white"
              : "bg-slate-100 text-slate-600 hover:bg-slate-200"
          }`}
        >
          Ảnh trong kho
        </button>
      </div>

      {mode === "upload" ? (
        <>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <UploadPanel
              file={leftFile}
              previewUrl={leftPreview}
              label="a"
              onFile={handleLeft}
              disabled={busy}
            />
            <UploadPanel
              file={rightFile}
              previewUrl={rightPreview}
              label="b"
              onFile={handleRight}
              disabled={busy}
            />
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleCompare}
              disabled={!leftFile || !rightFile || busy}
              className="rounded-md bg-sky-600 px-5 py-2 text-sm font-semibold text-white hover:bg-sky-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {busy ? "Đang tính toán…" : "So sánh"}
            </button>
            {leftFile && rightFile ? (
              <span className="text-xs text-slate-500">
                {leftFile.name} ↔ {rightFile.name}
              </span>
            ) : null}
          </div>

          {error ? (
            <div className="rounded-md bg-rose-50 p-3 text-sm text-rose-700">
              {error}
            </div>
          ) : null}

          {result ? (
            <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-4 flex items-baseline justify-between">
                <h2 className="text-lg font-semibold text-slate-800">
                  Kết quả so sánh
                </h2>
                <span className="text-xs text-slate-400">
                  {result.elapsed_ms} ms
                </span>
              </div>

              <div className="mb-6 flex items-center gap-4">
                <div className="flex flex-col">
                  <span className="text-xs uppercase tracking-wide text-slate-400">
                    Điểm tổng hợp
                  </span>
                  <span className="text-3xl font-bold text-sky-600">
                    {result.fused_score.toFixed(4)}
                  </span>
                </div>
                <div className="h-10 w-px bg-slate-200" />
                <div className="flex flex-col">
                  <span className="text-xs uppercase tracking-wide text-slate-400">
                    Đánh giá
                  </span>
                  <span className="text-sm font-medium text-slate-700">
                    {result.fused_score >= 0.8
                      ? "Rất giống nhau"
                      : result.fused_score >= 0.5
                        ? "Tương đối giống"
                        : result.fused_score >= 0.3
                          ? "Ít tương đồng"
                          : "Khác biệt rõ rệt"}
                  </span>
                </div>
              </div>

              <FeatureBars perFeature={result.per_feature} />
            </section>
          ) : null}
        </>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <IdComparePanel id={a} label="a" />
            <IdComparePanel id={b} label="b" />
          </div>
          {a !== null && b !== null ? (
            <p
              data-testid="compare-info"
              className="rounded-md bg-slate-100 p-3 text-xs text-slate-600"
            >
              Điểm tương đồng từng đặc trưng nằm trong kết quả tìm kiếm — mở một
              lượt tìm kiếm có chứa ảnh này để xem chi tiết.
            </p>
          ) : null}
        </>
      )}
    </main>
  );
}

export default function ComparePage() {
  return (
    <Suspense
      fallback={
        <main className="p-6 text-sm text-slate-500">
          Đang tải giao diện so sánh…
        </main>
      }
    >
      <CompareContent />
    </Suspense>
  );
}
