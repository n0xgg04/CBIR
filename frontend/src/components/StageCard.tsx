"use client";

import { useState } from "react";

import type { SearchTraceStage } from "@/lib/api";

export interface StageCardProps {
  index: number;
  title: string;
  stage: SearchTraceStage | null;
  hint?: string;
  featureName?: string;
  queryFile?: File | null;
}

function formatDetail(value: unknown, depth = 0): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  }
  if (typeof value === "string" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    if (depth > 1 || value.length > 6) {
      return `array(${value.length})`;
    }
    return `[${value.map((v) => formatDetail(v, depth + 1)).join(", ")}]`;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (depth > 1 || entries.length > 6) {
      return `{…${entries.length}}`;
    }
    return `{${entries
      .map(([k, v]) => `${k}: ${formatDetail(v, depth + 1)}`)
      .join(", ")}}`;
  }
  return String(value);
}

export function StageCard({
  index,
  title,
  stage,
  hint,
  featureName,
  queryFile,
}: StageCardProps) {
  const filled = stage !== null;
  const [expanded, setExpanded] = useState(false);
  const [plotUrl, setPlotUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const hasVisual = featureName !== undefined && queryFile !== null;

  const toggle = async () => {
    const next = !expanded;
    setExpanded(next);
    if (next && hasVisual && !plotUrl && featureName) {
      setLoading(true);
      try {
        const { visualizeQueryFeature } = await import("@/lib/api");
        const blob = await visualizeQueryFeature(queryFile!, featureName);
        setPlotUrl(URL.createObjectURL(blob));
      } catch {
        // silently ignore — user can retry by toggling again
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div
      data-testid="stage-card"
      data-stage-name={stage?.name ?? `pending-${index}`}
      className={`rounded-lg border transition ${
        filled
          ? "border-emerald-200 bg-emerald-50/40"
          : "border-slate-200 bg-slate-50"
      }`}
    >
      <button
        type="button"
        onClick={toggle}
        className="flex w-full items-start gap-4 rounded-lg p-4 text-left hover:bg-black/5 transition"
      >
        <div
          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-semibold ${
            filled ? "bg-emerald-500 text-white" : "bg-slate-300 text-slate-700"
          }`}
          aria-hidden
        >
          {index}
        </div>
        <div className="flex-1">
          <div className="flex items-baseline justify-between gap-2">
            <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
            <span className="text-xs font-mono text-slate-500">
              {filled ? `${stage.elapsed_ms} ms` : "đang chờ"}
            </span>
          </div>
          {hint !== undefined ? (
            <p className="mt-1 text-xs text-slate-500">{hint}</p>
          ) : null}
          {filled && Object.keys(stage.detail).length > 0 ? (
            <dl className="mt-2 grid grid-cols-1 gap-1 text-xs sm:grid-cols-2">
              {Object.entries(stage.detail).map(([k, v]) => (
                <div key={k} className="flex justify-between gap-2">
                  <dt className="font-mono text-slate-500">{k}</dt>
                  <dd className="truncate font-mono text-slate-700">
                    {formatDetail(v)}
                  </dd>
                </div>
              ))}
            </dl>
          ) : null}
        </div>
        <span className="ml-2 text-xs text-slate-400">
          {expanded ? "▲" : "▼"}
        </span>
      </button>

      {expanded ? (
        <div className="border-t border-slate-200/60 px-4 pb-4">
          {loading ? (
            <p className="py-3 text-xs text-slate-500">Đang tải hình ảnh…</p>
          ) : plotUrl ? (
            <div className="mt-3 flex flex-col gap-2">
              <span className="text-xs font-medium text-slate-600">
                Trực quan {title.toLowerCase()}
              </span>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={plotUrl}
                alt={`${title} plot`}
                className="w-full rounded-md border border-slate-200 bg-white object-contain"
                loading="lazy"
              />
            </div>
          ) : hasVisual ? (
            <p className="py-3 text-xs text-rose-600">Không thể tải hình ảnh.</p>
          ) : (
            <div className="py-3 text-xs text-slate-500">
              {stage?.name === "decode" && (
                <span>Đọc bytes ảnh upload, chuyển sang định dạng BGR uint8 bằng OpenCV.</span>
              )}
              {stage?.name === "preprocess" && (
                <span>Resize 128×128 → Gaussian blur 3×3 → CLAHE trên kênh L.</span>
              )}
              {stage?.name === "extract" && (
                <span>Trích xuất đồng thời 6 đặc trưng thủ công từ ảnh đã tiền xử lý.</span>
              )}
              {stage?.name === "load_corpus" && (
                <span>Nạp ma trận đặc trưng toàn bộ corpus từ cache/DB vào bộ nhớ.</span>
              )}
              {stage?.name === "cosine" && (
                <span>Tính cosine similarity = dot product (đã L2-normalized) giữa query và từng ảnh corpus.</span>
              )}
              {stage?.name === "rank" && (
                <span>Weighted fusion + argsort để chọn top-K ảnh giống nhất.</span>
              )}
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
