"use client";

import { useState } from "react";
import Link from "next/link";

import {
  type SearchResultItem,
  originalImageUrl,
  visualizationUrl,
} from "@/lib/api";
import { DEFAULT_WEIGHTS, FEATURE_TITLE } from "@/lib/weights";

export interface ResultGridProps {
  results: SearchResultItem[];
  queryPreview?: string | null;
  /** Triggered when user clicks "Compare" — passes the result image id. */
  onCompare?: (imageId: number) => void;
}

function formatScore(value: number): string {
  return value.toFixed(4);
}

const FEATURE_ORDER = ["hsv", "cm", "lbp", "glcm", "hog", "hu"] as const;

/* ------------------------------------------------------------------ */
/*  Radar chart (SVG)                                                 */
/* ------------------------------------------------------------------ */

function RadarChart({ data }: { data: Record<string, number> }) {
  const size = 240;
  const cx = size / 2;
  const cy = size / 2;
  const radius = 90;
  const n = FEATURE_ORDER.length;

  const points = FEATURE_ORDER.map((name, i) => {
    const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
    const value = Math.max(0, Math.min(1, data[name] ?? 0));
    const r = radius * value;
    return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
  });

  const axisPoints = FEATURE_ORDER.map((_, i) => {
    const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
    return {
      x1: cx,
      y1: cy,
      x2: cx + radius * Math.cos(angle),
      y2: cy + radius * Math.sin(angle),
    };
  });

  const labelPoints = FEATURE_ORDER.map((name, i) => {
    const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
    const r = radius + 18;
    return {
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
      label: FEATURE_TITLE[name] ?? name,
    };
  });

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      className="mx-auto h-64 w-64"
      role="img"
      aria-label="Biểu đồ radar độ tương đồng từng đặc trưng"
    >
      {/* Background grid */}
      {[0.2, 0.4, 0.6, 0.8, 1.0].map((level) => {
        const gridPoints = FEATURE_ORDER.map((_, i) => {
          const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
          const r = radius * level;
          return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
        });
        return (
          <polygon
            key={level}
            points={gridPoints.join(" ")}
            fill="none"
            stroke="#e2e8f0"
            strokeWidth={0.5}
          />
        );
      })}
      {/* Axes */}
      {axisPoints.map((a, i) => (
        <line
          key={i}
          x1={a.x1}
          y1={a.y1}
          x2={a.x2}
          y2={a.y2}
          stroke="#e2e8f0"
          strokeWidth={0.5}
        />
      ))}
      {/* Data polygon */}
      <polygon
        points={points.join(" ")}
        fill="rgba(14, 165, 233, 0.2)"
        stroke="#0ea5e9"
        strokeWidth={2}
      />
      {/* Data points */}
      {FEATURE_ORDER.map((name, i) => {
        const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
        const value = Math.max(0, Math.min(1, data[name] ?? 0));
        const r = radius * value;
        const x = cx + r * Math.cos(angle);
        const y = cy + r * Math.sin(angle);
        return <circle key={name} cx={x} cy={y} r={3} fill="#0ea5e9" />;
      })}
      {/* Labels */}
      {labelPoints.map((lp, i) => (
        <text
          key={i}
          x={lp.x}
          y={lp.y}
          textAnchor="middle"
          dominantBaseline="middle"
          className="text-[10px] fill-slate-500"
        >
          {lp.label}
        </text>
      ))}
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Fusion detail table                                               */
/* ------------------------------------------------------------------ */

function FusionTable({
  perFeature,
  fused,
}: {
  perFeature: Record<string, number>;
  fused: number;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-slate-200 text-left">
            <th className="py-1 pr-3 text-slate-500">Đặc trưng</th>
            <th className="py-1 pr-3 text-right text-slate-500">Trọng số</th>
            <th className="py-1 pr-3 text-right text-slate-500">Cosine</th>
            <th className="py-1 text-right text-slate-500">Đóng góp</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          {FEATURE_ORDER.map((name) => {
            const weight = DEFAULT_WEIGHTS[name];
            const cosine = perFeature[name] ?? 0;
            const contrib = weight * cosine;
            return (
              <tr key={name} className="border-b border-slate-100">
                <td className="py-1 pr-3 text-slate-700">{FEATURE_TITLE[name]}</td>
                <td className="py-1 pr-3 text-right text-slate-600">{weight.toFixed(2)}</td>
                <td className="py-1 pr-3 text-right text-slate-600">{cosine.toFixed(4)}</td>
                <td className="py-1 text-right text-sky-700">{contrib.toFixed(4)}</td>
              </tr>
            );
          })}
          <tr className="font-semibold">
            <td colSpan={3} className="py-1 pr-3 text-right text-slate-700">
              Tổng hợp (fused score)
            </td>
            <td className="py-1 text-right text-sky-700">{fused.toFixed(4)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Result grid                                                       */
/* ------------------------------------------------------------------ */

export function ResultGrid({ results, queryPreview, onCompare }: ResultGridProps) {
  const [selected, setSelected] = useState<SearchResultItem | null>(null);

  if (results.length === 0) {
    return (
      <p className="text-sm text-slate-500" data-testid="results-empty">
        Chưa có kết quả — hãy thả ảnh và chạy tìm kiếm.
      </p>
    );
  }

  return (
    <>
      <ul
        data-testid="result-grid"
        className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3"
      >
        {results.map((item) => {
          const animal = item.image?.animal_type ?? "unknown";
          const filename = item.image?.filename ?? `ảnh #${item.image_id}`;
          const thumbnail = originalImageUrl(item.image_id);
          const compareHref = queryPreview
            ? `/compare?b=${item.image_id}`
            : `/compare?b=${item.image_id}`;
          return (
            <li
              key={item.image_id}
              data-testid="result-item"
              className="flex flex-col gap-2 rounded-xl border border-slate-200 bg-white p-3 shadow-sm"
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-xs font-mono text-slate-500">
                  #{item.rank}
                </span>
                <span className="text-xs font-mono text-emerald-700">
                  {formatScore(item.score)}
                </span>
              </div>
              <button
                type="button"
                onClick={() => setSelected(item)}
                className="aspect-square w-full overflow-hidden rounded-md bg-slate-100 text-left"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={thumbnail}
                  alt={`${animal} ${filename}`}
                  className="h-full w-full object-cover transition hover:scale-105"
                  loading="lazy"
                />
              </button>
              <div className="flex flex-col text-sm">
                <span className="truncate font-medium text-slate-800">
                  {filename}
                </span>
                <span className="text-xs text-slate-500">{animal}</span>
              </div>
              <details className="text-xs text-slate-600">
                <summary className="cursor-pointer text-slate-500">
                  Điểm từng đặc trưng
                </summary>
                <dl className="mt-2 grid grid-cols-2 gap-1">
                  {Object.entries(item.per_feature).map(([k, v]) => (
                    <div key={k} className="flex justify-between gap-2 font-mono">
                      <dt className="text-slate-500">{k}</dt>
                      <dd className="text-slate-700">{formatScore(v)}</dd>
                    </div>
                  ))}
                </dl>
              </details>
              <div className="flex items-center justify-end gap-2">
                {onCompare ? (
                  <button
                    type="button"
                    onClick={() => onCompare(item.image_id)}
                    className="rounded-md border border-slate-200 px-2 py-1 text-xs text-slate-700 hover:bg-slate-50"
                  >
                    So sánh
                  </button>
                ) : (
                  <Link
                    href={compareHref}
                    className="rounded-md border border-slate-200 px-2 py-1 text-xs text-slate-700 hover:bg-slate-50"
                  >
                    So sánh
                  </Link>
                )}
              </div>
            </li>
          );
        })}
      </ul>

      {selected ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          onClick={(e) => {
            if (e.target === e.currentTarget) setSelected(null);
          }}
        >
          <div className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-xl bg-white p-5 shadow-lg">
            <div className="flex items-baseline justify-between">
              <h3 className="text-lg font-semibold text-slate-800">
                #{selected.rank} — {selected.image?.filename ?? `ảnh #${selected.image_id}`}
              </h3>
              <button
                type="button"
                onClick={() => setSelected(null)}
                className="rounded-md px-2 py-1 text-sm text-slate-500 hover:bg-slate-100"
              >
                Đóng
              </button>
            </div>
            <p className="text-xs text-slate-500">
              Điểm tổng hợp:{" "}
              <span className="font-mono text-emerald-700">{formatScore(selected.score)}</span>
            </p>

            <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="aspect-square w-full overflow-hidden rounded-md bg-slate-100">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={visualizationUrl(selected.image_id, "preprocess")}
                  alt="Ảnh kết quả"
                  className="h-full w-full object-contain"
                />
              </div>
              <div className="flex flex-col gap-4">
                <div>
                  <h4 className="text-sm font-semibold text-slate-700">
                    Biểu đồ radar
                  </h4>
                  <p className="text-xs text-slate-500">
                    So sánh độ tương đồng 6 đặc trưng với ảnh truy vấn
                  </p>
                  <RadarChart data={selected.per_feature} />
                </div>
              </div>
            </div>

            <div className="mt-4">
              <h4 className="text-sm font-semibold text-slate-700">
                Chi tiết fusion
              </h4>
              <p className="mb-2 text-xs text-slate-500">
                Công thức: fused = Σ(weightᵢ × cosineᵢ)
              </p>
              <FusionTable
                perFeature={selected.per_feature}
                fused={selected.score}
              />
            </div>

            <div className="mt-4 flex justify-end">
              {onCompare ? (
                <button
                  type="button"
                  onClick={() => {
                    onCompare(selected.image_id);
                    setSelected(null);
                  }}
                  className="rounded-md border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
                >
                  So sánh
                </button>
              ) : (
                <Link
                  href={
                    queryPreview
                      ? `/compare?b=${selected.image_id}`
                      : `/compare?b=${selected.image_id}`
                  }
                  className="rounded-md border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
                  onClick={() => setSelected(null)}
                >
                  So sánh
                </Link>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
