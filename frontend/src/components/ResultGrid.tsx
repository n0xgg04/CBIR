"use client";

import { useState } from "react";
import Link from "next/link";

import { type SearchResultItem, visualizationUrl } from "@/lib/api";

export interface ResultGridProps {
  results: SearchResultItem[];
  queryPreview?: string | null;
  /** Triggered when user clicks "Compare" — passes the result image id. */
  onCompare?: (imageId: number) => void;
}

function formatScore(value: number): string {
  return value.toFixed(4);
}

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
          const thumbnail = visualizationUrl(item.image_id, "preprocess");
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
          <div className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-xl bg-white p-5 shadow-lg">
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
              Điểm tổng hợp: <span className="font-mono text-emerald-700">{formatScore(selected.score)}</span>
            </p>
            <div className="mt-4 aspect-square w-full overflow-hidden rounded-md bg-slate-100">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={visualizationUrl(selected.image_id, "preprocess")}
                alt="Ảnh kết quả"
                className="h-full w-full object-contain"
              />
            </div>
            <div className="mt-4">
              <h4 className="text-sm font-semibold text-slate-700">Điểm từng đặc trưng</h4>
              <dl className="mt-2 grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
                {Object.entries(selected.per_feature).map(([k, v]) => (
                  <div
                    key={k}
                    className="flex justify-between gap-2 rounded-md border border-slate-200 p-2 font-mono"
                  >
                    <dt className="text-slate-500">{k}</dt>
                    <dd className="text-slate-700">{formatScore(v)}</dd>
                  </div>
                ))}
              </dl>
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
                  href={queryPreview ? `/compare?b=${selected.image_id}` : `/compare?b=${selected.image_id}`}
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
