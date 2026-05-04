"use client";

import type { ReactNode } from "react";

import type { SearchTraceStage } from "@/lib/api";

export interface StageCardProps {
  index: number;
  title: string;
  stage: SearchTraceStage | null;
  hint?: ReactNode;
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

export function StageCard({ index, title, stage, hint }: StageCardProps) {
  const filled = stage !== null;
  return (
    <div
      data-testid="stage-card"
      data-stage-name={stage?.name ?? `pending-${index}`}
      className={`flex items-start gap-4 rounded-lg border p-4 transition ${
        filled
          ? "border-emerald-200 bg-emerald-50/40"
          : "border-slate-200 bg-slate-50"
      }`}
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
    </div>
  );
}
