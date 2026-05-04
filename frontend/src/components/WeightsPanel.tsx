"use client";

import { useMemo, useState } from "react";

import { type FeatureName } from "@/lib/api";
import {
  DEFAULT_WEIGHTS,
  FEATURE_LIST,
  FEATURE_TITLE,
  copyDefaultWeights,
} from "@/lib/weights";

export interface WeightsPanelProps {
  weights: Record<string, number>;
  disabled?: boolean;
  onApply: (next: Record<string, number>) => void;
}

const STEP = 0.05;
const MIN = 0;
const MAX = 1;

export function WeightsPanel({
  weights,
  disabled,
  onApply,
}: WeightsPanelProps) {
  const initial = useMemo(() => {
    const merged: Record<FeatureName, number> = { ...DEFAULT_WEIGHTS };
    for (const name of FEATURE_LIST) {
      if (typeof weights[name] === "number") {
        merged[name] = weights[name];
      }
    }
    return merged;
  }, [weights]);

  const [draft, setDraft] = useState<Record<FeatureName, number>>(initial);

  const total = useMemo(
    () => FEATURE_LIST.reduce((sum, name) => sum + (draft[name] ?? 0), 0),
    [draft]
  );

  const apply = (): void => {
    onApply({ ...draft });
  };

  const reset = (): void => {
    setDraft(copyDefaultWeights());
  };

  return (
    <section
      aria-labelledby="weights-heading"
      className="flex flex-col gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
    >
      <div className="flex items-baseline justify-between">
        <h2
          id="weights-heading"
          className="text-base font-semibold text-slate-800"
        >
          Trọng số
        </h2>
        <span
          data-testid="weights-total"
          className="text-xs font-mono text-slate-500"
        >
          Σ {total.toFixed(2)}
        </span>
      </div>
      <ul className="flex flex-col gap-3">
        {FEATURE_LIST.map((name) => (
          <li key={name} className="flex flex-col gap-1">
            <div className="flex items-baseline justify-between text-sm">
              <label
                htmlFor={`w-${name}`}
                className="font-medium text-slate-700"
              >
                {FEATURE_TITLE[name]}
              </label>
              <output
                htmlFor={`w-${name}`}
                className="font-mono text-xs text-slate-500"
              >
                {(draft[name] ?? 0).toFixed(2)}
              </output>
            </div>
            <input
              id={`w-${name}`}
              type="range"
              min={MIN}
              max={MAX}
              step={STEP}
              value={draft[name] ?? 0}
              disabled={disabled}
              onChange={(event) =>
                setDraft((prev) => ({
                  ...prev,
                  [name]: Number(event.target.value),
                }))
              }
              className="h-2 w-full cursor-pointer accent-brand"
            />
          </li>
        ))}
      </ul>
      <div className="flex items-center justify-end gap-2">
        <button
          type="button"
          onClick={reset}
          disabled={disabled}
          className="rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50 disabled:opacity-50"
        >
          Đặt lại mặc định
        </button>
        <button
          type="button"
          onClick={apply}
          disabled={disabled}
          className="rounded-md bg-brand px-3 py-1.5 text-sm font-semibold text-white hover:bg-brand-dark disabled:opacity-50"
        >
          Tìm kiếm
        </button>
      </div>
    </section>
  );
}
