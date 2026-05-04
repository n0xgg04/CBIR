"use client";

import { useMemo } from "react";

import { Dropzone } from "@/components/Dropzone";
import { ResultGrid } from "@/components/ResultGrid";
import { StageCard } from "@/components/StageCard";
import { WeightsPanel } from "@/components/WeightsPanel";
import { useSearch } from "@/hooks/useSearch";
import type { SearchTraceStage } from "@/lib/api";
import { FEATURE_LIST, FEATURE_TITLE } from "@/lib/weights";

interface TimelineRow {
  index: number;
  title: string;
  stageName: string;
  hint?: string;
}

const TIMELINE: TimelineRow[] = [
  { index: 1, title: "Giải mã", stageName: "decode", hint: "Đọc bytes ảnh truy vấn → BGR." },
  { index: 2, title: "Tiền xử lý", stageName: "preprocess", hint: "Resize 224×224, CLAHE, chuẩn hóa RGB." },
  { index: 3, title: "Trích xuất đặc trưng", stageName: "extract", hint: "Cả 6 đặc trưng thủ công." },
  ...FEATURE_LIST.map((name, idx) => ({
    index: 4 + idx,
    title: FEATURE_TITLE[name],
    stageName: `feature.${name}`,
    hint: undefined,
  })),
  { index: 10, title: "Rank", stageName: "rank", hint: "Cosine fuse → argsort top-K." },
];

function pickStage(
  trace: SearchTraceStage[] | undefined,
  name: string,
): SearchTraceStage | null {
  if (!trace) {
    return null;
  }
  return trace.find((stage) => stage.name === name) ?? null;
}

export default function SearchPage() {
  const { state, runSearch, applyWeights, setTopN, reset } = useSearch();
  const { phase, response, preview, weights, error, topN } = state;

  const busy = phase === "searching" || phase === "reranking";

  const handleFiles = (files: File[]): void => {
    if (files.length === 0) {
      return;
    }
    void runSearch(files[0]);
  };

  const summary = useMemo(() => {
    if (!response) {
      return null;
    }
    return {
      runId: response.run_id,
      corpus: response.corpus_size,
      elapsed: response.elapsed_ms,
      stages: response.pipeline_trace.length,
    };
  }, [response]);

  return (
    <main className="mx-auto flex max-w-6xl flex-col gap-6 p-6 lg:p-10">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Giám sát Pipeline</h1>
          <p className="mt-1 text-sm text-slate-600">
            Thả ảnh truy vấn. Xem từng giai đoạn của pipeline CBIR chạy trực tiếp.
          </p>
        </div>
        <a
          href="/"
          className="text-sm text-slate-500 hover:text-slate-700"
        >
          ← Trang chủ
        </a>
      </header>

      <section
        aria-labelledby="query-heading"
        className="flex flex-col gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
      >
        <div className="flex items-baseline justify-between">
          <h2 id="query-heading" className="text-base font-semibold text-slate-800">
            Truy vấn
          </h2>
          {phase !== "idle" ? (
            <button
              type="button"
              onClick={reset}
              className="text-xs text-slate-500 hover:text-slate-700"
            >
              Clear
            </button>
          ) : null}
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Dropzone
            label="Ảnh truy vấn"
            hint="JPEG/PNG/WEBP, tối đa 8 MB"
            onFiles={handleFiles}
            disabled={busy}
          />
          <div className="flex min-h-[12rem] items-center justify-center overflow-hidden rounded-xl border border-slate-200 bg-slate-50 p-3">
            {preview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={preview}
                alt="Xem trước truy vấn"
                data-testid="query-preview"
                className="max-h-64 w-auto rounded-md object-contain"
              />
            ) : (
              <span className="text-sm text-slate-500">
                Bản xem trước sẽ hiển thị khi bạn thả ảnh vào.
              </span>
            )}
          </div>
        </div>
        {summary ? (
          <dl className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
            <div className="rounded-md bg-slate-50 p-2">
              <dt className="text-slate-500">Lượt chạy</dt>
              <dd className="font-mono text-slate-800">#{summary.runId}</dd>
            </div>
            <div className="rounded-md bg-slate-50 p-2">
              <dt className="text-slate-500">Kho dữ liệu</dt>
              <dd className="font-mono text-slate-800">{summary.corpus}</dd>
            </div>
            <div className="rounded-md bg-slate-50 p-2">
              <dt className="text-slate-500">Thờì gian</dt>
              <dd className="font-mono text-slate-800">{summary.elapsed} ms</dd>
            </div>
            <div className="rounded-md bg-slate-50 p-2">
              <dt className="text-slate-500">Giai đoạn</dt>
              <dd className="font-mono text-slate-800">{summary.stages}</dd>
            </div>
          </dl>
        ) : null}
        {error ? (
          <div
            role="alert"
            data-testid="search-error"
            className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700"
          >
            {error}
          </div>
        ) : null}
        {busy ? (
          <p
            role="status"
            data-testid="search-status"
            className="text-sm text-slate-500"
          >
            {phase === "reranking" ? "Đang xếp hạng lại…" : "Đang chạy pipeline…"}
          </p>
        ) : null}
      </section>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[2fr_1fr]">
        <section
          aria-labelledby="timeline-heading"
          className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
        >
          <h2
            id="timeline-heading"
            className="text-base font-semibold text-slate-800"
          >
            Dòng thờì gian pipeline
          </h2>
          <ol className="flex flex-col gap-3">
            {TIMELINE.map((row) => (
              <li key={row.stageName}>
                <StageCard
                  index={row.index}
                  title={row.title}
                  stage={pickStage(response?.pipeline_trace, row.stageName)}
                  hint={row.hint}
                />
              </li>
            ))}
          </ol>
        </section>

        <WeightsPanel
          weights={weights}
          disabled={busy || response === null}
          onApply={(next) => {
            void applyWeights(next);
          }}
        />
      </div>

      <section
        aria-labelledby="results-heading"
        className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
      >
        <div className="flex items-baseline justify-between">
          <h2
            id="results-heading"
            className="text-base font-semibold text-slate-800"
          >
            Kết quả hàng đầu
          </h2>
          <div className="flex items-center gap-3">
            <label htmlFor="top-n" className="text-xs text-slate-500">
              Top N
            </label>
            <input
              id="top-n"
              type="number"
              min={1}
              max={50}
              value={topN}
              disabled={busy}
              onChange={(e) => {
                const n = Number(e.target.value);
                setTopN(n);
                if (phase === "ready" && response) {
                  void applyWeights(weights);
                }
              }}
              className="w-16 rounded-md border border-slate-300 px-2 py-1 text-right text-sm focus:border-brand focus:outline-none focus:ring-2 focus:ring-brand/30"
            />
            {response ? (
              <span className="text-xs text-slate-500">
                {response.results.length} / {response.corpus_size} đã xếp hạng
              </span>
            ) : null}
          </div>
        </div>
        <ResultGrid
          results={response?.results ?? []}
          queryPreview={preview}
        />
      </section>
    </main>
  );
}
