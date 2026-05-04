"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useMemo } from "react";

import { type VisualizationName, VISUALIZATION_NAMES, visualizationUrl } from "@/lib/api";

const TITLES: Record<VisualizationName, string> = {
  preprocess: "Tiền xử lý",
  hsv: "HSV histogram",
  cm: "Color moments",
  lbp: "LBP",
  glcm: "GLCM",
  hog: "HOG",
  hu: "Hu moments",
};

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

function ComparePanel({ id, label }: { id: number | null; label: string }) {
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
        {VISUALIZATION_NAMES.filter((f) => f !== "preprocess").map((feature) => (
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
        ))}
      </div>
    </article>
  );
}

function CompareContent() {
  const params = useSearchParams();
  const a = useMemo(() => parseId(params.get("a")), [params]);
  const b = useMemo(() => parseId(params.get("b")), [params]);

  return (
    <main className="mx-auto flex max-w-6xl flex-col gap-6 p-6 lg:p-10">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">So sánh ảnh</h1>
          <p className="mt-1 text-sm text-slate-600">
            Trực quan đặc trưng so sánh song song. Thêm <code>?a=ID&amp;b=ID</code> vào URL.
          </p>
        </div>
        <a href="/" className="text-sm text-slate-500 hover:text-slate-700">
          ← Trang chủ
        </a>
      </header>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <ComparePanel id={a} label="a" />
        <ComparePanel id={b} label="b" />
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
    </main>
  );
}

export default function ComparePage() {
  return (
    <Suspense
      fallback={
        <main className="p-6 text-sm text-slate-500">Đang tải giao diện so sánh…</main>
      }
    >
      <CompareContent />
    </Suspense>
  );
}
