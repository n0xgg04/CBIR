"use client";

import { useCallback, useState } from "react";

import { Dropzone } from "@/components/Dropzone";
import {
  type ImageWithFeatures,
  type VisualizationName,
  VISUALIZATION_NAMES,
  uploadImage,
  visualizationUrl,
} from "@/lib/api";

interface UploadOutcome {
  status: "ok" | "error";
  filename: string;
  payload?: ImageWithFeatures;
  error?: string;
}

const TITLES: Record<VisualizationName, string> = {
  preprocess: "Tiền xử lý",
  hsv: "Biểu đồ HSV",
  cm: "Mômen màu",
  lbp: "LBP",
  glcm: "GLCM",
  hog: "HOG",
  hu: "Mômen Hu",
};

export default function UploadPage() {
  const [animalType, setAnimalType] = useState("cat");
  const [outcomes, setOutcomes] = useState<UploadOutcome[]>([]);
  const [busy, setBusy] = useState(false);

  const onFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) {
        return;
      }
      const trimmed = animalType.trim();
      if (trimmed.length === 0) {
        setOutcomes((prev) => [
          ...prev,
          {
            status: "error",
            filename: files[0]?.name ?? "(không tên)",
            error: "Hãy nhập nhãn động vật trước khi tải lên.",
          },
        ]);
        return;
      }
      setBusy(true);
      try {
        const next: UploadOutcome[] = [];
        for (const file of files) {
          try {
            const payload = await uploadImage(file, trimmed);
            next.push({ status: "ok", filename: file.name, payload });
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            next.push({ status: "error", filename: file.name, error: message });
          }
        }
        setOutcomes((prev) => [...next, ...prev]);
      } finally {
        setBusy(false);
      }
    },
    [animalType],
  );

  return (
    <main className="mx-auto flex max-w-4xl flex-col gap-6 p-6 lg:p-10">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Tải ảnh lên</h1>
          <p className="mt-1 text-sm text-slate-600">
            Thêm ảnh mặt động vật vào kho dữ liệu CBIR. Các đặc trưng được
            trích xuất phía server khi tải lên.
          </p>
        </div>
        <a href="/" className="text-sm text-slate-500 hover:text-slate-700">
          ← Trang chủ
        </a>
      </header>

      <section className="flex flex-col gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <label
          htmlFor="animal-type"
          className="text-sm font-medium text-slate-700"
        >
          Nhãn động vật
        </label>
        <input
          id="animal-type"
          type="text"
          value={animalType}
          maxLength={64}
          onChange={(e) => setAnimalType(e.target.value)}
          disabled={busy}
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand focus:outline-none focus:ring-2 focus:ring-brand/30"
        />
        <Dropzone
          accept="image/*"
          multiple
          disabled={busy}
          label="Tệp ảnh"
          hint="Hỗ trợ nhiều tệp. Ảnh trùng lặp (cùng SHA-256) sẽ bị bỏ qua phía server."
          onFiles={onFiles}
        />
        {busy ? (
          <p data-testid="upload-status" className="text-sm text-slate-500">
            Đang tải lên…
          </p>
        ) : null}
      </section>

      <section
        aria-labelledby="upload-history"
        className="flex flex-col gap-3"
      >
        <h2 id="upload-history" className="text-base font-semibold text-slate-800">
          Tải lên gần đây
        </h2>
        {outcomes.length === 0 ? (
          <p className="text-sm text-slate-500" data-testid="upload-empty">
            Chưa có ảnh nào được tải lên.
          </p>
        ) : (
          <ul className="flex flex-col gap-3">
            {outcomes.map((row, idx) => (
              <li
                key={`${row.filename}-${idx}`}
                className={`rounded-xl border p-4 ${
                  row.status === "ok"
                    ? "border-emerald-200 bg-emerald-50/40"
                    : "border-rose-200 bg-rose-50"
                }`}
                data-testid={`upload-row-${row.status}`}
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-medium text-slate-800">
                    {row.filename}
                  </span>
                  <span
                    className={`text-xs font-mono ${
                      row.status === "ok"
                        ? "text-emerald-700"
                        : "text-rose-700"
                    }`}
                  >
                    {row.status === "ok"
                      ? `ảnh #${row.payload!.image.id}`
                      : "thất bại"}
                  </span>
                </div>
                {row.status === "error" ? (
                  <p className="mt-1 text-xs text-rose-700">{row.error}</p>
                ) : null}
                {row.status === "ok" && row.payload ? (
                  <details className="mt-2 text-sm">
                    <summary className="cursor-pointer text-slate-600">
                      Xem trực quan đặc trưng
                    </summary>
                    <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
                      {VISUALIZATION_NAMES.map((feature) => (
                        <figure
                          key={feature}
                          className="flex flex-col gap-1 rounded-md border border-slate-200 bg-white p-2"
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={visualizationUrl(
                              row.payload!.image.id,
                              feature,
                            )}
                            alt={`${TITLES[feature]} for ${row.filename}`}
                            className="aspect-square w-full rounded-sm object-contain"
                            loading="lazy"
                          />
                          <figcaption className="text-center text-xs text-slate-600">
                            {TITLES[feature]}
                          </figcaption>
                        </figure>
                      ))}
                    </div>
                  </details>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
