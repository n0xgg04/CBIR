"use client";

import { useState } from "react";
import { invalidateCache } from "@/lib/api";

export default function CacheInvalidateButton() {
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [message, setMessage] = useState("");

  async function handleClick() {
    setStatus("loading");
    setMessage("");
    try {
      await invalidateCache();
      setStatus("ok");
      setMessage("Đã xoá cache thành công.");
    } catch (err) {
      setStatus("error");
      setMessage(err instanceof Error ? err.message : "Lỗi không xác định");
    }
  }

  return (
    <div className="mt-6 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="text-lg font-semibold">Quản trị cache</h2>
      <p className="mt-1 text-sm text-slate-600">
        Nếu vừa thêm/xoá ảnh mà kết quả search chưa đúng, hãy xoá cache để
        backend rebuild feature matrix.
      </p>
      <button
        onClick={handleClick}
        disabled={status === "loading"}
        className="mt-3 inline-flex items-center rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-800 disabled:opacity-50"
      >
        {status === "loading" ? "Đang xoá..." : "Xoá cache"}
      </button>
      {status === "ok" && (
        <p className="mt-2 text-sm text-emerald-700">{message}</p>
      )}
      {status === "error" && (
        <p className="mt-2 text-sm text-rose-600">{message}</p>
      )}
    </div>
  );
}
