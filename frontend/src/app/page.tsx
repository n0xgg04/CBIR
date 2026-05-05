import CacheInvalidateButton from "@/components/CacheInvalidateButton";
import { fetchPing, type PingResponse } from "@/lib/api";

export const dynamic = "force-dynamic";

type PingState =
  | { kind: "ok"; payload: PingResponse }
  | { kind: "error"; message: string };

async function loadPing(): Promise<PingState> {
  try {
    const payload = await fetchPing();
    return { kind: "ok", payload };
  } catch (err) {
    return {
      kind: "error",
      message: err instanceof Error ? err.message : String(err),
    };
  }
}

export default async function Home() {
  const state = await loadPing();

  return (
    <main className="mx-auto max-w-3xl p-12">
      <h1 className="text-4xl font-bold tracking-tight">Nhóm 2</h1>
      <p className="mt-3 text-slate-600">
        Tìm kiếm ảnh dựa trên nội dung (CBIR) cho ảnh mặt động vật sử dụng các
        đặc trưng.
      </p>

      <section
        className="mt-10 rounded-xl border border-slate-200 bg-white p-6 shadow-sm"
        aria-labelledby="api-status-heading"
      >
        <h2 id="api-status-heading" className="text-lg font-semibold">
          Trạng thái backend
        </h2>
        {state.kind === "error" ? (
          <p data-testid="api-status" className="mt-2 text-rose-600">
            Không kết nối được: {state.message}
          </p>
        ) : (
          <p data-testid="api-status" className="mt-2 text-emerald-700">
            đang hoạt động — dịch vụ{" "}
            <code className="font-mono">{state.payload.service}</code> v
            <code className="font-mono">{state.payload.version}</code>
          </p>
        )}
      </section>

      <section className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <a
          href="/upload"
          className="rounded-xl border border-slate-200 bg-white p-5 transition hover:border-brand hover:shadow"
        >
          <h3 className="font-semibold">Tải ảnh lên</h3>
          <p className="text-sm text-slate-600">
            Thêm ảnh mặt động vật vào kho dữ liệu.
          </p>
        </a>
        <a
          href="/search"
          className="rounded-xl border border-slate-200 bg-white p-5 transition hover:border-brand hover:shadow"
        >
          <h3 className="font-semibold">Chạy truy vấn</h3>
          <p className="text-sm text-slate-600">
            Thả ảnh truy vấn và xem pipeline chạy.
          </p>
        </a>
      </section>

      <CacheInvalidateButton />
    </main>
  );
}
