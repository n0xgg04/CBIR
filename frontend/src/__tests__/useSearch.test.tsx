import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useSearch } from "@/hooks/useSearch";

type FetchInit = RequestInit;

const originalFetch = global.fetch;

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
    ...init,
  });
}

beforeEach(() => {
  if (typeof URL.createObjectURL !== "function") {
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      writable: true,
      value: vi.fn(() => "blob:mock"),
    });
  } else {
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:mock");
  }
  if (typeof URL.revokeObjectURL !== "function") {
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      writable: true,
      value: vi.fn(),
    });
  } else {
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  }
});

afterEach(() => {
  global.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("useSearch", () => {
  it("transitions through searching → ready when /search succeeds", async () => {
    const payload = {
      run_id: 3,
      weights: { hsv: 0.4, hog: 0.6 },
      results: [
        {
          image_id: 1,
          rank: 1,
          score: 0.9,
          per_feature: { hsv: 0.5 },
          image: null,
        },
      ],
      pipeline_trace: [
        { name: "preprocess", elapsed_ms: 4, detail: { resize: "224x224" } },
      ],
      elapsed_ms: 12,
      corpus_size: 5,
      query_dims: { hsv: 768 },
    };
    const fetchMock = vi.fn(async (_url: RequestInfo | URL, _init?: FetchInit) =>
      jsonResponse(payload),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const { result } = renderHook(() => useSearch());
    expect(result.current.state.phase).toBe("idle");

    const file = new File(["x"], "q.jpg", { type: "image/jpeg" });
    await act(async () => {
      await result.current.runSearch(file);
    });

    await waitFor(() => {
      expect(result.current.state.phase).toBe("ready");
    });
    expect(result.current.state.response).toEqual(payload);
    expect(result.current.state.weights).toEqual(payload.weights);
    expect(result.current.state.preview).toBe("blob:mock");
    expect(result.current.state.error).toBeNull();
  });

  it("captures the error message when /search fails", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ detail: "corpus is empty" }), {
        status: 409,
        headers: { "content-type": "application/json" },
      }),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const { result } = renderHook(() => useSearch());
    const file = new File(["x"], "q.jpg");
    await act(async () => {
      await result.current.runSearch(file);
    });

    await waitFor(() => {
      expect(result.current.state.phase).toBe("error");
    });
    expect(result.current.state.error).toMatch(/corpus is empty/);
    expect(result.current.state.response).toBeNull();
  });

  it("reranks via PATCH after a successful search", async () => {
    const search = {
      run_id: 7,
      weights: { hsv: 0.4 },
      results: [],
      pipeline_trace: [],
      elapsed_ms: 0,
      corpus_size: 0,
      query_dims: {},
    };
    const rerank = {
      ...search,
      weights: { hsv: 0.1, hog: 0.9 },
      results: [{ image_id: 9, rank: 1, score: 0.99, per_feature: {}, image: null }],
    };

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(search))
      .mockResolvedValueOnce(jsonResponse(rerank));
    global.fetch = fetchMock as unknown as typeof fetch;

    const { result } = renderHook(() => useSearch());
    const file = new File(["x"], "q.jpg");
    await act(async () => {
      await result.current.runSearch(file);
    });
    await waitFor(() => expect(result.current.state.phase).toBe("ready"));

    await act(async () => {
      await result.current.applyWeights({ hsv: 0.1, hog: 0.9 }, 5);
    });

    await waitFor(() => expect(result.current.state.phase).toBe("ready"));
    expect(result.current.state.response).toEqual(rerank);
    expect(fetchMock.mock.calls).toHaveLength(2);
    const patchCall = fetchMock.mock.calls[1];
    expect((patchCall[1] as FetchInit).method).toBe("PATCH");
  });

  it("reset clears state", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        run_id: 1,
        weights: {},
        results: [],
        pipeline_trace: [],
        elapsed_ms: 0,
        corpus_size: 0,
        query_dims: {},
      }),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const { result } = renderHook(() => useSearch());
    const file = new File(["x"], "q.jpg");
    await act(async () => {
      await result.current.runSearch(file);
    });
    await waitFor(() => expect(result.current.state.phase).toBe("ready"));

    act(() => {
      result.current.reset();
    });
    expect(result.current.state.phase).toBe("idle");
    expect(result.current.state.response).toBeNull();
    expect(result.current.state.preview).toBeNull();
  });
});
