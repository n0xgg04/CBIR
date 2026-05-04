import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  apiUrl,
  evaluateCorpus,
  rerankSearch,
  searchImages,
  uploadImage,
  visualizationUrl,
} from "@/lib/api";

type FetchSpy = ReturnType<typeof vi.fn>;

const originalFetch = global.fetch;

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
    ...init,
  });
}

beforeEach(() => {
  // each test installs its own fetch mock
});

afterEach(() => {
  global.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("apiUrl", () => {
  it("joins paths against the active base URL", () => {
    expect(apiUrl("/ping")).toMatch(/\/api\/v1\/ping$/);
  });

  it("normalises a missing leading slash", () => {
    expect(apiUrl("ping")).toMatch(/\/api\/v1\/ping$/);
  });

  it("strips a trailing slash on the base URL", () => {
    const result = apiUrl("/health");
    expect(result.includes("//api")).toBe(false);
  });
});

describe("visualizationUrl", () => {
  it("targets the visualize/<id>/<feature> endpoint", () => {
    expect(visualizationUrl(7, "hsv")).toMatch(/\/visualize\/7\/hsv$/);
  });
});

describe("uploadImage", () => {
  it("posts multipart form data and returns parsed JSON", async () => {
    const payload = {
      image: { id: 1 },
      features: { image_id: 1 },
    };
    const fetchMock: FetchSpy = vi.fn(async () => jsonResponse(payload));
    global.fetch = fetchMock as unknown as typeof fetch;

    const file = new File(["xyz"], "cat.jpg", { type: "image/jpeg" });
    const result = await uploadImage(file, "cat");

    expect(result).toEqual(payload);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/images$/);
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    const form = init.body as FormData;
    expect(form.get("animal_type")).toBe("cat");
    expect(form.get("role")).toBe("corpus");
    expect(form.get("file")).toBeInstanceOf(File);
  });

  it("throws an Error with the backend detail on failure", async () => {
    const fetchMock: FetchSpy = vi.fn(async () =>
      new Response(JSON.stringify({ detail: "bad image" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      }),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const file = new File(["x"], "broken.jpg");
    await expect(uploadImage(file, "cat")).rejects.toThrow(/bad image/);
  });
});

describe("searchImages", () => {
  it("encodes top_k and weights JSON in the multipart body", async () => {
    const fetchMock: FetchSpy = vi.fn(async () =>
      jsonResponse({
        run_id: 9,
        weights: { hsv: 0.2 },
        results: [],
        pipeline_trace: [],
        elapsed_ms: 17,
        corpus_size: 4,
        query_dims: { hsv: 768 },
      }),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const file = new File(["q"], "query.jpg", { type: "image/jpeg" });
    await searchImages(file, { topK: 7, weights: { hsv: 0.2 } });

    const [, init] = fetchMock.mock.calls[0];
    const form = init.body as FormData;
    expect(form.get("top_k")).toBe("7");
    expect(form.get("weights")).toBe(JSON.stringify({ hsv: 0.2 }));
  });

  it("omits optional fields by default", async () => {
    const fetchMock: FetchSpy = vi.fn(async () =>
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

    const file = new File(["q"], "x.jpg");
    await searchImages(file);

    const [, init] = fetchMock.mock.calls[0];
    const form = init.body as FormData;
    expect(form.has("top_k")).toBe(false);
    expect(form.has("weights")).toBe(false);
    expect(form.has("stream_id")).toBe(false);
  });
});

describe("rerankSearch", () => {
  it("PATCHes the run with JSON body", async () => {
    const fetchMock: FetchSpy = vi.fn(async () =>
      jsonResponse({
        run_id: 12,
        weights: { hog: 0.5 },
        results: [],
        pipeline_trace: [],
        elapsed_ms: 1,
        corpus_size: 0,
        query_dims: {},
      }),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    await rerankSearch(12, { hog: 0.5 }, { topK: 3 });

    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/search\/12\/weights$/);
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body as string)).toEqual({
      weights: { hog: 0.5 },
      top_k: 3,
    });
  });
});

describe("evaluateCorpus", () => {
  it("posts the evaluation body with sensible defaults", async () => {
    const fetchMock: FetchSpy = vi.fn(async () =>
      jsonResponse({
        run_id: 3,
        method: "default",
        top_k: 10,
        corpus_size: 6,
        report: null,
        ablation: null,
        elapsed_ms: 1,
      }),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    await evaluateCorpus();

    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body as string)).toEqual({
      method: "default",
      top_k: 10,
    });
  });

  it("propagates ablation method + custom top_k", async () => {
    const fetchMock: FetchSpy = vi.fn(async () =>
      jsonResponse({
        run_id: 4,
        method: "ablation",
        top_k: 5,
        corpus_size: 6,
        report: null,
        ablation: null,
        elapsed_ms: 1,
      }),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    await evaluateCorpus({ method: "ablation", topK: 5 });

    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body as string)).toEqual({
      method: "ablation",
      top_k: 5,
    });
  });
});
