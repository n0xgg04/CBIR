import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import SearchPage from "@/app/search/page";

const originalFetch = global.fetch;

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
});

afterEach(() => {
  global.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("SearchPage", () => {
  it("renders the dropzone, an empty timeline, and the empty results message", () => {
    render(<SearchPage />);
    expect(
      screen.getByRole("heading", { level: 1, name: /Pipeline Inspector/i }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("dropzone")).toBeInTheDocument();
    expect(screen.getByTestId("results-empty")).toBeInTheDocument();
    // 10 timeline rows expected — 3 pre + 6 features + ranking
    const cards = screen.getAllByTestId("stage-card");
    expect(cards.length).toBeGreaterThanOrEqual(10);
  });

  it("disables the Re-rank button until a search has succeeded", () => {
    render(<SearchPage />);
    expect(
      screen.getByRole("button", { name: /Re-rank/i }),
    ).toBeDisabled();
  });
});
