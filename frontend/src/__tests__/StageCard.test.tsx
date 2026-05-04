import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StageCard } from "@/components/StageCard";

describe("StageCard", () => {
  it("renders pending state when stage is null", () => {
    render(<StageCard index={1} title="Decode" stage={null} />);
    expect(screen.getByText("Decode")).toBeInTheDocument();
    expect(screen.getByText(/pending/i)).toBeInTheDocument();
  });

  it("renders elapsed time and detail entries when stage is filled", () => {
    render(
      <StageCard
        index={2}
        title="Preprocess"
        stage={{
          name: "preprocess",
          elapsed_ms: 17,
          detail: { resize: "224x224", clahe: true },
        }}
      />,
    );
    expect(screen.getByText(/17 ms/)).toBeInTheDocument();
    expect(screen.getByText("resize")).toBeInTheDocument();
    expect(screen.getByText("224x224")).toBeInTheDocument();
  });
});
