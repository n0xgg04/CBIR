import userEvent from "@testing-library/user-event";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { WeightsPanel } from "@/components/WeightsPanel";

describe("WeightsPanel", () => {
  it("renders the live total of all sliders", () => {
    render(
      <WeightsPanel
        weights={{ hsv: 0.2, cm: 0.1, lbp: 0.15, glcm: 0.15, hog: 0.25, hu: 0.15 }}
        onApply={() => {}}
      />,
    );
    expect(screen.getByTestId("weights-total")).toHaveTextContent(/Σ\s+1\.00/);
  });

  it("emits the modified weights when the user clicks Re-rank", async () => {
    const onApply = vi.fn();
    render(
      <WeightsPanel
        weights={{ hsv: 0.2, cm: 0.1, lbp: 0.15, glcm: 0.15, hog: 0.25, hu: 0.15 }}
        onApply={onApply}
      />,
    );

    const hogSlider = screen.getByLabelText(/Histogram of Oriented Gradients/i);
    // RTL's fireEvent.change is React-aware; userEvent doesn't support range sliders well.
    fireEvent.change(hogSlider, { target: { value: "0.5" } });

    await userEvent.click(screen.getByRole("button", { name: /Re-rank/i }));

    expect(onApply).toHaveBeenCalledOnce();
    const submitted = onApply.mock.calls[0][0] as Record<string, number>;
    expect(submitted.hog).toBeCloseTo(0.5, 5);
  });

  it("disables the Re-rank button when busy", () => {
    render(
      <WeightsPanel
        weights={{ hsv: 0.2, cm: 0.1, lbp: 0.15, glcm: 0.15, hog: 0.25, hu: 0.15 }}
        disabled
        onApply={() => {}}
      />,
    );
    expect(
      screen.getByRole("button", { name: /Re-rank/i }),
    ).toBeDisabled();
  });
});
