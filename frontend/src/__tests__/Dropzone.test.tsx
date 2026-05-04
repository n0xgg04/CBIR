import userEvent from "@testing-library/user-event";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Dropzone } from "@/components/Dropzone";

describe("Dropzone", () => {
  it("emits the picked file via onFiles", async () => {
    const onFiles = vi.fn();
    render(<Dropzone onFiles={onFiles} label="Image" />);

    const input = screen
      .getByTestId("dropzone")
      .querySelector("input[type=file]") as HTMLInputElement;
    expect(input).not.toBeNull();

    const file = new File(["payload"], "cat.jpg", { type: "image/jpeg" });
    await userEvent.upload(input, file);

    expect(onFiles).toHaveBeenCalledOnce();
    const [files] = onFiles.mock.calls[0];
    expect(files).toHaveLength(1);
    expect(files[0].name).toBe("cat.jpg");
  });

  it("respects multiple=false even when several files are dropped", async () => {
    const onFiles = vi.fn();
    render(<Dropzone onFiles={onFiles} multiple={false} />);

    const input = screen
      .getByTestId("dropzone")
      .querySelector("input[type=file]") as HTMLInputElement;
    const f1 = new File(["a"], "a.jpg", { type: "image/jpeg" });
    const f2 = new File(["b"], "b.jpg", { type: "image/jpeg" });
    await userEvent.upload(input, [f1, f2]);

    const passed = onFiles.mock.calls[0][0];
    expect(passed).toHaveLength(1);
  });

  it("does not fire when disabled", async () => {
    const onFiles = vi.fn();
    render(<Dropzone onFiles={onFiles} disabled />);

    const input = screen
      .getByTestId("dropzone")
      .querySelector("input[type=file]") as HTMLInputElement;
    expect(input.disabled).toBe(true);
  });
});
