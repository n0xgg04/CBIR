import { test, expect } from "@playwright/test";

test("home page renders the hero and API status", async ({ page }) => {
  await page.goto("/");

  await expect(
    page.getByRole("heading", { level: 1, name: "Animal Face CBIR" }),
  ).toBeVisible();

  await expect(page.getByTestId("api-status")).toBeVisible();
});
