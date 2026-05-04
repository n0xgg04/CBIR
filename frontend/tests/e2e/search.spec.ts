import { expect, test } from "@playwright/test";

test("search page renders the inspector skeleton", async ({ page }) => {
  await page.goto("/search");

  await expect(
    page.getByRole("heading", { level: 1, name: /Pipeline Inspector/i }),
  ).toBeVisible();

  // 10 timeline stage cards should be present even before a search runs.
  const cards = page.getByTestId("stage-card");
  await expect(cards).toHaveCount(10);

  // Empty results state is rendered.
  await expect(page.getByTestId("results-empty")).toBeVisible();

  // Re-rank button is disabled until a search succeeds.
  await expect(page.getByRole("button", { name: /Re-rank/i })).toBeDisabled();
});

test("compare page handles missing query params gracefully", async ({ page }) => {
  await page.goto("/compare");
  await expect(
    page.getByRole("heading", { level: 1, name: /Compare images/i }),
  ).toBeVisible();
  await expect(page.getByTestId("compare-empty-a")).toBeVisible();
  await expect(page.getByTestId("compare-empty-b")).toBeVisible();
});
