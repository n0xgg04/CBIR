/** Default per-feature weights used when the user has not customised them yet. */

import { FEATURE_NAMES, type FeatureName } from "@/lib/api";

export const DEFAULT_WEIGHTS: Readonly<Record<FeatureName, number>> = {
  hsv: 0.2,
  cm: 0.1,
  lbp: 0.15,
  glcm: 0.15,
  hog: 0.25,
  hu: 0.15,
};

export function copyDefaultWeights(): Record<FeatureName, number> {
  return { ...DEFAULT_WEIGHTS };
}

export function withWeight(
  weights: Record<string, number>,
  name: FeatureName,
  value: number,
): Record<string, number> {
  return { ...weights, [name]: value };
}

/** Friendly title shown next to each weight slider / pipeline stage. */
export const FEATURE_TITLE: Readonly<Record<FeatureName, string>> = {
  hsv: "Biểu đồ HSV",
  cm: "Mômen màu",
  lbp: "Mẫu nhị phân cục bộ (LBP)",
  glcm: "GLCM Haralick",
  hog: "Histogram of Oriented Gradients (HOG)",
  hu: "Mômen Hu",
};

export const FEATURE_LIST: ReadonlyArray<FeatureName> = FEATURE_NAMES;
