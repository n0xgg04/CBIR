"use client";

import { useCallback, useState } from "react";

import {
  type SearchResponse,
  rerankSearch,
  searchImages,
} from "@/lib/api";
import { copyDefaultWeights } from "@/lib/weights";

export type SearchPhase = "idle" | "searching" | "reranking" | "ready" | "error";

export interface SearchState {
  phase: SearchPhase;
  response: SearchResponse | null;
  preview: string | null;
  weights: Record<string, number>;
  error: string | null;
  /** Last query file kept so the user can re-query without re-dropping. */
  queryFile: File | null;
  /** Number of top results to fetch. */
  topN: number;
}

const INITIAL_STATE: SearchState = {
  phase: "idle",
  response: null,
  preview: null,
  weights: copyDefaultWeights(),
  error: null,
  queryFile: null,
  topN: 5,
};

export interface UseSearchApi {
  state: SearchState;
  runSearch: (file: File) => Promise<void>;
  applyWeights: (weights: Record<string, number>) => Promise<void>;
  setTopN: (n: number) => void;
  reset: () => void;
}

export function useSearch(): UseSearchApi {
  const [state, setState] = useState<SearchState>(INITIAL_STATE);

  const setTopN = useCallback((n: number): void => {
    setState((prev) => ({ ...prev, topN: Math.max(1, Math.min(50, n)) }));
  }, []);

  const runSearch = useCallback(
    async (file: File): Promise<void> => {
      const preview = URL.createObjectURL(file);
      setState((prev) => {
        if (prev.preview) {
          URL.revokeObjectURL(prev.preview);
        }
        return {
          ...prev,
          phase: "searching",
          response: null,
          preview,
          error: null,
          queryFile: file,
        };
      });
      try {
        const response = await searchImages(file, {
          topK: state.topN,
          weights: state.weights,
        });
        setState((prev) => ({
          ...prev,
          phase: "ready",
          response,
          weights: { ...response.weights },
          error: null,
        }));
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setState((prev) => ({
          ...prev,
          phase: "error",
          response: null,
          error: message,
        }));
      }
    },
    [state.weights, state.topN],
  );

  const applyWeights = useCallback(
    async (weights: Record<string, number>): Promise<void> => {
      setState((prev) => ({
        ...prev,
        phase: "reranking",
        weights,
        error: null,
      }));
      const runId = state.response?.run_id;
      if (runId === undefined) {
        setState((prev) => ({
          ...prev,
          phase: "error",
          error: "no active search run",
        }));
        return;
      }
      try {
        const response = await rerankSearch(runId, weights, { topK: state.topN });
        setState((prev) => ({
          ...prev,
          phase: "ready",
          response,
          weights: { ...response.weights },
          error: null,
        }));
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setState((prev) => ({
          ...prev,
          phase: "error",
          error: message,
        }));
      }
    },
    [state.response?.run_id, state.topN],
  );

  const reset = useCallback((): void => {
    setState((prev) => {
      if (prev.preview) {
        URL.revokeObjectURL(prev.preview);
      }
      return {
        ...INITIAL_STATE,
        weights: copyDefaultWeights(),
      };
    });
  }, []);

  return { state, runSearch, applyWeights, setTopN, reset };
}
