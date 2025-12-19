"use client";

import { create } from "zustand";

export type ReviewScope =
  | "Structure"
  | "Contexte"
  | "Fonds"
  | "Disclaimers"
  | "Glossaires";

export type Violation = {
  id: string;
  page: number;
  scope: ReviewScope;
  title: string;
  description: string;
  severity: "low" | "medium" | "high";
  ruleId?: string; // e.g., "RS1", "RC25"
  location?: string; // e.g., "Slide 15 - slide_15_shape_9"
  type?: "violation" | "missing";
  bbox?: { x: number; y: number; w: number; h: number }; // pour overlay
};

type ReviewState = {
  fileName?: string;
  fileType?: "pdf" | "pptx" | "docx";
  totalPages: number;
  currentPage: number;
  analyzed: boolean;
  loading: boolean;

  // Scopes and Filtering
  scopes: Record<ReviewScope, boolean>;
  filterScopes: ReviewScope[];

  // Data
  violations: Violation[];
  docStructure: any; // Raw document structure from backend (useful for PPTX fallback)
  auditResults: any; // Full audit report from checker_memory

  // UI State
  showAnnotations: boolean;

  // Actions
  setFile: (info: Partial<ReviewState>) => void;
  setPage: (page: number) => void;
  toggleScope: (scope: ReviewScope) => void;
  setFilterScopes: (scopes: ReviewScope[]) => void;
  setShowAnnotations: (show: boolean) => void;
  startAnalysis: () => void;
  finishAnalysis: (violations: Violation[], docStructure?: any, auditResults?: any) => void;
};

export const useReviewStore = create<ReviewState>((set) => ({
  fileName: "document",
  fileType: "pdf",
  totalPages: 0,
  currentPage: 1,
  analyzed: false,
  loading: false,
  scopes: {
    Structure: true,
    Contexte: true,
    Fonds: true,
    Disclaimers: true,
    Glossaires: true,
  },
  violations: [],
  docStructure: null,
  auditResults: null,
  showAnnotations: false,
  filterScopes: [],

  setFile: (info) => set((state) => ({ ...state, ...info })),

  setPage: (p) =>
    set((state) => ({ currentPage: Math.max(1, Math.min(state.totalPages, p)) })),

  toggleScope: (s) => set((state) => ({ scopes: { ...state.scopes, [s]: !state.scopes[s] } })),

  setFilterScopes: (scopes) => set({ filterScopes: scopes }),

  setShowAnnotations: (show) => set({ showAnnotations: show }),

  startAnalysis: () => set({ loading: true, analyzed: false, violations: [], docStructure: null, auditResults: null, totalPages: 0 }),

  finishAnalysis: (violations, docStructure = null, auditResults = null) =>
    set({
      loading: false,
      analyzed: true,
      violations,
      docStructure,
      auditResults,
      showAnnotations: true,
      totalPages: docStructure?.total_slides || undefined,
    }),
}));