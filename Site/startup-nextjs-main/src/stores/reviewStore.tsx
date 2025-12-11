"use client";
import { create } from "zustand";

export type ReviewScope = "Structure" | "Contexte" | "Fonds" | "Disclaimers" | "Prospectus" | "Registration";

export type Violation = {
  id: string;
  page: number;
  scope: ReviewScope;
  title: string;
  description: string;
  severity: "low" | "medium" | "high";
  ruleId?: string; // e.g., "RS1", "RC25"
  location?: string; // e.g., "Slide 15 - slide_15_shape_9"
  type?: "violation" | "missing"; // Type of issue
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

  // UI State
  showAnnotations: boolean;

  // Actions
  setFile: (info: Partial<ReviewState>) => void;
  setPage: (page: number) => void;
  toggleScope: (scope: ReviewScope) => void;
  setFilterScopes: (scopes: ReviewScope[]) => void;
  setShowAnnotations: (show: boolean) => void;
  startAnalysis: () => void;
  // Updated finishAnalysis to accept optional docStructure
  finishAnalysis: (violations: Violation[], docStructure?: any) => void;
};

export const useReviewStore = create<ReviewState>((set) => ({
  fileName: "document",
  fileType: "pdf",
  totalPages: 1,
  currentPage: 1,
  analyzed: false,
  loading: false,

  scopes: {
    Structure: true,
    Contexte: true,
    Fonds: true,
    Disclaimers: true,
    Prospectus: true,
    Registration: true,
  },
  violations: [],
  docStructure: null,
  showAnnotations: false,
  filterScopes: [],

  setFile: (info) => set((state) => ({ ...state, ...info })),
  setPage: (p) => set((state) => ({ currentPage: Math.max(1, Math.min(state.totalPages, p)) })),
  toggleScope: (s) => set((state) => ({ scopes: { ...state.scopes, [s]: !state.scopes[s] } })),
  setFilterScopes: (scopes) => set({ filterScopes: scopes }),
  setShowAnnotations: (show) => set({ showAnnotations: show }),

  startAnalysis: () => set({ loading: true, analyzed: false, violations: [], docStructure: null }),
  finishAnalysis: (violations, docStructure = null) => set({
    loading: false,
    analyzed: true,
    violations,
    docStructure,
    showAnnotations: true,
    // If docStructure is present (PPTX), update total pages from it
    totalPages: docStructure?.total_slides || undefined
  }),
}));
