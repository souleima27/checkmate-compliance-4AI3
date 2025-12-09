"use client";
import { create } from "zustand";

type ReviewScope = "Structure" | "Contexte" | "Fonds" | "Disclaimers";

type Violation = {
  id: string;
  page: number;
  scope: ReviewScope;
  title: string;
  description: string;
  severity: "low" | "medium" | "high";
  bbox?: { x: number; y: number; w: number; h: number }; // pour overlay
};

type ReviewState = {
  fileName?: string;
  fileType?: "pdf" | "pptx" | "docx";
  totalPages: number;
  currentPage: number;
  analyzed: boolean;
  loading: boolean;
  scopes: Record<ReviewScope, boolean>;
  violations: Violation[];
  showAnnotations: boolean;
  filterScopes: ReviewScope[];

  setFile: (f: { fileName: string; fileType: ReviewState["fileType"]; totalPages: number }) => void;
  setPage: (p: number) => void;
  toggleScope: (s: ReviewScope) => void;
  setScopes: (next: Record<ReviewScope, boolean>) => void;
  startAnalysis: () => void;
  finishAnalysis: (violations: Violation[]) => void;
  setShowAnnotations: (v: boolean) => void;
  setFilterScopes: (arr: ReviewScope[]) => void;
};

export const useReviewStore = create<ReviewState>((set) => ({
  fileName: "presentation.pdf",
  fileType: "pdf",
  totalPages: 12,
  currentPage: 1,
  analyzed: false,
  loading: false,
  scopes: {
    Structure: true,
    Contexte: true,
    Fonds: true,
    Disclaimers: true,  
  },
  violations: [],
  showAnnotations: false,
  filterScopes: [],

  setFile: (f) => set({ fileName: f.fileName, fileType: f.fileType, totalPages: f.totalPages, currentPage: 1 }),
  setPage: (p) => set((st) => ({ currentPage: Math.max(1, Math.min(st.totalPages, p)) })),
  toggleScope: (s) => set((st) => ({ scopes: { ...st.scopes, [s]: !st.scopes[s] } })),
  setScopes: (next) => set({ scopes: next }),
  startAnalysis: () => set({ loading: true, analyzed: false }),
  finishAnalysis: (violations) => set({ loading: false, analyzed: true, violations, showAnnotations: true }),
  setShowAnnotations: (v) => set({ showAnnotations: v }),
  setFilterScopes: (arr) => set({ filterScopes: arr }),
}));
