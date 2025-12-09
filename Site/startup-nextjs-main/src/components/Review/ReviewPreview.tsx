"use client";
import { useReviewStore } from "@/stores/reviewStore";
import { ChevronLeft, ChevronRight, Eye, EyeOff } from "lucide-react";

export default function ReviewPreview() {
  const {
    fileName, fileType, totalPages, currentPage,
    analyzed, showAnnotations, violations, filterScopes,
    setPage, setShowAnnotations
  } = useReviewStore();

  const pageViolations = violations.filter(
    (v) => v.page === currentPage && (filterScopes.length === 0 || filterScopes.includes(v.scope))
  );

  return (
    <div className="rounded-xl p-6 shadow-sm mb-8 border border-green-200 bg-[#F1F8F4]">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-black">Aperçu de la présentation</h2>
        <p className="text-sm text-black/70">
          {fileName} — {fileType?.toUpperCase()}
        </p>
      </div>

      {/* Viewer surface */}
      <div className="relative h-[480px] rounded-lg border border-green-300 bg-green-50 overflow-hidden">
        {/* Placeholder rendu: à remplacer par PDF.js / images slides */}
        <div className="absolute inset-0 flex items-center justify-center text-black/60">
          Page {currentPage} / {totalPages} — rendu {fileType?.toUpperCase()} (slides/pages)
        </div>

        {/* Overlay annotations après analyse */}
        {analyzed && showAnnotations && pageViolations.map((v) => (
          <div
            key={v.id}
            className="absolute border-2 border-red-500/80 bg-red-200/20"
            style={{
              left: v.bbox?.x ?? 40,
              top: v.bbox?.y ?? 80,
              width: v.bbox?.w ?? 160,
              height: v.bbox?.h ?? 60
            }}
            title={`${v.scope}: ${v.title}`}
          />
        ))}
      </div>

      {/* Pager + toggle annotations */}
      <div className="mt-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            className="px-3 py-2 rounded-lg bg-green-700 text-white hover:bg-green-800 flex items-center gap-2"
            onClick={() => setPage(currentPage - 1)}
          >
            <ChevronLeft size={18} /> Précédent
          </button>
          <button
            className="px-3 py-2 rounded-lg bg-green-700 text-white hover:bg-green-800 flex items-center gap-2"
            onClick={() => setPage(currentPage + 1)}
          >
            Suivant <ChevronRight size={18} />
          </button>
          <p className="text-sm text-black/70 ml-3">
            Slide/Page {currentPage} / {totalPages}
          </p>
        </div>

        <div className="flex items-center gap-3">
          {analyzed && (
            <button
              className="px-3 py-2 rounded-lg bg-green-600 text-white hover:bg-green-700 flex items-center gap-2"
              onClick={() => setShowAnnotations(!showAnnotations)}
            >
              {showAnnotations ? <EyeOff size={18} /> : <Eye size={18} />}
              {showAnnotations ? "Masquer les annotations" : "Afficher les annotations"}
            </button>
          )}
        </div>
      </div>

      {analyzed && pageViolations.length > 0 && (
        <p className="mt-3 text-sm text-red-700">
          {pageViolations.length} violation(s) sur cette page.
        </p>
      )}

      {!analyzed && (
        <p className="mt-3 text-sm text-black/60 text-center">
          Les annotations s’afficheront après l’analyse.
        </p>
      )}
    </div>
  );
}
