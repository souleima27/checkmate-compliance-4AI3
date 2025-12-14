"use client";

import { useReviewStore } from "@/stores/reviewStore";
import { ChevronLeft, ChevronRight, Eye, EyeOff } from "lucide-react";

export default function ReviewPreview() {
  const {
    fileName,
    fileType,
    totalPages,
    currentPage,
    analyzed,
    showAnnotations,
    violations,
    filterScopes,
    docStructure,
    setPage,
    setShowAnnotations,
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

      {/* JSON-based viewer surface */}
      <div className="relative h-[600px] flex justify-center rounded-lg border border-green-300 bg-gray-100 overflow-auto">
        {fileType === "pptx" && docStructure ? (
          <div className="w-full h-full bg-white p-8 overflow-auto">
            {(() => {
              const slideData = docStructure.slides?.[currentPage - 1];
              if (!slideData)
                return <div className="text-center text-gray-400 mt-20">Slide {currentPage} introuvable</div>;

              return (
                <div className="space-y-6">
                  <div className="border-b pb-2 mb-4">
                    <h3 className="text-lg font-bold text-gray-800">Slide {slideData.slide_number}</h3>
                    <p className="text-xs text-gray-500">Layout: {slideData.layout}</p>
                  </div>

                  {slideData.content?.map((item: any) => (
                    <div key={item.id} className="mb-4">
                      {item.type === "text" && (
                        <p className="text-gray-800 whitespace-pre-wrap leading-relaxed">{item.text}</p>
                      )}
                      {item.type === "table" && (
                        <div className="overflow-x-auto border rounded-lg">
                          <table className="min-w-full divide-y divide-gray-200">
                            <tbody className="bg-white divide-y divide-gray-200">
                              {item.data?.map((row: any[], rIdx: number) => (
                                <tr key={rIdx}>
                                  {row.map((cell: any, cIdx: number) => (
                                    <td key={cIdx} className="px-3 py-2 text-sm text-gray-700 border-r">
                                      {cell}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                      {item.type === "image_text" && (
                        <div className="bg-gray-50 p-3 italic text-xs text-gray-600 border-l-4 border-gray-300">
                          {item.text}
                        </div>
                      )}
                    </div>
                  ))}

                  {slideData.images?.length > 0 && (
                    <div className="mt-6 pt-4 border-t border-dashed">
                      <p className="text-xs font-semibold text-gray-500 mb-2">Images détectées sur ce slide :</p>
                      <div className="grid grid-cols-2 gap-4">
                        {slideData.images.map((img: any) => (
                          <div key={img.id} className="bg-gray-100 rounded p-2 text-center text-xs text-gray-600">
                            Image: {img.name}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-black/60">
            Aucun aperçu disponible — utilisez l’analyse JSON.
          </div>
        )}

        {/* Overlay annotations */}
        {analyzed && showAnnotations && pageViolations.map((v) => (
          <div
            key={v.id}
            className="absolute border-2 border-red-500/80 bg-red-200/20 z-10"
            style={{ left: v.bbox?.x ?? 40, top: v.bbox?.y ?? 80, width: v.bbox?.w ?? 160, height: v.bbox?.h ?? 60 }}
            title={`${v.scope}: ${v.title}`}
          />
        ))}
      </div>

      {/* Pager */}
      <div className="mt-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            className="px-3 py-2 rounded-lg bg-green-700 text-white hover:bg-green-800 flex items-center gap-2 disabled:opacity-50"
            onClick={() => setPage(currentPage - 1)}
            disabled={currentPage <= 1}
          >
            <ChevronLeft size={18} /> Précédent
          </button>

          <button
            className="px-3 py-2 rounded-lg bg-green-700 text-white hover:bg-green-800 flex items-center gap-2 disabled:opacity-50"
            onClick={() => setPage(currentPage + 1)}
            disabled={currentPage >= totalPages}
          >
            Suivant <ChevronRight size={18} />
          </button>

          <p className="text-sm text-black/70 ml-3">Slide/Page {currentPage} / {totalPages}</p>
        </div>

        <div className="flex items-center gap-3">
          {analyzed && (
            <button
              className="px-3 py-2 rounded-lg bg-green-600 text-white hover:bg-green-700 flex items-center gap-2"
              onClick={() => setShowAnnotations(!showAnnotations)}
            >
              {showAnnotations ? <EyeOff size={18} /> : <Eye size={18} />} {showAnnotations ? "Masquer les annotations" : "Afficher les annotations"}
            </button>
          )}
        </div>
      </div>

      {analyzed && pageViolations.length > 0 && (
        <p className="mt-3 text-sm text-red-700">{pageViolations.length} violation(s) sur cette page.</p>
      )}

      {!analyzed && (
        <p className="mt-3 text-sm text-black/60 text-center">Les annotations s’afficheront après l’analyse.</p>
      )}
    </div>
  );
}
