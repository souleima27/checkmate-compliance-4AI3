"use client";

import { useReviewStore } from "@/stores/reviewStore";
import { ChevronLeft, ChevronRight, Eye, EyeOff, Loader2 } from "lucide-react";
import { useState, useEffect, useMemo } from "react";
import dynamic from "next/dynamic";

// ✅ Dynamic import of react-pdf components with SSR disabled
const Document = dynamic(() => import("react-pdf").then(mod => mod.Document), { ssr: false });
const Page = dynamic(() => import("react-pdf").then(mod => mod.Page), { ssr: false });

import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
    setPage,
    setShowAnnotations,
    setFile,
  } = useReviewStore();

  const [loading, setLoading] = useState(false);
  const [pdfSource, setPdfSource] = useState<string | null>(null);

  // ✅ Configure pdfjs worker only on client
  useEffect(() => {
    import("react-pdf").then(mod => {
      mod.pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${mod.pdfjs.version}/build/pdf.worker.min.mjs`;
    });
  }, []);

  const encodedName = encodeURIComponent(fileName ?? "");
  const remoteUrl = fileName ? `${API_URL}/uploads/${encodedName}` : null;
  const officeEmbedUrl = `https://view.officeapps.live.com/op/embed.aspx?src=${API_URL}/uploads/${encodedName}`;


  // Only fetch blob for PDFs
  useEffect(() => {
    if (!remoteUrl || fileType !== "pdf") return;
    setLoading(true);
    fetch(remoteUrl)
      .then((res) => res.blob())
      .then((blob) => {
        setPdfSource(URL.createObjectURL(blob));
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching PDF blob:", err);
        setLoading(false);
      });
  }, [remoteUrl, fileType]);

  const pageViolations = violations.filter(
    (v) =>
      v.page === currentPage &&
      (filterScopes.length === 0 || filterScopes.includes(v.scope))
  );

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    setFile({
      fileName: fileName || "",
      fileType: "pdf", // only set when actually PDF
      totalPages: numPages,
    });
    setLoading(false);
  }

  return (
    <div className="rounded-xl p-6 shadow-sm mb-8 border border-green-200 bg-[#F1F8F4]">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-black">Aperçu de la présentation</h2>
        <p className="text-sm text-black/70">
          {fileName} — {fileType?.toUpperCase()}
        </p>
      </div>

      {/* Viewer surface */}
      <div className="relative h-[600px] flex justify-center rounded-lg border border-green-300 bg-gray-100 overflow-auto">
        {!pdfSource && fileType === "pdf" && (
          <div className="flex items-center justify-center h-full text-black/60">
            {loading ? (
              <div className="flex gap-2">
                <Loader2 className="animate-spin" /> Chargement...
              </div>
            ) : (
              "Aucun fichier chargé"
            )}
          </div>
        )}

        {/* PDF viewer */}
        {fileType === "pdf" && pdfSource && (
          <Document
            file={pdfSource}
            onLoadSuccess={onDocumentLoadSuccess}
            className="flex justify-center"
          >
            <Page
              pageNumber={currentPage}
              width={600}
              renderTextLayer={false}
              renderAnnotationLayer={false}
            />
          </Document>
        )}

        {/* DOCX/PPTX viewer */}
        {(fileType === "pptx" || fileType === "docx") && officeEmbedUrl && (
          <iframe
            src={officeEmbedUrl}
            width="100%"
            height="100%"
            style={{ border: "none" }}
            sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
          />
        )}

        {/* Overlay annotations */}
        {analyzed &&
          showAnnotations &&
          pageViolations.map((v) => (
            <div
              key={v.id}
              className="absolute border-2 border-red-500/80 bg-red-200/20 z-10"
              style={{
                left: v.bbox?.x ?? 40,
                top: v.bbox?.y ?? 80,
                width: v.bbox?.w ?? 160,
                height: v.bbox?.h ?? 60,
              }}
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
