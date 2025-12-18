"use client";

import { useReviewStore } from "@/stores/reviewStore";
import { ChevronLeft, ChevronRight, Eye, EyeOff, Loader2, Upload, Download } from "lucide-react";
import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";

// ✅ Dynamic imports to disable SSR for react-pdf (fixes DOMMatrix error)
const Document = dynamic(() => import("react-pdf").then((mod) => mod.Document), { ssr: false });
const Page = dynamic(() => import("react-pdf").then((mod) => mod.Page), { ssr: false });

// ✅ Styles (safe on client)
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";

const getApiBaseUrl = (): string => {
  if (typeof window !== "undefined") {
    const hostname = window.location.hostname;
    return `http://${hostname}:8000`;
  }
  return "http://localhost:8000";
};

export default function ReviewPreview() {
  const router = useRouter();
  const {
    fileName,
    fileType: storeFileType,
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

  // Derive fileType from fileName extension (more reliable than store)
  const fileType = fileName?.toLowerCase().endsWith(".pptx") ? "pptx"
    : fileName?.toLowerCase().endsWith(".pdf") ? "pdf"
      : storeFileType || "pdf";

  const [loading, setLoading] = useState(false);
  const [pdfSource, setPdfSource] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ✅ Configure worker only on client
  useEffect(() => {
    import("react-pdf").then((mod) => {
      const pdfjs = mod.pdfjs;
      // Precise URL for pdfjs v5 ESM worker
      const workerUrl = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.mjs`;
      console.log(`[ReviewPreview] Setting up PDF worker: ${workerUrl}`);
      pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
    }).catch(err => {
      console.error("[ReviewPreview] Failed to load react-pdf/pdfjs:", err);
    });
  }, []);

  // Fetch PDF source (Convert PPTX -> PDF or fetch PDF directly)
  useEffect(() => {
    // Check for invalid or default filename
    if (!fileName || fileName === "document") {
      setPdfSource(null);
      return;
    }

    setLoading(true);
    setError(null);

    const API_URL = getApiBaseUrl();

    // Determine the URL: direct file or conversion endpoint
    // IMPORTANT: use percent encoding for filenames with spaces
    // For PPTX/DOCX, use the new LibreOffice conversion endpoint
    const url = fileType === "pptx" || (fileName.endsWith(".docx"))
      ? `${API_URL}/api/convert/${encodeURIComponent(fileName)}`
      : `${API_URL}/uploads/${encodeURIComponent(fileName)}`;

    console.log(`[ReviewPreview] fetching document: ${url}`);

    fetch(url)
      .then((res) => {
        if (!res.ok) {
          console.error(`[ReviewPreview] Fetch failed: ${res.status} ${res.statusText} for URL: ${url}`);
          throw new Error(`Fetch failed: ${res.statusText}`);
        }
        return res.blob();
      })
      .then((blob) => {
        console.log(`[ReviewPreview] blob received: size=${blob.size}, type=${blob.type}`);
        if (blob.size < 100) {
          console.warn("[ReviewPreview] warning: blob size is very small, might be an error message instead of PDF");
        }
        const objectUrl = URL.createObjectURL(blob);
        setPdfSource(objectUrl);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching document blob:", err);
        setError(err.message);
        setLoading(false);
      });
  }, [fileName, fileType]);

  const pageViolations = violations.filter(
    (v) => v.page === currentPage && (filterScopes.length === 0 || filterScopes.includes(v.scope))
  );

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    // Keep fileType generally as is, but we are viewing it as a PDF
    setFile({ fileName: fileName || "", fileType: fileType, totalPages: numPages });
    setLoading(false);
  }

  const handleDownloadAnnotated = async () => {
    if (!fileName) return;
    const API_URL = getApiBaseUrl();
    try {
      setDownloading(true);
      const res = await fetch(`${API_URL}/api/download-annotated`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fileName: fileName,
          violations: violations
        })
      });

      if (!res.ok) throw new Error("Download failed");

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `annotated_${fileName}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (e) {
      console.error(e);
      alert("Erreur lors du téléchargement");
    } finally {
      setDownloading(false);
    }
  };

  // Handle Empty State / Refresh (file lost from memory)
  if (!fileName || fileName === "document") {
    return (
      <div className="rounded-xl p-10 shadow-sm mb-8 border border-green-200 bg-[#F1F8F4] flex flex-col items-center text-center">
        <h2 className="text-xl font-semibold text-gray-800 mb-2">Aucun document chargé</h2>
        <p className="text-gray-600 mb-6 max-w-md">
          Le document n'est plus en mémoire (probablement suite à un rafraîchissement de page).
          Veuillez le recharger.
        </p>
        <button
          onClick={() => router.push('/upload')}
          className="inline-flex items-center gap-2 px-6 py-3 bg-green-700 text-white rounded-lg hover:bg-green-800 font-medium transition shadow-sm"
        >
          <Upload size={20} />
          Retourner à l'upload
        </button>
      </div>
    );
  }

  return (
    <div className="rounded-xl p-6 shadow-sm mb-8 border border-green-200 bg-[#F1F8F4]">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-4">
        <div>
          <h2 className="text-xl font-semibold text-black">Aperçu de la présentation</h2>
          <p className="text-sm text-black/70">
            {fileName} — {fileType?.toUpperCase()}
          </p>
        </div>

        {/* Download Button */}
        {analyzed && (
          <button
            onClick={handleDownloadAnnotated}
            disabled={downloading}
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium transition"
          >
            {downloading ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
            {downloading ? "Génération..." : "Télécharger avec Notes"}
          </button>
        )}
      </div>

      {/* Viewer surface */}
      <div className="relative h-[600px] flex justify-center rounded-lg border border-green-300 bg-gray-100 overflow-auto">

        {/* Loading State */}
        {loading && !pdfSource && (
          <div className="flex flex-col items-center justify-center h-full text-black/60 gap-4">
            <Loader2 className="animate-spin text-green-700" size={48} />
            <div className="text-center">
              <p className="font-semibold text-green-800 text-lg mb-1">
                {fileType === "pptx" ? "Conversion du PPTX en PDF..." : "Chargement du document..."}
              </p>
              {fileType === "pptx" && (
                <p className="text-sm text-gray-500 max-w-xs mx-auto">
                  Conversion serveur via LibreOffice pour un rendu fidèle.
                </p>
              )}
            </div>
          </div>
        )}

        {/* Error State */}
        {(!pdfSource && !loading) || error ? (
          <div className="flex flex-col items-center justify-center h-full text-red-500 gap-3">
            <p className="font-medium">Erreur lors du chargement de l'aperçu/conversion.</p>
            {error && <p className="text-sm text-red-400">{error}</p>}
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-white border border-gray-300 rounded hover:bg-gray-50 text-sm text-gray-700"
            >
              Réessayer
            </button>
          </div>
        ) : null}

        {/* PDF Viewer (Used for both PDF and Converted PPTX) */}
        {pdfSource && (
          <Document
            file={pdfSource}
            onLoadSuccess={onDocumentLoadSuccess}
            onLoadError={(error) => console.error("PDF Load Error:", error)}
            onSourceError={(error) => console.error("PDF Source Error:", error)}
            loading={<div className="flex items-center justify-center p-10"><Loader2 className="animate-spin text-green-700" size={32} /></div>}
            error={<div className="text-red-500 mt-10">Erreur de rendu PDF</div>}
            className="flex justify-center"
          >
            {totalPages > 0 && (
              <Page
                key={`page_${currentPage}`} // Force re-render on page change to avoid stale state
                pageNumber={currentPage}
                width={600}
                renderTextLayer={false}
                renderAnnotationLayer={false}
              />
            )}
          </Document>
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
