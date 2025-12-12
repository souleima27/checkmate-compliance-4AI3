"use client";
import { Download, CheckCircle } from "lucide-react";
import { useReviewStore } from "@/stores/reviewStore";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ReviewActions() {
  const { analyzed, fileName } = useReviewStore();

  const handleDownloadAnnotated = async () => {
    if (!fileName) return;

    try {
      const res = await fetch(`${API_URL}/api/download/${encodeURIComponent(fileName)}`);
      if (!res.ok) {
        alert("Erreur lors du téléchargement du fichier annoté.");
        return;
      }
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      // ⚠️ plus tard tu peux changer en `${basename}_annotated.ext`
      a.download = fileName;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Download error:", err);
      alert("Impossible de télécharger le fichier annoté.");
    }
  };

  return (
    <div className="text-center my-10">
      <div className="flex items-center justify-center gap-4">
        <button
          className="px-8 py-4 bg-green-700 text-white font-bold rounded-lg hover:bg-green-800 shadow flex items-center gap-3"
          disabled={!analyzed}
          onClick={handleDownloadAnnotated}
        >
          <Download size={20} /> Télécharger la version annotée
        </button>

        <button
          className="px-8 py-4 bg-green-600 text-white font-bold rounded-lg hover:bg-green-700 shadow flex items-center gap-3"
          onClick={() => {
            // CTA secondaire (ex: poursuivre la revue, ou aller vers une page dédiée annotate)
            alert("Continuer la revue (simulation).");
          }}
        >
          <CheckCircle size={20} /> Continuer la revue
        </button>
      </div>

      {!analyzed && (
        <p className="mt-3 text-sm text-black/60">
          Les actions seront disponibles après l’analyse.
        </p>
      )}
    </div>
  );
}
