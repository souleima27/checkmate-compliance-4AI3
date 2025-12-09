"use client";
import { Download, CheckCircle } from "lucide-react";
import { useReviewStore } from "@/stores/reviewStore";

export default function ReviewActions() {
  const { analyzed } = useReviewStore();

  return (
    <div className="text-center my-10">
      <div className="flex items-center justify-center gap-4">
        <button
          className="px-8 py-4 bg-green-700 text-white font-bold rounded-lg hover:bg-green-800 shadow flex items-center gap-3"
          disabled={!analyzed}
          onClick={() => {
            // Simuler téléchargement
            alert("Téléchargement de la version annotée (simulation).");
          }}
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
