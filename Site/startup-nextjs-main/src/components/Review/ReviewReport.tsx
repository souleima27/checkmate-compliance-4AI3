"use client";
import { useReviewStore } from "@/stores/reviewStore";

export default function ReviewReport() {
  const { analyzed, violations } = useReviewStore();

  if (!analyzed) return null;

  const total = violations.length;
  const critical = violations.filter(v => v.severity === "high").length;

  return (
    <div className="bg-[#F1F8F4] border border-green-200 rounded-xl p-6 shadow-sm mb-8">
      <h2 className="text-xl font-semibold mb-4 text-black">Rapport final</h2>
      <p className="text-black/80">
        Le document a été analysé selon les règles du corpus réglementaire. {total > 0 ? `Nous avons identifié ${total} violation(s), dont ${critical} critique(s).` : "Aucune violation détectée."}
      </p>
      <p className="text-black/70 mt-2">
        Veuillez corriger les éléments signalés avant la validation. Les annotations sont visibles directement dans l’aperçu par page/slide.
      </p>
    </div>
  );
}
