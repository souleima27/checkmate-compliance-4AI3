"use client";
import { useReviewStore } from "@/stores/reviewStore";

export default function ReviewMetrics() {
  const { analyzed, violations, auditResults } = useReviewStore();

  const counts = {
    Structure: violations.filter(v => v.scope === "Structure").length,
    Contexte: violations.filter(v => v.scope === "Contexte").length,
    Fonds: violations.filter(v => v.scope === "Fonds").length,
    Disclaimers: violations.filter(v => v.scope === "Disclaimers").length,
    Total: violations.length
  };

  if (!analyzed) {
    return (
      <div className="rounded-xl p-4 mb-8 border border-green-200 bg-[#F1F8F4]">
        <p className="text-sm text-black/70">Les KPI apparaîtront après l’analyse.</p>
      </div>
    );
  }

  const metrics = [
    { label: "Violations Structurelles", value: counts.Structure },
    { label: "Violations Contexte", value: counts.Contexte },
    { label: "Violations Fonds", value: counts.Fonds },
    { label: "Violations Disclaimers", value: counts.Disclaimers },
    { label: "Total violations", value: counts.Total },
  ];

  if (auditResults && auditResults.global_metrics) {
    const gm = auditResults.global_metrics;
    metrics.push({ label: "Taux de Conformité", value: `${(gm.conformity_rate * 100).toFixed(1)}%` });
    metrics.push({ label: "Similarité Moyenne", value: gm.avg_cosine_similarity?.toFixed(2) || "N/A" });
    metrics.push({ label: "Recall@3 Moyen", value: gm.avg_recall_at_3?.toFixed(2) || "N/A" });
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
      {metrics.map((m) => (
        <div
          key={m.label}
          className="bg-green-50 border border-green-300 p-4 rounded-xl shadow-sm text-center"
        >
          <p className="text-3xl font-bold text-green-700">{m.value}</p>
          <p className="text-sm text-black/70">{m.label}</p>
        </div>
      ))}
    </div>
  );
}
