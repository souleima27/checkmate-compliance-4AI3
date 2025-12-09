"use client";
import { AlertTriangle } from "lucide-react";
import { useReviewStore } from "@/stores/reviewStore";

export default function ReviewViolations() {
  const { analyzed, violations } = useReviewStore();

  if (!analyzed) return null;

  return (
    <div className="bg-[#F1F8F4] border border-green-200 rounded-xl p-6 shadow-sm mb-8">
      <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-black">
        <AlertTriangle className="text-yellow-700" /> Violations détectées
      </h2>

      {violations.length === 0 ? (
        <p className="text-black/70">Aucune violation détectée.</p>
      ) : (
        <ul className="space-y-4">
          {violations.map((v) => (
            <li key={v.id} className="border rounded-lg p-4 bg-red-50 border-red-300">
              <p className="font-bold text-red-700">
                Page {v.page} — {v.scope}: {v.title}
              </p>
              <p className="text-sm text-black/80">{v.description}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
