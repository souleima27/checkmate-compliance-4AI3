"use client";

import { Download, CheckCircle, FileText, LayoutDashboard } from "lucide-react";
import { useReviewStore } from "@/stores/reviewStore";
import { useRouter } from "next/navigation";
import { generatePDFReport } from "@/utils/pdfGenerator";

export default function ReviewActions() {
  const { analyzed, fileName, violations } = useReviewStore();
  const router = useRouter();

  return (
    <div className="text-center my-10">
      <div className="flex items-center justify-center gap-4">


        <button
          className="px-8 py-4 bg-green-700 text-white font-bold rounded-lg hover:bg-green-800 shadow flex items-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={!analyzed}
          onClick={() => {
            if (!fileName) return;
            generatePDFReport(fileName, violations);
          }}
        >
          <FileText size={20} /> Télécharger Rapport
        </button>

        <button
          className="px-8 py-4 bg-green-600 text-white font-bold rounded-lg hover:bg-green-700 shadow flex items-center gap-3"
          onClick={() => {
            router.push('/dashboard');
          }}
        >
          <LayoutDashboard size={20} /> Tableau de bord
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
