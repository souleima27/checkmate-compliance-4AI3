"use client";
import { useReviewStore } from "@/stores/reviewStore";
import { PlayCircle, LoaderCircle } from "lucide-react";

const scopes = [
  "Structure", "Contexte", "Fonds", "Disclaimers"
] as const;

export default function ReviewSidebar() {
  const { scopes: chosen, toggleScope, startAnalysis, finishAnalysis, loading, analyzed, setFilterScopes } = useReviewStore();

  const handleAnalyze = async () => {
    startAnalysis();
    // Simulation: calcule violations dummy
    setTimeout(() => {
      const fakeViolations = [
        { id: "v1", page: 2, scope: "Disclaimers", title: "Disclaimer manquant", description: "Le disclaimer ESG doit être présent.", severity: "medium", bbox: { x: 60, y: 120, w: 180, h: 60 } },
        { id: "v2", page: 4, scope: "Prospectus", title: "Contradiction Prospectus", description: "La performance annoncée dépasse la limite.", severity: "high", bbox: { x: 120, y: 200, w: 220, h: 80 } },
        { id: "v3", page: 1, scope: "Structure", title: "Titre non conforme", description: "Structure de page non conforme aux standards.", severity: "low", bbox: { x: 40, y: 80, w: 140, h: 50 } },
      ];
      finishAnalysis(fakeViolations as any);
      setFilterScopes([]);
    }, 1200);
  };

  return (
    <aside className="border rounded-xl p-6 bg-[#F1F8F4] shadow-sm sticky top-10">
      <h3 className="text-lg font-bold mb-4 text-black">Compliance Sidebar</h3>

      <div className="mb-4">
        <p className="text-sm text-black/70 mb-2">Sélection des contrôles</p>
        <ul className="space-y-2">
          {scopes.map((s) => (
            <li key={s} className="flex items-center justify-between">
              <label className="flex items-center gap-3">
                <input
                  type="checkbox"
                  className="accent-green-700 w-4 h-4"
                  checked={!!chosen[s]}
                  onChange={() => toggleScope(s)}
                />
                <span className="text-black">{s}</span>
              </label>
            </li>
          ))}
        </ul>
      </div>

      <button
        onClick={handleAnalyze}
        disabled={loading}
        className="w-full py-3 flex items-center gap-2 justify-center bg-green-700 text-white font-bold rounded-lg hover:bg-green-800 transition"
      >
        {loading ? <LoaderCircle size={20} className="animate-spin" /> : <PlayCircle size={20} />}
        {loading ? "Analyse en cours..." : "Lancer l’analyse"}
      </button>

      {loading && (
        <p className="text-center mt-3 text-sm text-black/70">Veuillez patienter ⏳</p>
      )}

      {analyzed && (
        <div className="mt-6">
          <p className="text-sm text-black/70 mb-2">Filtrer les annotations</p>
          <div className="flex flex-wrap gap-2">
            {scopes.map((s) => (
              <button
                key={s}
                className="px-3 py-1 rounded-full border border-green-300 bg-green-50 text-black hover:bg-green-100"
                onClick={() => {
                  // toggle filtre
                  const exists = s in chosen && chosen[s];
                  if (exists) {
                    setFilterScopes(Object.keys(chosen).filter((x) => x !== s) as any);
                  } else {
                    setFilterScopes([...Object.keys(chosen), s as any] as any);
                  }
                }}
              >
                {s}
              </button>
            ))}
            <button
              className="px-3 py-1 rounded-full bg-green-700 text-white hover:bg-green-800"
              onClick={() => setFilterScopes([])}
            >
              Réinitialiser
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}
