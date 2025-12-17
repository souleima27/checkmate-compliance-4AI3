"use client";
import { useReviewStore } from "@/stores/reviewStore";
import {
  CheckCircle,
  RotateCcw
} from "lucide-react";

export default function ReviewSidebar() {
  const {
    violations,
    scopes: allScopes, // Typo in store definition? 'scopes' is boolean map. 
    // Actually store has 'scopes' as Record<Scope, boolean>.
    // But we also have filterScopes.
    filterScopes,
    setFilterScopes
  } = useReviewStore();

  const definedScopes = ["Structure", "Contexte", "Fonds", "Disclaimers", "Glossaires"];

  // Group violations by scope for counts
  const counts: Record<string, number> = {};
  violations.forEach(v => {
    counts[v.scope] = (counts[v.scope] || 0) + 1;
  });

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm sticky top-24">
      <h2 className="text-xl font-bold mb-6">Compliance Sidebar</h2>

      {/* Sélection des contrôles (Filter) */}
      <div className="mb-8">
        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-4">
          Filtres par Scope
        </h3>
        <div className="space-y-3">
          {definedScopes.map((scope) => (
            <label key={scope} className="flex items-center gap-3 cursor-pointer hover:bg-gray-50 p-2 rounded-lg transition">
              <div
                className={`w-5 h-5 rounded border flex items-center justify-center transition ${!filterScopes.length || filterScopes.includes(scope as any)
                  ? "bg-green-600 border-green-600 text-white"
                  : "border-gray-300 bg-white"
                  }`}
                onClick={(e) => {
                  e.preventDefault();
                  // Logic: if empty, clicking one selects it.
                  // If in list, clicking removes it.
                  // Logic optimized:
                  let newScopes = [...filterScopes];
                  if (newScopes.length === 0) {
                    newScopes = [scope as any];
                  } else {
                    if (newScopes.includes(scope as any)) {
                      newScopes = newScopes.filter(s => s !== scope);
                    } else {
                      newScopes.push(scope as any);
                    }
                  }
                  setFilterScopes(newScopes);
                }}
              >
                {(!filterScopes.length || filterScopes.includes(scope as any)) && <CheckCircle size={14} />}
              </div>
              <span className="font-medium text-gray-700">{scope}</span>
              {counts[scope] > 0 && (
                <span className="ml-auto bg-red-100 text-red-700 text-xs font-bold px-2 py-1 rounded-full">
                  {counts[scope]}
                </span>
              )}
            </label>
          ))}
        </div>
      </div>

      {/* Liste des annotations / Violations Summary */}
      <div>
        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-4">
          Violations Détectées ({violations.length})
        </h3>
        <div className="space-y-4 max-h-[400px] overflow-y-auto pr-2">
          {violations.length === 0 ? (
            <p className="text-sm text-gray-500 italic">Aucune violation détectée (ou analyse non lancée).</p>
          ) : (
            violations
              .filter(v => filterScopes.length === 0 || filterScopes.includes(v.scope))
              .map((v) => (
                <div key={v.id} className={`p-3 border rounded-lg text-sm ${v.type === 'missing'
                  ? 'bg-amber-50 border-amber-200'
                  : 'bg-red-50 border-red-100'
                  }`}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      {v.type === 'missing' && (
                        <span className="font-bold text-xs px-2 py-0.5 rounded text-white bg-amber-500">
                          MANQUANT
                        </span>
                      )}
                      <span className={`font-bold text-xs px-2 py-0.5 rounded text-white ${v.scope === 'Disclaimers' ? 'bg-orange-500' :
                          v.scope === 'Structure' ? 'bg-blue-600' :
                            v.scope === 'Contexte' ? 'bg-purple-500' :
                              v.scope === 'Fonds' ? 'bg-green-600' :
                                v.scope === 'Glossaires' ? 'bg-teal-500' : 'bg-red-500'
                        }`}>
                        {v.scope}
                      </span>
                      {v.ruleId && (
                        <span className="text-xs bg-gray-200 text-gray-700 px-2 py-0.5 rounded font-mono">
                          {v.ruleId}
                        </span>
                      )}
                    </div>
                    <span className="text-gray-400 text-xs">
                      {v.type === 'missing' ? 'Document' : `Page ${v.page}`}
                    </span>
                  </div>
                  <p className="font-semibold text-gray-800 mb-1">{v.title}</p>
                  <p className="text-gray-600 leading-relaxed">{v.description}</p>
                </div>
              ))
          )}
        </div>
      </div>

      <div className="mt-8 pt-6 border-t border-gray-100">
        <button
          onClick={() => setFilterScopes([])}
          className="w-full py-2 bg-gray-100 text-gray-700 font-semibold rounded-lg hover:bg-gray-200 transition flex items-center justify-center gap-2"
        >
          <RotateCcw size={18} /> Réinitialiser filtres
        </button>
      </div>
    </div>
  );
}
