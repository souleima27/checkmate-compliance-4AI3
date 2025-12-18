"use client";
import React, { useState } from "react";
import { AlertTriangle, ThumbsUp, ThumbsDown } from "lucide-react";
import { useReviewStore, Violation } from "@/stores/reviewStore";

const getApiBaseUrl = (): string => {
  if (typeof window !== "undefined") {
    const hostname = window.location.hostname;
    return `http://${hostname}:8000/api`;
  }
  return "http://localhost:8000/api";
};

export default function ReviewViolations() {
  const { analyzed, violations } = useReviewStore();
  const [feedbackState, setFeedbackState] = useState<Record<string, "like" | "dislike">>({});

  const handleFeedback = async (violation: Violation, type: "like" | "dislike") => {
    setFeedbackState(prev => ({ ...prev, [violation.id]: type }));
    try {
      const baseUrl = getApiBaseUrl();
      await fetch(`${baseUrl}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "violation",
          id: violation.id,
          feedback: type,
          details: {
            description: violation.description,
            rules: violation.ruleId || violation.title,
            agent: violation.scope, // Use scope as agent name (e.g. "Disclaimers", "Structure")
            timestamp: new Date().toISOString()
          }
        })
      });
    } catch (error) {
      console.error("Error sending feedback:", error);
    }
  };

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
              <div className="flex justify-between items-start">
                <p className="font-bold text-red-700">
                  Page {v.page} — {v.scope}: {v.title}
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleFeedback(v, "like")}
                    className={`p-1 rounded hover:bg-white/50 transition ${feedbackState[v.id] === 'like' ? 'text-green-600' : 'text-gray-400'}`}
                    title="Correct"
                  >
                    <ThumbsUp size={16} />
                  </button>
                  <button
                    onClick={() => handleFeedback(v, "dislike")}
                    className={`p-1 rounded hover:bg-white/50 transition ${feedbackState[v.id] === 'dislike' ? 'text-red-600' : 'text-gray-400'}`}
                    title="Incorrect"
                  >
                    <ThumbsDown size={16} />
                  </button>
                </div>
              </div>
              <p className="text-sm text-black/80 mt-1">{v.description}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
