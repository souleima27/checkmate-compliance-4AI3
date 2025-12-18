"use client";
import React, { useState } from "react";
import { CheckCircle, AlertTriangle, XCircle, ThumbsUp, ThumbsDown } from "lucide-react";
import { useReviewStore } from "@/stores/reviewStore";

const getApiBaseUrl = (): string => {
    if (typeof window !== "undefined") {
        const hostname = window.location.hostname;
        return `http://${hostname}:8000/api`;
    }
    return "http://localhost:8000/api";
};

export default function ReviewAlignment() {
    const { analyzed, auditResults } = useReviewStore();
    const [feedbackState, setFeedbackState] = useState<Record<string, "like" | "dislike">>({});

    if (!analyzed || !auditResults) return null;

    const findings = auditResults.audit_results || {};

    const handleFeedback = async (question: string, type: "like" | "dislike", finding: any) => {
        setFeedbackState(prev => ({ ...prev, [question]: type }));
        try {
            const baseUrl = getApiBaseUrl();
            await fetch(`${baseUrl}/feedback`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    type: "violation",
                    id: question,
                    feedback: type,
                    details: {
                        description: finding.justification,
                        rules: question,
                        agent: "Alignment Agent",
                        timestamp: new Date().toISOString()
                    }
                })
            });
        } catch (error) {
            console.error("Error sending feedback:", error);
        }
    };

    return (
        <div className="bg-[#F1F8F4] border border-green-200 rounded-xl p-6 shadow-sm mb-8">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-black">
                <CheckCircle className="text-green-700" /> Alignement & Cohérence
            </h2>

            {Object.keys(findings).length === 0 ? (
                <p className="text-black/70">Aucune donnée d'alignement disponible.</p>
            ) : (
                <ul className="space-y-4">
                    {Object.entries(findings).map(([question, finding]: [string, any]) => {
                        let borderColor = "border-gray-300";
                        let bgColor = "bg-gray-50";
                        let icon = <CheckCircle className="text-gray-500" />;

                        if (finding.verdict === "conforme") {
                            borderColor = "border-green-300";
                            bgColor = "bg-green-50";
                            icon = <CheckCircle className="text-green-600" />;
                        } else if (finding.verdict === "conforme avec réserves") {
                            borderColor = "border-yellow-300";
                            bgColor = "bg-yellow-50";
                            icon = <AlertTriangle className="text-yellow-600" />;
                        } else {
                            borderColor = "border-red-300";
                            bgColor = "bg-red-50";
                            icon = <XCircle className="text-red-600" />;
                        }

                        return (
                            <li key={question} className={`border rounded-lg p-4 ${bgColor} ${borderColor}`}>
                                <div className="flex items-start gap-3">
                                    <div className="mt-1">{icon}</div>
                                    <div className="flex-1">
                                        <div className="flex justify-between items-start">
                                            <div>
                                                <p className="font-bold text-black">{question}</p>
                                                <p className="text-sm font-semibold mt-1 text-black/80">Verdict: {finding.verdict}</p>
                                            </div>
                                            <div className="flex gap-2">
                                                <button
                                                    onClick={() => handleFeedback(question, "like", finding)}
                                                    className={`p-1 rounded hover:bg-white/50 transition ${feedbackState[question] === 'like' ? 'text-green-600' : 'text-gray-400'}`}
                                                    title="Correct"
                                                >
                                                    <ThumbsUp size={16} />
                                                </button>
                                                <button
                                                    onClick={() => handleFeedback(question, "dislike", finding)}
                                                    className={`p-1 rounded hover:bg-white/50 transition ${feedbackState[question] === 'dislike' ? 'text-red-600' : 'text-gray-400'}`}
                                                    title="Incorrect"
                                                >
                                                    <ThumbsDown size={16} />
                                                </button>
                                            </div>
                                        </div>
                                        <p className="text-sm text-black/70 mt-1">{finding.justification}</p>
                                        {finding.alerts && finding.alerts.length > 0 && (
                                            <div className="mt-2 text-xs text-red-600">
                                                {finding.alerts.map((alert: string, i: number) => (
                                                    <p key={i}>⚠️ {alert}</p>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </li>
                        );
                    })}
                </ul>
            )}
        </div>
    );
}
