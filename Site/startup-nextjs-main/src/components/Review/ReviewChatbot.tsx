"use client";
import React, { useEffect, useState, useRef } from "react";
import { Send, Bot, User, Loader2, MessageSquare, X, Minimize2, Maximize2, FileText, Shield, ThumbsUp, ThumbsDown } from "lucide-react";
import { useReviewStore } from "@/stores/reviewStore";

interface Message {
  role: "user" | "bot";
  text: string;
  timestamp: Date;
  sources?: any[];
  metrics?: any;
  id: string;
  feedback?: "like" | "dislike";
}

const getApiBaseUrl = (): string => {
  if (typeof window !== "undefined") {
    const hostname = window.location.hostname;
    return `http://${hostname}:8000/api`;
  }
  return "http://localhost:8000/api";
};

export default function ReviewChatbot() {
  const { auditResults } = useReviewStore();
  const [ephemeral, setEphemeral] = useState<any>(null);
  const [persistent, setPersistent] = useState<any>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "init-1",
      role: "bot",
      text: "Bonjour ! Je suis votre assistant de conformité réglementaire. Je peux vous aider à comprendre les violations détectées, clarifier les règles applicables, et répondre à vos questions sur l'analyse en cours.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Update greeting when audit results are available
  useEffect(() => {
    if (auditResults && auditResults.global_metrics) {
      const conformityRate = (auditResults.global_metrics.conformity_rate * 100).toFixed(1);
      const avgSimilarity = (auditResults.global_metrics.avg_similarity * 100).toFixed(1);

      const alignmentMessage = `Analyse terminée.\n\n📊 **Score de Conformité Global**: ${conformityRate}%\n🔗 **Cohérence Documentaire**: ${avgSimilarity}%\n\nJe suis prêt à répondre à vos questions sur les détails de l'audit.`;

      setMessages(prev => {
        // Avoid adding duplicate alignment messages
        if (prev.some(m => m.text.includes("Score de Conformité Global"))) return prev;
        return [...prev, {
          id: `align-${Date.now()}`,
          role: "bot",
          text: alignmentMessage,
          timestamp: new Date()
        }];
      });
    }
  }, [auditResults]);

  useEffect(() => {
    async function fetchContext() {
      try {
        const baseUrl = getApiBaseUrl();
        const res = await fetch(`${baseUrl}/context`);
        const data = await res.json();
        setEphemeral(data.ephemeral);
        setPersistent(data.persistent);
      } catch (error) {
        console.error("Erreur lors du chargement du contexte:", error);
      }
    }
    fetchContext();
  }, []);

  const handleFeedback = async (messageId: string, type: "like" | "dislike") => {
    setMessages(prev => prev.map(m => m.id === messageId ? { ...m, feedback: type } : m));
    try {
      const baseUrl = getApiBaseUrl();
      await fetch(`${baseUrl}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "chatbot",
          id: messageId,
          feedback: type,
          details: { timestamp: new Date().toISOString() }
        })
      });
    } catch (error) {
      console.error("Error sending feedback:", error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      text: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const baseUrl = getApiBaseUrl();
      const res = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: input,
          ephemeral,
          persistent,
        }),
      });

      const data = await res.json();

      const botMessage: Message = {
        id: `bot-${Date.now()}`,
        role: "bot",
        text: data.answer,
        timestamp: new Date(),
        sources: data.sources,
        metrics: data.metrics,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Erreur chatbot:", error);
      const errorMessage: Message = {
        id: `err-${Date.now()}`,
        role: "bot",
        text: "Désolé, une erreur s'est produite. Veuillez réessayer.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Suggestions de questions prédéfinies
  const suggestedQuestions = [
    "Quelles violations ont été détectées ?",
    "Explique-moi la règle contextuelle appliquée",
    "Comment puis-je corriger ces non-conformités ?",
    "Quelle est la définition selon le glossaire ?",
  ];

  const handleSuggestionClick = (question: string) => {
    setInput(question);
    textareaRef.current?.focus();
  };

  return (
    <>
      {/* Bouton flottant */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 bg-gradient-to-r from-blue-600 to-blue-700 text-white p-4 rounded-full shadow-2xl hover:shadow-blue-500/50 hover:scale-110 transition-all duration-300 z-50 group"
          aria-label="Ouvrir l'assistant conformité"
        >
          <MessageSquare size={28} className="group-hover:rotate-12 transition-transform" />
          <span className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
        </button>
      )}

      {/* Fenêtre du chatbot */}
      {isOpen && (
        <div
          className={`fixed bottom-6 right-6 bg-white rounded-2xl shadow-2xl flex flex-col z-50 border border-gray-200 transition-all duration-300 ${isMinimized ? "w-96 h-16" : "w-[400px] h-[600px]"
            }`}
        >
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-700 text-white p-5 rounded-t-2xl flex items-center justify-between shadow-lg">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="bg-white/20 backdrop-blur-sm p-2.5 rounded-xl">
                  <Shield size={24} className="text-white" />
                </div>
                <span className="absolute -bottom-1 -right-1 w-3 h-3 bg-green-400 rounded-full border-2 border-white"></span>
              </div>
              <div>
                <h3 className="font-bold text-lg">Assistant Conformité</h3>
                <p className="text-xs text-blue-100 flex items-center gap-1">
                  <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                  En ligne • Analyse active
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsMinimized(!isMinimized)}
                className="hover:bg-white/20 p-2 rounded-lg transition"
                aria-label={isMinimized ? "Agrandir" : "Réduire"}
              >
                {isMinimized ? <Maximize2 size={18} /> : <Minimize2 size={18} />}
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="hover:bg-white/20 p-2 rounded-lg transition"
                aria-label="Fermer"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          {!isMinimized && (
            <>
              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-5 space-y-4 bg-gradient-to-b from-gray-50 to-white">
                {messages.length === 1 && (
                  <div className="mb-4">
                    <p className="text-sm text-gray-600 mb-3 font-medium">Questions suggérées :</p>
                    <div className="space-y-2">
                      {suggestedQuestions.map((question, i) => (
                        <button
                          key={i}
                          onClick={() => handleSuggestionClick(question)}
                          className="w-full text-left px-4 py-2.5 bg-white border border-blue-200 rounded-lg hover:border-blue-400 hover:shadow-md transition-all text-sm text-gray-700 hover:text-blue-700 flex items-center gap-2 group"
                        >
                          <FileText size={16} className="text-blue-500 group-hover:scale-110 transition-transform" />
                          {question}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex gap-3 animate-fade-in ${msg.role === "user" ? "flex-row-reverse" : "flex-row"
                      }`}
                  >
                    <div
                      className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center shadow-md ${msg.role === "user"
                        ? "bg-gradient-to-br from-blue-600 to-blue-700 text-white"
                        : "bg-gradient-to-br from-gray-100 to-gray-200 text-gray-700"
                        }`}
                    >
                      {msg.role === "user" ? <User size={20} /> : <Bot size={20} />}
                    </div>
                    <div className="flex-1 max-w-[85%]">
                      <div
                        className={`px-4 py-3 rounded-2xl shadow-sm ${msg.role === "user"
                          ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-tr-none"
                          : "bg-white text-gray-800 border border-gray-200 rounded-tl-none"
                          }`}
                      >
                        <p className="text-[15px] leading-relaxed whitespace-pre-wrap">
                          {msg.text}
                        </p>

                        {/* Affichage des sources */}
                        {msg.sources && msg.sources.length > 0 && (
                          <div className={`mt-3 pt-3 border-t ${msg.role === 'user' ? 'border-blue-500/30' : 'border-gray-200'}`}>
                            <p className={`text-xs font-semibold mb-1 ${msg.role === 'user' ? 'text-blue-100' : 'text-gray-600'}`}>Sources:</p>
                            <ul className="space-y-1">
                              {msg.sources.slice(0, 3).map((s: any, idx: number) => (
                                <li key={idx} className={`text-xs ${msg.role === 'user' ? 'text-blue-100' : 'text-gray-500'}`}>
                                  • {s.document_type}: {s.document_name} ({s.location})
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Affichage des métriques */}
                        {msg.metrics && (
                          <div className={`mt-2 pt-2 border-t ${msg.role === 'user' ? 'border-blue-500/30' : 'border-gray-200'}`}>
                            <p className={`text-xs ${msg.role === 'user' ? 'text-blue-100' : 'text-gray-400'}`}>
                              Recall@3: {msg.metrics.recall_at_3?.toFixed(2)} | Sim: {msg.metrics.avg_relevance_score?.toFixed(2)}
                            </p>
                          </div>
                        )}

                        {/* Feedback Buttons */}
                        {msg.role === "bot" && (
                          <div className="mt-2 flex items-center justify-end gap-2 pt-2 border-t border-gray-100">
                            <button
                              onClick={() => handleFeedback(msg.id, "like")}
                              className={`p-1 rounded hover:bg-gray-100 transition ${msg.feedback === 'like' ? 'text-green-600' : 'text-gray-400'}`}
                              title="Utile"
                            >
                              <ThumbsUp size={14} />
                            </button>
                            <button
                              onClick={() => handleFeedback(msg.id, "dislike")}
                              className={`p-1 rounded hover:bg-gray-100 transition ${msg.feedback === 'dislike' ? 'text-red-600' : 'text-gray-400'}`}
                              title="Pas utile"
                            >
                              <ThumbsDown size={14} />
                            </button>
                          </div>
                        )}
                      </div>
                      <p
                        className={`text-xs mt-1.5 px-1 ${msg.role === "user" ? "text-right text-gray-500" : "text-gray-400"
                          }`}
                      >
                        {msg.timestamp.toLocaleTimeString("fr-FR", {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </p>
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="flex gap-3 animate-fade-in">
                    <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-gray-100 to-gray-200 text-gray-700 flex items-center justify-center shadow-md">
                      <Bot size={20} />
                    </div>
                    <div className="bg-white px-4 py-3 rounded-2xl rounded-tl-none shadow-sm border border-gray-200 flex items-center gap-2">
                      <Loader2 size={18} className="animate-spin text-blue-600" />
                      <span className="text-sm text-gray-600">Analyse en cours...</span>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div className="p-4 border-t border-gray-200 bg-white rounded-b-2xl">
                <div className="flex gap-3">
                  <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Posez votre question sur la conformité..."
                    className="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm placeholder:text-gray-400"
                    rows={2}
                    disabled={isLoading}
                  />
                  <button
                    onClick={handleSend}
                    disabled={!input.trim() || isLoading}
                    className={`px-5 rounded-xl transition-all duration-200 flex items-center justify-center shadow-md ${!input.trim() || isLoading
                      ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                      : "bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:shadow-lg hover:scale-105"
                      }`}
                  >
                    {isLoading ? (
                      <Loader2 size={20} className="animate-spin" />
                    ) : (
                      <Send size={20} />
                    )}
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-2 text-center flex items-center justify-center gap-1">
                  <kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs border border-gray-300">
                    Entrée
                  </kbd>
                  pour envoyer •
                  <kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs border border-gray-300">
                    Shift + Entrée
                  </kbd>
                  nouvelle ligne
                </p>
              </div>
            </>
          )}
        </div>
      )}

      <style jsx>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fade-in {
          animation: fade-in 0.3s ease-out;
        }
      `}</style>
    </>
  );
}