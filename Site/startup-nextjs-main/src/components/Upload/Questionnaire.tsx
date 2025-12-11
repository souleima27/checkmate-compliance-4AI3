"use client";

import React from "react";
import { useUploadStore } from "@/stores/uploadStore";
import { Building2, Users, Sparkles, Package, Shield } from "lucide-react";

interface QuestionItemProps {
  icon: React.ReactNode;
  question: string;
  value: boolean;
  onChange: (value: boolean) => void;
}

function QuestionItem({ icon, question, value, onChange }: QuestionItemProps) {
  return (
    <div className="flex items-center justify-between py-4 border-b border-gray-100 dark:border-gray-700 last:border-b-0">
      <div className="flex items-center gap-3">
        <div className="text-green-600 dark:text-green-400">{icon}</div>
        <span className="text-gray-700 dark:text-gray-200 text-sm font-medium">
          {question}
        </span>
      </div>
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => onChange(true)}
          className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200 ${
            value
              ? "bg-green-600 text-white shadow-md shadow-green-200 dark:shadow-green-900"
              : "bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600"
          }`}
        >
          Oui
        </button>
        <button
          type="button"
          onClick={() => onChange(false)}
          className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200 ${
            !value
              ? "bg-red-500 text-white shadow-md shadow-red-200 dark:shadow-red-900"
              : "bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600"
          }`}
        >
          Non
        </button>
      </div>
    </div>
  );
}

export default function Questionnaire() {
  const { metadata, setMetadata } = useUploadStore();

  const updateMetadataField = (field: string, value: string | boolean) => {
    setMetadata({ ...metadata, [field]: value });
  };

  return (
    <div className="max-w-3xl mx-auto mt-12 mb-8">
      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6 transition-all duration-300 hover:shadow-2xl">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6 pb-4 border-b border-gray-200 dark:border-gray-700">
          <div className="p-2 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl shadow-lg">
            <Shield className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-800 dark:text-white">
              Questionnaire pré-analyse
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Ces informations permettent d'affiner l'analyse de conformité
            </p>
          </div>
        </div>

        {/* Société de Gestion */}
        <div className="mb-6">
          <label className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            <Building2 size={16} className="text-green-600" />
            Société de Gestion
          </label>
          <input
            type="text"
            value={metadata["Société de Gestion"] || "ODDO BHF ASSET MANAGEMENT SAS"}
            onChange={(e) => updateMetadataField("Société de Gestion", e.target.value)}
            className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 text-gray-800 dark:text-white focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all duration-200"
            placeholder="Nom de la société de gestion"
          />
        </div>

        {/* Questions with toggles */}
        <div className="space-y-1">
          <QuestionItem
            icon={<Shield size={18} />}
            question="Est-ce que le produit fait partie de la Sicav d'Oddo ?"
            value={metadata["Est ce que le produit fait partie de la Sicav d'Oddo"] as boolean}
            onChange={(v) => updateMetadataField("Est ce que le produit fait partie de la Sicav d'Oddo", v)}
          />
          <QuestionItem
            icon={<Users size={18} />}
            question="Le client est-il un professionnel ?"
            value={metadata["Le client est-il un professionnel"] as boolean}
            onChange={(v) => updateMetadataField("Le client est-il un professionnel", v)}
          />
          <QuestionItem
            icon={<Sparkles size={18} />}
            question="Le document fait-il référence à une nouvelle Stratégie ?"
            value={metadata["Le document fait-il référence à une nouvelle Stratégie"] as boolean}
            onChange={(v) => updateMetadataField("Le document fait-il référence à une nouvelle Stratégie", v)}
          />
          <QuestionItem
            icon={<Package size={18} />}
            question="Le document fait-il référence à un nouveau Produit ?"
            value={metadata["Le document fait-il référence à un nouveau Produit"] as boolean}
            onChange={(v) => updateMetadataField("Le document fait-il référence à un nouveau Produit", v)}
          />
        </div>
      </div>
    </div>
  );
}
