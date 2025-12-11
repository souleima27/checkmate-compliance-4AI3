"use client";

import React from "react";
import { Upload, FileText } from "lucide-react";
import { useUploadStore } from "@/stores/uploadStore";

export default function UploadProspectus() {
  const { prospectusFile, setProspectusFile } = useUploadStore();

  return (
    <div className="border border-blue-300 bg-blue-50 rounded-xl p-6 shadow-sm hover:shadow transition">
      <div className="flex items-center gap-2 mb-4">
        <FileText className="text-blue-600" size={22} />
        <h2 className="font-semibold text-lg">Prospectus (Document Réglementaire)</h2>
      </div>

      <p className="text-sm text-gray-700 mb-3">
        Formats : <span className="font-medium">PDF, DOCX</span>
      </p>

      <label className="cursor-pointer flex flex-col items-center justify-center border-2 border-dashed border-blue-400 rounded-lg h-40 hover:bg-blue-100 transition">
        <Upload className="text-blue-600 mb-2" size={32} />
        <span className="font-medium">
          {prospectusFile ? prospectusFile.name : "Déposer ou sélectionner un fichier"}
        </span>
        <input
          type="file"
          accept=".pdf,.docx"
          className="hidden"
          onChange={(e) => setProspectusFile(e.target.files?.[0] || null)}
        />
      </label>
    </div>
  );
}
