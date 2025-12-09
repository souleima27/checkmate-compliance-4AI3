"use client";

import React, { useState } from "react";
import { Upload, FileText } from "lucide-react";

export default function UploadMarketing() {
  const [file, setFile] = useState<File | null>(null);

  return (
    <div className="border border-green-300 bg-green-50 rounded-xl p-6 shadow-sm hover:shadow transition">
      <div className="flex items-center gap-2 mb-4">
        <FileText className="text-green-600" size={22} />
        <h2 className="font-semibold text-lg">Document Marketing</h2>
      </div>

      <p className="text-sm text-gray-700 mb-3">
        Formats : <span className="font-medium">PPTX, DOCX, PDF</span>
      </p>

      <label className="cursor-pointer flex flex-col items-center justify-center border-2 border-dashed border-green-400 rounded-lg h-40 hover:bg-green-100 transition">
        <Upload className="text-green-600 mb-2" size={32} />
        <span className="font-medium">
          {file ? file.name : "Déposer ou sélectionner un fichier"}
        </span>
        <input
          type="file"
          accept=".pptx,.docx,.pdf"
          className="hidden"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
      </label>
    </div>
  );
}
