"use client";

import { Upload, FileText, X } from "lucide-react";
import { useUploadStore } from "@/stores/uploadStore";

export default function UploadMarketing() {
  const { marketingFile, setMarketingFile } = useUploadStore();

  const getExtension = (filename: string) => {
    return filename.split('.').pop()?.toUpperCase() || '';
  };

  return (
    <div className="border border-green-300 bg-green-50 rounded-xl p-6 shadow-sm hover:shadow transition">
      <div className="flex items-center gap-2 mb-4">
        <FileText className="text-green-600" size={22} />
        <h2 className="font-semibold text-lg">Document Marketing</h2>
      </div>

      <p className="text-sm text-gray-700 mb-3">
        Formats : <span className="font-medium">PPTX, DOCX, PDF</span>
      </p>

      <div className="border-2 border-dashed border-green-400 rounded-lg h-40 flex items-center justify-center hover:bg-green-100 transition relative">
        {!marketingFile ? (
          <label className="cursor-pointer flex flex-col items-center justify-center">
            <Upload className="text-green-600 mb-2" size={32} />
            <span className="font-medium text-center px-4">
              Déposer ou sélectionner un fichier
            </span>
            <input
              type="file"
              accept=".pptx,.docx,.pdf"
              className="hidden"
              onChange={(e) => setMarketingFile(e.target.files?.[0] || null)}
            />
          </label>
        ) : (
          <div className="flex items-center gap-3 px-4 py-2 bg-white rounded-lg shadow-sm border border-gray-300">
            <FileText className="text-green-600" size={20} />
            <span className="text-sm font-medium text-gray-800">
              {marketingFile.name} ({getExtension(marketingFile.name)})
            </span>
            <button
              onClick={(e) => {
                e.preventDefault();
                setMarketingFile(null);
              }}
              className="ml-auto bg-red-500 text-white rounded-full p-1 hover:bg-red-600"
            >
              <X size={16} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}