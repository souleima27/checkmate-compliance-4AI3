"use client";

import React from "react";
import { ScanSearch } from "lucide-react";

export default function UploadActions() {
  return (
    <div className="mt-12 text-center">
      <button
        className="px-8 py-4 flex items-center gap-3 mx-auto bg-green-700 text-white font-bold rounded-lg hover:bg-green-800 shadow transition"
      >
        <ScanSearch size={22} />
        Lancer l’analyse
      </button>
    </div>
  );
}
