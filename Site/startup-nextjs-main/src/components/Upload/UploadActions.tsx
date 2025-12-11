"use client";

import React, { useState } from "react";
import { ScanSearch, Loader2 } from "lucide-react";
import { useUploadStore } from "@/stores/uploadStore";
import { analyzeDocument } from "@/services/api";
import { useReviewStore } from "@/stores/reviewStore";
import { useRouter } from "next/navigation";

export default function UploadActions() {
  // ✅ pull only metadata from store
  const { marketingFile, metadata } = useUploadStore();
  const { startAnalysis, finishAnalysis } = useReviewStore();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const router = useRouter();

  const handleAnalyze = async () => {
    if (!marketingFile) {
      alert("Veuillez sélectionner un document marketing.");
      return;
    }

    setIsAnalyzing(true);
    startAnalysis();

    try {
      // ✅ Pass metadata only to the API call
      const result = await analyzeDocument(marketingFile, metadata);
      console.log("Analysis Result:", result);

      // 1. Update File Info in Store
      let serverFileName = result.file_name || result.filename || marketingFile.name;
      let serverFileType: "pdf" | "pptx" | "docx" =
        serverFileName.toLowerCase().endsWith(".pdf") ? "pdf" : "pptx";

      if (result.converted_preview) {
        serverFileName = result.converted_preview;
        serverFileType = "pdf";
      }

      useReviewStore.getState().setFile({
        fileName: serverFileName,
        fileType: serverFileType,
        totalPages: 0,
      });

      // ========================================
      // 2. Map STRUCTURAL Violations (from doc_analyzer)
      // ========================================
      const rawViolations = result.analysis?.elements_non_conformes || result.issues || [];
      const rawMissing = result.analysis?.elements_manquants || [];

      function mapSeverity(sev: string): string {
        const s = (sev || "").toLowerCase();
        if (s === "haute" || s === "high" || s === "critical") return "high";
        if (s === "moyenne" || s === "medium") return "medium";
        if (s === "faible" || s === "low") return "low";
        return "medium";
      }

      function detectScope(regleId: string): string {
        if (regleId.startsWith("RS")) return "Structure";
        if (regleId.startsWith("RC")) return "Contexte";
        return "Contexte";
      }

      const mappedViolations = rawViolations.map((issue: any, index: number) => {
        const regleId = issue.regle_id || "";
        let scope = detectScope(regleId);

        const cat = (issue.categorie || issue.category || "").toLowerCase();
        if (cat.includes("structur")) scope = "Structure";
        else if (cat.includes("fond") || cat.includes("fund")) scope = "Fonds";
        else if (cat.includes("disclaimer") || cat.includes("legal")) scope = "Disclaimers";
        else if (cat.includes("prospectus") || cat.includes("reglementaire")) scope = "Prospectus";
        else if (cat.includes("registration")) scope = "Registration";

        const locationMatch = issue.location?.match(/Slide\s+(\d+)/i);
        const pageNum = locationMatch ? parseInt(locationMatch[1]) : 1;

        return {
          id: `struct-${index}`,
          page: issue.page || pageNum,
          scope,
          title: issue.element || issue.titre || "Non-Conformité Structurelle",
          description:
            issue.violation ||
            issue.observation ||
            issue.description ||
            issue.details ||
            "Description manquante",
          severity: mapSeverity(issue.gravite || issue.severity || "moyenne"),
          ruleId: regleId,
          location: issue.location || "",
          type: "violation" as const,
          analysisType: "structural" as const,
          bbox: issue.bbox
            ? { x: issue.bbox[0], y: issue.bbox[1], w: issue.bbox[2], h: issue.bbox[3] }
            : undefined,
        };
      });

      const mappedMissing = rawMissing.map((missing: any, index: number) => {
        const regleId = missing.regle_id || "";
        const scope = detectScope(regleId);

        return {
          id: `missing-${index}`,
          page: 0,
          scope,
          title: missing.element_requis || "Élément Manquant",
          description: `Élément requis manquant selon la règle ${regleId}`,
          severity: "high" as const,
          ruleId: regleId,
          location: "Document entier",
          type: "missing" as const,
          analysisType: "structural" as const,
          bbox: undefined,
        };
      });

      // ========================================
      // 3. Map CONTEXTUAL Violations (from theorist)
      // ========================================
      const contextualAnalysis = result.contextual_analysis || {};
      const complianceDetails = contextualAnalysis.compliance_details || [];

      const nonCompliantDetails = complianceDetails.filter(
        (detail: any) => detail.status === "non_compliant"
      );

      const mappedContextual = nonCompliantDetails.map((detail: any, index: number) => {
        const regleId = detail.rule_id || detail.regle_id || "";
        let pageNum = 0;
        const loc = detail.location || "";
        if (loc && loc !== "-") {
          const slideMatch = loc.match(/slide[_\s]?(\d+)/i);
          if (slideMatch) {
            pageNum = parseInt(slideMatch[1]);
          }
        }

        return {
          id: `ctx-${index}`,
          page: pageNum,
          scope: "Contexte",
          title: `Règle ${regleId}`,
          description:
            detail.evidence ||
            detail.observation ||
            detail.description ||
            "Violation contextuelle détectée",
          severity: mapSeverity(detail.severity || detail.gravite || "moyenne"),
          ruleId: regleId,
          location: detail.location || "",
          type: "violation" as const,
          analysisType: "contextual" as const,
          confidence: detail.confidence,
          bbox: undefined,
        };
      });

      const allViolations = [...mappedViolations, ...mappedMissing, ...mappedContextual];

      finishAnalysis(allViolations, result.analysis?.doc_structure || result.doc_structure);
      router.push("/review");
    } catch (error) {
      console.error("Analysis error:", error);
      alert("Erreur lors de l'analyse");
      setIsAnalyzing(false);
    } finally {
      if (
        typeof document !== "undefined" &&
        document.body.contains(document.getElementById("upload-actions"))
      ) {
        setIsAnalyzing(false);
      }
    }
  };

  return (
    <div id="upload-actions" className="mt-12 text-center">
      <button
        onClick={handleAnalyze}
        disabled={isAnalyzing || !marketingFile}
        className={`px-8 py-4 flex items-center gap-3 mx-auto font-bold rounded-lg shadow transition ${
          isAnalyzing || !marketingFile
            ? "bg-gray-400 cursor-not-allowed"
            : "bg-green-700 text-white hover:bg-green-800"
        }`}
      >
        {isAnalyzing ? <Loader2 className="animate-spin" size={22} /> : <ScanSearch size={22} />}
        {isAnalyzing ? "Analyse en cours..." : "Lancer l’analyse"}
      </button>
    </div>
  );
}
