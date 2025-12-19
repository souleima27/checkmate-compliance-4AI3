"use client";

import React, { useState } from "react";
import { ScanSearch, Loader2, CheckCircle2, Circle } from "lucide-react";
import { useUploadStore } from "@/stores/uploadStore";
import { analyzeDocument, auditDocuments } from "@/services/api";
import { useReviewStore } from "@/stores/reviewStore";
import { useRouter } from "next/navigation";

export default function UploadActions() {
  // ✅ pull only metadata from store
  const { marketingFile, prospectusFile, metadata } = useUploadStore();
  const { startAnalysis, finishAnalysis } = useReviewStore();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progressStep, setProgressStep] = useState<number>(0);
  const router = useRouter();

  const steps = [
    "Initialisation",
    "Structure et contexte",
    "Fond, glossaire et disclaimers",
    "Système d'audit et assistant",
    "Génération du rapport"
  ];

  const handleAnalyze = async () => {
    if (!marketingFile) {
      alert("Veuillez sélectionner un document marketing (PPTX).");
      return;
    }

    setIsAnalyzing(true);
    setProgressStep(0);
    startAnalysis();

    try {
      // Step 1: Structural Analysis
      // Step 1: Structure & Context (Started)
      setProgressStep(1);

      // We run analysis and audit in parallel, but conceptually we map them to steps
      const analysisPromise = analyzeDocument(marketingFile, metadata).then(res => {
        // When analysis finishes, Structure(1) and Fond/Glos(2) are done.
        // We can move to Step 3 (Audit) if audit isn't done, or just mark progress.
        setProgressStep(prev => Math.max(prev, 3));
        return res;
      });

      let auditPromise = Promise.resolve(null);
      if (prospectusFile) {
        auditPromise = auditDocuments(marketingFile, prospectusFile).then(res => {
          // When audit finishes, Step 3 (Audit) is done.
          setProgressStep(prev => Math.max(prev, 4));
          return res;
        });
      } else {
        // If no prospectus, skip audit step visually
        setProgressStep(prev => Math.max(prev, 4));
      }

      const [result, auditResult] = await Promise.all([analysisPromise, auditPromise]);

      setProgressStep(4); // Generating report (Finalizing)

      console.log("🚀 Analysis result received:", result);
      if (result.version) {
        console.log("🔔 Backend version:", result.version);
      }
      console.log("Audit Result:", auditResult);

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
        else if (cat.includes("fond") || cat.includes("fund") || cat.includes("registration")) scope = "Fonds";
        else if (cat.includes("disclaimer") || cat.includes("legal")) scope = "Disclaimers";
        else if (cat.includes("prospectus") || cat.includes("reglementaire")) scope = "Fonds"; // Merging small prospectus issues into Fonds if any
        else scope = "Contexte";

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

      // ========================================
      // 4. Map DISCLAIMER & GLOSSARY Violations (from new_dis_glos)
      // ========================================
      const disGlosAnalysis = result.disclaimer_analysis || {};
      const resultsDetailles = disGlosAnalysis.resultats_detailles || {};
      const disGlosViolations: any[] = [];

      // A. Registration issues (Mapped to "Fonds")
      const regInfo = resultsDetailles.enregistrement || {};
      (regInfo.pays_manquants || []).forEach((p: any, i: number) => {
        disGlosViolations.push({
          id: `dis-reg-miss-${i}`,
          page: 0,
          scope: "Fonds",
          title: `Enregistrement manquant: ${p.pays_base || p.pays_document || "Inconnu"}`,
          description: p.raison || "Pays mentionné mais non enregistré dans la base de données centrale.",
          severity: "high",
          type: "violation",
          analysisType: "dis_glos"
        });
      });
      (regInfo.pays_exces || []).forEach((p: any, i: number) => {
        disGlosViolations.push({
          id: `dis-reg-exc-${i}`,
          page: 0,
          scope: "Fonds",
          title: `Pays non enregistré: ${p.pays_document || "Inconnu"}`,
          description: "Ce pays est mentionné dans le document mais n'est pas dans la liste officielle des pays de commercialisation.",
          severity: "medium",
          type: "violation",
          analysisType: "dis_glos"
        });
      });

      // B. Disclaimer issues (Restored category)
      const disInfo = resultsDetailles.disclaimers || {};
      (disInfo.obligatoires?.absents || []).forEach((d: any, i: number) => {
        disGlosViolations.push({
          id: `dis-miss-${i}`,
          page: 0,
          scope: "Disclaimers",
          title: `Disclaimer obligatoire absent: ${d.titre}`,
          description: d.raison || "Un disclaimer requis pour cette langue/cible n'a pas été détecté.",
          severity: "high",
          type: "missing",
          analysisType: "dis_glos"
        });
      });
      (disInfo.alignement?.non_conformes || []).forEach((d: any, i: number) => {
        disGlosViolations.push({
          id: `dis-align-${i}`,
          page: d.slide || 0,
          scope: "Disclaimers",
          title: "Alignement Disclaimer incorrect",
          description: d.raison || "Le texte du disclaimer ne semble pas correspondre au contenu de ce slide.",
          severity: "medium",
          type: "violation",
          analysisType: "dis_glos"
        });
      });

      // C. Glossary issues
      const glosInfo = resultsDetailles.glossaires || {};
      (glosInfo.obligatoires?.absents || []).forEach((g: any, i: number) => {
        let pageNum = 0;
        if (typeof glosInfo.glossary_slide === "number") pageNum = glosInfo.glossary_slide;

        disGlosViolations.push({
          id: `glos-miss-${i}`,
          page: pageNum,
          scope: "Glossaires",
          title: `Terme glossaire manquant: ${g.terme}`,
          description: g.raison || "Ce terme technique obligatoire n'est pas défini dans la section Glossaire.",
          severity: "low",
          type: "missing",
          analysisType: "dis_glos"
        });
      });

      // D. Source issues
      const srcInfo = resultsDetailles.sources || {};
      (srcInfo.non_conformes || []).forEach((s: any, i: number) => {
        disGlosViolations.push({
          id: `src-bad-${i}`,
          page: s.slide || 0,
          scope: "Contexte",
          title: "Qualité de source insuffisante",
          description: `${s.raison}. ${s.suggestion || ""}`,
          severity: "low",
          type: "violation",
          analysisType: "dis_glos"
        });
      });

      // E. Fund Characteristics issues
      const charInfo = resultsDetailles.caracteristiques || {};
      if (charInfo.completeness_score < 0.8) {
        (charInfo.missing_elements || []).forEach((m: any, i: number) => {
          disGlosViolations.push({
            id: `char-miss-${i}`,
            page: disGlosAnalysis.statistiques_globales?.slide_caracteristiques || 0,
            scope: "Fonds",
            title: `Caractéristique manquante: ${m}`,
            description: `L'élément "${m}" est manquant ou incomplet dans la présentation des caractéristiques du fonds.`,
            severity: "medium",
            type: "missing",
            analysisType: "dis_glos"
          });
        });
      }

      const finalViolations = [...allViolations, ...disGlosViolations];

      setProgressStep(3); // Finalizing

      finishAnalysis(
        finalViolations,
        result.analysis?.doc_structure || result.doc_structure,
        auditResult // ✅ Pass audit results
      );
      router.push("/review");
    } catch (error) {
      console.error("Analysis error:", error);
      alert("Erreur lors de l'analyse");
      setIsAnalyzing(false);
      setProgressStep(0);
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
      {isAnalyzing && (
        <div className="mb-6 max-w-md mx-auto bg-white p-6 rounded-xl shadow-lg border border-green-100">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Analyse en cours...</h3>
          <div className="space-y-4">
            {steps.map((step, index) => (
              <div key={index} className="flex items-center gap-3">
                {index < progressStep ? (
                  <CheckCircle2 className="text-green-600" size={20} />
                ) : index === progressStep ? (
                  <Loader2 className="animate-spin text-blue-600" size={20} />
                ) : (
                  <Circle className="text-gray-300" size={20} />
                )}
                <span className={`text-sm ${index === progressStep ? 'font-medium text-blue-700' : index < progressStep ? 'text-green-700' : 'text-gray-400'}`}>
                  {step}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <button
        onClick={handleAnalyze}
        disabled={isAnalyzing || !marketingFile}
        className={`px-8 py-4 flex items-center gap-3 mx-auto font-bold rounded-lg shadow transition ${isAnalyzing || !marketingFile
          ? "bg-gray-400 cursor-not-allowed"
          : "bg-green-700 text-white hover:bg-green-800"
          }`}
      >
        {isAnalyzing ? <Loader2 className="animate-spin" size={22} /> : <ScanSearch size={22} />}
        {isAnalyzing ? "Traitement..." : "Lancer l’analyse"}
      </button>
    </div>
  );
}
