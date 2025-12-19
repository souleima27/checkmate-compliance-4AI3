"use client";

import React, { useState } from "react";
import { UploadCloud, FileText, CheckCircle, AlertCircle, Loader2 } from "lucide-react";

export default function AuditPage() {
    const [pptxFile, setPptxFile] = useState<File | null>(null);
    const [pdfFile, setPdfFile] = useState<File | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, type: "pptx" | "pdf") => {
        if (e.target.files && e.target.files[0]) {
            if (type === "pptx") setPptxFile(e.target.files[0]);
            else setPdfFile(e.target.files[0]);
            setError(null);
        }
    };

    const handleAudit = async () => {
        if (!pptxFile || !pdfFile) {
            setError("Please upload both a PPTX presentation and a PDF reference document.");
            return;
        }

        setIsLoading(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        formData.append("pptx_file", pptxFile);
        formData.append("pdf_file", pdfFile);

        try {
            const response = await fetch("http://localhost:8000/api/audit", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Audit failed: ${response.statusText}`);
            }

            const data = await response.json();
            setResult(data);
        } catch (err: any) {
            console.error(err);
            setError(err.message || "An error occurred during the audit.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="p-8 max-w-7xl mx-auto space-y-8 animate-fade-in">
            <div className="flex flex-col gap-2">
                <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
                    <span className="bg-blue-600 text-white p-2 rounded-lg">
                        <CheckCircle size={24} />
                    </span>
                    Compliance Audit (Agent Checker)
                </h1>
                <p className="text-gray-500">
                    Upload a presentation and a reference document (PDF) to perform a deep compliance audit.
                </p>
            </div>

            {/* Upload Section */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* PPTX Upload */}
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200 flex flex-col items-center justify-center gap-4 hover:border-blue-400 transition-colors">
                    <div className="p-4 bg-orange-50 text-orange-600 rounded-full">
                        <FileText size={32} />
                    </div>
                    <div className="text-center">
                        <h3 className="font-semibold text-gray-700">Presentation (PPTX)</h3>
                        <p className="text-xs text-gray-400 mt-1">The marketing deck to audit</p>
                    </div>
                    <input
                        type="file"
                        accept=".pptx"
                        onChange={(e) => handleFileChange(e, "pptx")}
                        className="hidden"
                        id="pptx-upload"
                    />
                    <label
                        htmlFor="pptx-upload"
                        className="cursor-pointer px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition text-sm font-medium"
                    >
                        {pptxFile ? pptxFile.name : "Select PPTX File"}
                    </label>
                </div>

                {/* PDF Upload */}
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200 flex flex-col items-center justify-center gap-4 hover:border-blue-400 transition-colors">
                    <div className="p-4 bg-red-50 text-red-600 rounded-full">
                        <FileText size={32} />
                    </div>
                    <div className="text-center">
                        <h3 className="font-semibold text-gray-700">Reference Document (PDF)</h3>
                        <p className="text-xs text-gray-400 mt-1">Prospectus / KID / Factsheet</p>
                    </div>
                    <input
                        type="file"
                        accept=".pdf,.docx"
                        onChange={(e) => handleFileChange(e, "pdf")}
                        className="hidden"
                        id="pdf-upload"
                    />
                    <label
                        htmlFor="pdf-upload"
                        className="cursor-pointer px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition text-sm font-medium"
                    >
                        {pdfFile ? pdfFile.name : "Select PDF/DOCX File"}
                    </label>
                </div>
            </div>

            {/* Action Button */}
            <div className="flex justify-center">
                <button
                    onClick={handleAudit}
                    disabled={isLoading || !pptxFile || !pdfFile}
                    className={`px-8 py-4 rounded-xl text-lg font-bold flex items-center gap-3 transition-all ${isLoading || !pptxFile || !pdfFile
                            ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                            : "bg-blue-600 text-white hover:bg-blue-700 hover:scale-105 shadow-lg shadow-blue-200"
                        }`}
                >
                    {isLoading ? (
                        <>
                            <Loader2 className="animate-spin" /> Running Audit...
                        </>
                    ) : (
                        <>
                            <UploadCloud /> Start Compliance Audit
                        </>
                    )}
                </button>
            </div>

            {/* Error Message */}
            {error && (
                <div className="p-4 bg-red-50 text-red-700 rounded-xl flex items-center gap-3 border border-red-100">
                    <AlertCircle size={20} />
                    {error}
                </div>
            )}

            {/* Results Section */}
            {result && (
                <div className="space-y-6 animate-fade-in">
                    <div className="bg-green-50 border border-green-100 p-6 rounded-2xl">
                        <h2 className="text-xl font-bold text-green-800 mb-4 flex items-center gap-2">
                            <CheckCircle size={24} /> Audit Completed Successfully
                        </h2>
                        <p className="text-green-700 mb-4">
                            The agent has analyzed your documents. You can now use the <strong>Chat Assistant</strong> (bottom right) to ask specific questions about the audit or the documents.
                        </p>

                        <div className="grid grid-cols-3 gap-4">
                            {/* Metrics Cards */}
                            <div className="bg-white p-4 rounded-xl shadow-sm">
                                <p className="text-sm text-gray-500">Conformity Rate</p>
                                <p className="text-2xl font-bold text-blue-600">
                                    {result.global_metrics ? (result.global_metrics.conformity_rate * 100).toFixed(1) : "N/A"}%
                                </p>
                            </div>
                            <div className="bg-white p-4 rounded-xl shadow-sm">
                                <p className="text-sm text-gray-500">Avg Similarity</p>
                                <p className="text-2xl font-bold text-blue-600">
                                    {result.global_metrics ? (result.global_metrics.avg_similarity * 100).toFixed(1) : "N/A"}%
                                </p>
                            </div>
                            <div className="bg-white p-4 rounded-xl shadow-sm">
                                <p className="text-sm text-gray-500">Documents Processed</p>
                                <p className="text-2xl font-bold text-gray-700">2</p>
                            </div>
                        </div>
                    </div>

                    <div className="bg-white border border-gray-200 rounded-2xl overflow-hidden shadow-sm">
                        <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
                            <h3 className="font-bold text-gray-700">Raw Audit Report</h3>
                        </div>
                        <div className="p-6 max-h-96 overflow-y-auto bg-gray-50/50">
                            <pre className="text-xs text-gray-600 whitespace-pre-wrap">
                                {JSON.stringify(result, null, 2)}
                            </pre>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
