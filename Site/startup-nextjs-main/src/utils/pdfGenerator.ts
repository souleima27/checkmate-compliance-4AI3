import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

interface Violation {
    id: string;
    scope: string;
    title: string;
    description: string;
    severity: "high" | "medium" | "low";
    page?: number;
}

export const generatePDFReport = (fileName: string, violations: Violation[]) => {
    const doc = new jsPDF();

    // Document Title
    doc.setFontSize(22);
    doc.setTextColor(33, 150, 83); // Green color
    doc.text("Rapport de Conformité", 14, 22);

    // Metadata
    doc.setFontSize(11);
    doc.setTextColor(100);
    doc.text(`Document: ${fileName}`, 14, 32);
    doc.text(`Date: ${new Date().toLocaleDateString()} à ${new Date().toLocaleTimeString()}`, 14, 38);
    doc.text(`Total Violations: ${violations.length}`, 14, 44);

    // Prepare Table Data
    const tableData = violations.map((v) => [
        v.page ? `Page ${v.page}` : "Global",
        v.scope || "N/A",
        v.title,
        v.severity.toUpperCase(),
        v.description
    ]);

    // Generate Table
    autoTable(doc, {
        startY: 55,
        head: [["Localisation", "Type", "Règle", "Gravité", "Détail"]],
        body: tableData,
        headStyles: {
            fillColor: [33, 150, 83], // Green header
            textColor: 255,
            fontSize: 10,
        },
        bodyStyles: {
            fontSize: 9,
        },
        alternateRowStyles: {
            fillColor: [240, 248, 240], // Light green alternate
        },
        columnStyles: {
            0: { cellWidth: 25 }, // Loc
            1: { cellWidth: 25 }, // Type
            2: { cellWidth: 40 }, // Règle
            3: { cellWidth: 20 }, // Gravité
            4: { cellWidth: 'auto' }, // Détail
        },
    });

    // Save the PDF
    doc.save(`Rapport_Conformite_${fileName.replace(/\.pptx$/i, "")}.pdf`);
};
