"use client";

export default function DashboardTable() {
  const documents = [
    { name: "Présentation Janvier.pdf", date: "08/12/2025", status: "Terminé" },
    { name: "Prospectus Fonds.docx", date: "07/12/2025", status: "Annoté" },
    { name: "Présentation ESG.pptx", date: "06/12/2025", status: "En cours" },
  ];

  return (
    <div className="bg-[var(--color-page-bg-alt)] border border-[var(--color-stroke-stroke)] rounded-xl shadow-sm p-6">
      <h2 className="text-xl font-semibold mb-4 text-black">Historique des documents</h2>
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="text-black">
            <th className="border-b p-3">Nom du fichier</th>
            <th className="border-b p-3">Date</th>
            <th className="border-b p-3">Statut</th>
            <th className="border-b p-3">Actions</th>
          </tr>
        </thead>
        <tbody>
          {documents.map((doc, i) => (
            <tr key={i} className="text-black">
              <td className="border-b p-3">{doc.name}</td>
              <td className="border-b p-3">{doc.date}</td>
              <td className="border-b p-3">{doc.status}</td>
              <td className="border-b p-3 space-x-2">
                <button className="px-3 py-1 bg-[var(--color-light-green)] rounded hover:bg-[var(--color-light-green-alt)]">
                  Revoir
                </button>
                <button className="px-3 py-1 bg-[var(--color-primary)] text-white rounded hover:bg-[var(--color-primary-dark)]">
                  Télécharger
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
