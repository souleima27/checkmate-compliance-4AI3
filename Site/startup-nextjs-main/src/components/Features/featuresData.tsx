import { Feature } from "@/types/feature";

const featuresData: Feature[] = [
  {
    id: 1,
    icon: <span className="text-4xl">📄</span>,
    title: "Analyse automatique",
    paragraph: "Transforme vos fichiers (PDF, PPTX, DOCX) en données structurées prêtes pour la vérification réglementaire."
  },
  {
    id: 2,
    icon: <span className="text-4xl">🕵️‍♀️</span>,
    title: "Détection des erreurs",
    paragraph: "Identifie les incohérences structurelles et signale les problèmes avant qu’ils ne deviennent critiques."
  },
  {
    id: 3,
    icon: <span className="text-4xl">🔗</span>,
    title: "Alignement des documents",
    paragraph: "Compare vos présentations et prospectus pour garantir une cohérence parfaite entre les deux."
  },
  {
    id: 4,
    icon: <span className="text-4xl">⚖️</span>,
    title: "Vérification réglementaire",
    paragraph: "Analyse vos contenus selon les règles légales et réglementaires en vigueur (Glossaire, Synthèse)."
  },
  {
    id: 5,
    icon: <span className="text-4xl">🧠</span>,
    title: "Analyse contextuelle",
    paragraph: "Détecte les erreurs liées au contexte, aux fonds et aux disclaimers pour une conformité totale."
  },
  {
    id: 6,
    icon: <span className="text-4xl">📊</span>,
    title: "Rapport final",
    paragraph: "Génère un rapport clair avec annotations et statistiques, prêt pour validation et archivage."
  },
];

export default featuresData;
