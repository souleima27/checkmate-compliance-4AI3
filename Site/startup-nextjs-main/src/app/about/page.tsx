import AboutSectionOne from "@/components/About/AboutSectionOne";
import AboutSectionTwo from "@/components/About/AboutSectionTwo";
import Breadcrumb from "@/components/Common/Breadcrumb";

import { Metadata } from "next";

export const metadata: Metadata = {
  title: "À propos | CheckMate",
  description: "Découvrez la mission de CheckMate : simplifier la conformité des documents marketing grâce à l’analyse intelligente et transparente.",
  // autres métadonnées si nécessaire
};

const AboutPage = () => {
  return (
    <>
      <Breadcrumb
        pageName="À propos"
        description="CheckMate est une plateforme conçue pour aider les équipes marketing, juridiques et conformité à analyser, vérifier et harmoniser leurs documents selon les règles réglementaires en vigueur."
      />
      <AboutSectionOne />
      <AboutSectionTwo />
    </>
  );
};

export default AboutPage;
