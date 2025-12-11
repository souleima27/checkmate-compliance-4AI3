import Breadcrumb from "@/components/Common/Breadcrumb";
import ReviewPreview from "@/components/Review/ReviewPreview";
import ReviewSidebar from "@/components/Review/ReviewSidebar";
import ReviewMetrics from "@/components/Review/ReviewMetrics";
import ReviewViolations from "@/components/Review/ReviewViolations";
import ReviewReport from "@/components/Review/ReviewReport";
import ReviewActions from "@/components/Review/ReviewActions";
import ReviewChatbot from "@/components/Review/ReviewChatbot";

import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Analyse  | CheckMate",
  // autres métadonnées si nécessaire
};

export default function ReviewPage() {
  return (
    <>
      {/* Breadcrumb */}
      <Breadcrumb
        pageName="Analyse et Vérification"
        description="Consultez votre présentation, lancez la vérification, visualisez les violations détectées et générez le rapport final."
      />

      <div className="max-w-7xl mx-auto mt-10 px-4">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Aperçu de la présentation */}
          <div className="lg:col-span-2">
            <ReviewPreview />
            <ReviewMetrics />
            <ReviewViolations />
            <ReviewReport />
          </div>
          {/* Sidebar (options d'analyse) */}
          <div className="lg:col-span-1">
            <ReviewSidebar />
          </div>
        </div>

        {/* Boutons actions */}
        <ReviewActions />
        <ReviewChatbot />
      </div>
    </>
  );
}
