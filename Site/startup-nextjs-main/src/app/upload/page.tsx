// app/upload/page.tsx
import Breadcrumb from "@/components/Common/Breadcrumb";
import UploadMarketing from "@/components/Upload/UploadMarketing";
import UploadProspectus from "@/components/Upload/UploadProspectus";
import UploadActions from "@/components/Upload/UploadActions";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Import | CheckMate",
  description: "Importez vos documents marketing et prospectus pour une analyse IA conforme.",
};

export default function UploadPage() {
  return (
    <>
      <Breadcrumb
        pageName="Importer Documents "
        description="Importez votre présentation marketing et le prospectus associé. Les deux documents sont requis pour une analyse réglementaire complète par notre IA."
      />

      {/* Section marketing + prospectus */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-10 max-w-5xl mx-auto mt-10">
        <UploadMarketing />
        <UploadProspectus />
      </div>

      {/* Bouton d’analyse */}
      <UploadActions />
    </>
  );
}
